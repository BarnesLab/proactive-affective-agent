"""PilotSimulator: runs the pilot study for CALLM, V1, V2, V3, V4.

Iterates users -> EMA entries chronologically -> calls agent.predict() ->
collects results. Supports checkpointing, dry-run mode, and per-user output.
No group dependency: uses combined test data from all 5 splits.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.agent.personal_agent import PersonalAgent
from src.data.loader import DataLoader
from src.data.preprocessing import align_sensing_to_ema, get_user_trait_profile, prepare_pilot_data
from src.evaluation.metrics import compute_all
from src.evaluation.reporter import Reporter
from src.remember.retriever import MultiModalRetriever, TFIDFRetriever
from src.think.llm_client import ClaudeCodeClient

logger = logging.getLogger(__name__)


class PilotSimulator:
    """Runs the pilot study: N users x all EMA entries x specified versions."""

    def __init__(
        self,
        loader: DataLoader,
        output_dir: Path,
        pilot_user_ids: list[int] | None = None,
        dry_run: bool = False,
        model: str = "sonnet",
        delay: float = 2.0,
    ) -> None:
        self.loader = loader
        self.output_dir = output_dir
        self.pilot_user_ids = pilot_user_ids
        self.dry_run = dry_run
        self.model = model
        self.delay = delay

        self.reporter = Reporter(output_dir)

        # Will be populated during setup
        self._users_data: list[dict] = []
        self._sensing_dfs: dict[str, pd.DataFrame] = {}
        self._all_ema: pd.DataFrame | None = None
        self._train_df: pd.DataFrame | None = None
        self._retriever: TFIDFRetriever | None = None
        self._mm_retriever: MultiModalRetriever | None = None

    def setup(self) -> None:
        """Load and prepare all data for the pilot."""
        logger.info("Loading pilot data...")

        # Load all EMA data (combined from 5 test splits)
        self._all_ema = self.loader.load_all_ema()
        logger.info(f"Loaded {len(self._all_ema)} EMA entries, {self._all_ema['Study_ID'].nunique()} users")

        # Load sensing data
        self._sensing_dfs = self.loader.load_all_sensing()
        logger.info(f"Loaded {len(self._sensing_dfs)} sensing sources")

        # Load training data for TF-IDF retriever (CALLM)
        self._train_df = self.loader.load_all_train()
        logger.info(f"Loaded training data: {len(self._train_df)} entries")

        # Build TF-IDF retriever (for CALLM)
        self._retriever = TFIDFRetriever()
        self._retriever.fit(self._train_df)
        logger.info("TF-IDF retriever fitted")

        # Build MultiModal retriever (for V3/V4: diary search → return diary + sensing)
        self._mm_retriever = MultiModalRetriever()
        self._mm_retriever.fit(self._train_df, sensing_dfs=self._sensing_dfs)
        logger.info("MultiModal retriever fitted")

        # Determine pilot users
        if self.pilot_user_ids is None:
            # Auto-select top 5 by EMA count
            counts = self._all_ema.groupby("Study_ID").size()
            self.pilot_user_ids = counts.nlargest(5).index.tolist()

        # Prepare per-user data (all EMA entries, no cap)
        self._users_data = prepare_pilot_data(
            self.loader,
            pilot_user_ids=self.pilot_user_ids,
            ema_df=self._all_ema,
        )
        logger.info(f"Prepared data for {len(self._users_data)} pilot users")
        for ud in self._users_data:
            logger.info(f"  User {ud['study_id']}: {len(ud['ema_entries'])} EMA entries")

    def run_version(self, version: str) -> dict[str, Any]:
        """Run the pilot for a single version (callm/v1/v2).

        Supports resuming from checkpoint. Each entry is checkpointed immediately.

        Returns:
            Dict with predictions, ground_truths, metadata, metrics.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {version.upper()} on {len(self._users_data)} users")
        logger.info(f"{'='*60}")

        # Try to resume from checkpoint
        all_predictions, all_ground_truths, all_metadata, resume_user, resume_entry = \
            self._load_checkpoint(version)

        per_user_metrics = {}

        llm_client = ClaudeCodeClient(
            model=self.model,
            dry_run=self.dry_run,
            delay_between_calls=self.delay,
        )

        for user_data in self._users_data:
            sid = user_data["study_id"]
            n_entries = len(user_data["ema_entries"])

            # Skip fully completed users when resuming
            if resume_user is not None and sid < resume_user:
                logger.info(f"\n--- User {sid}: skipped (already completed) ---")
                continue

            logger.info(f"\n--- User {sid} ({version.upper()}, {n_entries} entries) ---")

            # Select retriever based on version
            if version == "callm":
                retriever = self._retriever
            elif version in ("v3", "v4"):
                retriever = self._mm_retriever
            else:
                retriever = None

            agent = PersonalAgent(
                study_id=sid,
                version=version,
                llm_client=llm_client,
                profile=user_data["profile"],
                memory_doc=user_data["memory"],
                retriever=retriever,
            )

            user_preds = []
            user_gts = []
            user_meta = []

            for i, (ema_row, sensing_day) in enumerate(
                zip(user_data["ema_entries"], user_data["sensing_days"])
            ):
                # Skip already-processed entries when resuming
                if resume_user is not None and sid == resume_user and i <= resume_entry:
                    continue

                date_str = str(ema_row.get("date_local", ""))

                logger.info(f"  Entry {i+1}/{n_entries} ({date_str})")

                try:
                    pred = agent.predict(
                        ema_row=ema_row,
                        sensing_day=sensing_day,
                        date_str=date_str,
                        sensing_dfs=self._sensing_dfs if version in ("v2",) else None,
                    )
                except Exception as e:
                    logger.error(f"  Error predicting: {e}")
                    pred = {"_error": str(e)}

                # Extract ground truth
                gt = _extract_ground_truth(ema_row)

                # Save trace immediately
                trace_data = {k: v for k, v in pred.items() if k.startswith("_")}
                if trace_data:
                    self.reporter.save_trace(trace_data, sid, i, version)

                # Clean prediction for evaluation
                clean_pred = {k: v for k, v in pred.items() if not k.startswith("_")}

                user_preds.append(clean_pred)
                user_gts.append(gt)
                user_meta.append({"study_id": sid, "entry_idx": i, "date": date_str})

                all_predictions.append(clean_pred)
                all_ground_truths.append(gt)
                all_metadata.append({"study_id": sid, "entry_idx": i, "date": date_str})

                # Checkpoint after EVERY entry
                self._checkpoint(version, all_predictions, all_ground_truths, all_metadata,
                                 current_user=sid, current_entry=i)

            # Clear resume state after first resumed user is done
            resume_user = None
            resume_entry = -1

            # Per-user metrics
            if user_preds:
                try:
                    per_user_metrics[sid] = compute_all(user_preds, user_gts, user_meta)
                except Exception as e:
                    logger.warning(f"  Metrics error for user {sid}: {e}")

            logger.info(f"  User {sid} done. Total LLM calls: {llm_client.call_count}")

        # Save final outputs
        self.reporter.save_predictions_csv(
            all_predictions, all_ground_truths, all_metadata,
            f"{version}_predictions.csv",
        )

        # Compute overall metrics (with metadata for personal threshold)
        overall_metrics = {}
        if all_predictions:
            try:
                overall_metrics = compute_all(all_predictions, all_ground_truths, all_metadata)
            except Exception as e:
                logger.error(f"Overall metrics error: {e}")

        return {
            "version": version,
            "predictions": all_predictions,
            "ground_truths": all_ground_truths,
            "metadata": all_metadata,
            "metrics": overall_metrics,
            "per_user_metrics": per_user_metrics,
            "total_llm_calls": llm_client.call_count,
        }

    def run_all(self, versions: list[str] | None = None) -> dict[str, Any]:
        """Run all specified versions sequentially and generate comparison report.

        Args:
            versions: List of versions to run. Default: ["callm", "v1", "v2"].

        Returns:
            Dict with per-version results and comparison metrics.
        """
        if versions is None:
            versions = ["callm", "v1", "v2", "v3", "v4"]

        results = {}
        version_metrics = {}

        for version in versions:
            result = self.run_version(version)
            results[version] = result
            if result["metrics"]:
                version_metrics[version] = result["metrics"]

        # Generate comparison outputs
        if len(version_metrics) > 1:
            self.reporter.save_comparison_table(version_metrics)
            self.reporter.save_metrics_json(version_metrics, "comparison_metrics.json")

        # Save per-user metrics
        per_user = {v: r.get("per_user_metrics", {}) for v, r in results.items()}
        self.reporter.save_metrics_json(per_user, "per_user_metrics.json")

        logger.info(f"\n{'='*60}")
        logger.info("PILOT COMPLETE")
        logger.info(f"Output directory: {self.output_dir}")
        for v, r in results.items():
            logger.info(f"  {v}: {r['total_llm_calls']} LLM calls")
        logger.info(f"{'='*60}")

        return results

    def _checkpoint(
        self,
        version: str,
        predictions: list[dict],
        ground_truths: list[dict],
        metadata: list[dict],
        current_user: int | None = None,
        current_entry: int | None = None,
    ) -> None:
        """Save intermediate checkpoint with resume info (per-user-per-version)."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        # Per-user checkpoint to support parallel user processing
        suffix = f"_user{current_user}" if current_user is not None else ""
        path = checkpoint_dir / f"{version}{suffix}_checkpoint.json"
        with open(path, "w") as f:
            json.dump({
                "version": version,
                "n_entries": len(predictions),
                "current_user": current_user,
                "current_entry": current_entry,
                "predictions": predictions,
                "ground_truths": ground_truths,
                "metadata": metadata,
            }, f, default=str)

    def _load_checkpoint(self, version: str) -> tuple:
        """Load checkpoint for resume. Returns (preds, gts, meta, resume_user, resume_entry).

        Loads all per-user checkpoints and merges them.
        """
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return [], [], [], None, -1

        # Look for per-user checkpoints
        all_preds = []
        all_gts = []
        all_meta = []
        last_user = None
        last_entry = -1

        import glob
        pattern = str(checkpoint_dir / f"{version}_user*_checkpoint.json")
        files = sorted(glob.glob(pattern))

        if not files:
            # Try legacy single checkpoint
            legacy = checkpoint_dir / f"{version}_checkpoint.json"
            if legacy.exists():
                files = [str(legacy)]

        for fpath in files:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                all_preds.extend(data.get("predictions", []))
                all_gts.extend(data.get("ground_truths", []))
                all_meta.extend(data.get("metadata", []))
                cu = data.get("current_user")
                ce = data.get("current_entry", -1)
                if cu is not None:
                    if last_user is None or cu > last_user or (cu == last_user and ce > last_entry):
                        last_user = cu
                        last_entry = ce
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {fpath}: {e}")

        if all_preds and last_user is not None:
            logger.info(f"Resuming {version}: {len(all_preds)} entries from checkpoint, last user={last_user} entry={last_entry}")
            return all_preds, all_gts, all_meta, last_user, last_entry

        return [], [], [], None, -1


def _extract_ground_truth(ema_row) -> dict[str, Any]:
    """Extract ground truth targets from an EMA row (pandas Series)."""
    from src.utils.mappings import BINARY_STATE_TARGETS, CONTINUOUS_TARGETS

    gt = {}

    # Continuous
    for target in CONTINUOUS_TARGETS:
        val = ema_row.get(target)
        if pd.notna(val):
            gt[target] = float(val)
        else:
            gt[target] = None

    # Binary states
    for target in BINARY_STATE_TARGETS:
        val = ema_row.get(target)
        if pd.notna(val):
            if isinstance(val, bool):
                gt[target] = val
            elif isinstance(val, str):
                gt[target] = val.lower().strip() in ("true", "1", "yes")
            else:
                gt[target] = bool(val)
        else:
            gt[target] = None

    # Availability
    avail = ema_row.get("INT_availability")
    gt["INT_availability"] = str(avail).lower().strip() if pd.notna(avail) else None

    return gt
