"""V1 Structured Workflow: sensing data → memory → single LLM call → prediction.

For the pilot: simplified from 4-step to a single LLM call with all context
pre-assembled. The LLM does step-by-step reasoning within one call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.think.prompts import (
    build_trait_summary,
    format_sensing_summary,
    v1_prompt,
    v1_system_prompt,
)

logger = logging.getLogger(__name__)

# Sensing features used for peer fingerprint matching
_SENSING_FEATURE_COLS = [
    "screen_total_min", "screen_n_sessions",
    "motion_stationary_min", "motion_walking_min", "motion_automotive_min",
    "motion_tracked_hours",
    "app_total_min", "app_social_min", "app_comm_min", "app_entertainment_min",
    "keyboard_n_sessions", "keyboard_total_chars",
    "light_mean_lux_raw",
]


class StructuredWorkflow:
    """V1: Fixed pipeline — assemble context, single LLM call, parse output."""

    def __init__(
        self,
        llm_client,
        peer_db_path: str | None = None,
        study_id: int | None = None,
    ) -> None:
        self.llm = llm_client
        self.study_id = study_id
        self._peer_db: pd.DataFrame | None = None
        self._peer_sensing_matrix = None
        self._peer_sensing_features: list[str] = []

        if peer_db_path and Path(peer_db_path).exists():
            self._load_peer_db(peer_db_path)

    def _load_peer_db(self, path: str) -> None:
        """Load peer database and build sensing fingerprint index."""
        db = pd.read_parquet(path)
        # Exclude current participant
        if self.study_id is not None and "Study_ID" in db.columns:
            db = db[db["Study_ID"] != self.study_id].reset_index(drop=True)
        if db.empty:
            return
        self._peer_db = db

        available = [c for c in _SENSING_FEATURE_COLS if c in db.columns]
        if available:
            self._peer_sensing_features = available
            data = db[available].fillna(0).values.astype(np.float64)
            means = data.mean(axis=0)
            stds = data.std(axis=0)
            stds[stds == 0] = 1.0
            normed = (data - means) / stds
            self._peer_sensing_matrix = np.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
            self._peer_means = means
            self._peer_stds = stds

    def _get_peer_sensing_reference(self, sensing_day, top_k: int = 5) -> str:
        """Find peer cases with similar sensing patterns and return formatted reference."""
        if self._peer_db is None or self._peer_sensing_matrix is None:
            return ""
        if sensing_day is None:
            return ""

        # Build current sensing vector from SensingDay
        data = sensing_day.to_summary_dict() if sensing_day else {}
        if not data:
            return ""

        # Map SensingDay fields to filtered feature names
        field_map = {
            "screen_total_min": data.get("screen_minutes", 0),
            "screen_n_sessions": data.get("screen_sessions", 0),
            "motion_stationary_min": data.get("stationary_min", 0),
            "motion_walking_min": data.get("walking_min", 0),
            "motion_automotive_min": data.get("automotive_min", 0),
            "motion_tracked_hours": 0,
            "app_total_min": (data.get("total_app_seconds", 0) or 0) / 60,
            "app_social_min": 0,
            "app_comm_min": 0,
            "app_entertainment_min": 0,
            "keyboard_n_sessions": 0,
            "keyboard_total_chars": data.get("words_typed", 0),
            "light_mean_lux_raw": 0,
        }

        vector = np.array(
            [float(field_map.get(c, 0) or 0) for c in self._peer_sensing_features],
            dtype=np.float64,
        )

        # Skip if vector is all zeros (no sensing data)
        if np.allclose(vector, 0):
            return ""

        # Normalize using peer stats
        norm_vec = ((vector - self._peer_means) / self._peer_stds).reshape(1, -1)
        # Replace any NaN/inf from normalization
        norm_vec = np.nan_to_num(norm_vec, nan=0.0, posinf=0.0, neginf=0.0)

        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(norm_vec, self._peer_sensing_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        lines = ["## Similar Cases from Other Participants (by behavioral patterns, with ground truth)",
                  "These are days from other cancer survivors with similar sensing patterns and their actual outcomes:\n"]
        for rank, idx in enumerate(top_idx, 1):
            if sims[idx] <= 0:
                break
            row = self._peer_db.iloc[idx]
            lines.append(f"Peer Case {rank} (similarity: {sims[idx]:.3f}):")
            # Sensing features
            parts = []
            for feat in self._peer_sensing_features:
                val = row.get(feat)
                if val is not None and pd.notna(val) and val != 0:
                    parts.append(f"{feat}={val:.1f}")
            if parts:
                lines.append(f"  Behavior: {'; '.join(parts)}")
            # Ground truth outcomes
            outcome_parts = []
            pa = row.get("PANAS_Pos")
            if pa is not None and pd.notna(pa):
                outcome_parts.append(f"PA={pa:.1f}")
            na = row.get("PANAS_Neg")
            if na is not None and pd.notna(na):
                outcome_parts.append(f"NA={na:.1f}")
            er = row.get("ER_desire")
            if er is not None and pd.notna(er):
                outcome_parts.append(f"ER={er:.1f}")
            if outcome_parts:
                lines.append(f"  Outcomes: {', '.join(outcome_parts)}")
            lines.append("")

        return "\n".join(lines) if len(lines) > 2 else ""

    def run(
        self,
        sensing_day,
        memory_doc: str,
        profile,
        date_str: str = "",
    ) -> dict[str, Any]:
        """Execute V1 prediction pipeline.

        Args:
            sensing_day: SensingDay dataclass (or None).
            memory_doc: Pre-generated memory document text.
            profile: UserProfile dataclass.
            date_str: Date string for context.

        Returns:
            Dict with predictions + metadata.
        """
        # Step 1: Format sensing data
        sensing_summary = format_sensing_summary(sensing_day)

        # Step 1b: Get peer sensing reference (cross-user)
        peer_ref = self._get_peer_sensing_reference(sensing_day)

        # Step 2: Build prompt with all context
        trait_text = build_trait_summary(profile)
        prompt = v1_prompt(
            sensing_summary=sensing_summary,
            memory_doc=memory_doc,
            trait_profile=trait_text,
            date_str=date_str,
            peer_reference=peer_ref,
        )
        system = v1_system_prompt()

        # Step 3: Single LLM call
        logger.debug("V1: Calling LLM with sensing + memory context")
        raw_response = self.llm.generate(prompt=prompt, system_prompt=system)
        usage = getattr(self.llm, "last_usage", {})
        from src.think.parser import parse_prediction
        result = parse_prediction(raw_response)

        # Comprehensive trace
        result["_version"] = "v1"
        result["_model"] = self.llm.model
        result["_prompt_length"] = len(prompt) + len(system)
        result["_sensing_summary"] = sensing_summary
        result["_full_prompt"] = prompt
        result["_system_prompt"] = system
        result["_full_response"] = raw_response
        result["_memory_excerpt"] = memory_doc[:500] if memory_doc else ""
        result["_trait_summary"] = trait_text
        result["_input_tokens"] = usage.get("input_tokens", 0)
        result["_output_tokens"] = usage.get("output_tokens", 0)
        result["_total_tokens"] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        result["_cost_usd"] = usage.get("cost_usd", 0)
        result["_llm_calls"] = 1

        logger.info(
            f"V1: tokens={usage.get('input_tokens', '?')}in+"
            f"{usage.get('output_tokens', '?')}out, "
            f"confidence={result.get('confidence', '?')}"
        )

        return result
