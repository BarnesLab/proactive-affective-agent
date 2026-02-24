#!/usr/bin/env python3
"""Run all four advanced baseline pipelines and print a combined summary table.

Pipelines:
  - TextBaselinePipeline    : TF-IDF + BoW on diary text (emotion_driver)
  - DLBaselinePipeline      : MLP on hourly Parquet features (requires PyTorch)
  - TransformerBaselinePipeline : Sentence-transformer embeddings on diary text
  - CombinedBaselinePipeline : Parquet features + sentence-transformer embeddings (late fusion)

Usage:
    python scripts/run_dl_baselines.py [--splits-dir PATH] [--output PATH] [--hourly-dir PATH]
    python scripts/run_dl_baselines.py --pipelines text,transformer  # run subset

Results are saved to outputs/advanced_baselines/<pipeline_name>/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _print_summary(pipeline_name: str, results: dict) -> None:
    """Print a compact per-pipeline summary table."""
    print(f"\n{'='*64}")
    print(f"  {pipeline_name}")
    print(f"{'='*64}")
    print(f"  {'Model':<20} {'Mean MAE':>10} {'Mean BA':>10} {'Mean F1':>10}")
    print(f"  {'-'*52}")
    for model_name, targets in results.items():
        agg = targets.get("_aggregate", {})
        mae_str = f"{agg['mean_mae']:.3f}" if agg.get("mean_mae") is not None else "  —  "
        ba_str  = f"{agg['mean_ba']:.3f}"  if agg.get("mean_ba")  is not None else "  —  "
        f1_str  = f"{agg['mean_f1']:.3f}"  if agg.get("mean_f1")  is not None else "  —  "
        print(f"  {model_name:<20} {mae_str:>10} {ba_str:>10} {f1_str:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run advanced (DL / text / transformer / combined) baselines"
    )
    parser.add_argument(
        "--pipelines", type=str, default="text,dl,transformer,combined",
        help="Comma-separated list of pipelines to run: text, dl, transformer, combined"
    )
    parser.add_argument(
        "--splits-dir", type=str, default=None,
        help="Path to EMA splits directory (default: data/processed/splits/)"
    )
    parser.add_argument(
        "--hourly-dir", type=str, default=None,
        help="Path to processed hourly Parquet directory (default: data/processed/hourly/)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Base output directory (default: outputs/advanced_baselines/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run only this fold (1-5). If not set, runs all 5 folds. Use for parallel fold execution."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Paths
    splits_dir = Path(args.splits_dir) if args.splits_dir else PROJECT_ROOT / "data" / "processed" / "splits"
    hourly_dir = Path(args.hourly_dir) if args.hourly_dir else PROJECT_ROOT / "data" / "processed" / "hourly"
    base_output = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "advanced_baselines"

    pipelines_to_run = [p.strip().lower() for p in args.pipelines.split(",")]
    folds = [args.fold] if args.fold is not None else None
    fold_suffix = f"/fold_{args.fold}" if args.fold is not None else ""

    all_summaries: dict[str, dict] = {}

    # --- Text baseline ---
    if "text" in pipelines_to_run:
        logging.info("Running TextBaselinePipeline...")
        try:
            from src.baselines.text_baselines import TextBaselinePipeline
            pipeline = TextBaselinePipeline(
                splits_dir=splits_dir,
                output_dir=base_output / f"text{fold_suffix}",
            )
            results = pipeline.run_all_folds(folds=folds)
            all_summaries["Text (TF-IDF / BoW)"] = results
            _print_summary("Text Baseline (TF-IDF / BoW)", results)
        except Exception as e:
            logging.error(f"TextBaselinePipeline failed: {e}")

    # --- Deep learning (MLP) baseline ---
    if "dl" in pipelines_to_run:
        logging.info("Running DLBaselinePipeline (MLP)...")
        try:
            from src.baselines.deep_learning_baselines import DLBaselinePipeline
            pipeline = DLBaselinePipeline(
                splits_dir=splits_dir,
                output_dir=base_output / f"dl{fold_suffix}",
                processed_hourly_dir=hourly_dir,
            )
            results = pipeline.run_all_folds(folds=folds)
            all_summaries["Deep Learning (MLP)"] = results
            _print_summary("Deep Learning Baseline (MLP)", results)
        except ImportError as e:
            logging.warning(f"DLBaselinePipeline skipped (missing dependency): {e}")
        except Exception as e:
            logging.error(f"DLBaselinePipeline failed: {e}")

    # --- Transformer (sentence-transformer) baseline ---
    if "transformer" in pipelines_to_run:
        logging.info("Running TransformerBaselinePipeline...")
        try:
            from src.baselines.transformer_baselines import TransformerBaselinePipeline
            pipeline = TransformerBaselinePipeline(
                splits_dir=splits_dir,
                output_dir=base_output / f"transformer{fold_suffix}",
            )
            results = pipeline.run_all_folds(folds=folds)
            all_summaries["Transformer (MiniLM)"] = results
            _print_summary("Transformer Baseline (MiniLM + Ridge/LR)", results)
        except ImportError as e:
            logging.warning(f"TransformerBaselinePipeline skipped (missing dependency): {e}")
        except Exception as e:
            logging.error(f"TransformerBaselinePipeline failed: {e}")

    # --- Combined (late fusion) baseline ---
    if "combined" in pipelines_to_run:
        logging.info("Running CombinedBaselinePipeline...")
        try:
            from src.baselines.combined_baselines import CombinedBaselinePipeline
            pipeline = CombinedBaselinePipeline(
                splits_dir=splits_dir,
                output_dir=base_output / f"combined{fold_suffix}",
                processed_hourly_dir=hourly_dir,
            )
            results = pipeline.run_all_folds(folds=folds)
            all_summaries["Combined (sensor + text)"] = results
            _print_summary("Combined Baseline (Parquet + Transformer, late fusion)", results)
        except ImportError as e:
            logging.warning(f"CombinedBaselinePipeline skipped (missing dependency): {e}")
        except Exception as e:
            logging.error(f"CombinedBaselinePipeline failed: {e}")

    # --- Master summary ---
    if all_summaries:
        print(f"\n{'='*64}")
        print("  COMBINED SUMMARY (mean across all models in each pipeline)")
        print(f"{'='*64}")
        print(f"  {'Pipeline':<30} {'Best MAE':>10} {'Best BA':>10} {'Best F1':>10}")
        print(f"  {'-'*56}")
        for pipeline_label, results in all_summaries.items():
            best_mae, best_ba, best_f1 = None, None, None
            for model_name, targets in results.items():
                agg = targets.get("_aggregate", {})
                if agg.get("mean_mae") is not None:
                    if best_mae is None or agg["mean_mae"] < best_mae:
                        best_mae = agg["mean_mae"]
                if agg.get("mean_ba") is not None:
                    if best_ba is None or agg["mean_ba"] > best_ba:
                        best_ba = agg["mean_ba"]
                if agg.get("mean_f1") is not None:
                    if best_f1 is None or agg["mean_f1"] > best_f1:
                        best_f1 = agg["mean_f1"]
            mae_s = f"{best_mae:.3f}" if best_mae is not None else "  —  "
            ba_s  = f"{best_ba:.3f}"  if best_ba  is not None else "  —  "
            f1_s  = f"{best_f1:.3f}"  if best_f1  is not None else "  —  "
            print(f"  {pipeline_label:<30} {mae_s:>10} {ba_s:>10} {f1_s:>10}")

    print(f"\nAll results saved under: {base_output}")


if __name__ == "__main__":
    main()
