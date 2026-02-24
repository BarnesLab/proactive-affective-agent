#!/usr/bin/env python3
"""Run traditional ML baselines (RF, XGBoost, LogisticRegression) on sensor features.

No LLM calls — uses existing 5-fold CV splits. Can run anytime without API limits.

Usage:
    # Run all models with daily features
    python scripts/run_ml_baselines.py

    # Specific models
    python scripts/run_ml_baselines.py --models rf,xgboost

    # Custom output directory
    python scripts/run_ml_baselines.py --output outputs/ml_baselines/
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import feature_builder
from src.baselines.ml_pipeline import MLBaselinePipeline
from src.data.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Run ML baselines on sensor features (no LLM calls)"
    )
    parser.add_argument(
        "--features", type=str, default="parquet",
        choices=["daily", "parquet"],
        help="Feature type: parquet=hourly Parquet files (default), daily=legacy aggregate CSVs"
    )
    parser.add_argument(
        "--models", type=str, default="rf,xgboost,logistic,ridge",
        help="Comma-separated model names: rf, xgboost, logistic, ridge"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: outputs/ml_baselines/)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory (default: data/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run only this fold (1-5). If not set, runs all 5 folds sequentially."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="n_jobs for GridSearchCV and RF. Set to 8 when running 5 folds in parallel."
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse models
    model_names = [m.strip() for m in args.models.split(",")]

    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "outputs" / "ml_baselines"
    # When running a single fold, isolate output to avoid concurrent write conflicts
    if args.fold is not None:
        output_dir = output_dir / f"fold_{args.fold}"

    # Load data
    loader = DataLoader(data_dir=data_dir)
    processed_hourly_dir = data_dir / "processed" / "hourly"

    # Choose feature source
    if args.features == "parquet":
        if not processed_hourly_dir.exists():
            logging.error(
                f"Parquet directory not found: {processed_hourly_dir}\n"
                "Run Phase 1 scripts first: bash scripts/offline/run_phase1.sh"
            )
            sys.exit(1)
        logging.info(f"Using Parquet features from {processed_hourly_dir}")
        sensing_dfs = None  # not needed for Parquet path
    else:
        sensing_dfs = loader.load_all_sensing()
        logging.info(f"Loaded {len(sensing_dfs)} sensing sources (legacy CSV)")

    # Run pipeline
    pipeline = MLBaselinePipeline(
        splits_dir=loader.splits_dir,
        sensing_dfs=sensing_dfs,
        feature_builder=feature_builder,
        output_dir=output_dir,
        model_names=model_names,
        processed_hourly_dir=processed_hourly_dir if args.features == "parquet" else None,
        n_jobs=args.n_jobs,
    )

    logging.info("Starting ML baseline pipeline...")
    logging.info(f"  Models: {model_names}")
    logging.info(f"  Features: {args.features}")
    logging.info(f"  Output: {output_dir}")
    if args.fold is not None:
        logging.info(f"  Fold: {args.fold} (single-fold mode)")

    folds = [args.fold] if args.fold is not None else None
    results = pipeline.run_all_folds(folds=folds)

    # Print summary
    print(f"\n{'='*60}")
    print("ML BASELINE RESULTS (5-fold CV)")
    print(f"{'='*60}")
    for model_name, targets in results.items():
        agg = targets.get("_aggregate", {})
        print(f"\n{model_name.upper()}:")
        if agg.get("mean_mae") is not None:
            print(f"  Mean MAE: {agg['mean_mae']:.3f}")
        if agg.get("mean_ba") is not None:
            print(f"  Mean Balanced Accuracy: {agg['mean_ba']:.3f}")
        if agg.get("mean_f1") is not None:
            print(f"  Mean F1: {agg['mean_f1']:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
