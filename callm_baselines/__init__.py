"""Baselines: traditional ML, text-only, transformer, DL, and combined models.

Submodules:
- feature_builder: Sensing feature extraction (20 daily aggregates).
- ml_pipeline: RF, XGBoost, LogReg, Ridge on sensing features.
- text_baselines: Majority, Temporal, BoW, TF-IDF, LIWC + LogReg on text.
- transformer_baselines: BERT, RoBERTa, XLNet, etc. fine-tuned on text (requires torch).
- dl_baselines: MLP on sensing features (requires torch).
- combined_baselines: Late fusion and deep fusion (deep fusion requires torch).
"""

from src.baselines.feature_builder import (
    DAILY_FEATURE_NAMES,
    build_daily_features,
    impute_features,
)
from src.baselines.ml_pipeline import MLBaseline, MLBaselinePipeline
from src.baselines.text_baselines import (
    MajorityBaseline,
    TemporalBaseline,
    TextMLBaseline,
    run_text_baselines,
)

__all__ = [
    # Feature building
    "DAILY_FEATURE_NAMES",
    "build_daily_features",
    "impute_features",
    # ML baselines (sensing)
    "MLBaseline",
    "MLBaselinePipeline",
    # Text baselines
    "MajorityBaseline",
    "TemporalBaseline",
    "TextMLBaseline",
    "run_text_baselines",
]

# Optional imports (require torch/transformers) -- only attempt if torch is installed
try:
    import torch as _torch  # noqa: F401

    from src.baselines.transformer_baselines import (
        TransformerBaseline,
        run_transformer_baselines,
    )
    from src.baselines.dl_baselines import MLPBaseline, run_dl_baselines
    from src.baselines.combined_baselines import (
        LateFusionBaseline,
        DeepFusionBaseline,
        run_combined_baselines,
    )

    __all__ += [
        "TransformerBaseline",
        "run_transformer_baselines",
        "MLPBaseline",
        "run_dl_baselines",
        "LateFusionBaseline",
        "DeepFusionBaseline",
        "run_combined_baselines",
    ]
except ImportError:
    pass
