"""ML baselines: Traditional, Text-based, Deep Learning, and Transformer models.

Provides:
- MLBaselinePipeline: Traditional ML (RF, XGBoost, LogisticRegression) on sensor features
- TextBaselinePipeline: Text-based features (BoW, TF-IDF, LIWC, Temporal)
- DLBaselinePipeline: Deep learning models (MLP, CNN) on sensor features
- TransformerBaselinePipeline: Pre-trained transformers (BERT, RoBERTa, etc.) on text
- CombinedBaselinePipeline: Late fusion of ML and deep learning
- feature_builder: Shared utilities for feature extraction
"""

from src.baselines.ml_pipeline import MLBaselinePipeline

try:
    from src.baselines.text_baselines import TextBaselinePipeline
except ImportError:
    TextBaselinePipeline = None

try:
    from src.baselines.deep_learning_baselines import DLBaselinePipeline
except ImportError:
    DLBaselinePipeline = None

try:
    from src.baselines.transformer_baselines import TransformerBaselinePipeline
except ImportError:
    TransformerBaselinePipeline = None

try:
    from src.baselines.combined_baselines import CombinedBaselinePipeline
except ImportError:
    CombinedBaselinePipeline = None

from src.baselines import feature_builder

__all__ = [
    "MLBaselinePipeline",
    "TextBaselinePipeline",
    "DLBaselinePipeline",
    "TransformerBaselinePipeline",
    "CombinedBaselinePipeline",
    "feature_builder",
]
