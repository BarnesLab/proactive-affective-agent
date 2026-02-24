#\!/usr/bin/env python3
"""ML Baseline Runner: Traditional, Text, Deep Learning, Transformer models.

Comprehensive baseline comparison across multiple modalities and architectures.

Usage:
    # Run all modes (sensing, text, deep learning, transformers, combined)
    python scripts/run_ml_baselines.py --mode all

    # Run only sensing baselines
    python scripts/run_ml_baselines.py --mode sensing

    # Run text baselines with specific models
    python scripts/run_ml_baselines.py --mode text --models bow,tfidf,liwc

    # Run transformers on text
    python scripts/run_ml_baselines.py --mode transformers --models bert,roberta,sentencebert

    # Run deep learning on sensor features
    python scripts/run_ml_baselines.py --mode deep --models mlp,cnn

    # Run late fusion approaches
    python scripts/run_ml_baselines.py --mode combined --models late_ml_logistic,late_deep
"""
