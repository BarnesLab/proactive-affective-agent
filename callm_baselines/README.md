# CALLM Baseline Models

Baseline models from the CALLM paper (CHI). These are the text-based and transformer-based baselines evaluated against CALLM.

## Files

### Primary (Monolithic)
- **`baseline_monolithic.py`** — Original all-in-one baseline file from `cancer_survival/code/baseline.py` (139KB). Contains `BaselineExperiment` class with methods for every model. Designed for multi-GPU training with hyperparameter tuning.

### Modular (Refactored)
- **`text_baselines.py`** — BoW, TF-IDF, LIWC (lexicon-based features + LogisticRegression)
- **`transformer_baselines.py`** — BERT, SentenceBERT, XLNet, RoBERTa, EmoBERT, RoBERTa-Emotion, DeBERTa-Emotion, DistilBERT-Emotion (`MultiHeadClassifier`)
- **`ml_pipeline.py`** — Traditional ML (SVM, RandomForest, LogisticRegression, Ridge, XGBoost)
- **`dl_baselines.py`** — MLP deep learning baselines
- **`combined_baselines.py`** — Late fusion and deep fusion (sensing + text)
- **`feature_builder.py`** — Feature extraction utilities

## Model → File Mapping

| Model (Paper Table) | Monolithic Method | Modular File |
|---|---|---|
| BoW | `evaluate_traditional_ml('BoW', CountVectorizer, ...)` | `text_baselines.py` |
| TF-IDF | `evaluate_traditional_ml('TF-IDF', TfidfVectorizer, ...)` | `text_baselines.py` |
| LIWC | *(not in monolithic)* | `text_baselines.py` |
| BERT | `evaluate_bert()` | `transformer_baselines.py` |
| SentenceBERT | `evaluate_sentence_transformer()` | `transformer_baselines.py` |
| XLNet | `evaluate_xlnet()` | `transformer_baselines.py` |
| EmoBERT | `evaluate_emobert()` | `transformer_baselines.py` |
| RoBERTa | `evaluate_roberta()` | `transformer_baselines.py` |
| RoBERTa-Emotion | `evaluate_roberta_emotion()` | `transformer_baselines.py` |
| DeBERTa-Emotion | `evaluate_deberta_emotion()` | `transformer_baselines.py` |
| Emotion-Transformer | `evaluate_emotion_transformer()` | *(monolithic only)* |

## HuggingFace Model IDs

| Model | HF Path |
|---|---|
| BERT | `bert-base-uncased` |
| SentenceBERT | `sentence-transformers/all-MiniLM-L6-v2` |
| XLNet | `xlnet-base-cased` |
| RoBERTa | `roberta-base` |
| EmoBERT | `bhadresh-savani/bert-base-go-emotion` / `monologg/bert-base-cased-goemotions-original` |
| RoBERTa-Emotion | `SamLowe/roberta-base-go_emotions` |
| DeBERTa-Emotion | `microsoft/deberta-base` |
| DistilBERT | `distilbert-base-uncased` |

## Dependencies
- torch, transformers, sentence-transformers
- sklearn, numpy, pandas
- vaderSentiment (monolithic only)
