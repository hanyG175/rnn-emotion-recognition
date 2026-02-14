# Emotion Recognition Models:

## RNN-Based Emotion Classification (PyTorch)

A modular, reproducible PyTorch pipeline for training and evaluating Recurrent Neural Networks (LSTM) on the Hugging Face dair-ai/emotion dataset.

This project demonstrates:

- Clean ML system design
- Proper dataset handling and validation
- Reproducible experimentation
- Structured training & evaluation workflows
- Production-style repository organization

## 📌 Problem Overview

Emotion classification is a multi-class text classification task where short text inputs are mapped to emotional labels such as:

- joy
- sadness
- anger
- fear
- love
- surprise

This project implements and compares recurrent architectures for sequence modeling.

## 🏗 Project Architecture

The repository follows a production-style ML structure with strict separation of concerns:
```
rnn_project/
│
├── data/
│   ├── raw/              ## Downloaded dataset (cached locally)
│   └── processed/        ## Tokenized & numericalized splits
│
├── artifacts/            ## Saved vocabulary, models, metrics
│
├── src/
│   └── rnn_pipeline/
│       ├── data/         ## Download, validation, preprocessing
│       ├── models/       ## RNN/LSTM/GRU model definitions
│       ├── training/     ## Training loop, early stopping, scheduler
│       ├── evaluation/   ## Metrics & inference pipeline
│       └── utils/        ## Centralized path management
│
├── Makefile              ## Reproducible CLI entry points
└── README.md
```
Design Philosophy

- Code and data are strictly separated
- Paths are centralized via a paths.py utility
- Data pipeline is idempotent
- Raw data is cached locally after first download
- Vocabulary is fit only on training split (no leakage)

## 📊 Dataset

Source: dair-ai/emotion (Hugging Face)

The dataset is automatically downloaded on first run and stored in:
`
data/raw/
`

Subsequent runs reuse the local copy (offline-safe after initial download).

## 🔄 Data Pipeline

Run:

`make data`

or:

`python -m rnn_pipeline.data.make_dataset`


**Pipeline steps:**

1. Ensure raw dataset exists (download if missing)
2. Validate schema and split integrity
3. Fit tokenizer/vocabulary on training split
4. Transform train/validation/test splits
5. Save processed parquet files
6. Save vocabulary artifact

**Outputs:**

```data/processed/train.parquet
data/processed/val.parquet
data/processed/test.parquet
artifacts/vocab.json
```

## 🧬 Model Architecture

The classifier consists of:

- Embedding layer
- Recurrent backbone:
  - LSTM
- Dropout regularization
- Fully connected classification head

## 🏋️ Training

Run:

`make train`

or:

`python -m rnn_pipeline.training.train`

**Training features:**

- Configurable RNN type
- Deterministic seed control
- Early stopping
- Learning rate scheduler (cosine annealing)
- Validation loop per epoch
- Best model checkpoint saving
- CSV metric logging
- Artifacts saved to: `artifacts/`

## 📈 Evaluation

Run:

`make evaluate`

or:

`python -m rnn_pipeline.evaluation.evaluate_model`


**Evaluation metrics include:**

- Accuracy
- F1 Score
- Confusion Matrix
- Saved CSV metrics report

Evaluation is fully decoupled from training.

## 🔁 Reproducibility

This project enforces reproducibility via:

- Fixed random seeds (Python, NumPy, PyTorch)
- Controlled DataLoader generators
- Deterministic preprocessing
- Explicit vocabulary saving
- Structured experiment artifacts

## 🧪 Engineering Highlights

This repository demonstrates:

- Proper separation between:
  - Data ingestion
  - Feature engineering
  - Model definition
  - Training loop
  - Evaluation logic
- Idempotent data pipeline
- Centralized path management
- Clean CLI workflow via Makefile
- Modular and extensible design

The architecture is intentionally structured to scale toward:

- Config-driven experiments
- Hyperparameter tuning
- Experiment tracking (MLflow / W&B)
- CI/CD integration

## 🚀 Future Improvements

- YAML-based configuration system
- Automated experiment versioning
- Hyperparameter search
- Transformer baseline comparison
- Unit tests for pipeline components

## CNN-Based Emotion Classification Project Soon...

## 📜 License
MIT License