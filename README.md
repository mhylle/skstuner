# SKS Tuner

Fine-tuned LLM system for classifying Danish clinical text into SKS codes.

## Features

- Multi-task hierarchical classification
- Model-agnostic architecture (XLM-RoBERTa, Gemma, Phi-3)
- Synthetic data generation via LLM
- Comprehensive evaluation metrics
- FastAPI inference service

## Setup

```bash
# Install dependencies
poetry install

# Download SKS codes
python scripts/download_sks.py

# Generate synthetic data
python scripts/generate_synthetic_data.py

# Train model
python scripts/train.py --model xlm-roberta-large

# Run API
uvicorn src.skstuner.api.main:app --reload
```

## Project Structure

```
skstuner/
├── src/skstuner/
│   ├── data/           # Data processing
│   ├── models/         # Model architectures
│   ├── training/       # Training loops
│   ├── evaluation/     # Metrics and evaluation
│   ├── api/           # FastAPI service
│   └── utils/         # Utilities
├── tests/             # Test suite
├── data/              # Data storage
├── models/            # Model configs and checkpoints
├── scripts/           # CLI scripts
└── notebooks/         # Jupyter notebooks
```
