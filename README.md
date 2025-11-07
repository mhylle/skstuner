# SKS Tuner

Fine-tuned LLM system for classifying Danish clinical text into SKS codes.

## Features

- Multi-task hierarchical classification
- Model-agnostic architecture (XLM-RoBERTa, Gemma, Phi-3)
- Synthetic data generation via LLM
- Comprehensive evaluation metrics
- FastAPI inference service

## Setup

### Prerequisites

1. **Install dependencies**
```bash
poetry install
```

2. **Configure environment variables**

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

#### Option A: Using Anthropic Claude (default)
```env
ANTHROPIC_API_KEY=your_anthropic_key_here
```

#### Option B: Using Azure OpenAI
```env
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Usage

```bash
# Download SKS codes
python scripts/download_sks.py

# Generate synthetic data with Anthropic (default)
python scripts/generate_synthetic_data.py

# Generate synthetic data with Azure OpenAI
python scripts/generate_synthetic_data.py --provider azure

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
