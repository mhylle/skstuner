# SKS Tuner

Data processing and synthetic data generation system for Danish SKS (Sundhedsvæsenets Klassifikations System) codes.

## Current Status

**Phase 1: Data Pipeline** ✓ Complete
**Phase 2: Model Training** ✓ Complete

This project implements end-to-end data processing and model training for SKS classification codes. The following modules are **implemented and production-ready**:

- ✓ SKS code downloading and parsing
- ✓ Data processing and export
- ✓ Synthetic clinical note generation using Claude AI or Ollama
- ✓ Model training pipeline with support for multiple architectures
- ✓ LoRA/PEFT support for efficient fine-tuning
- ✓ Comprehensive evaluation metrics
- ✓ Comprehensive test suite
- ✓ CLI scripts for all operations

**Planned Features** (Not Yet Implemented):
- FastAPI inference service

## Features

- **SKS Code Management**: Download and parse official SKS classification codes
- **Data Processing**: Convert SKS codes to JSON with hierarchical relationships
- **Synthetic Data Generation**: Generate realistic Danish clinical notes using LLMs
- **Multiple LLM Providers**: Support for both Claude AI and Ollama (local models)
- **Checkpoint & Resume**: Automatic checkpointing with ability to resume interrupted generation
- **Quality Validation**: Comprehensive quality metrics and filtering for generated data
- **Model Training**: Complete training pipeline for SKS code classification
- **Multiple Architectures**: Support for XLM-RoBERTa, Phi-3, Gemma, and other models
- **LoRA/PEFT**: Efficient fine-tuning with Parameter-Efficient Fine-Tuning
- **Comprehensive Metrics**: Detailed evaluation with precision, recall, F1, and top-k accuracy
- **Production-Ready**: Comprehensive error handling, logging, and validation
- **Type-Safe**: Full type hints throughout the codebase
- **Well-Tested**: Unit tests with mocking for external services

## Setup

### Basic Installation

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env and configure your LLM provider (see Configuration section below)
```

### Option 1: Using Claude AI

```bash
# Edit .env and set:
# LLM_PROVIDER=claude
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Download SKS codes
python scripts/download_sks.py

# Process SKS codes
python scripts/process_sks.py

# Generate synthetic training data
python scripts/generate_synthetic_data.py --max-codes 10 --examples-per-code 5
```

### Option 2: Using Ollama (Local LLM)

```bash
# 1. Start your Ollama server at https://ollama:11434
# 2. Pull the llama3.3 model (or your preferred model)
#    ollama pull llama3.3

# 3. Edit .env and set:
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=https://ollama:11434
# OLLAMA_MODEL=llama3.3

# 4. Download and process SKS codes
python scripts/download_sks.py
python scripts/process_sks.py

# 5. Generate synthetic training data using Ollama
python scripts/generate_synthetic_data.py \
    --provider ollama \
    --max-codes 10 \
    --examples-per-code 5
```

## Project Structure

```
skstuner/
├── src/skstuner/
│   ├── data/              # Data processing modules
│   │   ├── sks_downloader.py   # Download SKS codes
│   │   ├── sks_parser.py       # Parse SKS format
│   │   ├── sks_processor.py    # Process and export
│   │   ├── synthetic_generator.py  # Generate synthetic data
│   │   ├── prompt_templates.py     # LLM prompt templates
│   │   ├── llm_providers.py        # LLM provider abstraction
│   │   ├── checkpoint_manager.py   # Checkpoint handling
│   │   └── quality_validator.py    # Data quality validation
│   ├── training/          # Model training modules
│   │   ├── dataset.py     # Dataset preparation
│   │   ├── model.py       # Model creation and loading
│   │   ├── trainer.py     # Training loop
│   │   └── metrics.py     # Evaluation metrics
│   ├── utils/             # Utility modules
│   │   └── logging_config.py  # Centralized logging
│   └── config.py          # Configuration management
├── tests/                 # Test suite
│   ├── data/             # Data module tests
│   ├── training/         # Training module tests
│   └── test_config.py    # Config tests
├── scripts/              # CLI scripts
│   ├── download_sks.py
│   ├── process_sks.py
│   ├── generate_synthetic_data.py
│   ├── validate_synthetic_data.py
│   └── train_model.py    # Model training script
├── data/                 # Data storage
│   ├── raw/             # Downloaded SKS files
│   ├── processed/       # Processed JSON files
│   └── synthetic/       # Generated synthetic data
├── models/              # Model storage
│   ├── configs/         # YAML config files
│   └── trained/         # Trained model checkpoints
└── docs/                # Documentation
```

## Usage Examples

### Download SKS Codes
```bash
python scripts/download_sks.py --output-dir data/raw --force
```

### Process SKS Codes
```bash
python scripts/process_sks.py \
    --input-file data/raw/SKScomplete.txt \
    --output-dir data/processed
```

### Generate Synthetic Data

**Using Claude AI (default):**
```bash
# Generate for all codes (expensive!)
python scripts/generate_synthetic_data.py

# Generate for specific category with limits and quality filtering
python scripts/generate_synthetic_data.py \
    --category D \
    --max-codes 100 \
    --examples-per-code 10 \
    --output-file data/synthetic/train_data.json \
    --quality-threshold 0.5 \
    --enable-quality-filter
```

**Using Ollama (local LLM):**
```bash
# Set LLM_PROVIDER=ollama in .env, or use --provider flag
python scripts/generate_synthetic_data.py \
    --provider ollama \
    --category D \
    --max-codes 100 \
    --examples-per-code 10 \
    --output-file data/synthetic/train_data_ollama.json

# Or set in environment
export LLM_PROVIDER=ollama
python scripts/generate_synthetic_data.py \
    --max-codes 10 \
    --examples-per-code 5
```

**Advanced Options:**
```bash
# Resume interrupted generation from checkpoint
python scripts/generate_synthetic_data.py \
    --resume \
    --output-file data/synthetic/train_data.json

# Custom checkpoint interval and quality settings
python scripts/generate_synthetic_data.py \
    --provider ollama \
    --max-codes 50 \
    --checkpoint-interval 5 \
    --quality-threshold 0.7 \
    --show-quality-report
```

### Validate Existing Data
```bash
# Validate quality of existing dataset
python scripts/validate_synthetic_data.py \
    --input-file data/synthetic/train_data.json \
    --quality-threshold 0.5 \
    --report-file data/synthetic/quality_report.json

# Filter and save only high-quality examples
python scripts/validate_synthetic_data.py \
    --input-file data/synthetic/train_data.json \
    --output-file data/synthetic/filtered_data.json \
    --filter \
    --quality-threshold 0.6
```

### Train Classification Model

**Prerequisites:**
1. Install training dependencies: `poetry install`
2. Generate or prepare synthetic training data

**Available Model Configurations:**
- `models/configs/xlm_roberta_large.yaml` - Encoder model (XLM-RoBERTa)
- `models/configs/phi3_mini.yaml` - Decoder model with LoRA (Phi-3)
- `models/configs/gemma_7b.yaml` - Decoder model with LoRA (Gemma)

**Quick Start (using shell script):**
```bash
# Train XLM-RoBERTa (default)
./train-model.sh

# Train Phi-3 with LoRA
./train-model.sh --phi3

# Train Gemma with LoRA
./train-model.sh --gemma
```

**Basic Training (using Python script directly):**
```bash
# Train XLM-RoBERTa on synthetic data
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/xlm_roberta_large.yaml \
    --output-dir models/trained/xlm_roberta

# Train Phi-3 with LoRA (memory efficient)
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/phi3_mini.yaml \
    --output-dir models/trained/phi3

# Train Gemma with LoRA
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/gemma_7b.yaml \
    --output-dir models/trained/gemma
```

**Advanced Training Options:**
```bash
# Custom hyperparameters
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/xlm_roberta_large.yaml \
    --output-dir models/trained/xlm_roberta_custom \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --num-epochs 5 \
    --max-length 256

# Custom train/validation/test split
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/phi3_mini.yaml \
    --output-dir models/trained/phi3 \
    --test-size 0.15 \
    --val-size 0.1

# Use cached models to save download time
python scripts/train_model.py \
    --data-file data/synthetic/train_data.json \
    --config models/configs/xlm_roberta_large.yaml \
    --output-dir models/trained/xlm_roberta \
    --cache-dir models/cache
```

**Training Output:**
After training completes, you'll find:
- `models/trained/<model_name>/final_model/` - Final trained model
- `models/trained/<model_name>/checkpoints/` - Training checkpoints
- `models/trained/<model_name>/test_metrics.json` - Detailed evaluation metrics
- `models/trained/<model_name>/label_mapping.json` - SKS code to ID mappings
- `models/trained/<model_name>/training_config.json` - Training configuration

**Evaluation Metrics:**
The training script provides comprehensive metrics:
- Top-1 and Top-K accuracy
- Weighted and macro-averaged precision, recall, F1
- Per-class performance
- Best and worst performing classes
- Full classification report

## Development

```bash
# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=skstuner --cov-report=html

# Format code
poetry run black src/ tests/ scripts/

# Lint code
poetry run ruff check src/ tests/ scripts/
```

## Configuration

The project uses environment variables for configuration. See `.env.example` for required variables:

### LLM Provider Settings

Choose between Claude AI or Ollama for synthetic data generation:

**Common Settings:**
- `LLM_PROVIDER`: Provider to use - `claude` or `ollama` (default: `claude`)

**Claude Settings (required if using Claude):**
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `CLAUDE_MODEL`: Claude model to use (default: `claude-sonnet-4-20250514`)

**Ollama Settings (required if using Ollama):**
- `OLLAMA_BASE_URL`: Ollama server URL (default: `https://ollama:11434`)
- `OLLAMA_MODEL`: Model name to use (default: `llama3.3`)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: `120`)

**Other Settings:**
- `WANDB_API_KEY`: Optional, for experiment tracking (future feature)
- `WANDB_PROJECT`: W&B project name
- `WANDB_ENTITY`: W&B entity name

## Advanced Features

### Checkpoint & Resume

Synthetic data generation now supports automatic checkpointing:

- **Automatic Checkpoints**: Progress is saved every N codes (default: 5)
- **Resume Capability**: Use `--resume` to continue from last checkpoint
- **Cost Savings**: Prevents loss of progress and wasted API calls
- **Atomic Saves**: Checkpoint files are saved atomically to prevent corruption

Checkpoint files are stored as `.{output_file}_checkpoint.json` by default.

### Quality Validation

Generated clinical notes are automatically validated for quality:

**Validation Metrics:**
- **Length Score**: Checks for appropriate text length (50-2000 chars)
- **Language Score**: Validates Danish language indicators and special characters
- **Medical Relevance**: Detects Danish medical terminology and relevance to SKS codes
- **Overall Score**: Weighted combination of all metrics (default threshold: 0.5)

**Quality Features:**
- Automatic filtering during generation
- Standalone validation script for existing datasets
- Detailed quality reports with statistics
- Diversity metrics (vocabulary, uniqueness, common terms)

**Customization:**
```python
from skstuner.data.quality_validator import QualityValidator

validator = QualityValidator(
    min_length=100,
    max_length=1500,
    min_danish_ratio=0.4,
    quality_threshold=0.7
)
```

## Model Architectures

The training pipeline supports multiple model architectures optimized for different use cases:

### XLM-RoBERTa Large (Encoder Model)
- **Best for**: High accuracy, multilingual understanding
- **Model size**: ~560M parameters
- **Memory**: ~8GB GPU RAM (training)
- **Speed**: Fast inference
- **Configuration**: `models/configs/xlm_roberta_large.yaml`
- **Use case**: Production deployments where accuracy is critical

### Phi-3 Mini (Decoder Model with LoRA)
- **Best for**: Balanced performance and efficiency
- **Model size**: ~3.8B parameters (only ~50M trained with LoRA)
- **Memory**: ~12GB GPU RAM (with LoRA)
- **Speed**: Moderate inference
- **Configuration**: `models/configs/phi3_mini.yaml`
- **Use case**: Resource-constrained environments

### Gemma 7B (Decoder Model with LoRA)
- **Best for**: Maximum capability
- **Model size**: ~7B parameters (only ~50M trained with LoRA)
- **Memory**: ~20GB GPU RAM (with LoRA)
- **Speed**: Slower inference
- **Configuration**: `models/configs/gemma_7b.yaml`
- **Use case**: Research and experimentation

### LoRA (Low-Rank Adaptation)
LoRA enables efficient fine-tuning by only training a small number of parameters:
- Reduces memory requirements by 3-4x
- Faster training times
- Maintains model quality
- Enabled by default for decoder models (Phi-3, Gemma)

## Contributing

Contributions are welcome! Areas for contribution:

1. **API Service**: Create FastAPI inference endpoint with model serving
2. **Additional Tests**: Expand test coverage for edge cases
3. **Model Optimization**: Quantization and optimization for production
4. **Additional Architectures**: Add support for more models (LLaMA, Mistral, etc.)
5. **Data Augmentation**: Implement data augmentation strategies

## License

[Add your license here]
