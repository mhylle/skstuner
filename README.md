# SKS Tuner

Data processing and synthetic data generation system for Danish SKS (Sundhedsvæsenets Klassifikations System) codes.

## Current Status

**Phase 1: Data Pipeline** ✓ Complete

This project currently implements the data ingestion and synthetic data generation pipeline for SKS classification codes. The following modules are **implemented and production-ready**:

- ✓ SKS code downloading and parsing
- ✓ Data processing and export
- ✓ Synthetic clinical note generation using Claude AI
- ✓ Comprehensive test suite
- ✓ CLI scripts for all operations

**Planned Features** (Not Yet Implemented):
- Model training pipeline
- Evaluation metrics
- FastAPI inference service

## Features

- **SKS Code Management**: Download and parse official SKS classification codes
- **Data Processing**: Convert SKS codes to JSON with hierarchical relationships
- **Synthetic Data Generation**: Generate realistic Danish clinical notes using Claude AI
- **Checkpoint & Resume**: Automatic checkpointing with ability to resume interrupted generation
- **Quality Validation**: Comprehensive quality metrics and filtering for generated data
- **Production-Ready**: Comprehensive error handling, logging, and validation
- **Type-Safe**: Full type hints throughout the codebase
- **Well-Tested**: Unit tests with mocking for external services

## Setup

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Download SKS codes
python scripts/download_sks.py

# Process SKS codes
python scripts/process_sks.py

# Generate synthetic training data
python scripts/generate_synthetic_data.py --max-codes 10 --examples-per-code 5
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
│   │   └── prompt_templates.py     # LLM prompt templates
│   ├── utils/             # Utility modules
│   │   └── logging_config.py  # Centralized logging
│   └── config.py          # Configuration management
├── tests/                 # Test suite
│   ├── data/             # Data module tests
│   └── test_config.py    # Config tests
├── scripts/              # CLI scripts
│   ├── download_sks.py
│   ├── process_sks.py
│   └── generate_synthetic_data.py
├── data/                 # Data storage
│   ├── raw/             # Downloaded SKS files
│   ├── processed/       # Processed JSON files
│   └── synthetic/       # Generated synthetic data
├── models/              # Model configurations
│   └── configs/         # YAML config files
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

# Resume interrupted generation from checkpoint
python scripts/generate_synthetic_data.py \
    --resume \
    --output-file data/synthetic/train_data.json

# Advanced: Custom checkpoint interval and quality settings
python scripts/generate_synthetic_data.py \
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

- `ANTHROPIC_API_KEY`: Required for synthetic data generation
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

## Contributing

Contributions are welcome! Areas for contribution:

1. **Model Training Pipeline**: Implement training for XLM-RoBERTa, Gemma, or Phi-3
2. **Evaluation Metrics**: Add comprehensive evaluation for multi-label classification
3. **API Service**: Create FastAPI inference endpoint
4. **Additional Tests**: Expand test coverage for edge cases

## License

[Add your license here]
