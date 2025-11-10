#!/bin/bash
# Train SKS classification model
#
# Usage:
#   ./train-model.sh                           # Train XLM-RoBERTa (default)
#   ./train-model.sh --phi3                    # Train Phi-3 with LoRA
#   ./train-model.sh --gemma                   # Train Gemma with LoRA
#   ./train-model.sh --custom [args]           # Pass custom arguments

set -e  # Exit on error

echo "🚀 Training SKS classification model..."

# Check if training data exists
if [ ! -f "data/synthetic/train_data.json" ]; then
    echo "❌ Error: data/synthetic/train_data.json not found"
    echo "Please generate training data first:"
    echo "  ./generate-data.sh --full"
    exit 1
fi

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  Warning: No GPU detected. Training will be slow on CPU."
    echo "Consider using a machine with GPU for faster training."
fi

# Parse mode
MODE="${1:-xlm}"

case "$MODE" in
    --xlm|--xlm-roberta)
        echo "📖 Training XLM-RoBERTa Large (encoder model)"
        poetry run python scripts/train_model.py \
            --data-file data/synthetic/train_data.json \
            --config models/configs/xlm_roberta_large.yaml \
            --output-dir models/trained/xlm_roberta \
            --log-level INFO
        ;;

    --phi3)
        echo "🔮 Training Phi-3 Mini with LoRA (decoder model)"
        poetry run python scripts/train_model.py \
            --data-file data/synthetic/train_data.json \
            --config models/configs/phi3_mini.yaml \
            --output-dir models/trained/phi3 \
            --log-level INFO
        ;;

    --gemma)
        echo "💎 Training Gemma 7B with LoRA (decoder model)"
        poetry run python scripts/train_model.py \
            --data-file data/synthetic/train_data.json \
            --config models/configs/gemma_7b.yaml \
            --output-dir models/trained/gemma \
            --log-level INFO
        ;;

    --custom)
        shift  # Remove --custom argument
        echo "🔧 Custom training with args: $@"
        poetry run python scripts/train_model.py "$@"
        ;;

    *)
        # Default: XLM-RoBERTa
        echo "📖 Training XLM-RoBERTa Large (encoder model)"
        echo "Tip: Use --phi3 or --gemma for other models, or --custom for custom arguments"
        poetry run python scripts/train_model.py \
            --data-file data/synthetic/train_data.json \
            --config models/configs/xlm_roberta_large.yaml \
            --output-dir models/trained/xlm_roberta \
            --log-level INFO
        ;;
esac

echo ""
echo "✅ Training complete!"
echo "📊 Check models/trained/ for model files and metrics"
