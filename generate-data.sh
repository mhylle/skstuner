#!/bin/bash
# Generate synthetic clinical notes for SKS codes using Claude AI
#
# Usage:
#   ./generate-data.sh                           # Quick test (10 codes, 5 examples each)
#   ./generate-data.sh --full                    # Full generation (all codes)
#   ./generate-data.sh --category D              # Generate for specific category
#   ./generate-data.sh --resume                  # Resume from checkpoint
#   ./generate-data.sh --custom [args]           # Pass custom arguments

set -e  # Exit on error

echo "🤖 Generating synthetic training data..."

# Check if processed codes exist
if [ ! -f "data/processed/sks_codes.json" ]; then
    echo "❌ Error: data/processed/sks_codes.json not found"
    echo "Please run ./process-sks.sh first"
    exit 1
fi

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ ! -f ".env" ]; then
        echo "❌ Error: ANTHROPIC_API_KEY not set and .env file not found"
        echo "Please create a .env file with your API key:"
        echo "  cp .env.example .env"
        echo "  # Edit .env and add your ANTHROPIC_API_KEY"
        exit 1
    fi
fi

# Parse mode
MODE="${1:-test}"

case "$MODE" in
    --full)
        echo "📊 Full generation mode (WARNING: This will use significant API credits!)"
        poetry run python scripts/generate_synthetic_data.py \
            --examples-per-code 10 \
            --output-file data/synthetic/train_data.json \
            --quality-threshold 0.5 \
            --enable-quality-filter \
            --show-quality-report
        ;;

    --resume)
        echo "🔄 Resuming from checkpoint..."
        poetry run python scripts/generate_synthetic_data.py \
            --resume \
            --output-file data/synthetic/train_data.json \
            --enable-quality-filter \
            --show-quality-report
        ;;

    --category)
        CATEGORY="$2"
        if [ -z "$CATEGORY" ]; then
            echo "❌ Error: Please specify a category (D, K, B, N, U, ZZ)"
            echo "Usage: ./generate-data.sh --category D"
            exit 1
        fi
        echo "📊 Generating for category: $CATEGORY"
        poetry run python scripts/generate_synthetic_data.py \
            --category "$CATEGORY" \
            --max-codes 100 \
            --examples-per-code 10 \
            --output-file "data/synthetic/train_data_${CATEGORY}.json" \
            --quality-threshold 0.5 \
            --enable-quality-filter \
            --show-quality-report
        ;;

    --custom)
        shift  # Remove --custom argument
        echo "🔧 Custom mode with args: $@"
        poetry run python scripts/generate_synthetic_data.py "$@"
        ;;

    *)
        # Default: Quick test mode
        echo "🧪 Test mode (10 codes, 5 examples each)"
        echo "Tip: Use --full for complete generation, --resume to continue, or --category [D/K/B/N/U/ZZ] for specific category"
        poetry run python scripts/generate_synthetic_data.py \
            --max-codes 10 \
            --examples-per-code 5 \
            --output-file data/synthetic/train_data_test.json \
            --quality-threshold 0.5 \
            --enable-quality-filter \
            --show-quality-report
        ;;
esac

echo "✅ Generation complete!"
