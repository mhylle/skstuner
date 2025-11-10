#!/bin/bash
# Validate quality of existing synthetic data
#
# Usage:
#   ./validate-data.sh [file]                    # Validate dataset
#   ./validate-data.sh [file] --filter           # Filter low-quality examples

set -e  # Exit on error

echo "🔍 Validating synthetic data..."

# Get input file (default to test data if not specified)
INPUT_FILE="${1:-data/synthetic/train_data_test.json}"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: File not found: $INPUT_FILE"
    echo ""
    echo "Available datasets:"
    if [ -d "data/synthetic" ]; then
        ls -lh data/synthetic/*.json 2>/dev/null || echo "  No datasets found"
    else
        echo "  data/synthetic directory not found"
    fi
    exit 1
fi

# Check if filtering is requested
if [[ "$2" == "--filter" ]]; then
    BASENAME=$(basename "$INPUT_FILE" .json)
    OUTPUT_FILE="data/synthetic/${BASENAME}_filtered.json"

    echo "🔧 Filtering mode enabled"
    echo "   Input:  $INPUT_FILE"
    echo "   Output: $OUTPUT_FILE"

    poetry run python scripts/validate_synthetic_data.py \
        --input-file "$INPUT_FILE" \
        --output-file "$OUTPUT_FILE" \
        --filter \
        --quality-threshold 0.5 \
        --report-file "data/synthetic/${BASENAME}_report.json"
else
    echo "📊 Validation mode (no filtering)"
    echo "   File: $INPUT_FILE"

    poetry run python scripts/validate_synthetic_data.py \
        --input-file "$INPUT_FILE" \
        --quality-threshold 0.5
fi

echo "✅ Validation complete!"
