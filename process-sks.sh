#!/bin/bash
# Process downloaded SKS codes into JSON format
#
# Usage:
#   ./process-sks.sh              # Process using default paths

set -e  # Exit on error

echo "⚙️  Processing SKS codes..."

# Check if raw data exists
if [ ! -f "data/raw/SKScomplete.txt" ]; then
    echo "❌ Error: data/raw/SKScomplete.txt not found"
    echo "Please run ./download-sks.sh first"
    exit 1
fi

# Run the processing script
poetry run python scripts/process_sks.py \
    --input-file data/raw/SKScomplete.txt \
    --output-dir data/processed

echo "✅ Processing complete!"
