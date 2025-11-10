#!/bin/bash
# Download SKS classification codes from Sundhedsdatastyrelsen
#
# Usage:
#   ./download-sks.sh              # Download to default location (data/raw)
#   ./download-sks.sh --force      # Force re-download even if file exists

set -e  # Exit on error

echo "🔽 Downloading SKS codes..."

# Default options
FORCE=""

# Parse arguments
if [[ "$1" == "--force" ]]; then
    FORCE="--force"
    echo "Force mode enabled - will re-download if file exists"
fi

# Run the download script
poetry run python scripts/download_sks.py --output-dir data/raw $FORCE

echo "✅ Download complete!"
