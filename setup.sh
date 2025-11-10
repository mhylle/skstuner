#!/bin/bash
# Initial setup script for SKS Tuner project
#
# This script will:
#   1. Install dependencies using Poetry
#   2. Create necessary directories
#   3. Set up environment configuration
#   4. Optionally download and process SKS codes

set -e  # Exit on error

echo "🚀 Setting up SKS Tuner project..."
echo ""

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry is not installed"
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install
echo "✅ Dependencies installed"
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/synthetic
mkdir -p models/configs
mkdir -p docs
echo "✅ Directories created"
echo ""

# Set up environment file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your ANTHROPIC_API_KEY"
    echo "   You can get an API key from: https://console.anthropic.com/"
    echo ""
else
    echo "ℹ️  .env file already exists, skipping..."
    echo ""
fi

# Make all shell scripts executable
echo "🔧 Making shell scripts executable..."
chmod +x download-sks.sh
chmod +x process-sks.sh
chmod +x generate-data.sh
chmod +x validate-data.sh
chmod +x run-tests.sh
chmod +x setup.sh
echo "✅ Shell scripts are now executable"
echo ""

# Ask if user wants to download SKS codes
echo "Would you like to download and process SKS codes now? (y/n)"
read -r DOWNLOAD

if [[ "$DOWNLOAD" =~ ^[Yy]$ ]]; then
    echo ""
    echo "📥 Downloading SKS codes..."
    ./download-sks.sh
    echo ""
    echo "⚙️  Processing SKS codes..."
    ./process-sks.sh
    echo ""
fi

echo "=" | tr '=' '='
echo "✨ Setup complete!"
echo "="
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ANTHROPIC_API_KEY (if not done already)"
echo "  2. Run ./download-sks.sh to download SKS codes (if not done already)"
echo "  3. Run ./process-sks.sh to process codes into JSON"
echo "  4. Run ./generate-data.sh to create synthetic training data"
echo "  5. Run ./run-tests.sh to verify everything works"
echo ""
echo "Available scripts:"
echo "  ./download-sks.sh      - Download SKS codes"
echo "  ./process-sks.sh       - Process SKS codes"
echo "  ./generate-data.sh     - Generate synthetic data"
echo "  ./validate-data.sh     - Validate data quality"
echo "  ./run-tests.sh         - Run test suite"
echo ""
