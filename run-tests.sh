#!/bin/bash
# Run test suite with various options
#
# Usage:
#   ./run-tests.sh                 # Run all tests
#   ./run-tests.sh --coverage      # Run with coverage report
#   ./run-tests.sh --verbose       # Run with verbose output

set -e  # Exit on error

echo "🧪 Running tests..."

MODE="${1:-default}"

case "$MODE" in
    --coverage)
        echo "📊 Running tests with coverage..."
        poetry run pytest --cov=skstuner --cov-report=html --cov-report=term
        echo ""
        echo "✅ Coverage report generated in htmlcov/index.html"
        ;;

    --verbose|-v)
        echo "📝 Running tests with verbose output..."
        poetry run pytest -v
        ;;

    --watch)
        echo "👀 Running tests in watch mode..."
        poetry run pytest-watch
        ;;

    *)
        echo "🚀 Running all tests..."
        poetry run pytest
        ;;
esac

echo "✅ Tests complete!"
