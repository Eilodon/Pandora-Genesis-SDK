#!/bin/bash
set -e

echo "🧪 Running test coverage analysis..."
echo ""

cd sdk

# Clean previous coverage data
cargo clean
rm -rf coverage/

# Run tests with coverage
cargo llvm-cov --workspace --all-features --html --output-dir ../coverage

echo ""
echo "✅ Coverage report generated in coverage/index.html"
echo ""

# Print summary
cargo llvm-cov --workspace --all-features --summary-only

# Check if we meet minimum coverage
echo ""
echo "📊 Checking coverage thresholds..."

cargo llvm-cov --workspace --all-features --fail-under-lines 80 || {
    echo "❌ Line coverage below 80%"
    exit 1
}

echo "✅ Coverage meets minimum thresholds"


