#!/bin/bash
set -e

echo "💾 Profiling memory usage with valgrind..."
echo ""

# Already in sdk directory

# Create profiles directory
mkdir -p benchmark_results/profiles

# Check if valgrind is available
if ! command -v valgrind &> /dev/null; then
    echo "❌ valgrind not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install valgrind"
    echo "   Fedora: sudo dnf install valgrind"
    exit 1
fi

# Build test binary
echo "Building test binary..."
cargo build --tests --release

# Find the test binary
TEST_BINARY=$(find target/release/deps -name "load_scenarios*" -type f -executable | head -1)

if [ -z "$TEST_BINARY" ]; then
    echo "❌ Could not find test binary"
    exit 1
fi

echo "Using test binary: $TEST_BINARY"

# Run with valgrind massif
echo "Running memory profiling with valgrind massif..."
valgrind \
    --tool=massif \
    --massif-out-file=benchmark_results/profiles/massif.out \
    --time-unit=B \
    --pages-as-heap=yes \
    $TEST_BINARY --test test_memory_stability_under_load --ignored

# Also run with memcheck for leak detection
echo "Running memory leak detection..."
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --show-leak-kinds=all \
    --track-origins=yes \
    --log-file=benchmark_results/profiles/memcheck.out \
    $TEST_BINARY --test test_memory_stability_under_load --ignored

echo ""
echo "✅ Memory profiles generated in benchmark_results/profiles/:"
echo "   - massif.out (heap usage over time)"
echo "   - memcheck.out (leak detection)"
echo ""
echo "Analyze with:"
echo "   ms_print benchmark_results/profiles/massif.out"
echo "   cat benchmark_results/profiles/memcheck.out"


