#!/bin/bash
set -e

echo "🔥 Generating flamegraph profile..."
echo ""

# Already in sdk directory

# Install cargo-flamegraph if not present
if ! command -v cargo-flamegraph &> /dev/null; then
    echo "Installing cargo-flamegraph..."
    cargo install flamegraph
fi

# Create profiles directory
mkdir -p benchmark_results/profiles

# Build with optimizations and debug symbols
export CARGO_PROFILE_RELEASE_DEBUG=true
export CARGO_PROFILE_BENCH_DEBUG=true

# Profile different components
echo "Profiling Skandha pipeline..."
cargo flamegraph --bench comprehensive_bench -- --bench "skandha_pipeline/full_cycle/100" --output benchmark_results/profiles/skandha_flamegraph.svg

echo "Profiling orchestrator load test..."
cargo flamegraph --bench load_test -- --bench "concurrent_requests/10" --output benchmark_results/profiles/orchestrator_flamegraph.svg 2>/dev/null || echo "Orchestrator profiling skipped"

echo "Profiling MCG enhanced monitoring..."
cargo flamegraph --bench enhanced_mcg_bench -- --bench "monitor_comprehensive" --output benchmark_results/profiles/mcg_flamegraph.svg 2>/dev/null || echo "MCG profiling skipped"

echo ""
echo "✅ Flamegraphs generated in benchmark_results/profiles/:"
echo "   - skandha_flamegraph.svg"
echo "   - orchestrator_flamegraph.svg (if available)"
echo "   - mcg_flamegraph.svg (if available)"
echo ""
echo "Open with: firefox benchmark_results/profiles/skandha_flamegraph.svg"


