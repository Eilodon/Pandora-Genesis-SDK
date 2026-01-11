#!/bin/bash
set -e

echo "🔬 Running comprehensive profiling analysis..."
echo ""

# Already in sdk directory

# Create comprehensive results directory
mkdir -p benchmark_results/comprehensive
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

echo "Profiling session started at: $TIMESTAMP"
echo ""

# Run all profiling tools
echo "1. Running benchmarks..."
./scripts/benchmark_dashboard.sh

echo ""
echo "2. Checking for performance regression..."
./scripts/performance_regression.sh || echo "⚠️  Performance regression detected (see above)"

echo ""
echo "3. Generating flamegraphs..."
./scripts/profile_flamegraph.sh

echo ""
echo "4. Profiling memory usage..."
./scripts/profile_memory.sh

echo ""
echo "5. Running load tests..."
./scripts/load_testing.sh

# Generate comprehensive report
echo ""
echo "📋 Generating comprehensive profiling report..."

cat > benchmark_results/comprehensive/profiling_report_${TIMESTAMP}.md << EOF
# Comprehensive Profiling Report

**Generated**: $TIMESTAMP  
**System**: $(uname -a)  
**Rust Version**: $(rustc --version)  
**Cargo Version**: $(cargo --version)

## 📊 Benchmark Results
- [Benchmark Dashboard](html/index.html)
- [Current Metrics](trends/current_metrics.json)

## 🔥 Performance Analysis
- [Flamegraphs](profiles/)
  - Skandha Pipeline
  - Orchestrator Load Test
  - MCG Enhanced Monitoring

## 💾 Memory Analysis
- [Massif Output](profiles/massif.out) - Heap usage over time
- [Memcheck Output](profiles/memcheck.out) - Leak detection

## 🚀 Load Test Results
- [Load Test Summary](load_tests/summary.md)
- [Load Test Logs](load_tests/)

## 📈 Performance Trends
- Compare with previous runs
- Regression detection results
- Performance recommendations

## 🔧 Tools Used
- **Criterion**: Benchmarking framework
- **Flamegraph**: CPU profiling
- **Valgrind**: Memory profiling
- **Load Testing**: Stress testing

## 📝 Recommendations
1. Monitor performance trends over time
2. Investigate any detected regressions
3. Optimize hot paths identified in flamegraphs
4. Address any memory leaks found
5. Consider load testing results for production planning

EOF

echo ""
echo "✅ Comprehensive profiling completed!"
echo "   Results available in: benchmark_results/"
echo "   Report: benchmark_results/comprehensive/profiling_report_${TIMESTAMP}.md"
echo ""
echo "To view results:"
echo "   python3 -m http.server 8000 --directory benchmark_results/html"
echo "   firefox benchmark_results/profiles/skandha_flamegraph.svg"
echo "   ms_print benchmark_results/profiles/massif.out"
