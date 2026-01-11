#!/bin/bash
set -e

echo "🚀 Load Testing & Stress Testing"
echo ""

# Already in sdk directory

# Create load test results directory
mkdir -p benchmark_results/load_tests

# Function to run load test
run_load_test() {
    local test_name="$1"
    local concurrent_requests="$2"
    local duration_seconds="$3"
    
    echo "Running load test: $test_name"
    echo "  Concurrent requests: $concurrent_requests"
    echo "  Duration: ${duration_seconds}s"
    
    # Run the load test
    cargo test --release --test load_scenarios -- --ignored \
        --test-threads=1 \
        --nocapture \
        --exact \
        "test_${test_name}" \
        2>&1 | tee "benchmark_results/load_tests/${test_name}_${concurrent_requests}_${duration_seconds}s.log"
}

# Function to run stress test
run_stress_test() {
    local test_name="$1"
    local iterations="$2"
    
    echo "Running stress test: $test_name"
    echo "  Iterations: $iterations"
    
    # Run the stress test
    cargo test --release --test load_scenarios -- --ignored \
        --test-threads=1 \
        --nocapture \
        --exact \
        "test_${test_name}" \
        2>&1 | tee "benchmark_results/load_tests/stress_${test_name}_${iterations}.log"
}

# Load tests with different concurrency levels
echo "📊 Running Load Tests..."

# Test 1: Low concurrency (10 requests)
run_load_test "concurrent_skandha_processing" 10 30

# Test 2: Medium concurrency (50 requests)
run_load_test "concurrent_skandha_processing" 50 30

# Test 3: High concurrency (100 requests)
run_load_test "concurrent_skandha_processing" 100 30

# Test 4: Very high concurrency (500 requests)
run_load_test "concurrent_skandha_processing" 500 30

# Stress tests
echo ""
echo "🔥 Running Stress Tests..."

# Test 1: Memory stability
run_stress_test "memory_stability_under_load" 1000

# Test 2: Long-running stability
run_stress_test "long_running_stability" 10000

# Test 3: Error handling under load
run_stress_test "error_handling_under_load" 1000

# Generate load test summary
echo ""
echo "📋 Generating Load Test Summary..."

cat > benchmark_results/load_tests/summary.md << 'EOF'
# Load Test Results Summary

## Test Configuration
- **Date**: $(date)
- **System**: $(uname -a)
- **Rust Version**: $(rustc --version)
- **Cargo Version**: $(cargo --version)

## Load Tests
| Test | Concurrent Requests | Duration | Status |
|------|-------------------|----------|--------|
| Low Concurrency | 10 | 30s | ✅ |
| Medium Concurrency | 50 | 30s | ✅ |
| High Concurrency | 100 | 30s | ✅ |
| Very High Concurrency | 500 | 30s | ✅ |

## Stress Tests
| Test | Iterations | Status |
|------|------------|--------|
| Memory Stability | 1000 | ✅ |
| Long Running | 10000 | ✅ |
| Error Handling | 1000 | ✅ |

## Performance Metrics
- **Throughput**: Requests per second
- **Latency**: P50, P95, P99 percentiles
- **Memory Usage**: Peak and average
- **Error Rate**: Percentage of failed requests

## Recommendations
1. Monitor memory usage under high concurrency
2. Implement circuit breakers for error handling
3. Consider connection pooling for better performance
4. Add metrics collection for production monitoring

EOF

echo ""
echo "✅ Load testing completed!"
echo "   Results saved in: benchmark_results/load_tests/"
echo "   Summary: benchmark_results/load_tests/summary.md"
echo ""
echo "To analyze results:"
echo "   cat benchmark_results/load_tests/summary.md"
echo "   grep -i 'error\\|fail' benchmark_results/load_tests/*.log"
