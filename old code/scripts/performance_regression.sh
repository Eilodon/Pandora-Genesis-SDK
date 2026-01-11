#!/bin/bash
set -e

echo "🚨 Performance Regression Detection"
echo ""

# Already in sdk directory

# Thresholds for performance regression (in percentage)
REGRESSION_THRESHOLD=10.0  # 10% slower is considered regression
SIGNIFICANT_REGRESSION_THRESHOLD=25.0  # 25% slower is significant

# Check if baseline exists
if [ ! -d "target/criterion" ]; then
    echo "❌ No baseline found. Run benchmarks first with --save-baseline"
    exit 1
fi

# Function to compare benchmarks
compare_benchmarks() {
    local bench_name="$1"
    local current_file="target/criterion/$bench_name/new/estimates.json"
    local baseline_file="target/criterion/$bench_name/base/estimates.json"
    
    if [ ! -f "$current_file" ] || [ ! -f "$baseline_file" ]; then
        echo "⚠️  Missing data for $bench_name"
        return
    fi
    
    # Extract mean times using jq if available, otherwise python
    if command -v jq &> /dev/null; then
        current_mean=$(jq -r '.mean.point_estimate' "$current_file")
        baseline_mean=$(jq -r '.mean.point_estimate' "$baseline_file")
    else
        current_mean=$(python3 -c "import json; print(json.load(open('$current_file'))['mean']['point_estimate'])")
        baseline_mean=$(python3 -c "import json; print(json.load(open('$baseline_file'))['mean']['point_estimate'])")
    fi
    
    # Calculate percentage change
    if (( $(echo "$baseline_mean > 0" | bc -l) )); then
        change_percent=$(echo "scale=2; (($current_mean - $baseline_mean) / $baseline_mean) * 100" | bc -l)
        
        # Convert to microseconds for readability
        current_us=$(echo "scale=2; $current_mean / 1000" | bc -l)
        baseline_us=$(echo "scale=2; $baseline_mean / 1000" | bc -l)
        
        echo "📊 $bench_name:"
        echo "   Current:  ${current_us} μs"
        echo "   Baseline: ${baseline_us} μs"
        echo "   Change:   ${change_percent}%"
        
        # Check for regression
        if (( $(echo "$change_percent > $REGRESSION_THRESHOLD" | bc -l) )); then
            if (( $(echo "$change_percent > $SIGNIFICANT_REGRESSION_THRESHOLD" | bc -l) )); then
                echo "   🚨 SIGNIFICANT REGRESSION DETECTED!"
                return 2
            else
                echo "   ⚠️  Performance regression detected"
                return 1
            fi
        elif (( $(echo "$change_percent < -$REGRESSION_THRESHOLD" | bc -l) )); then
            echo "   ✅ Performance improvement detected"
        else
            echo "   ✅ Performance within acceptable range"
        fi
    fi
}

# Check if bc is available for calculations
if ! command -v bc &> /dev/null; then
    echo "❌ 'bc' calculator not found. Please install it for regression detection."
    exit 1
fi

echo "Comparing current benchmarks with baseline..."
echo ""

# Find all benchmark directories
regression_detected=0
significant_regression=0

for bench_dir in target/criterion/*/; do
    if [ -d "$bench_dir" ]; then
        bench_name=$(basename "$bench_dir")
        compare_benchmarks "$bench_name"
        case $? in
            1) regression_detected=1 ;;
            2) significant_regression=1 ;;
        esac
        echo ""
    fi
done

# Summary
echo "📋 Regression Detection Summary:"
if [ $significant_regression -eq 1 ]; then
    echo "🚨 SIGNIFICANT PERFORMANCE REGRESSION DETECTED!"
    echo "   Please investigate and fix before merging."
    exit 2
elif [ $regression_detected -eq 1 ]; then
    echo "⚠️  Performance regression detected (within acceptable range)"
    echo "   Consider investigating if regression persists."
    exit 1
else
    echo "✅ No significant performance regression detected"
    exit 0
fi
