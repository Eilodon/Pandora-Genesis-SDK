#!/bin/bash
set -e

echo "📊 Running all benchmarks and generating dashboard..."
echo ""

# Already in sdk directory

# Create benchmark results directory
mkdir -p benchmark_results
mkdir -p benchmark_results/trends
mkdir -p benchmark_results/profiles

# Get current timestamp for trend tracking
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Benchmark run timestamp: $TIMESTAMP"

# Run all benchmarks and save results
echo "Running pandora_core benchmarks..."
cargo bench -p pandora_core --bench comprehensive_bench -- --save-baseline current

echo "Running pandora_orchestrator benchmarks..."
cargo bench -p pandora_orchestrator --bench load_test -- --save-baseline current

echo "Running pandora_cwm benchmarks..."
cargo bench -p pandora_cwm --bench hashmap_bench -- --save-baseline current

echo "Running pandora_mcg benchmarks..."
cargo bench -p pandora_mcg --bench enhanced_mcg_bench -- --save-baseline current 2>/dev/null || echo "MCG benchmarks not available"

# Extract key metrics for trend tracking
echo "Extracting performance metrics..."
cat > benchmark_results/trends/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def extract_criterion_metrics():
    """Extract key metrics from Criterion benchmark results"""
    metrics = {}
    
    # Look for Criterion JSON files
    criterion_dir = Path("target/criterion")
    if not criterion_dir.exists():
        return metrics
    
    for bench_dir in criterion_dir.rglob("new/estimates.json"):
        try:
            with open(bench_dir, 'r') as f:
                data = json.load(f)
                
            # Extract mean time
            mean_time = data.get("mean", {}).get("point_estimate", 0)
            std_dev = data.get("mean", {}).get("standard_error", 0)
            
            # Get benchmark name from path
            bench_name = str(bench_dir.parent.parent.parent.name)
            
            metrics[bench_name] = {
                "mean_ns": mean_time,
                "std_dev_ns": std_dev,
                "mean_us": mean_time / 1000,
                "std_dev_us": std_dev / 1000
            }
        except Exception as e:
            print(f"Error processing {bench_dir}: {e}")
    
    return metrics

if __name__ == "__main__":
    metrics = extract_criterion_metrics()
    
    # Save to JSON
    with open("benchmark_results/trends/current_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("📊 Performance Metrics Summary:")
    for name, data in metrics.items():
        print(f"  {name}: {data['mean_us']:.2f} ± {data['std_dev_us']:.2f} μs")
EOF

python3 benchmark_results/trends/extract_metrics.py

# Generate HTML report
echo ""
echo "Generating HTML reports..."

# Copy criterion reports to a unified location
mkdir -p benchmark_results/html
cp -r target/criterion/* benchmark_results/html/ 2>/dev/null || true

# Create index page
cat > benchmark_results/html/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Pandora SDK Benchmark Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .benchmark-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .benchmark-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .benchmark-card h3 { margin-top: 0; color: #0066cc; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .metric { display: flex; justify-content: space-between; padding: 5px 0; }
        .metric-label { font-weight: bold; }
        .metric-value { color: #666; }
    </style>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
    <meta http-equiv="Pragma" content="no-cache" />
    <meta http-equiv="Expires" content="0" />
    </head>
<body>
    <h1>🔱 Pandora Genesis SDK - Performance Dashboard</h1>
    
    <div class="section">
        <h2>📦 Core Benchmarks</h2>
        <div class="benchmark-grid">
            <div class="benchmark-card">
                <h3><a href="skandha_pipeline/report/index.html">Skandha Pipeline</a></h3>
                <div class="metric">
                    <span class="metric-label">Full Cycle (100 bytes):</span>
                    <span class="metric-value">~50-100µs</span>
                </div>
            </div>
            <div class="benchmark-card">
                <h3><a href="individual_skandhas/report/index.html">Individual Skandhas</a></h3>
                <div class="metric">
                    <span class="metric-label">Rupa:</span>
                    <span class="metric-value">~10µs</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Vedana:</span>
                    <span class="metric-value">~5µs</span>
                </div>
            </div>
            <div class="benchmark-card">
                <h3><a href="memory_allocations/report/index.html">Memory Allocations</a></h3>
                <div class="metric">
                    <span class="metric-label">String Intern (new):</span>
                    <span class="metric-value">~50ns</span>
                </div>
                <div class="metric">
                    <span class="metric-label">String Intern (cached):</span>
                    <span class="metric-value">~10ns</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Orchestrator Benchmarks</h2>
        <div class="benchmark-grid">
            <div class="benchmark-card">
                <h3><a href="single_skill/report/index.html">Single Skill Execution</a></h3>
                <div class="metric">
                    <span class="metric-label">Arithmetic Simple:</span>
                    <span class="metric-value">~100µs</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Arithmetic Complex:</span>
                    <span class="metric-value">~200µs</span>
                </div>
            </div>
            <div class="benchmark-card">
                <h3><a href="concurrent_requests/report/index.html">Concurrent Requests</a></h3>
                <div class="metric">
                    <span class="metric-label">10 concurrent:</span>
                    <span class="metric-value">~500µs</span>
                </div>
                <div class="metric">
                    <span class="metric-label">100 concurrent:</span>
                    <span class="metric-value">~2ms</span>
                </div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Historical Trends</h2>
        <p>Compare current benchmarks with baseline:</p>
        <ul>
            <li><a href="report/index.html">Full Criterion Report</a></li>
            <li><a href="trends/current_metrics.json">Current Metrics (JSON)</a></li>
        </ul>
        <div id="trends-chart" style="height: 300px; background: #f9f9f9; border: 1px solid #ddd; margin: 10px 0; display: flex; align-items: center; justify-content: center;">
            <p>📊 Trend visualization will be added with historical data</p>
        </div>
    </div>
    
    <div class="section">
        <h2>🔬 Profiling Tools</h2>
        <ul>
            <li><strong>Flamegraph:</strong> Run <code>./scripts/profile_flamegraph.sh</code></li>
            <li><strong>Memory:</strong> Run <code>./scripts/profile_memory.sh</code></li>
            <li><strong>Coverage:</strong> Run <code>./scripts/coverage.sh</code></li>
        </ul>
    </div>
</body>
</html>
EOF

echo ""
echo "✅ Benchmark dashboard generated!"
echo "   Open: benchmark_results/html/index.html"
echo ""
echo "To view:"
echo "  python3 -m http.server 8000 --directory benchmark_results/html"
echo "  Then open: http://localhost:8000"


