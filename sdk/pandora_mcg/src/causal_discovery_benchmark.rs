//! Benchmark suite for comparing causal discovery algorithms
//! 
//! This module provides comprehensive benchmarking for different causal discovery
//! approaches including classical methods (PC, GES, LiNGAM) and neural methods 
//! (NOTEARS, DECI).

use crate::causal_discovery::{discover_causal_links, CausalDiscoveryConfig, CausalAlgorithm, CausalHypothesis};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Benchmark results for a single algorithm run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub algorithm: String,
    pub execution_time: Duration,
    pub num_hypotheses: usize,
    pub avg_confidence: f32,
    pub avg_strength: f32,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Comprehensive benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Algorithms to test
    pub algorithms: Vec<CausalAlgorithm>,
    /// Number of benchmark runs per algorithm
    pub num_runs: usize,
    /// Whether to use synthetic data
    pub use_synthetic_data: bool,
    /// Size of synthetic dataset
    pub synthetic_data_size: usize,
    /// Number of variables in synthetic data
    pub synthetic_num_vars: usize,
    /// Noise level for synthetic data (0.0 - 1.0)
    pub synthetic_noise_level: f32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                CausalAlgorithm::DirectLiNGAM,
                CausalAlgorithm::PC,
                CausalAlgorithm::GES,
                CausalAlgorithm::NOTEARS,
                CausalAlgorithm::DECI,
            ],
            num_runs: 3,
            use_synthetic_data: true,
            synthetic_data_size: 1000,
            synthetic_num_vars: 5,
            synthetic_noise_level: 0.1,
        }
    }
}

/// Benchmark results aggregated across multiple runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedBenchmarkResult {
    pub algorithm: String,
    pub avg_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub success_rate: f32,
    pub avg_num_hypotheses: f32,
    pub avg_confidence: f32,
    pub avg_strength: f32,
    pub std_execution_time: Duration,
}

/// Run comprehensive benchmark comparing all causal discovery algorithms
pub fn run_causal_discovery_benchmark(
    data_matrix: Option<Vec<Vec<f32>>>,
    config: Option<BenchmarkConfig>,
) -> Result<Vec<AggregatedBenchmarkResult>, Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();
    
    // Generate or use provided data
    let data = if let Some(data) = data_matrix {
        data
    } else if config.use_synthetic_data {
        generate_synthetic_causal_data(&config)?
    } else {
        return Err("No data provided and synthetic data generation disabled".into());
    };
    
    info!("Starting causal discovery benchmark with {} algorithms on {}x{} dataset", 
          config.algorithms.len(), data.len(), data[0].len());
    
    let mut all_results = Vec::new();
    
    // Run benchmarks for each algorithm
    for algorithm in &config.algorithms {
        let mut algorithm_results = Vec::new();
        
        for run in 0..config.num_runs {
            info!("Running {:?} - iteration {}/{}", algorithm, run + 1, config.num_runs);
            
            let discovery_config = CausalDiscoveryConfig {
                min_strength_threshold: 0.1,
                min_confidence_threshold: 0.3,
                max_hypotheses: 20,
                algorithm: algorithm.clone(),
            };
            
            let result = benchmark_single_algorithm(&data, &discovery_config);
            algorithm_results.push(result);
        }
        
        // Aggregate results for this algorithm
        let aggregated = aggregate_benchmark_results(algorithm, algorithm_results);
        all_results.push(aggregated);
    }
    
    // Sort by average execution time for performance comparison
    all_results.sort_by(|a, b| a.avg_execution_time.cmp(&b.avg_execution_time));
    
    // Print summary
    print_benchmark_summary(&all_results);
    
    Ok(all_results)
}

/// Benchmark a single algorithm run
fn benchmark_single_algorithm(
    data: &[Vec<f32>],
    config: &CausalDiscoveryConfig,
) -> BenchmarkResult {
    let start_time = Instant::now();
    
    let result = discover_causal_links(data.to_vec(), config);
    let execution_time = start_time.elapsed();
    
    match result {
        Ok(hypotheses) => {
            let num_hypotheses = hypotheses.len();
            let avg_confidence = if !hypotheses.is_empty() {
                hypotheses.iter().map(|h| h.confidence).sum::<f32>() / num_hypotheses as f32
            } else { 0.0 };
            let avg_strength = if !hypotheses.is_empty() {
                hypotheses.iter().map(|h| h.strength.abs()).sum::<f32>() / num_hypotheses as f32
            } else { 0.0 };
            
            BenchmarkResult {
                algorithm: format!("{:?}", config.algorithm),
                execution_time,
                num_hypotheses,
                avg_confidence,
                avg_strength,
                success: true,
                error_message: None,
            }
        }
        Err(e) => {
            warn!("Algorithm {:?} failed: {}", config.algorithm, e);
            BenchmarkResult {
                algorithm: format!("{:?}", config.algorithm),
                execution_time,
                num_hypotheses: 0,
                avg_confidence: 0.0,
                avg_strength: 0.0,
                success: false,
                error_message: Some(format!("{}", e)),
            }
        }
    }
}

/// Aggregate multiple benchmark results for the same algorithm
fn aggregate_benchmark_results(
    algorithm: &CausalAlgorithm,
    results: Vec<BenchmarkResult>
) -> AggregatedBenchmarkResult {
    let successful_results: Vec<_> = results.iter().filter(|r| r.success).collect();
    let success_rate = successful_results.len() as f32 / results.len() as f32;
    
    if successful_results.is_empty() {
        return AggregatedBenchmarkResult {
            algorithm: format!("{:?}", algorithm),
            avg_execution_time: Duration::from_secs(0),
            min_execution_time: Duration::from_secs(0),
            max_execution_time: Duration::from_secs(0),
            success_rate: 0.0,
            avg_num_hypotheses: 0.0,
            avg_confidence: 0.0,
            avg_strength: 0.0,
            std_execution_time: Duration::from_secs(0),
        };
    }
    
    let execution_times: Vec<Duration> = successful_results.iter().map(|r| r.execution_time).collect();
    let avg_execution_time = Duration::from_nanos(
        execution_times.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / execution_times.len() as u64
    );
    let min_execution_time = *execution_times.iter().min().unwrap();
    let max_execution_time = *execution_times.iter().max().unwrap();
    
    // Calculate standard deviation of execution times
    let mean_nanos = avg_execution_time.as_nanos() as f64;
    let variance = execution_times.iter()
        .map(|d| {
            let diff = d.as_nanos() as f64 - mean_nanos;
            diff * diff
        })
        .sum::<f64>() / execution_times.len() as f64;
    let std_execution_time = Duration::from_nanos(variance.sqrt() as u64);
    
    let avg_num_hypotheses = successful_results.iter().map(|r| r.num_hypotheses as f32).sum::<f32>() 
        / successful_results.len() as f32;
    let avg_confidence = successful_results.iter().map(|r| r.avg_confidence).sum::<f32>() 
        / successful_results.len() as f32;
    let avg_strength = successful_results.iter().map(|r| r.avg_strength).sum::<f32>() 
        / successful_results.len() as f32;
    
    AggregatedBenchmarkResult {
        algorithm: format!("{:?}", algorithm),
        avg_execution_time,
        min_execution_time,
        max_execution_time,
        success_rate,
        avg_num_hypotheses,
        avg_confidence,
        avg_strength,
        std_execution_time,
    }
}

/// Generate synthetic causal data for benchmarking
fn generate_synthetic_causal_data(config: &BenchmarkConfig) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    info!("Generating synthetic causal data: {} samples, {} variables, noise level {}",
          config.synthetic_data_size, config.synthetic_num_vars, config.synthetic_noise_level);
    
    let mut data = Vec::new();
    
    // Create a simple linear causal model: X1 -> X2 -> X3 -> X4 -> X5
    for _ in 0..config.synthetic_data_size {
        let mut sample = vec![0.0; config.synthetic_num_vars];
        
        // X1 is exogenous (random)
        sample[0] = rand::random::<f32>() * 2.0 - 1.0; // Range [-1, 1]
        
        // Each subsequent variable depends on the previous one
        for i in 1..config.synthetic_num_vars {
            let causal_effect = 0.7 * sample[i - 1]; // Strong causal relationship
            let noise = (rand::random::<f32>() - 0.5) * config.synthetic_noise_level;
            sample[i] = causal_effect + noise;
        }
        
        data.push(sample);
    }
    
    Ok(data)
}

/// Print a formatted summary of benchmark results
fn print_benchmark_summary(results: &[AggregatedBenchmarkResult]) {
    info!("📊 Causal Discovery Benchmark Results:");
    info!("═══════════════════════════════════════════════════════════════");
    
    for (i, result) in results.iter().enumerate() {
        let rank_emoji = match i {
            0 => "🥇", // Gold
            1 => "🥈", // Silver  
            2 => "🥉", // Bronze
            _ => "📈",
        };
        
        info!("{} {} Algorithm:", rank_emoji, result.algorithm);
        info!("   ⏱️  Avg Time: {:.2}ms (±{:.2}ms)", 
              result.avg_execution_time.as_millis(), 
              result.std_execution_time.as_millis());
        info!("   ✅ Success Rate: {:.1}%", result.success_rate * 100.0);
        info!("   🔍 Avg Hypotheses: {:.1}", result.avg_num_hypotheses);
        info!("   📊 Avg Confidence: {:.3}", result.avg_confidence);
        info!("   💪 Avg Strength: {:.3}", result.avg_strength);
        info!("   ⚡ Range: {:.2}ms - {:.2}ms", 
              result.min_execution_time.as_millis(),
              result.max_execution_time.as_millis());
        info!("───────────────────────────────────────────────────────────────");
    }
    
    // Performance insights
    if results.len() >= 2 {
        let fastest = &results[0];
        let slowest = &results[results.len() - 1];
        let speedup = slowest.avg_execution_time.as_millis() as f32 / fastest.avg_execution_time.as_millis() as f32;
        
        info!("🚀 Performance Insights:");
        info!("   {} is {:.1}x faster than {}", fastest.algorithm, speedup, slowest.algorithm);
        
        // Find most accurate algorithm
        let most_accurate = results.iter().max_by(|a, b| 
            (a.avg_confidence * a.success_rate).partial_cmp(&(b.avg_confidence * b.success_rate)).unwrap()
        ).unwrap();
        
        info!("   🎯 Most Accurate: {} (confidence: {:.3}, success: {:.1}%)",
              most_accurate.algorithm, most_accurate.avg_confidence, most_accurate.success_rate * 100.0);
        
        // Find best balance of speed and accuracy
        let best_balance = results.iter().min_by(|a, b| {
            let score_a = a.avg_execution_time.as_millis() as f32 / (a.avg_confidence * a.success_rate + 0.001);
            let score_b = b.avg_execution_time.as_millis() as f32 / (b.avg_confidence * b.success_rate + 0.001);
            score_a.partial_cmp(&score_b).unwrap()
        }).unwrap();
        
        info!("   ⚖️  Best Balance: {} (speed + accuracy)", best_balance.algorithm);
    }
    
    info!("═══════════════════════════════════════════════════════════════");
}

/// Specialized benchmark for comparing classical vs neural approaches
pub fn compare_classical_vs_neural_methods(
    data_matrix: Vec<Vec<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🔬 Classical vs Neural Causal Discovery Comparison");
    
    let classical_config = BenchmarkConfig {
        algorithms: vec![
            CausalAlgorithm::DirectLiNGAM,
            CausalAlgorithm::PC,
            CausalAlgorithm::GES,
        ],
        num_runs: 5,
        use_synthetic_data: false,
        ..Default::default()
    };
    
    let neural_config = BenchmarkConfig {
        algorithms: vec![
            CausalAlgorithm::NOTEARS,
            CausalAlgorithm::DECI,
        ],
        num_runs: 3, // Neural methods are slower
        use_synthetic_data: false,
        ..Default::default()
    };
    
    info!("Testing Classical Methods...");
    let classical_results = run_causal_discovery_benchmark(Some(data_matrix.clone()), Some(classical_config))?;
    
    info!("Testing Neural Methods...");  
    let neural_results = run_causal_discovery_benchmark(Some(data_matrix), Some(neural_config))?;
    
    // Comparative analysis
    let avg_classical_time: Duration = Duration::from_nanos(
        classical_results.iter().map(|r| r.avg_execution_time.as_nanos() as u64).sum::<u64>() 
        / classical_results.len() as u64
    );
    
    let avg_neural_time: Duration = Duration::from_nanos(
        neural_results.iter().map(|r| r.avg_execution_time.as_nanos() as u64).sum::<u64>() 
        / neural_results.len() as u64
    );
    
    let avg_classical_accuracy = classical_results.iter()
        .map(|r| r.avg_confidence * r.success_rate).sum::<f32>() / classical_results.len() as f32;
    
    let avg_neural_accuracy = neural_results.iter()
        .map(|r| r.avg_confidence * r.success_rate).sum::<f32>() / neural_results.len() as f32;
    
    info!("📈 Comparative Analysis:");
    info!("   Classical Methods - Avg Time: {:.2}ms, Avg Accuracy: {:.3}", 
          avg_classical_time.as_millis(), avg_classical_accuracy);
    info!("   Neural Methods    - Avg Time: {:.2}ms, Avg Accuracy: {:.3}", 
          avg_neural_time.as_millis(), avg_neural_accuracy);
    
    if avg_neural_time > avg_classical_time {
        let slowdown = avg_neural_time.as_millis() as f32 / avg_classical_time.as_millis() as f32;
        info!("   ⚡ Neural methods are {:.1}x slower but potentially more flexible", slowdown);
    }
    
    if avg_neural_accuracy > avg_classical_accuracy {
        let improvement = (avg_neural_accuracy - avg_classical_accuracy) / avg_classical_accuracy * 100.0;
        info!("   🎯 Neural methods show {:.1}% accuracy improvement", improvement);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_causal_data() {
        let config = BenchmarkConfig {
            synthetic_data_size: 100,
            synthetic_num_vars: 3,
            synthetic_noise_level: 0.1,
            ..Default::default()
        };
        
        let data = generate_synthetic_causal_data(&config).unwrap();
        assert_eq!(data.len(), 100);
        assert_eq!(data[0].len(), 3);
        
        // Check that there's some variation in the data
        let var_x1: f32 = data.iter().map(|row| row[0]).sum::<f32>() / data.len() as f32;
        let var_x2: f32 = data.iter().map(|row| row[1]).sum::<f32>() / data.len() as f32;
        
        // Due to causal relationship X1 -> X2, means should be related
        assert!((var_x1 * 0.7 - var_x2).abs() < 0.2); // Allow some noise tolerance
    }

    #[test] 
    fn test_benchmark_single_algorithm() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        
        let config = CausalDiscoveryConfig {
            min_strength_threshold: 0.5,
            min_confidence_threshold: 0.5,
            max_hypotheses: 5,
            algorithm: CausalAlgorithm::DirectLiNGAM,
        };
        
        let result = benchmark_single_algorithm(&data, &config);
        assert!(!result.algorithm.is_empty());
        assert!(result.execution_time.as_nanos() > 0);
    }
}