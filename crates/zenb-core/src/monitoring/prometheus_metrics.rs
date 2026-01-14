//! Prometheus metrics implementation.
//!
//! Uses lazy_static for thread-safe metric singletons.

use lazy_static::lazy_static;
use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Registry, TextEncoder};
use std::sync::Once;

static INIT: Once = Once::new();

lazy_static! {
    /// Global metric registry
    pub static ref REGISTRY: Registry = Registry::new();

    // === Performance Metrics ===
    
    /// Task processing latency histogram
    pub static ref TASK_LATENCY: Histogram = Histogram::with_opts(
        HistogramOpts::new("zenb_task_latency_seconds", "Task processing latency in seconds")
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5])
    ).unwrap();
    
    /// Total tasks processed counter
    pub static ref TASK_THROUGHPUT: Counter = Counter::new(
        "zenb_task_throughput_total", "Total number of tasks processed"
    ).unwrap();
    
    /// Total errors counter
    pub static ref ERROR_RATE: Counter = Counter::new(
        "zenb_error_total", "Total number of errors occurred"
    ).unwrap();

    // === Resource Metrics ===
    
    /// CPU utilization gauge (percentage)
    pub static ref CPU_UTILIZATION: Gauge = Gauge::new(
        "zenb_cpu_utilization_percent", "CPU utilization percentage"
    ).unwrap();
    
    /// Memory usage gauge (MB)
    pub static ref MEMORY_USAGE: Gauge = Gauge::new(
        "zenb_memory_usage_mb", "Memory usage in megabytes"
    ).unwrap();

    // === Quality Metrics ===
    
    /// Confidence score distribution histogram
    pub static ref CONFIDENCE_SCORE: Histogram = Histogram::with_opts(
        HistogramOpts::new("zenb_confidence_score", "Distribution of confidence scores")
            .buckets(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ).unwrap();

    // === Meta-Cognitive Metrics ===
    
    /// Self-correction counter
    pub static ref SELF_CORRECTION_RATE: Counter = Counter::new(
        "zenb_self_correction_total", "Total number of self-corrections triggered"
    ).unwrap();

    // === Active Inference Metrics ===
    
    /// Free energy gauge
    pub static ref FREE_ENERGY: Gauge = Gauge::new(
        "zenb_free_energy", "Current free energy value"
    ).unwrap();
    
    /// Prediction error histogram
    pub static ref PREDICTION_ERROR: Histogram = Histogram::with_opts(
        HistogramOpts::new("zenb_prediction_error", "Distribution of prediction errors")
            .buckets(vec![0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0])
    ).unwrap();
}

/// Register all metrics with the global registry.
///
/// This is idempotent — safe to call multiple times.
pub fn register_metrics() {
    INIT.call_once(|| {
        // Performance
        REGISTRY.register(Box::new(TASK_LATENCY.clone())).ok();
        REGISTRY.register(Box::new(TASK_THROUGHPUT.clone())).ok();
        REGISTRY.register(Box::new(ERROR_RATE.clone())).ok();
        
        // Resources
        REGISTRY.register(Box::new(CPU_UTILIZATION.clone())).ok();
        REGISTRY.register(Box::new(MEMORY_USAGE.clone())).ok();
        
        // Quality
        REGISTRY.register(Box::new(CONFIDENCE_SCORE.clone())).ok();
        
        // Meta-cognitive
        REGISTRY.register(Box::new(SELF_CORRECTION_RATE.clone())).ok();
        
        // Active Inference
        REGISTRY.register(Box::new(FREE_ENERGY.clone())).ok();
        REGISTRY.register(Box::new(PREDICTION_ERROR.clone())).ok();
    });
}

/// Gather all metrics in Prometheus text format.
///
/// Returns a string ready to be served at `/metrics` endpoint.
pub fn gather_metrics() -> String {
    let mut buffer = vec![];
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

// === Convenience Functions ===

/// Record task latency.
pub fn record_task_latency(seconds: f64) {
    TASK_LATENCY.observe(seconds);
}

/// Increment task throughput counter.
pub fn inc_task_throughput() {
    TASK_THROUGHPUT.inc();
}

/// Increment error counter.
pub fn inc_error_rate() {
    ERROR_RATE.inc();
}

/// Set CPU utilization percentage.
pub fn set_cpu_utilization(percent: f64) {
    CPU_UTILIZATION.set(percent);
}

/// Set memory usage in MB.
pub fn set_memory_usage(mb: f64) {
    MEMORY_USAGE.set(mb);
}

/// Record confidence score.
pub fn record_confidence_score(score: f64) {
    CONFIDENCE_SCORE.observe(score);
}

/// Increment self-correction counter.
pub fn inc_self_correction() {
    SELF_CORRECTION_RATE.inc();
}

/// Set free energy value.
pub fn set_free_energy(value: f64) {
    FREE_ENERGY.set(value);
}

/// Record prediction error.
pub fn record_prediction_error(error: f64) {
    PREDICTION_ERROR.observe(error);
}

/// RAII timer for auto-recording latency.
pub struct LatencyTimer {
    start: std::time::Instant,
}

impl LatencyTimer {
    /// Start a new latency timer.
    pub fn start() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for LatencyTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        record_task_latency(elapsed);
        inc_task_throughput();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_registration_is_idempotent() {
        // Should not panic on multiple calls
        register_metrics();
        register_metrics();
        register_metrics();
    }

    #[test]
    fn gather_metrics_returns_valid_format() {
        register_metrics();
        
        // Record some data
        record_task_latency(0.05);
        inc_task_throughput();
        record_confidence_score(0.85);
        
        let output = gather_metrics();
        
        // Should contain our metric names
        assert!(output.contains("zenb_task_latency_seconds"));
        assert!(output.contains("zenb_task_throughput_total"));
        assert!(output.contains("zenb_confidence_score"));
    }

    #[test]
    fn latency_timer_records_on_drop() {
        register_metrics();
        
        let before = TASK_THROUGHPUT.get();
        
        {
            let _timer = LatencyTimer::start();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        
        let after = TASK_THROUGHPUT.get();
        assert!(after > before, "Throughput should increment on timer drop");
    }

    #[test]
    fn free_energy_gauge_works() {
        register_metrics();
        
        set_free_energy(42.5);
        assert!((FREE_ENERGY.get() - 42.5).abs() < 1e-6);
        
        set_free_energy(0.0);
        assert!((FREE_ENERGY.get() - 0.0).abs() < 1e-6);
    }
}
