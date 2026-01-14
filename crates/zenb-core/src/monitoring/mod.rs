//! Prometheus metrics for production observability.
//!
//! Provides application-level metrics:
//! - Performance: task latency, throughput, error rate
//! - Resources: CPU utilization, memory usage
//! - Quality: confidence scores
//! - Meta-cognitive: self-correction rate
//!
//! # Feature Gating
//! This module is only available with the `prometheus` feature enabled.

#[cfg(feature = "prometheus")]
mod prometheus_metrics;

#[cfg(feature = "prometheus")]
pub use prometheus_metrics::*;

// Fallback no-op implementation when prometheus feature is disabled
#[cfg(not(feature = "prometheus"))]
mod noop {
    /// No-op metrics registration (feature disabled)
    pub fn register_metrics() {}

    /// No-op metrics gathering (feature disabled)
    pub fn gather_metrics() -> String {
        String::new()
    }

    /// Record task latency (no-op when feature disabled)
    pub fn record_task_latency(_seconds: f64) {}

    /// Increment task throughput (no-op when feature disabled)
    pub fn inc_task_throughput() {}

    /// Increment error count (no-op when feature disabled)
    pub fn inc_error_rate() {}

    /// Set CPU utilization (no-op when feature disabled)
    pub fn set_cpu_utilization(_percent: f64) {}

    /// Set memory usage (no-op when feature disabled)
    pub fn set_memory_usage(_mb: f64) {}

    /// Record confidence score (no-op when feature disabled)
    pub fn record_confidence_score(_score: f64) {}

    /// Increment self-correction counter (no-op when feature disabled)
    pub fn inc_self_correction() {}
}

#[cfg(not(feature = "prometheus"))]
pub use noop::*;
