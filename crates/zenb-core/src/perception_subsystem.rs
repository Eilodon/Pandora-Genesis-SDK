//! Perception Subsystem — Sensor Processing and Consensus
//!
//! Extracted from the Engine god-object to improve modularity.
//! Encapsulates:
//! - Skandha pipeline (Rupa stage for sensor consensus)
//! - Anomaly detection
//! - Adaptive thresholds for physiological signals
//! - Circuit breaker for numerical stability
//!
//! # Invariants
//! - All outputs are finite (no NaN/Inf)
//! - Energy values are bounded
//! - Anomaly detection is context-aware

use crate::adaptive::{AdaptiveThreshold, AnomalyDetector};
use crate::circuit_breaker::{CircuitBreakerConfig, CircuitBreakerManager};
use crate::config::ZenbConfig;
use crate::perception::PhysiologicalContext;
use crate::skandha::zenb::{zenb_pipeline, ZenbPipeline, ZenbRupa};
use crate::skandha::{ProcessedForm, RupaSkandha, SensorInput};

/// Error types for perception subsystem
#[derive(Debug, Clone)]
pub enum PerceptionError {
    /// Numerical overflow or NaN detected
    NumericalInstability {
        component: &'static str,
        detail: String,
    },
    /// Input validation failed
    InvalidInput { reason: String },
    /// Circuit breaker triggered
    CircuitBreakerOpen { name: String },
}

impl std::fmt::Display for PerceptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumericalInstability { component, detail } => {
                write!(f, "Numerical instability in {}: {}", component, detail)
            }
            Self::InvalidInput { reason } => write!(f, "Invalid input: {}", reason),
            Self::CircuitBreakerOpen { name } => write!(f, "Circuit breaker {} is open", name),
        }
    }
}

impl std::error::Error for PerceptionError {}

/// Result of perception processing
#[derive(Debug, Clone)]
pub struct PerceptionResult {
    /// Processed sensor values after consensus
    pub values: [f32; 5],
    /// Whether an anomaly was detected
    pub is_anomalous: bool,
    /// Energy level (measure of sensor disagreement)
    pub energy: f32,
    /// Detected physiological context
    pub context: PhysiologicalContext,
    /// Whether the data is reliable
    pub is_reliable: bool,
}

/// Perception Subsystem — extracted from Engine
///
/// Handles all sensor processing, consensus, and anomaly detection.
/// Uses the Skandha pipeline's Rupa stage for sensor fusion.
#[derive(Debug)]
pub struct PerceptionSubsystem {
    /// Skandha pipeline (contains Rupa for sensor processing)
    pipeline: ZenbPipeline,

    /// Adaptive threshold for respiration rate minimum
    rr_min_threshold: AdaptiveThreshold,

    /// Adaptive threshold for respiration rate maximum
    rr_max_threshold: AdaptiveThreshold,

    /// Anomaly detector for sensor readings
    anomaly_detector: AnomalyDetector,

    /// Circuit breaker for numerical stability
    circuit_breaker: CircuitBreakerManager,

    /// Last detected energy level
    last_energy: f32,

    /// Last detected context
    last_context: PhysiologicalContext,
}

impl PerceptionSubsystem {
    /// Create a new perception subsystem with default configuration
    pub fn new() -> Self {
        Self::with_config(&ZenbConfig::default())
    }

    /// Create a new perception subsystem with configuration
    pub fn with_config(config: &ZenbConfig) -> Self {
        // CircuitBreakerManager creates circuits on-demand, no registration needed
        let circuit_breaker = CircuitBreakerManager::new(CircuitBreakerConfig {
            failure_threshold: 3,
            open_cooldown_ms: 5000,
            half_open_trial: 2,
            max_circuits: 100,
            state_ttl_secs: 3600,
        });

        // Default thresholds for respiration rate (breaths per minute)
        const RR_MIN_DEFAULT: f32 = 4.0;
        const RR_MIN_LOWER: f32 = 2.0;
        const RR_MIN_UPPER: f32 = 8.0;
        const RR_MAX_DEFAULT: f32 = 20.0;
        const RR_MAX_LOWER: f32 = 15.0;
        const RR_MAX_UPPER: f32 = 30.0;
        const ANOMALY_WINDOW_SIZE: usize = 20;
        const DEFAULT_LEARNING_RATE: f32 = 0.1;

        Self {
            pipeline: crate::skandha::zenb::zenb_pipeline(config),
            rr_min_threshold: AdaptiveThreshold::new(
                RR_MIN_DEFAULT,
                RR_MIN_LOWER,
                RR_MIN_UPPER,
                DEFAULT_LEARNING_RATE,
            ),
            rr_max_threshold: AdaptiveThreshold::new(
                RR_MAX_DEFAULT,
                RR_MAX_LOWER,
                RR_MAX_UPPER,
                DEFAULT_LEARNING_RATE,
            ),
            anomaly_detector: AnomalyDetector::new(ANOMALY_WINDOW_SIZE, 2.0),
            circuit_breaker,
            last_energy: 0.0,
            last_context: PhysiologicalContext::Rest,
        }
    }

    /// Process raw sensor values through the perception pipeline
    ///
    /// # Arguments
    /// * `hr_bpm` - Heart rate in BPM (optional)
    /// * `hrv_rmssd` - HRV RMSSD in ms (optional)
    /// * `rr_bpm` - Respiratory rate in BPM (optional)
    /// * `quality` - Signal quality (0.0 - 1.0)
    /// * `motion` - Motion level (0.0 - 1.0)
    /// * `timestamp_us` - Timestamp in microseconds
    ///
    /// # Returns
    /// `PerceptionResult` with processed values, anomaly status, and context
    pub fn process(
        &mut self,
        hr_bpm: Option<f32>,
        hrv_rmssd: Option<f32>,
        rr_bpm: Option<f32>,
        quality: f32,
        motion: f32,
        timestamp_us: i64,
    ) -> Result<PerceptionResult, PerceptionError> {
        // Validate inputs
        self.validate_inputs(hr_bpm, hrv_rmssd, rr_bpm, quality, motion)?;

        // Normalize sensor values
        let hr_norm = hr_bpm.unwrap_or(60.0) / 200.0;
        let hrv_norm = hrv_rmssd.unwrap_or(50.0) / 100.0;

        // Detect physiological context
        let context = ZenbRupa::detect_context(hr_norm, hrv_norm, motion);
        self.last_context = context;

        // Update context on pipeline's rupa stage
        self.pipeline.rupa.set_context(context);

        // Create sensor input
        let input = SensorInput {
            hr_bpm,
            hrv_rmssd,
            rr_bpm,
            quality,
            motion,
            timestamp_us,
        };

        // Process through Skandha Rupa stage with circuit breaker
        let processed = self.process_with_circuit_breaker(&input)?;

        // Update last energy
        self.last_energy = processed.energy;

        // Detect anomalies
        let anomaly_score = self.anomaly_detector.score(processed.energy);
        let is_anomalous = anomaly_score > 0.0;

        Ok(PerceptionResult {
            values: processed.values,
            is_anomalous,
            energy: processed.energy,
            context,
            is_reliable: processed.is_reliable && !is_anomalous,
        })
    }

    /// Process with circuit breaker protection
    fn process_with_circuit_breaker(
        &mut self,
        input: &SensorInput,
    ) -> Result<ProcessedForm, PerceptionError> {
        // Check if circuit breaker is open (requests should be rejected)
        if self.circuit_breaker.is_open("sheaf_perception") {
            return Err(PerceptionError::CircuitBreakerOpen {
                name: "sheaf_perception".to_string(),
            });
        }

        // Try processing with panic protection
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.pipeline.rupa.process_form(input)
        }));

        match result {
            Ok(form) => {
                // Validate output is finite
                if form.values.iter().any(|v| !v.is_finite()) {
                    self.circuit_breaker.record_failure("sheaf_perception");
                    return Err(PerceptionError::NumericalInstability {
                        component: "sheaf_perception",
                        detail: "Non-finite values in output".to_string(),
                    });
                }
                self.circuit_breaker.record_success("sheaf_perception");
                Ok(form)
            }
            Err(e) => {
                self.circuit_breaker.record_failure("sheaf_perception");
                let detail = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                Err(PerceptionError::NumericalInstability {
                    component: "sheaf_perception",
                    detail,
                })
            }
        }
    }

    /// Validate input values
    fn validate_inputs(
        &self,
        hr_bpm: Option<f32>,
        hrv_rmssd: Option<f32>,
        rr_bpm: Option<f32>,
        quality: f32,
        motion: f32,
    ) -> Result<(), PerceptionError> {
        // Check for NaN/Inf
        if let Some(hr) = hr_bpm {
            if !hr.is_finite() || hr < 0.0 || hr > 300.0 {
                return Err(PerceptionError::InvalidInput {
                    reason: format!("HR out of range: {}", hr),
                });
            }
        }
        if let Some(hrv) = hrv_rmssd {
            if !hrv.is_finite() || hrv < 0.0 {
                return Err(PerceptionError::InvalidInput {
                    reason: format!("HRV out of range: {}", hrv),
                });
            }
        }
        if let Some(rr) = rr_bpm {
            if !rr.is_finite() || rr < 0.0 || rr > 60.0 {
                return Err(PerceptionError::InvalidInput {
                    reason: format!("RR out of range: {}", rr),
                });
            }
        }
        if !quality.is_finite() || quality < 0.0 || quality > 1.0 {
            return Err(PerceptionError::InvalidInput {
                reason: format!("Quality out of range: {}", quality),
            });
        }
        if !motion.is_finite() || motion < 0.0 || motion > 1.0 {
            return Err(PerceptionError::InvalidInput {
                reason: format!("Motion out of range: {}", motion),
            });
        }
        Ok(())
    }

    /// Get the last detected energy level
    pub fn last_energy(&self) -> f32 {
        self.last_energy
    }

    /// Get the last detected physiological context
    pub fn last_context(&self) -> PhysiologicalContext {
        self.last_context
    }

    /// Get the current RR minimum threshold
    pub fn rr_min_threshold(&self) -> f32 {
        self.rr_min_threshold.get()
    }

    /// Get the current RR maximum threshold
    pub fn rr_max_threshold(&self) -> f32 {
        self.rr_max_threshold.get()
    }

    /// Adapt RR thresholds based on performance delta
    /// Positive delta = good performance, negative = bad
    pub fn adapt_rr_thresholds(&mut self, performance_delta: f32) {
        self.rr_min_threshold.adapt(performance_delta);
        self.rr_max_threshold.adapt(performance_delta);
    }

    /// Check if circuit breaker is healthy (not open)
    pub fn is_healthy(&self) -> bool {
        !self.circuit_breaker.is_open("sheaf_perception")
    }

    /// Get reference to the internal pipeline for advanced usage
    pub fn pipeline(&self) -> &ZenbPipeline {
        &self.pipeline
    }

    /// Get mutable reference to the internal pipeline
    pub fn pipeline_mut(&mut self) -> &mut ZenbPipeline {
        &mut self.pipeline
    }
}

impl Default for PerceptionSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_subsystem_basic() {
        let mut perception = PerceptionSubsystem::new();

        let result = perception.process(Some(70.0), Some(45.0), Some(12.0), 0.9, 0.1, 1_000_000);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.values.iter().all(|v| v.is_finite()));
        assert!(result.energy >= 0.0);
    }

    #[test]
    fn test_context_detection() {
        let mut perception = PerceptionSubsystem::new();

        // High motion should detect exercise
        let result = perception
            .process(Some(140.0), Some(25.0), Some(18.0), 0.8, 0.8, 1_000_000)
            .unwrap();

        assert!(matches!(
            result.context,
            PhysiologicalContext::ModerateExercise | PhysiologicalContext::IntenseExercise
        ));
    }

    #[test]
    fn test_invalid_input_rejected() {
        let mut perception = PerceptionSubsystem::new();

        // NaN HR should be rejected
        let result =
            perception.process(Some(f32::NAN), Some(45.0), Some(12.0), 0.9, 0.1, 1_000_000);

        assert!(result.is_err());
    }

    #[test]
    fn test_out_of_range_rejected() {
        let mut perception = PerceptionSubsystem::new();

        // HR > 300 should be rejected
        let result = perception.process(Some(400.0), Some(45.0), Some(12.0), 0.9, 0.1, 1_000_000);

        assert!(result.is_err());
    }
}
