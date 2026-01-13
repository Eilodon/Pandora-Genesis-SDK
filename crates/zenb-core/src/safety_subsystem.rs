//! Safety Subsystem — Runtime Verification and Ethical Filtering
//!
//! Extracted from the Engine god-object to improve modularity.
//! Encapsulates:
//! - SafetyMonitor for LTL runtime verification
//! - TraumaRegistry for trigger pattern detection
//! - DharmaFilter for phase-based ethical action filtering
//!
//! # Invariants
//! - Safety violations are always logged
//! - Tempo is always within safe bounds [0.8, 1.4]
//! - Ethically misaligned actions are vetoed

use crate::config::ZenbConfig;
use crate::safety::{
    AlignmentCategory, DharmaFilter, RuntimeState, SafetyMonitor, SafetyViolation, Severity,
};
use crate::safety_swarm::TraumaRegistry;
use crate::trauma_cache::TraumaCache;
use num_complex::Complex32;

/// Error types for safety subsystem
#[derive(Debug, Clone)]
pub enum SafetyError {
    /// Safety property violated
    Violation { violations: Vec<SafetyViolation> },
    /// Ethical alignment rejected (Dharma veto)
    DharmaVeto { action_phase: f32, alignment: f32 },
    /// Trauma trigger detected
    TraumaTriggered { pattern_id: String, intensity: f32 },
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Violation { violations } => {
                write!(f, "Safety violations: ")?;
                for v in violations {
                    write!(f, "[{}] ", v.property_name)?;
                }
                Ok(())
            }
            Self::DharmaVeto {
                action_phase,
                alignment,
            } => {
                write!(
                    f,
                    "Dharma veto: action phase={:.2}rad, alignment={:.2}",
                    action_phase, alignment
                )
            }
            Self::TraumaTriggered {
                pattern_id,
                intensity,
            } => {
                write!(
                    f,
                    "Trauma triggered: {} (intensity={:.2})",
                    pattern_id, intensity
                )
            }
        }
    }
}

impl std::error::Error for SafetyError {}

/// Result of safety check
#[derive(Debug, Clone)]
pub struct SafetyCheckResult {
    /// Whether all safety checks passed
    pub is_safe: bool,
    /// Sanctioned (filtered) tempo value
    pub tempo: f32,
    /// Dharma alignment score
    pub alignment: f32,
    /// Alignment category
    pub category: AlignmentCategory,
    /// Any violations (even non-critical)
    pub warnings: Vec<String>,
}

/// Safety Subsystem — extracted from Engine
///
/// Handles all safety verification, ethical filtering, and trauma detection.
pub struct SafetySubsystem {
    /// LTL safety monitor
    monitor: SafetyMonitor,

    /// Dharma filter for ethical action filtering
    dharma: DharmaFilter,

    /// Trauma registry for trigger detection
    trauma_registry: TraumaRegistry,

    /// Trauma cache for fast lookups
    trauma_cache: TraumaCache,

    /// Last check result
    last_alignment: f32,

    /// Count of violations in current session
    violation_count: u32,
}

impl SafetySubsystem {
    /// Create a new safety subsystem with default configuration
    pub fn new() -> Self {
        Self::with_config(&ZenbConfig::default())
    }

    /// Create a new safety subsystem with configuration
    pub fn with_config(config: &ZenbConfig) -> Self {
        // Wire device secret from config if persisted
        let trauma_registry = if let Some(secret) = config.safety.device_secret {
            TraumaRegistry::with_secret(secret)
        } else {
            TraumaRegistry::new()
        };

        Self {
            monitor: SafetyMonitor::new(),
            dharma: DharmaFilter::default_for_zenb(),
            trauma_registry,
            trauma_cache: TraumaCache::default(),
            last_alignment: 1.0,
            violation_count: 0,
        }
    }

    /// Run safety checks on current runtime state
    ///
    /// # Arguments
    /// * `state` - Current runtime state snapshot
    ///
    /// # Returns
    /// `SafetyCheckResult` with sanctioned values and any warnings
    pub fn check(&mut self, state: &RuntimeState) -> Result<SafetyCheckResult, SafetyError> {
        let mut warnings = Vec::new();

        // Run LTL safety monitor
        if let Err(violations) = self.monitor.check(state) {
            self.violation_count += violations.len() as u32;

            // Check if any critical
            let has_critical = violations.iter().any(|v| v.severity == Severity::Critical);

            if has_critical {
                return Err(SafetyError::Violation { violations });
            } else {
                // Non-critical - add to warnings
                for v in &violations {
                    warnings.push(format!("{}: {}", v.property_name, v.description));
                }
            }
        }

        // Shield tempo to safe bounds
        let safe_tempo = self.monitor.shield_tempo(state.tempo_scale);

        // Check Dharma alignment (using tempo as action magnitude)
        let action = Complex32::new(safe_tempo, 0.0);
        let alignment = self.dharma.check_alignment(action);
        let category = self.dharma.alignment_category(action);
        self.last_alignment = alignment;

        if alignment < 0.0 {
            warnings.push(format!("Low Dharma alignment: {:.2}", alignment));
        }

        Ok(SafetyCheckResult {
            is_safe: warnings.is_empty(),
            tempo: safe_tempo,
            alignment,
            category,
            warnings,
        })
    }

    /// Sanction an action through Dharma filter
    ///
    /// # Returns
    /// `Some(action)` if sanctioned, `None` if vetoed
    pub fn sanction_action(&mut self, action: Complex32) -> Result<Complex32, SafetyError> {
        match self.dharma.sanction(action) {
            Some(sanctioned) => Ok(sanctioned),
            None => {
                let alignment = self.dharma.check_alignment(action);
                Err(SafetyError::DharmaVeto {
                    action_phase: action.arg(),
                    alignment,
                })
            }
        }
    }

    /// Check for trauma triggers in content
    ///
    /// # Arguments
    /// * `sig_hash` - Signature hash of the context (32 bytes)
    /// * `now_ts_us` - Current timestamp in microseconds
    ///
    /// # Returns
    /// `Ok(())` if safe, `Err` if trauma triggered
    pub fn check_trauma(&self, sig_hash: &[u8; 32], now_ts_us: i64) -> Result<(), SafetyError> {
        use crate::safety_swarm::TraumaSource;

        // Check registry via TraumaSource trait
        if let Ok(Some(hit)) = self.trauma_registry.query_trauma(sig_hash, now_ts_us) {
            // Check if still inhibited
            if now_ts_us < hit.inhibit_until_ts_us {
                return Err(SafetyError::TraumaTriggered {
                    pattern_id: format!("{:x?}", &sig_hash[..8]),
                    intensity: hit.sev_eff,
                });
            }
        }

        Ok(())
    }

    /// Get current Dharma alignment
    pub fn alignment(&self) -> f32 {
        self.last_alignment
    }

    /// Get violation count
    pub fn violation_count(&self) -> u32 {
        self.violation_count
    }

    /// Get safety violations history
    pub fn get_violations(&self) -> &[SafetyViolation] {
        self.monitor.get_violations()
    }

    /// Clear all violations
    pub fn clear_violations(&mut self) {
        self.monitor.clear_violations();
        self.violation_count = 0;
    }

    /// Get reference to trauma registry
    pub fn trauma_registry(&self) -> &TraumaRegistry {
        &self.trauma_registry
    }

    /// Get mutable reference to trauma registry (for registration)
    pub fn trauma_registry_mut(&mut self) -> &mut TraumaRegistry {
        &mut self.trauma_registry
    }

    /// Get reference to trauma cache
    pub fn trauma_cache(&self) -> &TraumaCache {
        &self.trauma_cache
    }

    /// Get mutable reference to trauma cache
    pub fn trauma_cache_mut(&mut self) -> &mut TraumaCache {
        &mut self.trauma_cache
    }

    /// Get reference to Dharma filter
    pub fn dharma(&self) -> &DharmaFilter {
        &self.dharma
    }

    /// Get reference to safety monitor
    pub fn monitor(&self) -> &SafetyMonitor {
        &self.monitor
    }

    /// Get mutable reference to safety monitor (for check which updates violations)
    pub fn monitor_mut(&mut self) -> &mut SafetyMonitor {
        &mut self.monitor
    }
}

impl Default for SafetySubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SafetySubsystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetySubsystem")
            .field("last_alignment", &self.last_alignment)
            .field("violation_count", &self.violation_count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_check_valid_state() {
        let mut safety = SafetySubsystem::new();

        let state = RuntimeState {
            tempo_scale: 1.0,
            status: "RUNNING".to_string(),
            session_duration: 30.0,
            prediction_error: 0.1,
            last_update_timestamp: 1000,
        };

        let result = safety.check(&state);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.is_safe);
        assert_eq!(result.tempo, 1.0);
    }

    #[test]
    fn test_tempo_shielding() {
        let mut safety = SafetySubsystem::new();

        // Out of bounds tempo should be shielded
        let state = RuntimeState {
            tempo_scale: 2.0, // Too high!
            status: "RUNNING".to_string(),
            session_duration: 5.0, // Short session so panic_halt doesn't trigger
            prediction_error: 0.1,
            last_update_timestamp: 1000,
        };

        let result = safety.check(&state);
        // Should be error due to tempo_bounds violation
        assert!(result.is_err());
    }

    #[test]
    fn test_dharma_sanction() {
        let mut safety = SafetySubsystem::new();

        // Good action (positive real = calming)
        let good_action = Complex32::new(1.0, 0.0);
        assert!(safety.sanction_action(good_action).is_ok());

        // Bad action (negative real = harmful)
        let bad_action = Complex32::new(-1.0, 0.0);
        assert!(safety.sanction_action(bad_action).is_err());
    }

    #[test]
    fn test_alignment_tracking() {
        let mut safety = SafetySubsystem::new();

        let state = RuntimeState {
            tempo_scale: 1.0,
            status: "RUNNING".to_string(),
            session_duration: 30.0,
            prediction_error: 0.1,
            last_update_timestamp: 1000,
        };

        let _ = safety.check(&state);

        // Alignment should be tracked
        assert!(safety.alignment() >= -1.0 && safety.alignment() <= 1.0);
    }
}
