//! LTL Safety Monitor - Runtime verification
//!
//! Reference: SafetyMonitor.ts (~300 lines)

use std::sync::Arc;

/// Safety violation record
#[derive(Debug, Clone)]
pub struct SafetyViolation {
    pub timestamp: u64,
    pub property_name: String,
    pub description: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Critical,
    Warning,
}

/// Runtime state snapshot
#[derive(Debug, Clone)]
pub struct RuntimeState {
    pub tempo_scale: f32,
    pub status: String,
    pub session_duration: f32,
    pub prediction_error: f32,
    pub last_update_timestamp: u64,
}

/// Safety property (LTL formula)
pub struct SafetyProperty {
    pub name: &'static str,
    pub description: &'static str,
    pub predicate: Arc<dyn Fn(&RuntimeState) -> bool + Send + Sync>,
}

impl SafetyProperty {
    pub fn new(
        name: &'static str,
        description: &'static str,
        predicate: impl Fn(&RuntimeState) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            description,
            predicate: Arc::new(predicate),
        }
    }
}

/// Safety Monitor
pub struct SafetyMonitor {
    properties: Vec<SafetyProperty>,
    violations: Vec<SafetyViolation>,
    max_violations: usize,
}

impl SafetyMonitor {
    pub fn new() -> Self {
        let mut properties = Vec::new();

        // Property 1: Tempo bounds [0.8, 1.4]
        properties.push(SafetyProperty::new(
            "tempo_bounds",
            "Tempo must stay within [0.8, 1.4]",
            |state| state.tempo_scale >= 0.8 && state.tempo_scale <= 1.4,
        ));

        // Property 2: Safety lock immutable
        properties.push(SafetyProperty::new(
            "safety_lock_immutable",
            "Once in SAFETY_LOCK, cannot start new session",
            |state| state.status != "SAFETY_LOCK" || state.session_duration == 0.0,
        ));

        // Property 3: Panic halt
        properties.push(SafetyProperty::new(
            "panic_halt",
            "High prediction error must trigger halt",
            |state| {
                if state.prediction_error > 0.95 && state.session_duration > 10.0 {
                    state.status == "HALTED" || state.status == "SAFETY_LOCK"
                } else {
                    true
                }
            },
        ));

        Self {
            properties,
            violations: Vec::new(),
            max_violations: 100,
        }
    }

    /// Check if state satisfies all safety properties
    pub fn check(&mut self, state: &RuntimeState) -> Result<(), Vec<SafetyViolation>> {
        let mut violations = Vec::new();

        for prop in &self.properties {
            if !(prop.predicate)(state) {
                let violation = SafetyViolation {
                    timestamp: state.last_update_timestamp,
                    property_name: prop.name.to_string(),
                    description: prop.description.to_string(),
                    severity: Severity::Critical,
                };

                violations.push(violation);
            }
        }

        // Record all violations after iteration
        for v in &violations {
            self.record_violation(v.clone());
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    /// Shield (correct) a tempo value to be safe
    pub fn shield_tempo(&self, tempo: f32) -> f32 {
        tempo.clamp(0.8, 1.4)
    }

    fn record_violation(&mut self, violation: SafetyViolation) {
        self.violations.push(violation);
        if self.violations.len() > self.max_violations {
            self.violations.remove(0);
        }
    }

    pub fn get_violations(&self) -> &[SafetyViolation] {
        &self.violations
    }

    pub fn clear_violations(&mut self) {
        self.violations.clear();
    }
}

impl Default for SafetyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tempo_bounds_valid() {
        let mut monitor = SafetyMonitor::new();
        let state = RuntimeState {
            tempo_scale: 1.0,
            status: "RUNNING".into(),
            session_duration: 30.0,
            prediction_error: 0.1,
            last_update_timestamp: 1000,
        };

        assert!(monitor.check(&state).is_ok());
    }

    #[test]
    fn test_tempo_bounds_violation() {
        let mut monitor = SafetyMonitor::new();
        let state = RuntimeState {
            tempo_scale: 2.0, // Out of bounds!
            status: "RUNNING".into(),
            session_duration: 30.0,
            prediction_error: 0.1,
            last_update_timestamp: 1000,
        };

        let result = monitor.check(&state);
        assert!(result.is_err());

        let violations = result.unwrap_err();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].property_name, "tempo_bounds");
    }

    #[test]
    fn test_shield_tempo() {
        let monitor = SafetyMonitor::new();

        assert_eq!(monitor.shield_tempo(0.5), 0.8);
        assert_eq!(monitor.shield_tempo(1.0), 1.0);
        assert_eq!(monitor.shield_tempo(2.0), 1.4);
    }

    #[test]
    fn test_panic_halt() {
        let mut monitor = SafetyMonitor::new();

        // High error but session too short - should pass
        let state1 = RuntimeState {
            tempo_scale: 1.0,
            status: "RUNNING".into(),
            session_duration: 5.0,
            prediction_error: 0.96,
            last_update_timestamp: 1000,
        };
        assert!(monitor.check(&state1).is_ok());

        // High error, long session, still running - should fail
        let state2 = RuntimeState {
            tempo_scale: 1.0,
            status: "RUNNING".into(),
            session_duration: 15.0,
            prediction_error: 0.96,
            last_update_timestamp: 2000,
        };
        assert!(monitor.check(&state2).is_err());

        // High error, long session, halted - should pass
        let state3 = RuntimeState {
            tempo_scale: 1.0,
            status: "HALTED".into(),
            session_duration: 15.0,
            prediction_error: 0.96,
            last_update_timestamp: 3000,
        };
        assert!(monitor.check(&state3).is_ok());
    }
}
