//! Triple Guardians: Physical, Cognitive, and Ethical Safety Layers
//!
//! Implements a layered safety system inspired by biological homeostatic mechanisms.
//! Each guardian layer monitors a different aspect of system health:
//!
//! 1. **HamiltonianGuard** (Physical): Battery, temperature, hardware constraints
//! 2. **FepMonitor** (Cognitive): Free Energy/Prediction error monitoring
//! 3. **DharmaFilter** (Ethical): Already implemented, integrated here
//!
//! # VAJRA V5 Architecture: SAṄKHĀRA + VEDANĀ
//!
//! The guardians form a hierarchical safety plane:
//! - Physical layer has highest priority (prevents hardware damage)
//! - Cognitive layer protects against runaway inference
//! - Ethical layer ensures actions align with values
//!
//! # Safety Guarantees
//! - Physical constraints are checked before any action
//! - Cognitive overload triggers graceful degradation
//! - Ethical violations are hard-blocked

use crate::safety::DharmaFilter;
use num_complex::Complex32;
use serde::{Deserialize, Serialize};

/// Guardian decision levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardianDecision {
    /// Normal operation - all systems nominal
    Normal,
    /// Cautious mode - reduce gain, increase monitoring
    Cautious {
        /// Factor to reduce action intensity (0.0-1.0)
        gain_reduction: u8, // Stored as percentage (0-100)
    },
    /// Sleep/hibernation mode - minimal activity
    Sleep,
    /// Hard halt - stop all non-essential operations
    Halt,
}

impl GuardianDecision {
    /// Create cautious decision with gain reduction
    pub fn cautious(gain: f32) -> Self {
        GuardianDecision::Cautious {
            gain_reduction: (gain.clamp(0.0, 1.0) * 100.0) as u8,
        }
    }

    /// Get gain reduction factor (0.0-1.0)
    pub fn gain_factor(&self) -> f32 {
        match self {
            GuardianDecision::Normal => 1.0,
            GuardianDecision::Cautious { gain_reduction } => *gain_reduction as f32 / 100.0,
            GuardianDecision::Sleep => 0.0,
            GuardianDecision::Halt => 0.0,
        }
    }

    /// Check if operation should proceed
    pub fn should_proceed(&self) -> bool {
        matches!(self, GuardianDecision::Normal | GuardianDecision::Cautious { .. })
    }
}

/// Physical state for HamiltonianGuard
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicalState {
    /// Battery level (0.0-1.0)
    pub battery_level: f32,
    /// Temperature in Celsius
    pub temperature_c: f32,
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f32,
    /// Memory usage (0.0-1.0)
    pub memory_usage: f32,
    /// Whether device is charging
    pub is_charging: bool,
}

impl PhysicalState {
    /// Create mock state for testing
    pub fn mock_normal() -> Self {
        Self {
            battery_level: 0.8,
            temperature_c: 35.0,
            cpu_usage: 0.3,
            memory_usage: 0.4,
            is_charging: false,
        }
    }

    /// Create mock low battery state for testing
    pub fn mock_low_battery() -> Self {
        Self {
            battery_level: 0.05,
            temperature_c: 35.0,
            cpu_usage: 0.3,
            memory_usage: 0.4,
            is_charging: false,
        }
    }

    /// Create mock high temperature state for testing
    pub fn mock_high_temp() -> Self {
        Self {
            battery_level: 0.8,
            temperature_c: 48.0,
            cpu_usage: 0.9,
            memory_usage: 0.4,
            is_charging: false,
        }
    }
}

/// Configuration for HamiltonianGuard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HamiltonianConfig {
    /// Battery threshold for Sleep mode (default: 0.10 = 10%)
    pub battery_critical: f32,
    /// Battery threshold for Cautious mode (default: 0.20 = 20%)
    pub battery_low: f32,
    /// Temperature threshold for Sleep mode (default: 45.0°C)
    pub temp_critical: f32,
    /// Temperature threshold for Cautious mode (default: 40.0°C)
    pub temp_high: f32,
    /// CPU usage threshold for Cautious mode (default: 0.90)
    pub cpu_high: f32,
    /// Memory usage threshold for Cautious mode (default: 0.85)
    pub memory_high: f32,
}

impl Default for HamiltonianConfig {
    fn default() -> Self {
        Self {
            battery_critical: 0.10,
            battery_low: 0.20,
            temp_critical: 45.0,
            temp_high: 40.0,
            cpu_high: 0.90,
            memory_high: 0.85,
        }
    }
}

/// Physical layer guardian (Layer 1)
///
/// Monitors hardware constraints and enforces physical safety:
/// - Battery level
/// - Temperature
/// - CPU/Memory usage
#[derive(Debug, Clone)]
pub struct HamiltonianGuard {
    config: HamiltonianConfig,
    last_state: Option<PhysicalState>,
}

impl Default for HamiltonianGuard {
    fn default() -> Self {
        Self::new(HamiltonianConfig::default())
    }
}

impl HamiltonianGuard {
    /// Create new guard with configuration
    pub fn new(config: HamiltonianConfig) -> Self {
        Self {
            config,
            last_state: None,
        }
    }

    /// Evaluate physical state and return decision
    pub fn evaluate(&mut self, state: &PhysicalState) -> GuardianDecision {
        self.last_state = Some(state.clone());

        // Critical battery (not charging) -> Sleep
        if state.battery_level < self.config.battery_critical && !state.is_charging {
            return GuardianDecision::Sleep;
        }

        // Critical temperature -> Sleep
        if state.temperature_c > self.config.temp_critical {
            return GuardianDecision::Sleep;
        }

        // Low battery -> Cautious with reduced gain
        if state.battery_level < self.config.battery_low && !state.is_charging {
            let gain = state.battery_level / self.config.battery_low;
            return GuardianDecision::cautious(gain);
        }

        // High temperature -> Cautious
        if state.temperature_c > self.config.temp_high {
            let normalized = (state.temperature_c - self.config.temp_high)
                / (self.config.temp_critical - self.config.temp_high);
            let gain = 1.0 - normalized.clamp(0.0, 0.5);
            return GuardianDecision::cautious(gain);
        }

        // High CPU/Memory -> Cautious
        if state.cpu_usage > self.config.cpu_high || state.memory_usage > self.config.memory_high {
            let max_usage = state.cpu_usage.max(state.memory_usage);
            let gain = 1.0 - (max_usage - 0.8) * 2.0; // Scale down from 80%
            return GuardianDecision::cautious(gain.max(0.5));
        }

        GuardianDecision::Normal
    }

    /// Get last observed state
    pub fn last_state(&self) -> Option<&PhysicalState> {
        self.last_state.as_ref()
    }
}

/// Configuration for FepMonitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepConfig {
    /// Error spike threshold (relative increase from baseline)
    pub spike_threshold: f32,
    /// Baseline error EMA alpha
    pub ema_alpha: f32,
    /// Error threshold for Halt
    pub halt_threshold: f32,
    /// Error threshold for Cautious
    pub cautious_threshold: f32,
}

impl Default for FepConfig {
    fn default() -> Self {
        Self {
            spike_threshold: 0.5,   // 50% increase triggers Cautious
            ema_alpha: 0.1,         // Smoothing factor for baseline
            halt_threshold: 0.95,   // 95% error -> Halt
            cautious_threshold: 0.7, // 70% error -> Cautious
        }
    }
}

/// Cognitive layer guardian (Layer 2)
///
/// Monitors Free Energy Principle prediction error:
/// - Sudden spikes indicate model-reality mismatch
/// - Sustained high error indicates need for caution
#[derive(Debug, Clone)]
pub struct FepMonitor {
    config: FepConfig,
    /// Baseline error (EMA)
    baseline_error: f32,
    /// Last observed error
    last_error: f32,
    /// Number of observations
    observation_count: u64,
}

impl Default for FepMonitor {
    fn default() -> Self {
        Self::new(FepConfig::default())
    }
}

impl FepMonitor {
    /// Create new monitor with configuration
    pub fn new(config: FepConfig) -> Self {
        Self {
            config,
            baseline_error: 0.5,
            last_error: 0.0,
            observation_count: 0,
        }
    }

    /// Update with new prediction error and return decision
    ///
    /// # Arguments
    /// * `error` - Current prediction error in [0, 1]
    pub fn evaluate(&mut self, error: f32) -> GuardianDecision {
        self.observation_count += 1;
        self.last_error = error.clamp(0.0, 1.0);

        // Update baseline EMA
        if self.observation_count > 1 {
            self.baseline_error = self.config.ema_alpha * self.last_error
                + (1.0 - self.config.ema_alpha) * self.baseline_error;
        } else {
            self.baseline_error = self.last_error;
        }

        // Check for spike (sudden increase from baseline)
        let relative_increase = if self.baseline_error > 0.01 {
            (self.last_error - self.baseline_error) / self.baseline_error
        } else {
            0.0
        };

        // Halt threshold
        if self.last_error > self.config.halt_threshold {
            return GuardianDecision::Halt;
        }

        // Spike detection -> Cautious
        if relative_increase > self.config.spike_threshold {
            let gain = 1.0 - (relative_increase - self.config.spike_threshold).clamp(0.0, 0.5);
            return GuardianDecision::cautious(gain);
        }

        // Sustained high error -> Cautious
        if self.last_error > self.config.cautious_threshold {
            let normalized = (self.last_error - self.config.cautious_threshold)
                / (self.config.halt_threshold - self.config.cautious_threshold);
            let gain = 1.0 - normalized * 0.5;
            return GuardianDecision::cautious(gain);
        }

        GuardianDecision::Normal
    }

    /// Get current baseline error
    pub fn baseline_error(&self) -> f32 {
        self.baseline_error
    }

    /// Get last observed error
    pub fn last_error(&self) -> f32 {
        self.last_error
    }

    /// Reset the monitor
    pub fn reset(&mut self) {
        self.baseline_error = 0.5;
        self.last_error = 0.0;
        self.observation_count = 0;
    }
}

/// Triple Guardians: Unified safety controller
///
/// Combines all three safety layers with hierarchical evaluation:
/// 1. Physical (highest priority)
/// 2. Cognitive
/// 3. Ethical (lowest priority but hard-block on violations)
#[derive(Debug)]
pub struct TripleGuardians {
    /// Physical layer
    pub physical: HamiltonianGuard,
    /// Cognitive layer
    pub cognitive: FepMonitor,
    /// Ethical layer (immutable reference pattern)
    pub ethical: DharmaFilter,
    /// Last combined decision
    last_decision: GuardianDecision,
}

impl Default for TripleGuardians {
    fn default() -> Self {
        Self {
            physical: HamiltonianGuard::default(),
            cognitive: FepMonitor::default(),
            ethical: DharmaFilter::default(),
            last_decision: GuardianDecision::Normal,
        }
    }
}

impl TripleGuardians {
    /// Create with custom configurations
    pub fn new(
        hamiltonian_config: HamiltonianConfig,
        fep_config: FepConfig,
        dharma: DharmaFilter,
    ) -> Self {
        Self {
            physical: HamiltonianGuard::new(hamiltonian_config),
            cognitive: FepMonitor::new(fep_config),
            ethical: dharma,
            last_decision: GuardianDecision::Normal,
        }
    }

    /// Evaluate all layers and return most restrictive decision
    ///
    /// # Arguments
    /// * `physical_state` - Current hardware state
    /// * `prediction_error` - Current FEP prediction error [0, 1]
    /// * `action_vector` - Proposed action vector for ethical check
    pub fn evaluate(
        &mut self,
        physical_state: &PhysicalState,
        prediction_error: f32,
        action_vector: Complex32,
    ) -> GuardianDecision {
        // Layer 1: Physical (highest priority)
        let physical_decision = self.physical.evaluate(physical_state);
        if matches!(physical_decision, GuardianDecision::Sleep | GuardianDecision::Halt) {
            self.last_decision = physical_decision;
            return physical_decision;
        }

        // Layer 2: Cognitive
        let cognitive_decision = self.cognitive.evaluate(prediction_error);
        if matches!(cognitive_decision, GuardianDecision::Halt) {
            self.last_decision = cognitive_decision;
            return cognitive_decision;
        }

        // Layer 3: Ethical (hard-block on negative alignment)
        let alignment = self.ethical.check_alignment(action_vector);
        if alignment < 0.0 {
            // Hard block on negative alignment (anti-dharma action)
            self.last_decision = GuardianDecision::Halt;
            return GuardianDecision::Halt;
        }

        // Combine decisions (most restrictive wins)
        let decision = self.combine_decisions(physical_decision, cognitive_decision, alignment);
        self.last_decision = decision;
        decision
    }

    /// Combine multiple decisions, returning the most restrictive
    fn combine_decisions(
        &self,
        physical: GuardianDecision,
        cognitive: GuardianDecision,
        alignment: f32,
    ) -> GuardianDecision {
        // Get minimum gain from all sources
        let physical_gain = physical.gain_factor();
        let cognitive_gain = cognitive.gain_factor();
        let ethical_gain = alignment.clamp(0.0, 1.0); // Alignment as gain

        let min_gain = physical_gain.min(cognitive_gain).min(ethical_gain);

        if min_gain < 0.01 {
            GuardianDecision::Sleep
        } else if min_gain < 0.9 {
            GuardianDecision::cautious(min_gain)
        } else {
            GuardianDecision::Normal
        }
    }

    /// Get last decision
    pub fn last_decision(&self) -> GuardianDecision {
        self.last_decision
    }

    /// Quick check: should system proceed with operations?
    pub fn should_proceed(&self) -> bool {
        self.last_decision.should_proceed()
    }

    /// Get diagnostic report
    pub fn diagnostics(&self) -> GuardianDiagnostics {
        GuardianDiagnostics {
            last_decision: self.last_decision,
            physical_battery: self.physical.last_state().map(|s| s.battery_level),
            physical_temp: self.physical.last_state().map(|s| s.temperature_c),
            cognitive_baseline: self.cognitive.baseline_error(),
            cognitive_last: self.cognitive.last_error(),
            ethical_key: self.ethical.dharma_key(),
        }
    }
}

/// Diagnostic information from guardians
#[derive(Debug, Clone)]
pub struct GuardianDiagnostics {
    pub last_decision: GuardianDecision,
    pub physical_battery: Option<f32>,
    pub physical_temp: Option<f32>,
    pub cognitive_baseline: f32,
    pub cognitive_last: f32,
    pub ethical_key: Complex32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guardian_decision_gain() {
        assert_eq!(GuardianDecision::Normal.gain_factor(), 1.0);
        assert_eq!(GuardianDecision::Sleep.gain_factor(), 0.0);
        assert_eq!(GuardianDecision::Halt.gain_factor(), 0.0);

        let cautious = GuardianDecision::cautious(0.75);
        assert!((cautious.gain_factor() - 0.75).abs() < 0.02);
    }

    #[test]
    fn test_hamiltonian_normal() {
        let mut guard = HamiltonianGuard::default();
        let state = PhysicalState::mock_normal();

        let decision = guard.evaluate(&state);
        assert_eq!(decision, GuardianDecision::Normal);
    }

    #[test]
    fn test_hamiltonian_low_battery() {
        let mut guard = HamiltonianGuard::default();
        let state = PhysicalState::mock_low_battery();

        let decision = guard.evaluate(&state);
        assert_eq!(decision, GuardianDecision::Sleep);
    }

    #[test]
    fn test_hamiltonian_high_temp() {
        let mut guard = HamiltonianGuard::default();
        let state = PhysicalState::mock_high_temp();

        let decision = guard.evaluate(&state);
        assert_eq!(decision, GuardianDecision::Sleep);
    }

    #[test]
    fn test_hamiltonian_cautious_battery() {
        let mut guard = HamiltonianGuard::default();
        let state = PhysicalState {
            battery_level: 0.15, // Below 20% but above 10%
            temperature_c: 35.0,
            cpu_usage: 0.3,
            memory_usage: 0.4,
            is_charging: false,
        };

        let decision = guard.evaluate(&state);
        assert!(matches!(decision, GuardianDecision::Cautious { .. }));
    }

    #[test]
    fn test_fep_monitor_normal() {
        let mut monitor = FepMonitor::default();

        let decision = monitor.evaluate(0.3);
        assert_eq!(decision, GuardianDecision::Normal);
    }

    #[test]
    fn test_fep_monitor_high_error() {
        let mut monitor = FepMonitor::default();

        let decision = monitor.evaluate(0.96);
        assert_eq!(decision, GuardianDecision::Halt);
    }

    #[test]
    fn test_fep_monitor_spike() {
        let mut monitor = FepMonitor::default();

        // Establish baseline
        for _ in 0..10 {
            monitor.evaluate(0.2);
        }

        // Spike
        let decision = monitor.evaluate(0.8);
        assert!(matches!(decision, GuardianDecision::Cautious { .. }));
    }

    #[test]
    fn test_triple_guardians_normal() {
        let mut guardians = TripleGuardians::default();
        let state = PhysicalState::mock_normal();
        let action = Complex32::new(0.8, 0.2); // Positive action

        let decision = guardians.evaluate(&state, 0.2, action);
        assert!(decision.should_proceed());
    }

    #[test]
    fn test_triple_guardians_physical_override() {
        let mut guardians = TripleGuardians::default();
        let state = PhysicalState::mock_low_battery();
        let action = Complex32::new(0.8, 0.2);

        let decision = guardians.evaluate(&state, 0.2, action);
        assert_eq!(decision, GuardianDecision::Sleep);
    }

    #[test]
    fn test_triple_guardians_ethical_block() {
        let mut guardians = TripleGuardians::default();
        let state = PhysicalState::mock_normal();
        // Action opposite to dharma key (anti-ethical)
        let bad_action = Complex32::new(-1.0, 0.0);

        let decision = guardians.evaluate(&state, 0.2, bad_action);
        assert_eq!(decision, GuardianDecision::Halt);
    }

    #[test]
    fn test_diagnostics() {
        let mut guardians = TripleGuardians::default();
        let state = PhysicalState::mock_normal();
        let action = Complex32::new(0.8, 0.2);

        guardians.evaluate(&state, 0.3, action);
        let diag = guardians.diagnostics();

        assert_eq!(diag.physical_battery, Some(0.8));
        assert!((diag.cognitive_last - 0.3).abs() < 0.01);
    }
}
