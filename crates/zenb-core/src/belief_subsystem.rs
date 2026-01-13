//! Belief Subsystem: Encapsulated belief management for the Engine.
//!
//! This module extracts the belief-related concerns from the Engine god object,
//! providing a clean interface for belief state management, FEP updates, and
//! hysteresis-based mode transitions.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────┐
//! │            BeliefSubsystem              │
//! │  ┌───────────────┐  ┌───────────────┐  │
//! │  │ BeliefEngine  │  │  BeliefState  │  │
//! │  │ (model)       │  │  (current)    │  │
//! │  └───────────────┘  └───────────────┘  │
//! │  ┌───────────────┐  ┌───────────────┐  │
//! │  │   FepState    │  │ AdaptiveThres │  │
//! │  │ (free energy) │  │ (hysteresis)  │  │
//! │  └───────────────┘  └───────────────┘  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Extracted from Engine
//! This subsystem replaces the following Engine fields:
//! - `belief_engine`
//! - `belief_state`
//! - `fep_state`
//! - `belief_enter_threshold`

use crate::adaptive::AdaptiveThreshold;
use crate::belief::{
    BeliefBasis, BeliefDebug, BeliefEngine, BeliefState, Context, FepState, FepUpdateOut,
    PhysioState, SensorFeatures,
};
use crate::config::{BeliefConfig, ZenbConfig};
use crate::resonance::ResonanceFeatures;
use crate::skandha::{AffectiveState, ProcessedForm, VedanaSkandha};

/// Encapsulated belief management subsystem.
///
/// Provides a single point of control for all belief-related state and operations,
/// reducing the Engine's field count and improving cohesion.
#[derive(Debug, Clone)]
pub struct BeliefSubsystem {
    /// The belief update model (contains agent ensemble logic)
    engine: BeliefEngine,

    /// Current belief state (5-mode probability distribution)
    state: BeliefState,

    /// Free Energy Principle state (surprise tracking, learning rates)
    fep: FepState,

    /// Adaptive threshold for mode transitions (hysteresis)
    enter_threshold: AdaptiveThreshold,
}

impl Default for BeliefSubsystem {
    fn default() -> Self {
        Self::new(&BeliefConfig::default())
    }
}

impl BeliefSubsystem {
    /// Create a new BeliefSubsystem with the given configuration.
    pub fn new(config: &BeliefConfig) -> Self {
        Self {
            engine: BeliefEngine::from_config(config),
            state: BeliefState::default(),
            fep: FepState::default(),
            enter_threshold: AdaptiveThreshold::new(
                config.enter_threshold,
                0.2,  // min threshold
                0.8,  // max threshold
                0.05, // learning rate
            ),
        }
    }

    /// Create from full ZenbConfig (convenience method for Engine integration).
    pub fn from_zenb_config(cfg: &ZenbConfig) -> Self {
        Self::new(&cfg.belief)
    }

    /// Access internal belief engine for testing
    #[cfg(test)]
    pub fn belief_engine(&self) -> &BeliefEngine {
        &self.engine
    }

    // =========================================================================
    // State Access
    // =========================================================================

    /// Get the current belief mode (Calm, Stress, Focus, Sleepy, Energize).
    #[inline]
    pub fn mode(&self) -> BeliefBasis {
        self.state.mode
    }

    /// Get the current probability distribution [Calm, Stress, Focus, Sleepy, Energize].
    #[inline]
    pub fn probabilities(&self) -> &[f32; 5] {
        &self.state.p
    }

    /// Get the 5-mode array representation.
    #[inline]
    pub fn to_5mode_array(&self) -> [f32; 5] {
        self.state.to_5mode_array()
    }

    /// Get the current confidence level (0.0 - 1.0).
    #[inline]
    pub fn confidence(&self) -> f32 {
        self.state.conf
    }

    /// Get the current uncertainty (inverse of confidence).
    #[inline]
    pub fn uncertainty(&self) -> f32 {
        self.state.uncertainty()
    }

    /// Get the current FEP state.
    #[inline]
    pub fn fep_state(&self) -> &FepState {
        &self.fep
    }

    /// Get the current free energy EMA.
    #[inline]
    pub fn free_energy_ema(&self) -> f32 {
        self.fep.free_energy_ema
    }

    /// Get the current enter threshold value.
    #[inline]
    pub fn enter_threshold(&self) -> f32 {
        self.enter_threshold.get()
    }

    /// Get the base enter threshold value.
    #[inline]
    pub fn enter_threshold_base(&self) -> f32 {
        self.enter_threshold.base()
    }

    /// Get mutable reference to belief state (for direct manipulation).
    ///
    /// # Warning
    /// Use sparingly — prefer using update methods instead.
    #[inline]
    pub fn state_mut(&mut self) -> &mut BeliefState {
        &mut self.state
    }

    /// Get immutable reference to belief state.
    #[inline]
    pub fn state(&self) -> &BeliefState {
        &self.state
    }

    // =========================================================================
    // Update Methods
    // =========================================================================

    /// Update belief state from sensor/physio input (non-FEP path).
    pub fn update(
        &mut self,
        sf: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f64,
    ) -> BeliefDebug {
        let (new_state, debug) = self.engine.update(&self.state, sf, phys, ctx, dt_sec);
        self.state = new_state;
        debug
    }

    /// Update using Free Energy Principle dynamics.
    ///
    /// This is the primary update path for Active Inference.
    pub fn update_fep(
        &mut self,
        sf: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f32,
        resonance: ResonanceFeatures,
        cfg: &ZenbConfig,
    ) -> FepUpdateOut {
        let out = self.engine.update_fep_with_config(
            self.state.mode,
            &self.fep,
            sf,
            phys,
            ctx,
            dt_sec,
            resonance,
            cfg,
        );
        self.state = out.belief.clone();
        self.fep = out.fep;
        out
    }

    /// Process feedback from action outcomes (Active Inference learning).
    pub fn process_feedback(&mut self, success: bool, cfg: &mut crate::config::FepConfig) {
        BeliefEngine::process_feedback(&mut self.fep, cfg, success);
    }

    /// Adapt the enter threshold based on performance feedback.
    pub fn adapt_threshold(&mut self, performance_delta: f32) {
        self.enter_threshold.adapt(performance_delta);
    }

    /// Reset the enter threshold to its base value.
    pub fn reset_threshold(&mut self) {
        self.enter_threshold.reset();
    }

    // =========================================================================
    // Direct State Manipulation (for integration with ThermodynamicEngine etc.)
    // =========================================================================

    /// Set the probability distribution directly.
    ///
    /// # Note
    /// Caller is responsible for ensuring probabilities are normalized.
    pub fn set_probabilities(&mut self, p: [f32; 5]) {
        self.state.p = p;
    }

    /// Set the belief state directly.
    pub fn set_state(&mut self, state: BeliefState) {
        self.state = state;
    }

    /// Update free energy peak tracking.
    pub fn update_free_energy_peak(&mut self, current_peak: &mut f32) {
        if self.fep.free_energy_ema > *current_peak {
            *current_peak = self.fep.free_energy_ema;
        }
    }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// Get diagnostic summary for logging.
    pub fn diagnostics(&self) -> BeliefDiagnostics {
        BeliefDiagnostics {
            mode: self.state.mode,
            probabilities: self.state.p,
            confidence: self.state.conf,
            free_energy: self.fep.free_energy_ema,
            enter_threshold: self.enter_threshold.get(),
            threshold_drift: self.enter_threshold.drift_percent(),
        }
    }
}

/// Diagnostic summary for belief subsystem.
#[derive(Debug, Clone)]
pub struct BeliefDiagnostics {
    pub mode: BeliefBasis,
    pub probabilities: [f32; 5],
    pub confidence: f32,
    pub free_energy: f32,
    pub enter_threshold: f32,
    pub threshold_drift: f32,
}

// ============================================================================
// SKANDHA INTEGRATION: VedanaSkandha for BeliefSubsystem
// ============================================================================

/// Implements VedanaSkandha for BeliefSubsystem.
///
/// This enables the BeliefSubsystem to be used as the Vedana (feeling) stage
/// in the Skandha pipeline, extracting valence and arousal from sensor data.
///
/// # Mapping from Belief to Affect
/// - **Valence** = (Calm + Focus) - Stress + 0.5*(Energize - Sleepy)
/// - **Arousal** = Stress + Energize - 0.5*(Calm + Sleepy)
/// - **Confidence** = belief state confidence
impl VedanaSkandha for BeliefSubsystem {
    fn extract_affect(&mut self, form: &ProcessedForm) -> AffectiveState {
        // Get current belief probabilities [Calm, Stress, Focus, Sleepy, Energize]
        let p = self.state.p;

        // Map belief to valence: positive states contribute positively
        // valence = (Calm + Focus + 0.5*Energize) - (Stress + 0.5*Sleepy)
        let valence = (p[0] + p[2] + 0.5 * p[4]) - (p[1] + 0.5 * p[3]);
        let valence = valence.clamp(-1.0, 1.0);

        // Map belief to arousal: activated states contribute positively
        // arousal = (Stress + Energize + 0.3*Focus) - (Calm + Sleepy)
        let arousal = (p[1] + p[4] + 0.3 * p[2]) - (p[0] + p[3]);
        let arousal = (arousal + 1.0) / 2.0; // Normalize to 0-1
        let arousal = arousal.clamp(0.0, 1.0);

        // Use reliability from form and belief confidence
        let confidence = if form.is_reliable {
            self.state.conf * 0.8 + 0.2 * form.values[3]
        } else {
            self.state.conf * 0.5
        };

        AffectiveState {
            valence,
            arousal,
            confidence: confidence.clamp(0.1, 0.95),
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_subsystem_default() {
        let bs = BeliefSubsystem::default();
        assert_eq!(bs.mode(), BeliefBasis::Calm);
        assert_eq!(bs.probabilities().len(), 5);
    }

    #[test]
    fn test_belief_subsystem_from_config() {
        let cfg = ZenbConfig::default();
        let bs = BeliefSubsystem::from_zenb_config(&cfg);
        assert!(bs.enter_threshold() > 0.0);
    }

    #[test]
    fn test_diagnostics() {
        let bs = BeliefSubsystem::default();
        let diag = bs.diagnostics();
        assert_eq!(diag.mode, BeliefBasis::Calm);
        assert!(diag.probabilities.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_threshold_adaptation() {
        let mut bs = BeliefSubsystem::default();
        let initial = bs.enter_threshold();

        // Positive feedback should lower threshold (more aggressive)
        bs.adapt_threshold(0.5);
        assert!(bs.enter_threshold() < initial || bs.enter_threshold() == initial);

        // Negative feedback should raise threshold (more conservative)
        bs.adapt_threshold(-0.5);
        // After both, should be somewhere in range
        assert!(bs.enter_threshold() >= 0.2 && bs.enter_threshold() <= 0.8);
    }
}
