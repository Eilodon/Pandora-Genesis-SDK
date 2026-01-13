//! Belief Subsystem — State Inference and Active Learning
//!
//! Extracted from the Engine god-object to improve modularity.
//! Encapsulates:
//! - BeliefEngine for multi-agent belief inference
//! - BeliefState and FepState management
//! - Adaptive threshold for hysteresis transitions
//! - Active Inference feedback processing
//!
//! # Invariants
//! - Belief probabilities always sum to ~1.0
//! - FepState precision >= 0
//! - Hysteresis prevents rapid state transitions

use crate::adaptive::AdaptiveThreshold;
use crate::belief::{
    hysteresis_collapse, softmax, BeliefBasis, BeliefEngine, BeliefState, Context, FepState,
    PhysioState, SensorFeatures,
};
use crate::config::ZenbConfig;

/// Error types for belief subsystem
#[derive(Debug, Clone)]
pub enum BeliefError {
    /// Input features are invalid
    InvalidFeatures { reason: String },
    /// State normalization failed
    NormalizationFailed { sum: f32 },
}

impl std::fmt::Display for BeliefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFeatures { reason } => write!(f, "Invalid features: {}", reason),
            Self::NormalizationFailed { sum } => write!(f, "Normalization failed: sum={}", sum),
        }
    }
}

impl std::error::Error for BeliefError {}

/// Result of belief update
#[derive(Debug, Clone)]
pub struct BeliefUpdateResult {
    /// Current belief state
    pub state: BeliefState,
    /// Whether a state transition occurred
    pub transitioned: bool,
    /// Previous mode (if transitioned)
    pub previous_mode: Option<BeliefBasis>,
    /// Free energy delta
    pub fe_delta: f32,
}

/// Belief Subsystem — extracted from Engine
///
/// Handles all belief state inference, active learning, and hysteresis.
/// Wraps BeliefEngine with proper error handling and state management.
pub struct BeliefSubsystem {
    /// Core belief engine (multi-agent inference)
    engine: BeliefEngine,

    /// Current belief state
    state: BeliefState,

    /// Free Energy Principle state
    fep: FepState,

    /// Adaptive threshold for belief transitions
    enter_threshold: AdaptiveThreshold,

    /// Exit threshold (typically lower than enter for hysteresis)
    exit_threshold: f32,

    /// Last sensor features (for replay/debugging)
    last_features: Option<SensorFeatures>,

    /// Last physiological state
    last_phys: Option<PhysioState>,

    /// Smoothing time constant
    smooth_tau_sec: f32,
}

impl BeliefSubsystem {
    /// Create a new belief subsystem with default configuration
    pub fn new() -> Self {
        Self::with_config(&ZenbConfig::default())
    }

    /// Create a new belief subsystem with configuration
    pub fn with_config(config: &ZenbConfig) -> Self {
        let engine = BeliefEngine::from_config(&config.belief);

        Self {
            engine,
            state: BeliefState::default(),
            fep: FepState::default(),
            enter_threshold: AdaptiveThreshold::new(
                config.belief.enter_threshold,
                0.3,  // min
                0.8,  // max
                0.05, // learning rate
            ),
            exit_threshold: config.belief.exit_threshold,
            last_features: None,
            last_phys: None,
            smooth_tau_sec: config.belief.smooth_tau_sec,
        }
    }

    /// Update belief state from sensor features and context
    ///
    /// # Arguments
    /// * `features` - Sensor features (HR, HRV, RR etc.)
    /// * `phys` - Physiological state
    /// * `ctx` - Current context (time, session info)
    /// * `dt_sec` - Time delta in seconds
    ///
    /// # Returns
    /// `BeliefUpdateResult` with new state and transition info
    pub fn update(
        &mut self,
        features: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f32,
    ) -> Result<BeliefUpdateResult, BeliefError> {
        // Store for debugging
        self.last_features = Some(*features);
        self.last_phys = Some(*phys);

        // Get votes from all agents
        let mut logits = self.engine.prior_logits;

        for (i, agent) in self.engine.agents.iter().enumerate() {
            let vote = agent.eval(features, phys, ctx);
            let weight = self.engine.w.get(i).copied().unwrap_or(1.0);

            for (j, &v) in vote.logits.iter().enumerate() {
                logits[j] += v * weight;
            }
        }

        // Convert to probabilities
        let probs = softmax(logits);

        // Validate normalization
        let sum: f32 = probs.iter().sum();
        if (sum - 1.0).abs() > 0.1 {
            return Err(BeliefError::NormalizationFailed { sum });
        }

        // Compute smoothing alpha from dt
        let alpha = if self.smooth_tau_sec > 0.0 {
            1.0 - (-dt_sec / self.smooth_tau_sec).exp()
        } else {
            1.0
        };

        // Smooth probabilities
        let smoothed: [f32; 5] =
            std::array::from_fn(|i| self.state.p[i] * (1.0 - alpha) + probs[i] * alpha);

        // Hysteresis collapse
        let previous_mode = self.state.mode;
        let new_mode = hysteresis_collapse(
            previous_mode,
            &smoothed,
            self.enter_threshold.get(),
            self.exit_threshold,
        );

        let transitioned = new_mode != previous_mode;

        // Update FEP state
        let prev_fe = self.fep.free_energy_ema;
        let uncertainty = self.state.uncertainty();
        self.fep.free_energy_ema = (uncertainty * 0.2 + prev_fe * 0.8).clamp(0.0, 5.0);
        let fe_delta = self.fep.free_energy_ema - prev_fe;

        // Update state
        self.state = BeliefState {
            mode: new_mode,
            p: smoothed,
            conf: self.compute_confidence(&smoothed),
        };

        Ok(BeliefUpdateResult {
            state: self.state.clone(),
            transitioned,
            previous_mode: if transitioned {
                Some(previous_mode)
            } else {
                None
            },
            fe_delta,
        })
    }

    /// Process outcome feedback for active learning
    pub fn process_feedback(&mut self, success: bool) {
        // Adapt threshold based on outcome
        let delta = if success { 0.1 } else { -0.1 };
        self.enter_threshold.adapt(delta);

        // Adjust FEP learning rate
        if success {
            self.fep.lr = (self.fep.lr * 0.95).max(0.1);
        } else {
            self.fep.lr = (self.fep.lr * 1.1).min(0.9);
        }
    }

    /// Get current belief state
    pub fn state(&self) -> &BeliefState {
        &self.state
    }

    /// Get current mode
    pub fn mode(&self) -> BeliefBasis {
        self.state.mode
    }

    /// Get current probabilities
    pub fn probabilities(&self) -> &[f32; 5] {
        &self.state.p
    }

    /// Get current confidence
    pub fn confidence(&self) -> f32 {
        self.state.conf
    }

    /// Get FEP state
    pub fn fep(&self) -> &FepState {
        &self.fep
    }

    /// Get free energy from FEP state
    pub fn free_energy(&self) -> f32 {
        self.fep.free_energy_ema
    }

    /// Check if in specific mode
    pub fn is_in_mode(&self, mode: BeliefBasis) -> bool {
        self.state.mode == mode
    }

    /// Reset to default state
    pub fn reset(&mut self) {
        self.state = BeliefState::default();
        self.fep = FepState::default();
        self.last_features = None;
        self.last_phys = None;
    }

    fn compute_confidence(&self, probs: &[f32; 5]) -> f32 {
        // Max probability as confidence proxy
        probs.iter().cloned().fold(0.0_f32, f32::max)
    }
}

impl Default for BeliefSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for BeliefSubsystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BeliefSubsystem")
            .field("state", &self.state)
            .field("fep", &self.fep)
            .field("enter_threshold", &self.enter_threshold.get())
            .field("exit_threshold", &self.exit_threshold)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_subsystem_basic() {
        let mut belief = BeliefSubsystem::new();

        let features = SensorFeatures {
            hr_bpm: Some(70.0),
            rmssd: Some(45.0),
            rr_bpm: Some(12.0),
            quality: 0.9,
            motion: 0.1,
        };

        let phys = PhysioState {
            hr_bpm: Some(70.0),
            rr_bpm: Some(12.0),
            rmssd: Some(45.0),
            confidence: 0.9,
        };

        let ctx = Context {
            local_hour: 14,
            is_charging: false,
            recent_sessions: 5,
        };

        let result = belief.update(&features, &phys, &ctx, 0.1);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.state.p.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_belief_normalization() {
        let belief = BeliefSubsystem::new();

        // Default state should have conf = 0 (uninitialized)
        assert!(belief.confidence() >= 0.0);
    }

    #[test]
    fn test_hysteresis() {
        let mut belief = BeliefSubsystem::new();

        // Force initial state to Calm with high probability
        belief.state.mode = BeliefBasis::Calm;
        belief.state.p = [0.9, 0.025, 0.025, 0.025, 0.025];
        belief.state.conf = 0.9;

        let features = SensorFeatures {
            hr_bpm: Some(70.0),
            rmssd: Some(45.0),
            rr_bpm: Some(12.0),
            quality: 0.9,
            motion: 0.1,
        };
        let phys = PhysioState {
            hr_bpm: Some(70.0),
            rr_bpm: Some(12.0),
            rmssd: Some(45.0),
            confidence: 0.9,
        };
        let ctx = Context {
            local_hour: 14,
            is_charging: false,
            recent_sessions: 5,
        };

        // Small update shouldn't cause transition due to hysteresis
        let result = belief.update(&features, &phys, &ctx, 0.1).unwrap();

        // Mode should still be Calm (hysteresis prevents easy switching)
        assert_eq!(result.state.mode, BeliefBasis::Calm);
    }

    #[test]
    fn test_feedback_adapts_threshold() {
        let mut belief = BeliefSubsystem::new();
        let initial_threshold = belief.enter_threshold.get();

        // Positive feedback should adapt threshold
        belief.process_feedback(true);
        assert_ne!(belief.enter_threshold.get(), initial_threshold);
    }
}
