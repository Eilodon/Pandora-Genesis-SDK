//! Core traits for domain-agnostic adaptive control.
//!
//! These traits define the interfaces that domain-specific implementations must provide
//! to work with the AGOLOS engine. The `Domain` trait is the root abstraction that ties
//! together configuration, signal variables, and action types.
//!
//! # Example: Implementing a Custom Domain
//!
//! ```rust,ignore
//! use zenb_core::core::{Domain, OscillatorConfig, SignalVariable, ActionKind};
//!
//! struct MyConfig { frequency: f32 }
//! impl OscillatorConfig for MyConfig { /* ... */ }
//!
//! #[derive(Clone, Copy, PartialEq, Eq, Hash)]
//! enum MyVariable { Temperature, Pressure }
//! impl SignalVariable for MyVariable { /* ... */ }
//!
//! #[derive(Clone)]
//! enum MyAction { Heat, Cool }
//! impl ActionKind for MyAction { /* ... */ }
//!
//! struct IndustrialDomain;
//! impl Domain for IndustrialDomain {
//!     type Config = MyConfig;
//!     type Variable = MyVariable;
//!     type Action = MyAction;
//!     fn name() -> &'static str { "industrial" }
//! }
//! ```

use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;
use std::hash::Hash;

// =============================================================================
// OSCILLATOR CONFIGURATION
// =============================================================================

/// Domain-agnostic oscillator/rhythm configuration.
///
/// This trait abstracts the core frequency control parameters that drive
/// the adaptive control loop. In the biofeedback domain, this represents
/// breathing rate (BPM). In other domains, it could represent:
/// - Industrial control: cycle frequency, sampling rate
/// - Trading: rebalancing frequency, signal update rate
/// - IoT: sensor polling rate, actuation frequency
///
/// # Frequency Units
/// The `target_frequency()` method returns cycles per minute (CPM) for
/// consistency with the biofeedback reference implementation. Domains
/// using different units (e.g., Hz) should convert internally.
pub trait OscillatorConfig: Default + Clone + Debug + Send + Sync + 'static {
    /// Target frequency in cycles per minute.
    ///
    /// For biofeedback: breaths per minute (BPM)
    /// For other domains: convert from native units (1 Hz = 60 CPM)
    fn target_frequency(&self) -> f32;

    /// Update the target frequency.
    fn set_target_frequency(&mut self, freq: f32);

    /// Minimum allowed frequency (for safety bounds).
    fn min_frequency(&self) -> f32 {
        1.0 // Default: 1 CPM
    }

    /// Maximum allowed frequency (for safety bounds).
    fn max_frequency(&self) -> f32 {
        30.0 // Default: 30 CPM
    }

    /// Validate frequency is within bounds.
    fn validate_frequency(&self, freq: f32) -> bool {
        freq >= self.min_frequency() && freq <= self.max_frequency()
    }
}

// =============================================================================
// SIGNAL VARIABLE
// =============================================================================

/// A signal variable in the causal graph.
///
/// Signal variables represent observable or latent factors that the engine
/// tracks and models causal relationships between. In biofeedback, these
/// include HeartRate, HRV, RespiratoryRate, etc.
///
/// # Requirements
/// - Must be enumerable (all variants known at compile time)
/// - Must map to/from indices for matrix operations
/// - Must be hashable for use in state maps
///
/// # Example
/// ```rust,ignore
/// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// enum BioVariable {
///     HeartRate,
///     HeartRateVariability,
///     RespiratoryRate,
/// }
///
/// impl SignalVariable for BioVariable {
///     fn index(&self) -> usize {
///         match self {
///             Self::HeartRate => 0,
///             Self::HeartRateVariability => 1,
///             Self::RespiratoryRate => 2,
///         }
///     }
///     fn from_index(idx: usize) -> Option<Self> { /* ... */ }
///     fn count() -> usize { 3 }
///     fn all() -> &'static [Self] { &[Self::HeartRate, Self::HeartRateVariability, Self::RespiratoryRate] }
/// }
/// ```
pub trait SignalVariable:
    Clone + Copy + PartialEq + Eq + Hash + Debug + Send + Sync + Serialize + DeserializeOwned + 'static
{
    /// Get the index of this variable for matrix operations.
    fn index(&self) -> usize;

    /// Construct a variable from its index.
    fn from_index(idx: usize) -> Option<Self>;

    /// Total number of variables in this domain.
    fn count() -> usize;

    /// Get all variables as a static slice for iteration.
    fn all() -> &'static [Self];

    /// Human-readable name for this variable.
    fn name(&self) -> &'static str {
        "Variable"
    }
}

// =============================================================================
// ACTION KIND
// =============================================================================

/// An action type the system can perform as intervention.
///
/// Actions represent the output layer of the adaptive control loop. The engine
/// selects actions based on expected free energy minimization and executes them
/// to influence the state of the controlled system.
///
/// # Example
/// ```rust,ignore
/// #[derive(Clone)]
/// enum BioAction {
///     BreathGuidance { target_bpm: f32, duration_sec: u32 },
///     Notification { message: String },
///     DoNothing,
/// }
///
/// impl ActionKind for BioAction {
///     fn description(&self) -> String {
///         match self {
///             Self::BreathGuidance { target_bpm, .. } => format!("Breath at {} BPM", target_bpm),
///             Self::Notification { message } => format!("Notify: {}", message),
///             Self::DoNothing => "No action".to_string(),
///         }
///     }
///     fn intrusiveness(&self) -> f32 {
///         match self {
///             Self::BreathGuidance { .. } => 0.6,
///             Self::Notification { .. } => 0.3,
///             Self::DoNothing => 0.0,
///         }
///     }
/// }
/// ```
pub trait ActionKind: Clone + Debug + Send + Sync + Serialize + DeserializeOwned + 'static {
    /// Human-readable description for logging and UI.
    fn description(&self) -> String;

    /// Intrusiveness level from 0.0 (passive) to 1.0 (highly intrusive).
    ///
    /// Used for balancing intervention effectiveness vs user autonomy.
    fn intrusiveness(&self) -> f32;

    /// Whether this action requires explicit user permission.
    fn requires_permission(&self) -> bool {
        self.intrusiveness() > 0.5
    }

    /// Unique identifier for this action type (for trauma tracking).
    fn type_id(&self) -> String {
        format!("{:?}", self)
    }
}

// =============================================================================
// DOMAIN
// =============================================================================

/// A complete domain specification for the adaptive control engine.
///
/// The `Domain` trait ties together all domain-specific types into a coherent
/// package that the engine can work with. This enables the engine to be
/// generic over different application domains while maintaining type safety.
///
/// # Associated Types
/// - `Config`: Oscillator/rhythm configuration
/// - `Variable`: Signal variables for causal modeling
/// - `Action`: Intervention action types
/// - `Mode`: Belief modes for state estimation
///
/// # Built-in Domains
/// - `BiofeedbackDomain`: Breath guidance, HRV tracking, physiological signal processing
///
/// # Creating Custom Domains
/// See module-level documentation for a complete example.
pub trait Domain: 'static + Send + Sync {
    /// Configuration type for oscillator/rhythm control.
    type Config: OscillatorConfig;

    /// Signal variable type for causal modeling.
    type Variable: SignalVariable;

    /// Action type for interventions.
    type Action: ActionKind;

    /// Belief mode type for state estimation.
    type Mode: BeliefMode;

    /// Human-readable name for this domain.
    fn name() -> &'static str;

    /// Default prior weights for causal graph initialization.
    ///
    /// Returns a function that, given two variable indices (cause, effect),
    /// returns the prior causal weight (-1.0 to 1.0, 0.0 = no prior).
    ///
    /// Override this to encode domain-specific prior knowledge.
    fn default_priors() -> fn(cause: usize, effect: usize) -> f32 {
        |_, _| 0.0 // No priors by default
    }

    /// Map from belief mode to recommended default action.
    ///
    /// Override to specify domain-specific mode-action mappings.
    fn mode_to_default_action(_mode: Self::Mode) -> Option<Self::Action> {
        None
    }
}

// =============================================================================
// BELIEF MODE
// =============================================================================

/// A belief mode representing a discrete state the system can be in.
///
/// Belief modes capture the "type" of state the system believes it's in.
/// In biofeedback, modes might be Calm, Stress, Focus, etc.
/// In trading, modes might be Bullish, Bearish, Volatile, etc.
///
/// # Requirements
/// - Must be enumerable (all variants known at compile time)
/// - Must map to/from indices for probability distribution
/// - Must have a sensible default (neutral) mode
///
/// # Example
/// ```rust,ignore
/// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// enum MarketMode {
///     Bullish, Bearish, Sideways, Volatile,
/// }
///
/// impl BeliefMode for MarketMode {
///     fn count() -> usize { 4 }
///     fn index(&self) -> usize { *self as usize }
///     fn from_index(idx: usize) -> Option<Self> { /* ... */ }
///     fn default_mode() -> Self { MarketMode::Sideways }
/// }
/// ```
pub trait BeliefMode:
    Clone
    + Copy
    + PartialEq
    + Eq
    + std::hash::Hash
    + Debug
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
    + 'static
{
    /// Number of modes in this domain.
    fn count() -> usize;

    /// Convert mode to index for probability distribution.
    fn index(&self) -> usize;

    /// Convert index to mode.
    fn from_index(idx: usize) -> Option<Self>;

    /// Default (neutral) mode when no information is available.
    fn default_mode() -> Self;

    /// Human-readable name for this mode.
    fn name(&self) -> &'static str {
        "Mode"
    }

    /// All modes as a static slice for iteration.
    fn all() -> &'static [Self];
}

/// Generic belief state over any mode type.
///
/// This is the domain-agnostic representation of belief. The system maintains
/// a probability distribution over modes and tracks confidence.
///
/// # Type Parameters
/// - `M`: The belief mode type (e.g., BioBeliefMode, MarketMode)
#[derive(Clone, Debug)]
pub struct GenericBeliefState<M: BeliefMode> {
    /// Probability distribution over modes (sums to 1.0)
    pub distribution: Vec<f32>,
    /// Confidence in current belief (0.0 = uncertain, 1.0 = certain)
    pub confidence: f32,
    /// Current dominant mode (argmax of distribution)
    pub mode: M,
}

impl<M: BeliefMode> Default for GenericBeliefState<M> {
    fn default() -> Self {
        let n = M::count();
        let uniform = 1.0 / n as f32;
        Self {
            distribution: vec![uniform; n],
            confidence: 0.0,
            mode: M::default_mode(),
        }
    }
}

impl<M: BeliefMode> GenericBeliefState<M> {
    /// Create a new belief state with uniform distribution.
    pub fn uniform() -> Self {
        Self::default()
    }

    /// Create from a probability distribution.
    pub fn from_distribution(dist: Vec<f32>) -> Self {
        let mode = dist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(i, _)| M::from_index(i))
            .unwrap_or_else(M::default_mode);

        Self {
            distribution: dist,
            confidence: 0.5,
            mode,
        }
    }

    /// Get probability for a specific mode.
    pub fn probability(&self, mode: M) -> f32 {
        self.distribution.get(mode.index()).copied().unwrap_or(0.0)
    }

    /// Update the dominant mode based on current distribution.
    pub fn update_mode(&mut self) {
        if let Some((idx, _)) = self
            .distribution
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if let Some(m) = M::from_index(idx) {
                self.mode = m;
            }
        }
    }

    /// Convert distribution to fixed-size array (for domains with known size).
    pub fn to_array<const N: usize>(&self) -> [f32; N] {
        let mut arr = [0.0f32; N];
        for (i, &p) in self.distribution.iter().take(N).enumerate() {
            arr[i] = p;
        }
        arr
    }

    /// Uncertainty = 1 - confidence
    pub fn uncertainty(&self) -> f32 {
        (1.0 - self.confidence).clamp(0.0, 1.0)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal test config
    #[derive(Clone, Debug, Default)]
    struct TestConfig {
        freq: f32,
    }

    impl OscillatorConfig for TestConfig {
        fn target_frequency(&self) -> f32 {
            self.freq
        }
        fn set_target_frequency(&mut self, freq: f32) {
            self.freq = freq;
        }
    }

    #[test]
    fn test_oscillator_config_validation() {
        let config = TestConfig { freq: 6.0 };
        assert!(config.validate_frequency(6.0));
        assert!(config.validate_frequency(1.0));
        assert!(config.validate_frequency(30.0));
        assert!(!config.validate_frequency(0.5));
        assert!(!config.validate_frequency(31.0));
    }
}
