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
