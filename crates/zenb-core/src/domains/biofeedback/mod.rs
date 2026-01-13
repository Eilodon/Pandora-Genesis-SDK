//! Biofeedback Domain: Breath guidance, HRV tracking, and physiological signal processing.
//!
//! This is the **reference implementation** domain for AGOLOS, demonstrating
//! how to implement the core traits for a specific application area.
//!
//! # Components
//!
//! - [`BreathConfig`]: Oscillator configuration for breath guidance
//! - [`BioVariable`]: Signal variables (HeartRate, HRV, RespiratoryRate, etc.)
//! - [`BioAction`]: Intervention actions (breath guidance, notifications)
//! - [`BiofeedbackDomain`]: Domain implementation tying everything together
//!
//! # Usage
//!
//! ```rust,ignore
//! use zenb_core::domains::BiofeedbackDomain;
//! use zenb_core::Engine;
//!
//! // Engine defaults to BiofeedbackDomain
//! let engine = Engine::new(6.0);
//! ```

mod actions;
mod config;
mod variables;

pub use actions::BioAction;
pub use config::BreathConfig;
pub use variables::BioVariable;

use crate::core::Domain;

/// The Biofeedback domain for breath guidance and physiological signal processing.
///
/// This domain includes:
/// - **Oscillator control**: Breathing rate (BPM) for breath guidance
/// - **Signal variables**: HeartRate, HRV, RespiratoryRate, NotificationPressure, etc.
/// - **Actions**: BreathGuidance, NotificationBlock, SuggestBreak, etc.
///
/// # Causal Priors
///
/// The domain encodes physiological prior knowledge:
/// - Notifications → ↑HeartRate (+0.6)
/// - NotificationPressure → ↓HRV (-0.4)
/// - RespiratoryRate → HeartRate (RSA effect)
/// - RespiratoryRate → ↑HRV (breath coherence)
pub struct BiofeedbackDomain;

impl Domain for BiofeedbackDomain {
    type Config = BreathConfig;
    type Variable = BioVariable;
    type Action = BioAction;

    fn name() -> &'static str {
        "biofeedback"
    }

    fn default_priors() -> fn(cause: usize, effect: usize) -> f32 {
        |cause, effect| {
            // Encode physiological prior knowledge
            // Index mapping from BioVariable:
            // 0: NotificationPressure, 1: HeartRate, 2: HRV, 3: Location,
            // 4: TimeOfDay, 5: UserAction, 6: InteractionIntensity,
            // 7: RespiratoryRate, 8: NoiseLevel
            match (cause, effect) {
                // Notifications cause stress (↑HR, ↓HRV)
                (0, 1) => 0.6,  // NotificationPressure → HeartRate
                (0, 2) => -0.4, // NotificationPressure → HRV (negative)

                // Respiratory Sinus Arrhythmia (RSA)
                (7, 1) => -0.3, // RespiratoryRate → HeartRate (slow breathing lowers HR)
                (7, 2) => 0.4,  // RespiratoryRate → HRV (breath coherence)

                // User action affects respiratory rate (breath guidance)
                (5, 7) => 0.5, // UserAction → RespiratoryRate

                // Noise affects heart rate
                (8, 1) => 0.3, // NoiseLevel → HeartRate

                // No prior for other relationships
                _ => 0.0,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ActionKind, OscillatorConfig, SignalVariable};

    #[test]
    fn test_domain_name() {
        assert_eq!(BiofeedbackDomain::name(), "biofeedback");
    }

    #[test]
    fn test_config_trait() {
        let mut config = BreathConfig::default();
        assert!(config.target_frequency() > 0.0);
        config.set_target_frequency(8.0);
        assert_eq!(config.target_frequency(), 8.0);
    }

    #[test]
    fn test_variable_trait() {
        assert_eq!(BioVariable::count(), 12);
        assert!(BioVariable::all().len() > 0);

        let hr = BioVariable::HeartRate;
        let idx = hr.index();
        assert_eq!(BioVariable::from_index(idx), Some(hr));
    }

    #[test]
    fn test_action_trait() {
        let action = BioAction::BreathGuidance {
            target_bpm: 6.0,
            duration_sec: 60,
        };
        assert!(action.intrusiveness() > 0.0);
        assert!(!action.description().is_empty());
    }

    #[test]
    fn test_priors() {
        let priors = BiofeedbackDomain::default_priors();
        // NotificationPressure (0) → HeartRate (1) should be positive
        assert!(priors(0, 1) > 0.0);
        // NotificationPressure (0) → HRV (2) should be negative
        assert!(priors(0, 2) < 0.0);
        // No prior for unrelated pairs
        assert_eq!(priors(3, 4), 0.0);
    }
}
