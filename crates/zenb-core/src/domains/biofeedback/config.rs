//! Breath configuration for the biofeedback domain.

use crate::core::OscillatorConfig;
use serde::{Deserialize, Serialize};

/// Configuration for breath guidance oscillator.
///
/// This implements `OscillatorConfig` for the biofeedback domain,
/// controlling the target breathing rate (BPM) for guided breathing exercises.
///
/// # Example
///
/// ```rust,ignore
/// use zenb_core::domains::biofeedback::BreathConfig;
/// use zenb_core::core::OscillatorConfig;
///
/// let mut config = BreathConfig::default();
/// assert_eq!(config.target_frequency(), 6.0); // 6 BPM default
///
/// config.set_target_frequency(4.0); // Slow breathing
/// assert!(config.validate_frequency(4.0));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathConfig {
    /// Target breathing rate in breaths per minute.
    pub default_target_bpm: f32,
}

impl Default for BreathConfig {
    fn default() -> Self {
        Self {
            default_target_bpm: 6.0, // Resonance frequency for HRV coherence
        }
    }
}

impl OscillatorConfig for BreathConfig {
    fn target_frequency(&self) -> f32 {
        self.default_target_bpm
    }

    fn set_target_frequency(&mut self, freq: f32) {
        self.default_target_bpm = freq;
    }

    fn min_frequency(&self) -> f32 {
        3.0 // 3 BPM for deep meditation
    }

    fn max_frequency(&self) -> f32 {
        20.0 // 20 BPM for high stress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BreathConfig::default();
        assert_eq!(config.default_target_bpm, 6.0);
    }

    #[test]
    fn test_oscillator_config_impl() {
        let mut config = BreathConfig::default();

        assert_eq!(config.target_frequency(), 6.0);

        config.set_target_frequency(4.5);
        assert_eq!(config.target_frequency(), 4.5);

        assert!(config.validate_frequency(6.0));
        assert!(config.validate_frequency(3.0));
        assert!(config.validate_frequency(20.0));
        assert!(!config.validate_frequency(2.0)); // Below min
        assert!(!config.validate_frequency(25.0)); // Above max
    }
}
