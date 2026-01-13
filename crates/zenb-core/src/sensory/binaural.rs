//! Binaural beat frequency calculator
//!
//! Maps breath phases to brain wave frequencies

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainWaveState {
    Delta, // 1-4 Hz: Deep sleep
    Theta, // 4-8 Hz: Meditation
    Alpha, // 8-13 Hz: Relaxed
    Beta,  // 13-30 Hz: Active
}

impl BrainWaveState {
    /// Get (carrier_hz, beat_hz) for this state
    pub fn config(&self) -> (f32, f32) {
        match self {
            Self::Delta => (200.0, 2.5),
            Self::Theta => (200.0, 6.0),
            Self::Alpha => (200.0, 10.0),
            Self::Beta => (220.0, 18.0),
        }
    }
}

pub struct BinauralEngine {
    current_state: BrainWaveState,
}

impl BinauralEngine {
    pub fn new() -> Self {
        Self {
            current_state: BrainWaveState::Alpha,
        }
    }

    /// Update based on breath phase and arousal
    pub fn update(&mut self, phase: &str, arousal: f32) -> Option<(f32, f32)> {
        let next = match phase {
            "inhale" => {
                if arousal > 0.5 {
                    BrainWaveState::Alpha
                } else {
                    BrainWaveState::Theta
                }
            }
            "exhale" => BrainWaveState::Theta,
            "hold" => {
                if arousal < 0.3 {
                    BrainWaveState::Delta
                } else {
                    BrainWaveState::Theta
                }
            }
            _ => self.current_state,
        };

        if next != self.current_state {
            self.current_state = next;
            Some(next.config())
        } else {
            None
        }
    }

    pub fn current_state(&self) -> BrainWaveState {
        self.current_state
    }
}

impl Default for BinauralEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PSYCHOACOUSTIC CALIBRATION
// ============================================================================

/// Psychoacoustic calibration for optimal binaural beat perception.
///
/// Based on:
/// - ISO 226:2003 Equal-Loudness Contours
/// - Oster (1973) binaural beat perception thresholds
/// - Licensed professional audio engineering guidelines
#[derive(Debug, Clone)]
pub struct PsychoacousticCalibration {
    /// Reference loudness level in dB SPL
    pub reference_db_spl: f32,

    /// Individual hearing sensitivity adjustment (-10 to +10 dB)
    pub sensitivity_offset: f32,

    /// Ambient noise floor estimate in dB
    pub ambient_noise_floor: f32,

    /// Maximum safe listening level in dB
    pub max_safe_level: f32,
}

impl Default for PsychoacousticCalibration {
    fn default() -> Self {
        Self {
            reference_db_spl: 60.0,    // Comfortable listening level
            sensitivity_offset: 0.0,   // No adjustment
            ambient_noise_floor: 30.0, // Quiet room
            max_safe_level: 85.0,      // Safe for 8-hour exposure
        }
    }
}

impl PsychoacousticCalibration {
    /// Get optimal carrier frequency for a given beat frequency.
    ///
    /// Binaural beats are most effective when carrier is in 200-500 Hz range.
    /// Higher carriers improve spatial perception, lower carriers may cause fatigue.
    pub fn optimal_carrier_hz(&self, beat_hz: f32) -> f32 {
        // Carrier should be low enough for clear beat perception
        // but high enough to avoid interaural crosstalk
        let base_carrier = if beat_hz < 4.0 {
            // Delta range: use lower carrier for deeper effect
            180.0
        } else if beat_hz < 8.0 {
            // Theta range: standard carrier
            200.0
        } else if beat_hz < 13.0 {
            // Alpha range: slightly higher for clarity
            220.0
        } else {
            // Beta range: higher carrier needed
            250.0
        };

        // Adjust for individual hearing
        base_carrier + self.sensitivity_offset * 2.0
    }

    /// Compute equal-loudness corrected amplitude.
    ///
    /// Based on simplified ISO 226:2003 equal-loudness contours.
    /// Returns multiplier to achieve perceptually uniform loudness.
    pub fn equal_loudness_amplitude(&self, frequency_hz: f32) -> f32 {
        // Simplified A-weighting curve approximation
        // Full implementation would use complete ISO 226 tables
        let f = frequency_hz;
        let f2 = f * f;

        // A-weighting approximation (simplified)
        let ra = (12194.0_f32.powi(2) * f2.powi(2))
            / ((f2 + 20.6_f32.powi(2))
                * ((f2 + 107.7_f32.powi(2)) * (f2 + 737.9_f32.powi(2))).sqrt()
                * (f2 + 12194.0_f32.powi(2)));

        // Convert to amplitude correction (inverse of loudness curve)
        let correction = 1.0 / (ra + 0.1).sqrt();

        // Clamp to reasonable range
        correction.clamp(0.3, 2.0)
    }

    /// Calculate left and right channel frequencies for stereo binaural.
    pub fn stereo_frequencies(&self, beat_hz: f32) -> (f32, f32) {
        let carrier = self.optimal_carrier_hz(beat_hz);
        let half_beat = beat_hz / 2.0;

        // Left ear gets lower frequency
        let left_hz = carrier - half_beat;
        // Right ear gets higher frequency
        let right_hz = carrier + half_beat;

        (left_hz, right_hz)
    }

    /// Compute safe volume level considering ambient noise.
    /// Returns amplitude multiplier [0.0, 1.0].
    pub fn safe_amplitude(&self, desired_db: f32) -> f32 {
        // Must be louder than ambient but below max safe
        let effective_db = desired_db.clamp(
            self.ambient_noise_floor + 10.0, // At least 10dB above noise floor
            self.max_safe_level,
        );

        // Convert dB to linear amplitude (relative to reference)
        let db_diff = effective_db - self.reference_db_spl;
        10.0_f32.powf(db_diff / 20.0).clamp(0.0, 1.0)
    }

    /// Full calibration for a binaural beat session.
    /// Returns (left_hz, right_hz, amplitude_multiplier).
    pub fn calibrate(&self, beat_hz: f32) -> (f32, f32, f32) {
        let (left, right) = self.stereo_frequencies(beat_hz);
        let carrier = (left + right) / 2.0;

        // Apply equal-loudness correction
        let loudness_correction = self.equal_loudness_amplitude(carrier);

        // Combine with safe amplitude
        let base_amplitude = self.safe_amplitude(self.reference_db_spl);
        let final_amplitude = (base_amplitude * loudness_correction).clamp(0.1, 1.0);

        (left, right, final_amplitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_wave_configs() {
        assert_eq!(BrainWaveState::Delta.config(), (200.0, 2.5));
        assert_eq!(BrainWaveState::Theta.config(), (200.0, 6.0));
        assert_eq!(BrainWaveState::Alpha.config(), (200.0, 10.0));
        assert_eq!(BrainWaveState::Beta.config(), (220.0, 18.0));
    }

    #[test]
    fn test_binaural_update() {
        let mut engine = BinauralEngine::new();

        // Initial state is Alpha, so high arousal inhale returns None (no change)
        let result = engine.update("inhale", 0.7);
        assert!(result.is_none()); // Already in Alpha state
        assert_eq!(engine.current_state(), BrainWaveState::Alpha);

        // Exhale -> Theta (state change)
        let result = engine.update("exhale", 0.7);
        assert!(result.is_some());
        assert_eq!(engine.current_state(), BrainWaveState::Theta);

        // Same state -> None
        let result = engine.update("exhale", 0.5);
        assert!(result.is_none());
    }
}
