//! Belief modes for the biofeedback domain.
//!
//! This module defines the discrete belief states that the biofeedback
//! system can be in, corresponding to different physiological/psychological modes.

use crate::core::BeliefMode;
use serde::{Deserialize, Serialize};

/// Biofeedback belief modes representing physiological/psychological states.
///
/// These modes capture the system's belief about the user's current state.
/// The belief engine maintains a probability distribution over these modes
/// and selects interventions accordingly.
///
/// # Mode Descriptions
///
/// | Mode | Description | Typical Intervention |
/// |------|-------------|---------------------|
/// | Calm | Relaxed, balanced parasympathetic state | Maintain current state |
/// | Stress | Elevated sympathetic activation | Slow breathing (4-6 BPM) |
/// | Focus | Concentrated, engaged attention | Coherent breathing (6 BPM) |
/// | Sleepy | Low arousal, drowsy | Energizing breathing (8+ BPM) |
/// | Energize | Need for activation increase | Stimulating breathing |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BioBeliefMode {
    /// Relaxed, balanced state (parasympathetic dominant)
    #[default]
    Calm,
    /// Elevated stress/anxiety (sympathetic activation)
    Stress,
    /// Concentrated attention state
    Focus,
    /// Low arousal, drowsy
    Sleepy,
    /// Need for higher activation/energy
    Energize,
}

impl BioBeliefMode {
    /// Total number of modes
    pub const COUNT: usize = 5;

    /// All modes as static array
    const ALL: [BioBeliefMode; 5] = [
        BioBeliefMode::Calm,
        BioBeliefMode::Stress,
        BioBeliefMode::Focus,
        BioBeliefMode::Sleepy,
        BioBeliefMode::Energize,
    ];

    /// Get recommended breath rate (BPM) for this mode
    pub fn recommended_bpm(&self) -> f32 {
        match self {
            BioBeliefMode::Calm => 6.0,
            BioBeliefMode::Stress => 5.0, // Slower to activate vagus
            BioBeliefMode::Focus => 6.0,  // Coherent breathing
            BioBeliefMode::Sleepy => 8.0, // Faster to energize
            BioBeliefMode::Energize => 7.0,
        }
    }
}

impl BeliefMode for BioBeliefMode {
    fn count() -> usize {
        Self::COUNT
    }

    fn index(&self) -> usize {
        match self {
            BioBeliefMode::Calm => 0,
            BioBeliefMode::Stress => 1,
            BioBeliefMode::Focus => 2,
            BioBeliefMode::Sleepy => 3,
            BioBeliefMode::Energize => 4,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(BioBeliefMode::Calm),
            1 => Some(BioBeliefMode::Stress),
            2 => Some(BioBeliefMode::Focus),
            3 => Some(BioBeliefMode::Sleepy),
            4 => Some(BioBeliefMode::Energize),
            _ => None,
        }
    }

    fn default_mode() -> Self {
        BioBeliefMode::Calm
    }

    fn name(&self) -> &'static str {
        match self {
            BioBeliefMode::Calm => "Calm",
            BioBeliefMode::Stress => "Stress",
            BioBeliefMode::Focus => "Focus",
            BioBeliefMode::Sleepy => "Sleepy",
            BioBeliefMode::Energize => "Energize",
        }
    }

    fn all() -> &'static [Self] {
        &Self::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_count() {
        assert_eq!(BioBeliefMode::count(), 5);
        assert_eq!(BioBeliefMode::all().len(), 5);
    }

    #[test]
    fn test_index_roundtrip() {
        for mode in BioBeliefMode::all() {
            let idx = mode.index();
            assert_eq!(BioBeliefMode::from_index(idx), Some(*mode));
        }
    }

    #[test]
    fn test_default_mode() {
        assert_eq!(BioBeliefMode::default_mode(), BioBeliefMode::Calm);
    }

    #[test]
    fn test_recommended_bpm() {
        assert!(BioBeliefMode::Calm.recommended_bpm() > 0.0);
        assert!(BioBeliefMode::Stress.recommended_bpm() <= 6.0); // Slow for stress
    }
}
