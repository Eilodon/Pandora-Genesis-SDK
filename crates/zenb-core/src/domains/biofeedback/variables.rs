//! Signal variables for the biofeedback domain.

use crate::core::SignalVariable;
use serde::{Deserialize, Serialize};

/// Biological signal variables for causal modeling in the biofeedback domain.
///
/// These variables represent observable and latent factors that the engine
/// tracks and models causal relationships between.
///
/// # Causal Model
///
/// The biofeedback domain models relationships like:
/// - NotificationPressure → HeartRate (stress response)
/// - RespiratoryRate → HRV (respiratory sinus arrhythmia)
/// - UserAction → RespiratoryRate (breath guidance intervention)
///
/// # Index Mapping
///
/// Variables are indexed for matrix operations:
/// | Index | Variable |
/// |-------|----------|
/// | 0 | NotificationPressure |
/// | 1 | HeartRate |
/// | 2 | HeartRateVariability |
/// | 3 | Location |
/// | 4 | TimeOfDay |
/// | 5 | UserAction |
/// | 6 | InteractionIntensity |
/// | 7 | RespiratoryRate |
/// | 8 | NoiseLevel |
/// | 9 | CognitiveLoad |
/// | 10 | EmotionalValence |
/// | 11 | VoiceArousal |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BioVariable {
    /// Notification pressure: rate of incoming notifications (digital stressor)
    NotificationPressure,
    /// Heart rate in BPM (physiological arousal indicator)
    HeartRate,
    /// Heart rate variability in ms (parasympathetic tone indicator)
    HeartRateVariability,
    /// Location type (home, work, transit, etc.)
    Location,
    /// Time of day (0-23)
    TimeOfDay,
    /// User-initiated action (intervention trigger)
    UserAction,
    /// Interaction intensity (screen time, app switches)
    InteractionIntensity,
    /// Respiratory rate (breath pattern)
    RespiratoryRate,
    /// Ambient noise level
    NoiseLevel,
    /// Cognitive load (semantic/task switching burden)
    CognitiveLoad,
    /// Emotional valence from voice/text (-1 to 1, normalized)
    EmotionalValence,
    /// Voice arousal (intensity/activation)
    VoiceArousal,
}

impl BioVariable {
    /// Total number of variables.
    pub const COUNT: usize = 12;

    /// All variables as a static slice.
    const ALL: [BioVariable; 12] = [
        BioVariable::NotificationPressure,
        BioVariable::HeartRate,
        BioVariable::HeartRateVariability,
        BioVariable::Location,
        BioVariable::TimeOfDay,
        BioVariable::UserAction,
        BioVariable::InteractionIntensity,
        BioVariable::RespiratoryRate,
        BioVariable::NoiseLevel,
        BioVariable::CognitiveLoad,
        BioVariable::EmotionalValence,
        BioVariable::VoiceArousal,
    ];
}

impl SignalVariable for BioVariable {
    fn index(&self) -> usize {
        match self {
            BioVariable::NotificationPressure => 0,
            BioVariable::HeartRate => 1,
            BioVariable::HeartRateVariability => 2,
            BioVariable::Location => 3,
            BioVariable::TimeOfDay => 4,
            BioVariable::UserAction => 5,
            BioVariable::InteractionIntensity => 6,
            BioVariable::RespiratoryRate => 7,
            BioVariable::NoiseLevel => 8,
            BioVariable::CognitiveLoad => 9,
            BioVariable::EmotionalValence => 10,
            BioVariable::VoiceArousal => 11,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(BioVariable::NotificationPressure),
            1 => Some(BioVariable::HeartRate),
            2 => Some(BioVariable::HeartRateVariability),
            3 => Some(BioVariable::Location),
            4 => Some(BioVariable::TimeOfDay),
            5 => Some(BioVariable::UserAction),
            6 => Some(BioVariable::InteractionIntensity),
            7 => Some(BioVariable::RespiratoryRate),
            8 => Some(BioVariable::NoiseLevel),
            9 => Some(BioVariable::CognitiveLoad),
            10 => Some(BioVariable::EmotionalValence),
            11 => Some(BioVariable::VoiceArousal),
            _ => None,
        }
    }

    fn count() -> usize {
        Self::COUNT
    }

    fn all() -> &'static [Self] {
        &Self::ALL
    }

    fn name(&self) -> &'static str {
        match self {
            BioVariable::NotificationPressure => "NotificationPressure",
            BioVariable::HeartRate => "HeartRate",
            BioVariable::HeartRateVariability => "HRV",
            BioVariable::Location => "Location",
            BioVariable::TimeOfDay => "TimeOfDay",
            BioVariable::UserAction => "UserAction",
            BioVariable::InteractionIntensity => "InteractionIntensity",
            BioVariable::RespiratoryRate => "RespiratoryRate",
            BioVariable::NoiseLevel => "NoiseLevel",
            BioVariable::CognitiveLoad => "CognitiveLoad",
            BioVariable::EmotionalValence => "EmotionalValence",
            BioVariable::VoiceArousal => "VoiceArousal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_count() {
        assert_eq!(BioVariable::count(), 12);
        assert_eq!(BioVariable::all().len(), 12);
    }

    #[test]
    fn test_index_roundtrip() {
        for var in BioVariable::all() {
            let idx = var.index();
            assert_eq!(BioVariable::from_index(idx), Some(*var));
        }
    }

    #[test]
    fn test_invalid_index() {
        assert_eq!(BioVariable::from_index(100), None);
    }

    #[test]
    fn test_variable_names() {
        assert_eq!(BioVariable::HeartRate.name(), "HeartRate");
        assert_eq!(BioVariable::RespiratoryRate.name(), "RespiratoryRate");
    }
}
