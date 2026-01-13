//! Action types for the biofeedback domain.

use crate::core::ActionKind;
use serde::{Deserialize, Serialize};

/// Actions the system can perform in the biofeedback domain.
///
/// These represent interventions the engine can propose to influence
/// the user's physiological state or digital environment.
///
/// # Action Categories
///
/// 1. **Physiological**: Direct guidance (breath, movement)
/// 2. **Digital**: Environment modification (notifications, apps)
/// 3. **Passive**: Observation or minimal intervention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BioAction {
    /// Guided breathing exercise.
    ///
    /// Parameters:
    /// - `target_bpm`: Target breathing rate in breaths per minute
    /// - `duration_sec`: Duration of the exercise in seconds
    BreathGuidance { target_bpm: f32, duration_sec: u32 },

    /// Block or filter notifications.
    ///
    /// Parameters:
    /// - `duration_sec`: How long to block notifications
    /// - `allow_priority`: Allow priority notifications through
    NotificationBlock {
        duration_sec: u32,
        allow_priority: bool,
    },

    /// Suggest a break from activity.
    ///
    /// Parameters:
    /// - `suggested_duration_min`: Recommended break duration in minutes
    /// - `reason`: Human-readable reason for the suggestion
    SuggestBreak {
        suggested_duration_min: u32,
        reason: String,
    },

    /// Launch a specific app for intervention.
    ///
    /// Parameters:
    /// - `app_id`: Package name or app identifier
    /// - `action`: Specific action within the app
    LaunchApp { app_id: String, action: String },

    /// Play audio intervention (binaural beats, nature sounds).
    ///
    /// Parameters:
    /// - `audio_type`: Type of audio intervention
    /// - `duration_sec`: Duration in seconds
    PlayAudio {
        audio_type: AudioInterventionType,
        duration_sec: u32,
    },

    /// No intervention - continue observing.
    DoNothing,
}

/// Types of audio interventions for the biofeedback domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioInterventionType {
    /// Binaural beats for focus or relaxation
    BinauralBeats { target_frequency: f32 },
    /// Nature soundscapes
    NatureSounds,
    /// Guided meditation audio
    GuidedMeditation,
    /// White/pink/brown noise
    Noise { color: NoiseColor },
}

/// Noise colors for audio interventions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NoiseColor {
    White,
    Pink,
    Brown,
}

impl ActionKind for BioAction {
    fn description(&self) -> String {
        match self {
            BioAction::BreathGuidance {
                target_bpm,
                duration_sec,
            } => {
                format!(
                    "Breath guidance at {} BPM for {}s",
                    target_bpm, duration_sec
                )
            }
            BioAction::NotificationBlock {
                duration_sec,
                allow_priority,
            } => {
                if *allow_priority {
                    format!(
                        "Block notifications for {}s (priority allowed)",
                        duration_sec
                    )
                } else {
                    format!("Block all notifications for {}s", duration_sec)
                }
            }
            BioAction::SuggestBreak {
                suggested_duration_min,
                reason,
            } => {
                format!("Suggest {}min break: {}", suggested_duration_min, reason)
            }
            BioAction::LaunchApp { app_id, action } => {
                format!("Launch {} ({})", app_id, action)
            }
            BioAction::PlayAudio {
                audio_type,
                duration_sec,
            } => {
                let type_desc = match audio_type {
                    AudioInterventionType::BinauralBeats { target_frequency } => {
                        format!("binaural beats at {} Hz", target_frequency)
                    }
                    AudioInterventionType::NatureSounds => "nature sounds".to_string(),
                    AudioInterventionType::GuidedMeditation => "guided meditation".to_string(),
                    AudioInterventionType::Noise { color } => {
                        format!("{:?} noise", color).to_lowercase()
                    }
                };
                format!("Play {} for {}s", type_desc, duration_sec)
            }
            BioAction::DoNothing => "No intervention".to_string(),
        }
    }

    fn intrusiveness(&self) -> f32 {
        match self {
            BioAction::DoNothing => 0.0,
            BioAction::PlayAudio { .. } => 0.2,
            BioAction::SuggestBreak { .. } => 0.3,
            BioAction::NotificationBlock { .. } => 0.4,
            BioAction::LaunchApp { .. } => 0.5,
            BioAction::BreathGuidance { .. } => 0.6,
        }
    }

    fn requires_permission(&self) -> bool {
        matches!(
            self,
            BioAction::BreathGuidance { .. } | BioAction::LaunchApp { .. }
        )
    }

    fn type_id(&self) -> String {
        match self {
            BioAction::BreathGuidance { .. } => "BreathGuidance".to_string(),
            BioAction::NotificationBlock { .. } => "NotificationBlock".to_string(),
            BioAction::SuggestBreak { .. } => "SuggestBreak".to_string(),
            BioAction::LaunchApp { .. } => "LaunchApp".to_string(),
            BioAction::PlayAudio { .. } => "PlayAudio".to_string(),
            BioAction::DoNothing => "DoNothing".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breath_guidance_description() {
        let action = BioAction::BreathGuidance {
            target_bpm: 6.0,
            duration_sec: 120,
        };
        assert!(action.description().contains("6 BPM"));
        assert!(action.description().contains("120s"));
    }

    #[test]
    fn test_intrusiveness_ordering() {
        let nothing = BioAction::DoNothing;
        let breath = BioAction::BreathGuidance {
            target_bpm: 6.0,
            duration_sec: 60,
        };

        assert!(nothing.intrusiveness() < breath.intrusiveness());
    }

    #[test]
    fn test_permission_required() {
        let breath = BioAction::BreathGuidance {
            target_bpm: 6.0,
            duration_sec: 60,
        };
        let nothing = BioAction::DoNothing;

        assert!(breath.requires_permission());
        assert!(!nothing.requires_permission());
    }

    #[test]
    fn test_type_id() {
        let action = BioAction::NotificationBlock {
            duration_sec: 300,
            allow_priority: true,
        };
        assert_eq!(action.type_id(), "NotificationBlock");
    }
}
