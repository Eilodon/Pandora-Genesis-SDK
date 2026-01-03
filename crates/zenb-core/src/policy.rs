//! Policy Layer: Action Selection and Intervention Planning
//!
//! This module defines the output layer of the Homeostatic AI Kernel.
//! Policies map from belief states to concrete actions in both biological
//! (breath guidance) and digital (app interventions) domains.
//!
//! Active Inference Perspective:
//! - Policies are sequences of actions that minimize expected free energy
//! - Free energy = surprise + ambiguity (exploration vs exploitation)
//! - Actions serve dual purpose: regulation (reduce prediction error) and
//!   information gathering (reduce uncertainty about hidden states)

use serde::{Deserialize, Serialize};

// ============================================================================
// OUTPUT LAYER: Action Policy Space
// ============================================================================

/// Digital intervention action types.
/// These are concrete actions the system can take in the digital environment
/// to support homeostatic regulation and cognitive wellbeing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DigitalActionType {
    /// Block or suppress incoming notifications temporarily.
    /// Use case: High cognitive load, need for sustained focus.
    /// Expected effect: Reduce interruption-driven stress, improve flow state.
    BlockNotifications,

    /// Play ambient soundscape or calming audio.
    /// Use case: Elevated arousal, need for parasympathetic activation.
    /// Expected effect: Reduce physiological arousal, support relaxation.
    PlaySoundscape,

    /// Launch a specific application (wellness, meditation, etc.).
    /// Use case: Detected need for active intervention (e.g., guided meditation).
    /// Expected effect: Provide structured support for state regulation.
    LaunchApp,

    /// Suggest a break or rest period to the user.
    /// Use case: Prolonged high cognitive load or detected fatigue.
    /// Expected effect: Prevent burnout, support recovery.
    SuggestBreak,
}

/// Parameters for digital interventions.
/// Provides fine-grained control over intervention execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalIntervention {
    /// The type of digital action to perform.
    pub action: DigitalActionType,

    /// Optional duration in seconds for time-limited interventions.
    /// None = indefinite or user-controlled duration.
    /// Some(duration) = automatically revert after specified time.
    /// Example: BlockNotifications for 1800 seconds (30 minutes).
    pub duration_sec: Option<u32>,

    /// Optional target application identifier.
    /// Used when action requires specifying a particular app (e.g., LaunchApp).
    /// Format depends on platform: bundle ID (iOS), package name (Android), etc.
    pub target_app: Option<String>,

    /// Optional intensity or strength parameter normalized to [0, 1].
    /// Example: volume level for PlaySoundscape, urgency level for SuggestBreak.
    pub intensity: Option<f32>,
}

/// Parameters for breath guidance interventions.
/// Specifies the breathing pattern and target physiological state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceBreath {
    /// Identifier for the breathing pattern to use.
    /// Examples: "box_breathing", "4_7_8", "resonance_breathing", "coherent_breathing".
    /// Pattern definitions should be stored in a pattern library.
    pub pattern_id: String,

    /// Target breathing rate in breaths per minute (BPM).
    /// Typical range: 4-12 BPM for regulation, 6 BPM for resonance frequency.
    /// Lower rates generally promote parasympathetic activation.
    pub target_bpm: f32,

    /// Optional duration in seconds for the guided session.
    /// None = continue until user stops or state goal achieved.
    pub duration_sec: Option<u32>,

    /// Optional target heart rate variability (HRV) in milliseconds.
    /// System will adapt breathing guidance to achieve this HRV target.
    pub target_hrv: Option<f32>,
}

/// Root action policy enum representing all possible system actions.
/// This is the output of the policy selection process in Active Inference.
///
/// Policy Selection Process:
/// 1. Compute expected free energy for each policy under current beliefs
/// 2. Select policy that minimizes expected free energy (or sample probabilistically)
/// 3. Execute first action in selected policy
/// 4. Update beliefs based on new observations
/// 5. Repeat (receding horizon control)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPolicy {
    /// No intervention: system observes passively.
    /// Selected when: beliefs are uncertain, current state is acceptable,
    /// or no intervention is predicted to reduce free energy.
    /// This is NOT inaction - it's an active choice to gather information.
    NoAction,

    /// Breath guidance intervention: direct physiological regulation.
    /// Selected when: biological state requires regulation (arousal, stress),
    /// user context permits (not in meeting, not driving), and
    /// breath intervention is predicted to be effective.
    GuidanceBreath(GuidanceBreath),

    /// Digital environment intervention: modify digital context.
    /// Selected when: digital factors contribute to dysregulation,
    /// intervention is contextually appropriate, and
    /// expected to reduce cognitive load or improve state.
    DigitalIntervention(DigitalIntervention),
}

impl ActionPolicy {
    /// Check if this policy requires user permission before execution.
    /// Some actions are more intrusive and should request consent.
    pub fn requires_permission(&self) -> bool {
        match self {
            ActionPolicy::NoAction => false,
            ActionPolicy::GuidanceBreath(_) => false, // Guidance is opt-in by nature
            ActionPolicy::DigitalIntervention(intervention) => {
                // Blocking notifications or launching apps are intrusive
                matches!(
                    intervention.action,
                    DigitalActionType::BlockNotifications | DigitalActionType::LaunchApp
                )
            }
        }
    }

    /// Estimate the intrusiveness level of this policy on a scale of 0-1.
    /// Used for: balancing intervention effectiveness vs user autonomy.
    /// 0 = completely passive, 1 = highly intrusive.
    pub fn intrusiveness(&self) -> f32 {
        match self {
            ActionPolicy::NoAction => 0.0,
            ActionPolicy::GuidanceBreath(_) => 0.3, // User-initiated but requires attention
            ActionPolicy::DigitalIntervention(intervention) => match intervention.action {
                DigitalActionType::SuggestBreak => 0.4,
                DigitalActionType::PlaySoundscape => 0.5,
                DigitalActionType::BlockNotifications => 0.7,
                DigitalActionType::LaunchApp => 0.8,
            },
        }
    }

    /// Get a human-readable description of this policy for logging/UI.
    pub fn description(&self) -> String {
        match self {
            ActionPolicy::NoAction => "Passive observation".to_string(),
            ActionPolicy::GuidanceBreath(params) => {
                format!(
                    "Breath guidance: {} at {:.1} BPM",
                    params.pattern_id, params.target_bpm
                )
            }
            ActionPolicy::DigitalIntervention(intervention) => {
                let action_str = match intervention.action {
                    DigitalActionType::BlockNotifications => "Block notifications",
                    DigitalActionType::PlaySoundscape => "Play soundscape",
                    DigitalActionType::LaunchApp => "Launch app",
                    DigitalActionType::SuggestBreak => "Suggest break",
                };

                if let Some(duration) = intervention.duration_sec {
                    format!("{} for {} seconds", action_str, duration)
                } else {
                    action_str.to_string()
                }
            }
        }
    }
}

// ============================================================================
// Policy Evaluation and Selection Utilities
// ============================================================================

/// Expected free energy components for policy evaluation.
/// In Active Inference, policies are selected to minimize expected free energy,
/// which decomposes into pragmatic value (goal achievement) and epistemic value
/// (information gain).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvaluation {
    /// The policy being evaluated.
    pub policy: ActionPolicy,

    /// Pragmatic value: expected utility or reward.
    /// Higher = better expected outcome for homeostatic goals.
    /// Example: breath guidance when aroused has high pragmatic value.
    pub pragmatic_value: f32,

    /// Epistemic value: expected information gain.
    /// Higher = more uncertainty reduction about hidden states.
    /// Example: trying a new intervention has high epistemic value.
    pub epistemic_value: f32,

    /// Expected free energy (lower is better).
    /// G = -pragmatic_value - epistemic_value (simplified).
    /// Actual computation involves KL divergences and expected surprisal.
    pub expected_free_energy: f32,

    /// Probability of selecting this policy (softmax over -G).
    pub selection_probability: f32,
}

impl PolicyEvaluation {
    /// Create a new policy evaluation with computed expected free energy.
    pub fn new(policy: ActionPolicy, pragmatic_value: f32, epistemic_value: f32) -> Self {
        // Simplified EFE: negative of combined values
        // Real implementation would use proper KL divergences
        let expected_free_energy = -(pragmatic_value + epistemic_value);

        Self {
            policy,
            pragmatic_value,
            epistemic_value,
            expected_free_energy,
            selection_probability: 0.0, // Set by softmax over batch
        }
    }
}

/// Policy library: pre-defined policies for common scenarios.
/// These serve as a "policy prior" in Active Inference - the system's
/// innate repertoire of behaviors before learning.
pub struct PolicyLibrary;

impl PolicyLibrary {
    /// Get a calming breath guidance policy for arousal reduction.
    pub fn calming_breath() -> ActionPolicy {
        ActionPolicy::GuidanceBreath(GuidanceBreath {
            pattern_id: "resonance_breathing".to_string(),
            target_bpm: 6.0,         // Resonance frequency for most adults
            duration_sec: Some(300), // 5 minutes
            target_hrv: Some(60.0),  // Target HRV increase
        })
    }

    /// Get an energizing breath guidance policy for fatigue.
    pub fn energizing_breath() -> ActionPolicy {
        ActionPolicy::GuidanceBreath(GuidanceBreath {
            pattern_id: "bellows_breathing".to_string(),
            target_bpm: 20.0,        // Rapid breathing for activation
            duration_sec: Some(180), // 3 minutes
            target_hrv: None,
        })
    }

    /// Get a focus-enhancing digital intervention.
    pub fn focus_mode() -> ActionPolicy {
        ActionPolicy::DigitalIntervention(DigitalIntervention {
            action: DigitalActionType::BlockNotifications,
            duration_sec: Some(1800), // 30 minutes
            target_app: None,
            intensity: Some(1.0), // Maximum blocking
        })
    }

    /// Get a break suggestion policy for cognitive overload.
    pub fn suggest_rest() -> ActionPolicy {
        ActionPolicy::DigitalIntervention(DigitalIntervention {
            action: DigitalActionType::SuggestBreak,
            duration_sec: Some(600), // Suggest 10-minute break
            target_app: None,
            intensity: Some(0.7), // Moderate urgency
        })
    }

    /// Get a passive observation policy.
    pub fn observe() -> ActionPolicy {
        ActionPolicy::NoAction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_intrusiveness() {
        assert_eq!(PolicyLibrary::observe().intrusiveness(), 0.0);
        assert!(PolicyLibrary::calming_breath().intrusiveness() > 0.0);
        assert!(PolicyLibrary::focus_mode().intrusiveness() > 0.5);
    }

    #[test]
    fn test_policy_description() {
        let policy = PolicyLibrary::calming_breath();
        let desc = policy.description();
        assert!(desc.contains("Breath guidance"));
        assert!(desc.contains("6.0 BPM"));
    }

    #[test]
    fn test_permission_requirements() {
        assert!(!PolicyLibrary::observe().requires_permission());
        assert!(!PolicyLibrary::calming_breath().requires_permission());
        assert!(PolicyLibrary::focus_mode().requires_permission());
    }

    #[test]
    fn test_policy_evaluation() {
        let policy = PolicyLibrary::calming_breath();
        let eval = PolicyEvaluation::new(policy, 0.8, 0.2);

        assert_eq!(eval.pragmatic_value, 0.8);
        assert_eq!(eval.epistemic_value, 0.2);
        assert_eq!(eval.expected_free_energy, -1.0);
    }
}
