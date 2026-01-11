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
    /// Convert the policy into a low-level ControlDecision for the engine.
    ///
    /// # Arguments
    /// * `current_bpm`: Current target BPM (fallback for non-breath policies)
    /// * `default_confidence`: Confidence to assign if not specified by policy
    pub fn to_control_decision(&self, current_bpm: f32, default_confidence: f32) -> crate::domain::ControlDecision {
        match self {
            ActionPolicy::NoAction => crate::domain::ControlDecision {
                target_rate_bpm: current_bpm,
                confidence: default_confidence,
                recommended_poll_interval_ms: 1000, // Standard poll
            },
            ActionPolicy::GuidanceBreath(params) => crate::domain::ControlDecision {
                target_rate_bpm: params.target_bpm,
                confidence: 0.9, // High confidence in explicit guidance
                recommended_poll_interval_ms: 500, // Faster poll for breath guidance
            },
            ActionPolicy::DigitalIntervention(_) => crate::domain::ControlDecision {
                target_rate_bpm: current_bpm, // Digital action doesn't change breath target
                confidence: 0.8,
                recommended_poll_interval_ms: 2000, // Slower poll for digital actions (longer duration)
            },
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

// ============================================================================
// FULL EFE CALCULATOR WITH EPISTEMIC VALUE
// ============================================================================

/// Expected Free Energy Calculator for Active Inference policy selection.
/// 
/// EFE = G(π) = E_q[D_KL[q(o|s,π) || p(o|C)] + H[q(s|o,π)]]
///            = Risk (pragmatic) + Ambiguity (epistemic)
/// 
/// Where:
/// - Risk: Expected divergence from preferred outcomes (C = preferences)
/// - Ambiguity: Expected uncertainty about states given observations
#[derive(Debug, Clone)]
pub struct EFECalculator {
    /// Precision (inverse temperature) for softmax policy selection
    pub precision: f32,
    
    /// Weight for pragmatic value (goal achievement)
    pub pragmatic_weight: f32,
    
    /// Weight for epistemic value (information gain)
    pub epistemic_weight: f32,
    
    /// Prior probability of each policy (policy prior)
    pub policy_prior: f32,
    
    /// Homeostatic setpoint for arousal (target state)
    pub arousal_setpoint: f32,
    
    /// Homeostatic setpoint for HRV
    pub hrv_setpoint: f32,
}

impl Default for EFECalculator {
    fn default() -> Self {
        Self {
            precision: 4.0,          // Moderate exploration/exploitation balance
            pragmatic_weight: 0.7,   // Emphasize goal achievement
            epistemic_weight: 0.3,   // But also value information
            policy_prior: 0.25,      // Uniform over 4 policy types
            arousal_setpoint: 0.4,   // Calm default
            hrv_setpoint: 50.0,      // Target RMSSD in ms
        }
    }
}


impl EFECalculator {
    pub fn new(precision: f32) -> Self {
        let mut calc = Self::default();
        calc.precision = precision;
        calc
    }
}

impl EFECalculator {
    /// Compute full expected free energy for a policy given current beliefs.
    /// 
    /// # Arguments
    /// * `policy` - The policy to evaluate
    /// * `belief_state` - Current posterior beliefs (5-mode distribution)
    /// * `belief_uncertainty` - Variance/entropy of current beliefs
    /// * `predicted_state` - Expected state after executing policy
    /// * `predicted_uncertainty` - Expected uncertainty after observation
    pub fn compute_efe(
        &self,
        policy: &ActionPolicy,
        belief_state: &[f32; 5],
        belief_uncertainty: f32,
        predicted_state: &[f32; 5],
        predicted_uncertainty: f32,
    ) -> PolicyEvaluation {
        // 1. PRAGMATIC VALUE: How much does this policy reduce divergence from preferences?
        // R = -D_KL[q(s|π) || p(s|C)] ≈ negative distance to homeostatic setpoint
        let pragmatic_value = self.compute_pragmatic_value(policy, predicted_state);
        
        // 2. EPISTEMIC VALUE: How much information does this policy provide?
        // I = H[q(s)] - E[H[q(s|o,π)]] = current entropy - expected posterior entropy
        let epistemic_value = self.compute_epistemic_value(
            policy,
            belief_uncertainty,
            predicted_uncertainty,
        );
        
        // 3. Combine with weights
        let combined_value = self.pragmatic_weight * pragmatic_value 
                           + self.epistemic_weight * epistemic_value;
        
        // 4. EFE is negative value (minimize G = maximize value)
        let efe = -combined_value;
        
        let mut eval = PolicyEvaluation::new(policy.clone(), pragmatic_value, epistemic_value);
        eval.expected_free_energy = efe;
        eval
    }

    /// Compute selection probabilities using Softmax over Negative Expected Free Energy
    /// P(π) = σ(-γ * G(π))
    pub fn compute_selection_probabilities(&self, evaluations: &mut [PolicyEvaluation]) {
        // Compute negative EFE (Value)
        let values: Vec<f32> = evaluations.iter()
            .map(|e| -e.expected_free_energy * self.precision)
            .collect();
            
        // Compute Softmax
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        let mut probs = values.clone();
        
        for p in probs.iter_mut() {
            *p = (*p - max_val).exp();
            sum += *p;
        }
        
        // Normalize
        if sum > 0.0 {
            for (i, p) in probs.iter().enumerate() {
                evaluations[i].selection_probability = *p / sum;
            }
        } else {
             // Fallback to uniform
             let n = evaluations.len() as f32;
             for e in evaluations.iter_mut() {
                 e.selection_probability = 1.0 / n;
             }
        }
    }

    /// Sample a policy based on computed probabilities
    pub fn sample_policy<'a>(&self, evaluations: &'a [PolicyEvaluation], rand_seed: f32) -> &'a ActionPolicy {
        let mut cumulative = 0.0;
        // Use provided seed (0.0-1.0) to select
        // Ensure rand_seed is within [0, 1]
        let r = rand_seed.clamp(0.0, 0.9999);
        
        for eval in evaluations {
            cumulative += eval.selection_probability;
            if r < cumulative {
                return &eval.policy;
            }
        }
        
        // Fallback to last (or first)
        &evaluations.last().unwrap().policy
    }
    
    /// Compute pragmatic value: how well does policy achieve goals?
    fn compute_pragmatic_value(&self, policy: &ActionPolicy, predicted_state: &[f32; 5]) -> f32 {
        // Predicted arousal (mode 1 = Stress)
        let predicted_arousal = predicted_state[1];
        
        // Distance from homeostatic setpoint
        let arousal_error = (predicted_arousal - self.arousal_setpoint).abs();
        
        // Base value from state improvement
        let state_value = 1.0 - arousal_error.min(1.0);
        
        // Policy-specific adjustments
        let policy_bonus = match policy {
            ActionPolicy::NoAction => {
                // NoAction gets bonus if state is already good
                if arousal_error < 0.2 { 0.2 } else { -0.1 }
            }
            ActionPolicy::GuidanceBreath(ref params) => {
                // Breath guidance gets bonus proportional to expected regulation
                let regulation_strength = (6.0 / params.target_bpm).clamp(0.5, 1.5);
                regulation_strength * 0.3
            }
            ActionPolicy::DigitalIntervention(ref intervention) => {
                // Digital interventions get smaller bonus (indirect effect)
                match intervention.action {
                    DigitalActionType::BlockNotifications => 0.15,
                    DigitalActionType::PlaySoundscape => 0.2,
                    DigitalActionType::SuggestBreak => 0.1,
                    DigitalActionType::LaunchApp => 0.1,
                }
            }
        };
        
        (state_value + policy_bonus).clamp(0.0, 1.0)
    }
    
    /// Compute epistemic value: information gain from policy execution.
    fn compute_epistemic_value(
        &self,
        policy: &ActionPolicy,
        current_uncertainty: f32,
        predicted_uncertainty: f32,
    ) -> f32 {
        // Information gain = reduction in entropy
        // I(π) = H[q(s)] - E[H[q(s|o,π)]]
        let entropy_reduction = (current_uncertainty - predicted_uncertainty).max(0.0);
        
        // Policy-specific epistemic bonus
        let exploration_bonus = match policy {
            ActionPolicy::NoAction => {
                // NoAction has high epistemic value when uncertain
                // (pure observation to gather information)
                if current_uncertainty > 0.5 { 0.4 } else { 0.1 }
            }
            ActionPolicy::GuidanceBreath(_) => {
                // Breath guidance provides direct feedback (high epistemic value)
                0.25
            }
            ActionPolicy::DigitalIntervention(ref intervention) => {
                // Digital interventions provide indirect feedback
                match intervention.action {
                    DigitalActionType::PlaySoundscape => 0.15, // Some feedback possible
                    _ => 0.05, // Low direct feedback
                }
            }
        };
        
        (entropy_reduction + exploration_bonus).clamp(0.0, 1.0)
    }
    
    /// Select policy using softmax over negative EFE values.
    /// Returns policies sorted by selection probability (highest first).
    pub fn select_policy(&self, evaluations: &mut [PolicyEvaluation]) {
        if evaluations.is_empty() {
            return;
        }
        
        // Compute softmax over -EFE (minimizing EFE = maximizing -EFE)
        let neg_efes: Vec<f32> = evaluations.iter()
            .map(|e| -e.expected_free_energy * self.precision)
            .collect();
        
        // Stable softmax
        let max_neg_efe = neg_efes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = neg_efes.iter()
            .map(|&x| (x - max_neg_efe).exp())
            .sum();
        
        for (i, eval) in evaluations.iter_mut().enumerate() {
            eval.selection_probability = ((neg_efes[i] - max_neg_efe).exp()) / exp_sum;
        }
        
        // Sort by probability (descending)
        evaluations.sort_by(|a, b| b.selection_probability
            .partial_cmp(&a.selection_probability)
            .unwrap_or(std::cmp::Ordering::Equal));
    }
}
    


/// Policy library: pre-defined policies for common scenarios.
/// These serve as a "policy prior" in Active Inference - the system's
/// innate repertoire of behaviors before learning.
// ============================================================================
// META-LEARNING: Adaptive Precision (Beta)
// ============================================================================

/// adaptive meta-learner for EFE precision (beta).
/// Balances exploration vs exploitation based on recent success rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaMetaLearner {
    /// Score tracking recent exploration (high uncertainty policies)
    pub exploration_score: f32,
    
    /// Score tracking recent exploitation (high pragmatic value policies)
    pub exploitation_score: f32,
    
    /// Exponential moving average of policy success (0.0 - 1.0)
    pub success_rate_ema: f32,
    
    /// Target ratio of exploration to total activity (default: 0.2)
    pub target_exploration_ratio: f32,
}

impl Default for BetaMetaLearner {
    fn default() -> Self {
        Self {
            exploration_score: 1.0,
            exploitation_score: 4.0,
            success_rate_ema: 0.5,
            target_exploration_ratio: 0.2, // 20% exploration, 80% exploitation
        }
    }
}

impl BetaMetaLearner {
    /// Update beta based on recent policy outcome.
    /// 
    /// # Logic
    /// - If success rate is low and we are over-exploring -> Increase beta (Exploit more)
    /// - If exploration ratio is too low -> Decrease beta (Explore more)
    /// - If success rate is high -> Slightly increase beta (Stabilize)
    pub fn update_beta(
        &mut self,
        current_beta: f32,
        policy_was_exploratory: bool,
        policy_succeeded: bool,
    ) -> f32 {
        // Update success rate (alpha = 0.1)
        let alpha = 0.1;
        self.success_rate_ema = self.success_rate_ema * (1.0 - alpha)
            + (if policy_succeeded { 1.0 } else { 0.0 }) * alpha;
        
        // Update exploration tracking
        if policy_was_exploratory {
            self.exploration_score += 1.0;
        } else {
            self.exploitation_score += 1.0;
        }
        
        let total = self.exploration_score + self.exploitation_score;
        // Prevent division by zero (unlikely with default initialization)
        let current_ratio = if total > 0.0 { 
            self.exploration_score / total 
        } else { 
            0.0 
        };
        
        // Adapt beta
        let new_beta = if current_ratio > self.target_exploration_ratio && self.success_rate_ema < 0.6 {
            // Over-exploring with poor results -> exploit more (increase precision)
            (current_beta * 1.1).min(5.0)
        } else if current_ratio < self.target_exploration_ratio {
            // Under-exploring -> explore more (decrease precision)
            (current_beta * 0.9).max(0.1)
        } else if self.success_rate_ema > 0.8 {
            // Doing well -> slowly consolidate/exploit
            (current_beta * 1.02).min(5.0)
        } else {
            current_beta // No change
        };
        
        // Decay scores (forgetting factor)
        self.exploration_score *= 0.99;
        self.exploitation_score *= 0.99;
        
        new_beta
    }
}

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

    #[test]
    fn test_beta_meta_learner() {
        let mut learner = super::BetaMetaLearner::default();
        let initial_beta = 1.0;

        // 1. Simulate under-exploration -> Beta should decrease (encourage exploration)
        learner.exploration_score = 0.0;
        learner.exploitation_score = 10.0; // Ratio = 0.0 < 0.2 Target
        let new_beta = learner.update_beta(initial_beta, false, true);
        assert!(new_beta < initial_beta, "Beta should decrease to encourage exploration");

        // 2. Simulate over-exploration with failure -> Beta should increase (force exploitation)
        learner.exploration_score = 10.0;
        learner.exploitation_score = 0.0; // Ratio = 1.0 > 0.2
        learner.success_rate_ema = 0.4; // Low success
        let new_beta_2 = learner.update_beta(1.0, true, false);
        assert!(new_beta_2 > 1.0, "Beta should increase to stop failed exploration");
    }
}
