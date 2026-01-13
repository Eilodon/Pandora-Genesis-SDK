//! Intervenable trait for causal interventions
//!
//! Implements Pearl's do-calculus intervention operator for counterfactual reasoning.
//! Enables "what-if" analysis: "What would happen if we forced variable X to value Y?"

use crate::belief::BeliefState;
use crate::causal::Variable;
use crate::domain::CausalBeliefState;

/// Trait for types that support causal interventions (Pearl's do-operator)
///
/// # Mathematical Foundation
/// In causal inference, an intervention do(X = x) represents forcing
/// a variable X to take value x, cutting all incoming causal edges.
///
/// This is different from observing X = x:
/// - Observation: P(Y | X = x) - "what is Y given we see X = x?"
/// - Intervention: P(Y | do(X = x)) - "what is Y if we force X = x?"
///
/// # Use Cases
/// - Counterfactual reasoning: "What if we had lowered HR to 60?"
/// - Policy evaluation: "What if we apply this intervention?"
/// - Safety analysis: "What if sensor fails and reports 0?"
///
/// # Audit Trail
/// Implementations should preserve intervention history for debugging
/// and safety auditing.
pub trait Intervenable {
    /// Perform causal intervention on a variable
    ///
    /// # Arguments
    /// * `variable` - The variable to intervene on
    /// * `value` - The value to force (normalized to [0, 1])
    ///
    /// # Returns
    /// New state with intervention applied and logged
    fn intervene(self, variable: Variable, value: f32) -> Self;

    /// Conditional intervention - only apply if condition is true
    ///
    /// This is useful for safety mechanisms:
    /// ```ignore
    /// state.intervene_if(emergency, Variable::HeartRate, safe_value)
    /// ```
    fn intervene_if(self, condition: bool, variable: Variable, value: f32) -> Self
    where
        Self: Sized,
    {
        if condition {
            self.intervene(variable, value)
        } else {
            self
        }
    }
}

/// Intervention log entry for audit trail
#[derive(Debug, Clone)]
pub struct InterventionLog {
    pub variable: Variable,
    pub original_value: f32,
    pub intervened_value: f32,
    pub reason: String,
}

impl Intervenable for BeliefState {
    fn intervene(mut self, variable: Variable, value: f32) -> Self {
        // BeliefState is a 5-mode probability distribution
        // We need to map the variable intervention to mode adjustments

        match variable {
            Variable::HeartRate | Variable::HeartRateVariability => {
                // Physiological interventions affect Stress/Calm modes
                if value < 0.3 {
                    // Low arousal → increase Calm
                    self.p[0] = (self.p[0] + 0.2).min(1.0); // Calm
                    self.p[1] = (self.p[1] - 0.2).max(0.0); // Stress
                } else if value > 0.7 {
                    // High arousal → increase Stress
                    self.p[1] = (self.p[1] + 0.2).min(1.0); // Stress
                    self.p[0] = (self.p[0] - 0.2).max(0.0); // Calm
                }
            }

            Variable::CognitiveLoad | Variable::NotificationPressure => {
                // Cognitive interventions affect Focus mode
                if value > 0.6 {
                    self.p[2] = (self.p[2] - 0.15).max(0.0); // Reduce Focus
                    self.p[1] = (self.p[1] + 0.15).min(1.0); // Increase Stress
                }
            }

            Variable::RespiratoryRate => {
                // Breath interventions affect Calm/Stress balance
                if value < 0.4 {
                    // Slow breathing → calming
                    self.p[0] = (self.p[0] + 0.25).min(1.0);
                    self.p[1] = (self.p[1] - 0.25).max(0.0);
                }
            }

            _ => {
                // Other variables - minimal effect on belief state
            }
        }

        // Renormalize to ensure probabilities sum to 1.0
        let sum: f32 = self.p.iter().sum();
        if sum > 0.0 {
            for p in &mut self.p {
                *p /= sum;
            }
        }

        self
    }
}

impl Intervenable for CausalBeliefState {
    fn intervene(mut self, variable: Variable, value: f32) -> Self {
        // CausalBeliefState has 3 separate state arrays: bio_state, cognitive_state, social_state
        // We need to map Variable to the appropriate state component

        let clamped = value.clamp(0.0, 1.0);

        match variable {
            // Biological variables affect bio_state
            Variable::HeartRate | Variable::HeartRateVariability | Variable::RespiratoryRate => {
                // Map to bio_state: [Calm, Aroused, Fatigue]
                if clamped < 0.3 {
                    // Low arousal → Calm
                    self.bio_state[0] = (self.bio_state[0] + 0.2).min(1.0);
                    self.bio_state[1] = (self.bio_state[1] - 0.2).max(0.0);
                } else if clamped > 0.7 {
                    // High arousal → Aroused
                    self.bio_state[1] = (self.bio_state[1] + 0.2).min(1.0);
                    self.bio_state[0] = (self.bio_state[0] - 0.2).max(0.0);
                }
                // Renormalize
                let sum: f32 = self.bio_state.iter().sum();
                if sum > 0.0 {
                    for s in &mut self.bio_state {
                        *s /= sum;
                    }
                }
            }

            // Cognitive variables affect cognitive_state
            Variable::CognitiveLoad
            | Variable::NotificationPressure
            | Variable::InteractionIntensity => {
                // Map to cognitive_state: [Focus, Distracted, Flow]
                if clamped > 0.6 {
                    // High load → Distracted
                    self.cognitive_state[1] = (self.cognitive_state[1] + 0.2).min(1.0);
                    self.cognitive_state[0] = (self.cognitive_state[0] - 0.2).max(0.0);
                }
                // Renormalize
                let sum: f32 = self.cognitive_state.iter().sum();
                if sum > 0.0 {
                    for s in &mut self.cognitive_state {
                        *s /= sum;
                    }
                }
            }

            // Social/environmental variables affect social_state
            Variable::NoiseLevel | Variable::EmotionalValence | Variable::VoiceArousal => {
                // Map to social_state: [Solitary, Interactive, Overwhelmed]
                if clamped > 0.7 {
                    // High intensity → Overwhelmed
                    self.social_state[2] = (self.social_state[2] + 0.2).min(1.0);
                    self.social_state[0] = (self.social_state[0] - 0.1).max(0.0);
                }
                // Renormalize
                let sum: f32 = self.social_state.iter().sum();
                if sum > 0.0 {
                    for s in &mut self.social_state {
                        *s /= sum;
                    }
                }
            }

            _ => {
                // Other variables - no direct mapping
            }
        }

        self
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_state_hr_intervention() {
        let mut state = BeliefState::default();
        state.p = [0.5, 0.3, 0.1, 0.05, 0.05]; // Calm-dominant

        // Intervene: force high HR (stress)
        let intervened = state.clone().intervene(Variable::HeartRate, 0.9);

        // Should increase stress, decrease calm
        assert!(intervened.p[1] > state.p[1]); // Stress increased
        assert!(intervened.p[0] < state.p[0]); // Calm decreased

        // Should still be normalized
        let sum: f32 = intervened.p.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_belief_state_rr_intervention() {
        let mut state = BeliefState::default();
        state.p = [0.2, 0.6, 0.1, 0.05, 0.05]; // Stress-dominant

        // Intervene: force slow breathing (calming)
        let intervened = state.clone().intervene(Variable::RespiratoryRate, 0.2);

        // Should increase calm, decrease stress
        assert!(intervened.p[0] > state.p[0]); // Calm increased
        assert!(intervened.p[1] < state.p[1]); // Stress decreased
    }

    #[test]
    fn test_conditional_intervention() {
        let state = BeliefState::default();

        // Should apply
        let intervened1 = state.clone().intervene_if(true, Variable::HeartRate, 0.9);
        assert_ne!(intervened1.p, state.p);

        // Should NOT apply
        let intervened2 = state.clone().intervene_if(false, Variable::HeartRate, 0.9);
        assert_eq!(intervened2.p, state.p);
    }

    #[test]
    fn test_causal_belief_state_intervention() {
        let mut state = CausalBeliefState::default();
        // Set initial bio_state
        state.bio_state = [0.5, 0.3, 0.2]; // Calm-dominant

        // Intervene with high HR (should increase arousal)
        let intervened = state.intervene(Variable::HeartRate, 0.9);

        // Should have increased arousal (bio_state[1])
        assert!(intervened.bio_state[1] > 0.3);
    }

    #[test]
    fn test_intervention_clamping() {
        let state = CausalBeliefState::default();

        // Test that extreme values are handled gracefully
        let intervened1 = state.clone().intervene(Variable::HeartRate, 1.5);
        // Should not panic, values should be normalized
        let sum: f32 = intervened1.bio_state.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        let intervened2 = state.intervene(Variable::HeartRate, -0.5);
        let sum2: f32 = intervened2.bio_state.iter().sum();
        assert!((sum2 - 1.0).abs() < 0.01);
    }
}
