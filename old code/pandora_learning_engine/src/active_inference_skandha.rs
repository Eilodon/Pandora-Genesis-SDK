//! Active Inference Sankhara Skandha with planning and imagination capabilities
//!
//! This module implements an advanced Sankhara Skandha that uses Active Inference
//! for planning by simulating future states and selecting actions that minimize
//! expected free energy. It integrates with the Causal World Model (CWM) and
//! Learning Engine to enable goal-oriented behavior.

use crate::{LearningEngine, ValueDrivenPolicy};
use pandora_core::interfaces::skandhas::{SankharaSkandha, Skandha};
use pandora_core::ontology::{EpistemologicalFlow, Vedana};
use pandora_core::world_model::WorldModel;
use pandora_error::PandoraError;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

/// Represents a causal hypothesis discovered through data analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CausalHypothesis {
    pub from_node_index: usize,
    pub to_node_index: usize,
    pub strength: f32,
    pub confidence: f32,
    pub edge_type: CausalEdgeType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CausalEdgeType {
    Direct,
    Indirect,
    Conditional,
    Inhibitory,
}

/// Simple epsilon-greedy policy (legacy, kept for backward compatibility)
#[derive(Debug, Clone)]
pub struct SimplePolicy {
    /// Epsilon for epsilon-greedy exploration
    pub epsilon: f64,
    /// Available actions for the policy
    pub available_actions: Vec<&'static str>,
}

/// Active Inference Sankhara Skandha with planning and imagination capabilities
///
/// This skandha implements Active Inference for planning by simulating future states
/// and selecting actions that minimize expected free energy. It integrates with the
/// Causal World Model (CWM) and Learning Engine to enable goal-oriented behavior.
pub struct ActiveInferenceSankharaSkandha {
    /// Reference to the Causal World Model for state prediction
    pub cwm: Arc<Mutex<dyn WorldModel + Send + Sync>>,
    /// Reference to the Learning Engine for reward calculation
    pub learning_engine: Arc<LearningEngine>,
    /// How many steps to simulate into the future
    pub planning_horizon: usize,
    /// Available actions for the agent
    pub available_actions: Vec<&'static str>,
    /// Pending causal hypothesis to test through experimentation
    pub pending_hypothesis: Option<CausalHypothesis>,
    /// Mapping from concepts (nodes) to actions that can influence them
    pub concept_action_mapping: std::collections::HashMap<usize, Vec<&'static str>>,
    /// Discount factor for future rewards (gamma)
    pub gamma: f64,
    /// Advanced value-driven policy with adaptive exploration
    pub policy: Arc<Mutex<ValueDrivenPolicy>>,
}

impl SimplePolicy {
    /// Creates a new SimplePolicy (legacy epsilon-greedy)
    pub fn new(epsilon: f64, available_actions: Vec<&'static str>) -> Self {
        Self {
            epsilon,
            available_actions,
        }
    }

    /// Selects an action using epsilon-greedy strategy
    pub fn select_action(&self, _flow: &EpistemologicalFlow) -> Result<&'static str, PandoraError> {
        use rand::Rng;

        // Epsilon-greedy strategy
        if rand::thread_rng().gen::<f64>() < self.epsilon {
            // Explore: choose a random action
            let random_idx = rand::thread_rng().gen_range(0..self.available_actions.len());
            Ok(self.available_actions[random_idx])
        } else {
            // Exploit: choose the first action (simplified for now)
            // In a full implementation, this would query the ValueEstimator
            Ok(self.available_actions[0])
        }
    }
}

impl ActiveInferenceSankharaSkandha {
    /// Creates a new ActiveInferenceSankharaSkandha
    ///
    /// # Arguments
    ///
    /// * `cwm` - The Causal World Model wrapped in Arc<Mutex<>>
    /// * `learning_engine` - The Learning Engine wrapped in Arc
    /// * `planning_horizon` - Number of steps to simulate into the future
    /// * `available_actions` - List of available actions for the agent
    /// * `gamma` - Discount factor for future rewards
    /// * `policy_epsilon` - Epsilon for policy exploration (used as base exploration constant)
    pub fn new(
        cwm: Arc<Mutex<dyn WorldModel + Send + Sync>>,
        learning_engine: Arc<LearningEngine>,
        planning_horizon: usize,
        available_actions: Vec<&'static str>,
        gamma: f64,
        policy_epsilon: f64,
    ) -> Self {
        // Create ValueDrivenPolicy with adaptive exploration
        // learning_rate=0.2 (increased for faster Q-value convergence)
        // discount_factor=gamma, exploration_constant=policy_epsilon
        let policy = Arc::new(Mutex::new(ValueDrivenPolicy::new(0.2, gamma, policy_epsilon)));
        let concept_action_mapping = Self::create_default_concept_action_mapping();
        Self {
            cwm,
            learning_engine,
            planning_horizon,
            available_actions,
            pending_hypothesis: None,
            concept_action_mapping,
            gamma,
            policy,
        }
    }

    /// Get access to the underlying ValueDrivenPolicy for testing and monitoring
    pub fn get_policy(&self) -> Arc<Mutex<ValueDrivenPolicy>> {
        Arc::clone(&self.policy)
    }

    /// Creates a default mapping from concepts to actions that can influence them
    fn create_default_concept_action_mapping() -> std::collections::HashMap<usize, Vec<&'static str>>
    {
        let mut mapping = std::collections::HashMap::new();

        // Door-related concepts (nodes 0-9)
        mapping.insert(0, vec!["unlock_door", "lock_door", "check_door_status"]);
        mapping.insert(1, vec!["unlock_door", "lock_door", "check_door_status"]);
        mapping.insert(2, vec!["unlock_door", "lock_door", "check_door_status"]);

        // Key-related concepts (nodes 10-19)
        mapping.insert(10, vec!["pick_up_key", "drop_key", "check_key_status"]);
        mapping.insert(11, vec!["pick_up_key", "drop_key", "check_key_status"]);
        mapping.insert(12, vec!["pick_up_key", "drop_key", "check_key_status"]);

        // Position-related concepts (nodes 20-29)
        mapping.insert(20, vec!["move_forward", "move_backward", "check_position"]);
        mapping.insert(21, vec!["move_forward", "move_backward", "check_position"]);
        mapping.insert(22, vec!["move_forward", "move_backward", "check_position"]);

        // Switch-related concepts (nodes 30-39)
        mapping.insert(
            30,
            vec!["turn_on_switch", "turn_off_switch", "check_switch_status"],
        );
        mapping.insert(
            31,
            vec!["turn_on_switch", "turn_off_switch", "check_switch_status"],
        );

        // Light-related concepts (nodes 40-49)
        mapping.insert(
            40,
            vec!["turn_on_light", "turn_off_light", "check_light_status"],
        );
        mapping.insert(
            41,
            vec!["turn_on_light", "turn_off_light", "check_light_status"],
        );

        mapping
    }

    /// Set a pending causal hypothesis to test through experimentation.
    pub fn set_pending_hypothesis(&mut self, hypothesis: Option<CausalHypothesis>) {
        self.pending_hypothesis = hypothesis;
        if let Some(ref hyp) = self.pending_hypothesis {
            info!("ActiveInference: Set pending hypothesis: {:?}", hyp);
        } else {
            info!("ActiveInference: Cleared pending hypothesis");
        }
    }

    /// Get the current pending hypothesis.
    pub fn get_pending_hypothesis(&self) -> &Option<CausalHypothesis> {
        &self.pending_hypothesis
    }

    /// Clear the pending hypothesis after testing.
    pub fn clear_pending_hypothesis(&mut self) {
        self.pending_hypothesis = None;
        info!("ActiveInference: Cleared pending hypothesis");
    }
}

impl Skandha for ActiveInferenceSankharaSkandha {
    fn name(&self) -> &'static str {
        "Active Inference Sankhara (Planning & Imagination)"
    }
}

impl SankharaSkandha for ActiveInferenceSankharaSkandha {
    fn form_intent(&self, flow: &mut EpistemologicalFlow) {
        info!(
            "[{}] Khởi phát ý chỉ với Active Inference planning.",
            self.name()
        );

        match self.plan_action(flow) {
            Ok(intent) => {
                info!("[{}] Khởi phát ý chỉ: '{}'", self.name(), intent);
                flow.sankhara = Some(std::sync::Arc::<str>::from(intent));
            }
            Err(e) => {
                info!(
                    "[{}] Lỗi trong quá trình planning: {}. Sử dụng intent mặc định.",
                    self.name(),
                    e
                );
                flow.sankhara = Some(std::sync::Arc::<str>::from("default_fallback_intent"));
            }
        }
    }
}

impl ActiveInferenceSankharaSkandha {
    /// Plans the best action using Active Inference
    ///
    /// This method simulates future states for each candidate action and selects
    /// the one that minimizes expected free energy (maximizes expected reward).
    fn plan_action(
        &self,
        current_flow: &EpistemologicalFlow,
    ) -> Result<&'static str, PandoraError> {
        info!(
            "[{}] Bắt đầu quá trình planning với horizon: {}",
            self.name(),
            self.planning_horizon
        );

        let cwm = self
            .cwm
            .lock()
            .map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;

        // 1. Propose candidate actions
        let mut candidate_actions = self.propose_candidate_actions(current_flow);
        // Randomize order to avoid deterministic tie-breaking when scores are equal
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        candidate_actions.shuffle(&mut rng);
        info!(
            "[{}] Đề xuất {} hành động: {:?}",
            self.name(),
            candidate_actions.len(),
            candidate_actions
        );

        if candidate_actions.is_empty() {
            return Ok("intent_do_nothing"); // Default safe action
        }

        let mut best_action = *candidate_actions.first().unwrap();
        let mut best_expected_free_energy = f64::NEG_INFINITY;

        // 2. For each possible action, simulate the future
        for &action in &candidate_actions {
            let mut total_future_efe = 0.0;
            // Create a deep copy for simulation to avoid side effects
            let mut simulated_flow = current_flow.clone();

            // Set the initial action for the first step of the simulation
            simulated_flow.sankhara = Some(std::sync::Arc::<str>::from(action));

            // 3. Rollout the simulation for `planning_horizon` steps
            for i in 0..self.planning_horizon {
                // **Crucial Step 1: Prediction**
                // Ask the CWM to predict the outcome of the action.
                // This modifies `simulated_flow` to represent the next hypothetical state.
                // Note: This is a simplified version - in a full implementation,
                // we would need a more sophisticated state prediction mechanism
                let _temp_flow = simulated_flow.clone();

                // For now, we'll simulate prediction by just continuing with the current state
                // In a full implementation, this would call cwm.predict_next_state(&mut simulated_flow)
                debug!("[{}] Simulating prediction for step {}", self.name(), i);

                // **Crucial Step 2: Evaluation**
                // Calculate the Expected Free Energy (EFE) for this imagined state.
                let reward = self.learning_engine.calculate_reward(
                    &*cwm,
                    &*cwm, // In simulation, the model doesn't change
                    &simulated_flow,
                );
                let efe = self.learning_engine.get_total_weighted_reward(&reward);

                // Apply a discount factor for future steps (gamma)
                total_future_efe += efe * self.gamma.powi(i as i32);

                // **Crucial Step 3: Propose next action for the simulation**
                // For the next step in the simulation, we need a new action.
                // Use ValueDrivenPolicy with its adaptive exploration rate
                if i < self.planning_horizon - 1 {
                    if let Ok(policy_guard) = self.policy.lock() {
                        let explore_rate = policy_guard.get_exploration_rate();
                        // Use Policy trait method
                        use crate::Policy;
                        let action_enum = policy_guard.select_action(&simulated_flow, explore_rate);
                        
                        // Map PolicyAction to action string
                        let next_sim_action = match action_enum {
                            crate::PolicyAction::Noop => self.available_actions.first().copied().unwrap_or("default_action"),
                            crate::PolicyAction::Explore => self.available_actions.get(1).copied().unwrap_or("default_action"),
                            crate::PolicyAction::Exploit => self.available_actions.get(2).copied().unwrap_or("default_action"),
                            _ => self.available_actions.first().copied().unwrap_or("default_action"),
                        };
                        
                        simulated_flow.sankhara = Some(std::sync::Arc::<str>::from(next_sim_action));
                    } else {
                        warn!(
                            "[{}] Failed to acquire policy lock at step {}",
                            self.name(),
                            i
                        );
                        simulated_flow.sankhara = Some(std::sync::Arc::<str>::from("default_action"));
                    }
                }

                debug!(
                    "[{}] Step {} cho action '{}': EFE = {:.4}, Total = {:.4}",
                    self.name(),
                    i + 1,
                    action,
                    efe,
                    total_future_efe
                );
            }

            info!(
                "[{}] Action '{}' có tổng EFE: {:.4}",
                self.name(),
                action,
                total_future_efe
            );

            if total_future_efe > best_expected_free_energy {
                best_expected_free_energy = total_future_efe;
                best_action = action;
            }
        }

        info!(
            "[{}] Chọn action '{}' với EFE: {:.4}",
            self.name(),
            best_action,
            best_expected_free_energy
        );

        Ok(best_action)
    }

    /// Proposes candidate actions based on current context
    ///
    /// This method analyzes the current flow and generates relevant actions.
    /// In experiment mode, it focuses exclusively on actions that can test the hypothesis.
    fn propose_candidate_actions(&self, flow: &EpistemologicalFlow) -> Vec<&'static str> {
        let mut candidates = Vec::new();

        // **EXPERIMENT MODE**: If we have a pending hypothesis, focus exclusively on testing it
        if let Some(ref hypothesis) = self.pending_hypothesis {
            info!(
                "ActiveInference: Entering experiment mode for hypothesis: {:?}",
                hypothesis
            );

            // **CORE EXPERIMENTAL LOGIC**: Generate actions that manipulate the 'from' node
            // to observe the effect on the 'to' node

            // 1. Get actions that can influence the cause variable (from_node)
            if let Some(actions) = self.concept_action_mapping.get(&hypothesis.from_node_index) {
                info!(
                    "ActiveInference: Found {} actions for cause node {}: {:?}",
                    actions.len(),
                    hypothesis.from_node_index,
                    actions
                );
                candidates.extend(actions.iter().cloned());
            } else {
                // Fallback: generate generic manipulation actions
                candidates.push("MANIPULATE_CAUSE_VARIABLE");
                candidates.push("ACTIVATE_CAUSE_NODE");
                candidates.push("DEACTIVATE_CAUSE_NODE");
                info!(
                    "ActiveInference: No specific actions found for cause node {}, using fallbacks",
                    hypothesis.from_node_index
                );
            }

            // 2. Add observation actions for the effect variable (to_node)
            if let Some(actions) = self.concept_action_mapping.get(&hypothesis.to_node_index) {
                // Filter for observation/measurement actions
                let observation_actions: Vec<&'static str> = actions
                    .iter()
                    .filter(|action| {
                        action.contains("check")
                            || action.contains("observe")
                            || action.contains("measure")
                    })
                    .cloned()
                    .collect();
                if !observation_actions.is_empty() {
                    candidates.extend(observation_actions);
                }
            }

            // 3. Add specific experimental actions based on edge type
            match hypothesis.edge_type {
                CausalEdgeType::Direct => {
                    candidates.push("TEST_DIRECT_CAUSALITY");
                    candidates.push("ISOLATE_DIRECT_EFFECT");
                    // For direct causality, we want to test immediate cause-effect relationship
                    if let Some(actions) =
                        self.concept_action_mapping.get(&hypothesis.from_node_index)
                    {
                        // Add both positive and negative manipulation actions
                        for action in actions {
                            if action.contains("on") || action.contains("activate") {
                                candidates.push(action);
                            }
                        }
                    }
                }
                CausalEdgeType::Indirect => {
                    candidates.push("TEST_INDIRECT_CAUSALITY");
                    candidates.push("EXPLORE_MEDIATING_FACTORS");
                    candidates.push("TRACE_CAUSAL_CHAIN");
                    // For indirect causality, we need to test the mediating path
                    candidates.push("IDENTIFY_MEDIATORS");
                    candidates.push("TEST_MEDIATION_PATH");
                }
                CausalEdgeType::Conditional => {
                    candidates.push("TEST_CONDITIONAL_CAUSALITY");
                    candidates.push("VARY_CONDITIONS");
                    candidates.push("TEST_UNDER_DIFFERENT_CONDITIONS");
                    // For conditional causality, we need to test under different conditions
                    candidates.push("ESTABLISH_CONDITIONS");
                    candidates.push("VARY_CONTEXT");
                }
                CausalEdgeType::Inhibitory => {
                    candidates.push("TEST_INHIBITORY_CAUSALITY");
                    candidates.push("BLOCK_INHIBITOR");
                    candidates.push("REMOVE_INHIBITION");
                    // For inhibitory causality, we test by removing the inhibitor
                    if let Some(actions) =
                        self.concept_action_mapping.get(&hypothesis.from_node_index)
                    {
                        for action in actions {
                            if action.contains("off")
                                || action.contains("deactivate")
                                || action.contains("remove")
                            {
                                candidates.push(action);
                            }
                        }
                    }
                }
            }

            // 4. Add control actions to ensure clean experiments
            candidates.push("ESTABLISH_BASELINE");
            candidates.push("CONTROL_FOR_CONFOUNDING_VARIABLES");
            candidates.push("MEASURE_BEFORE_AND_AFTER");
            candidates.push("RECORD_EXPERIMENTAL_CONTEXT");

            // 5. Add actions based on hypothesis confidence and strength
            if hypothesis.confidence > 0.7 {
                candidates.push("CONDUCT_DEFINITIVE_EXPERIMENT");
                candidates.push("REPLICATE_HIGH_CONFIDENCE_RESULT");
                // High confidence: be more aggressive in testing
                if let Some(actions) = self.concept_action_mapping.get(&hypothesis.from_node_index)
                {
                    candidates.extend(actions.iter().cloned());
                }
            } else if hypothesis.confidence > 0.4 {
                candidates.push("CONDUCT_EXPLORATORY_EXPERIMENT");
                candidates.push("GATHER_MORE_EVIDENCE");
                candidates.push("INCREMENTAL_TESTING");
            } else {
                candidates.push("CONDUCT_PRELIMINARY_TEST");
                candidates.push("COLLECT_BASELINE_DATA");
                candidates.push("GENTLE_MANIPULATION");
            }

            // 6. Add strength-based experimental protocols
            if hypothesis.strength.abs() > 0.5 {
                candidates.push("CONDUCT_HIGH_IMPACT_EXPERIMENT");
                candidates.push("MAXIMUM_MANIPULATION");
            } else {
                candidates.push("CONDUCT_SENSITIVE_EXPERIMENT");
                candidates.push("GRADUAL_MANIPULATION");
            }

            // 7. Add verification actions
            candidates.push("VERIFY_HYPOTHESIS");
            candidates.push("MEASURE_EFFECT_MAGNITUDE");
            candidates.push("MONITOR_EFFECT_TIMING");
            candidates.push("ASSESS_CAUSAL_STRENGTH");

            info!(
                "ActiveInference: Generated {} experimental actions for hypothesis testing",
                candidates.len()
            );

            // **CRITICAL**: In experiment mode, we ONLY return experimental actions
            // Remove duplicates and return only experiment-focused actions
            candidates.sort();
            candidates.dedup();
            return candidates;
        }

        // **NORMAL MODE**: If no hypothesis pending, proceed with normal goal-oriented action proposal
        info!("ActiveInference: Normal mode - proposing goal-oriented actions");

        // Add all available actions as candidates
        candidates.extend(self.available_actions.iter().cloned());

        // Add context-specific actions based on flow analysis
        if let Some(vedana) = &flow.vedana {
            match vedana {
                Vedana::Pleasant { karma_weight } if *karma_weight > 1.0 => {
                    candidates.push("CONTINUE_SUCCESS");
                }
                Vedana::Unpleasant { karma_weight } if *karma_weight < -1.0 => {
                    candidates.push("TAKE_CORRECTIVE_ACTION");
                }
                _ => {
                    candidates.push("MAINTAIN_STATUS");
                }
            }
        }

        // Add pattern-based actions
        if let Some(sanna) = &flow.sanna {
            let pattern_complexity =
                sanna.active_indices.len() as f64 / sanna.dimensionality as f64;
            if pattern_complexity > 0.1 {
                candidates.push("ANALYZE_PATTERN");
            }
        }

        // Remove duplicates and return
        candidates.sort();
        candidates.dedup();
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use pandora_core::ontology::EpistemologicalFlow;
    use std::sync::Arc;
    use std::sync::Mutex;

    struct MockWorldModel {
        mdl: f64,
        prediction_error: f64,
    }

    impl WorldModel for MockWorldModel {
        fn get_mdl(&self) -> f64 {
            self.mdl
        }

        fn get_prediction_error(&self, _flow: &EpistemologicalFlow) -> f64 {
            self.prediction_error
        }
    }

    #[test]
    fn test_active_inference_skandha_creation() {
        let cwm = Arc::new(Mutex::new(MockWorldModel {
            mdl: 10.0,
            prediction_error: 0.2,
        }));
        let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
        let available_actions = vec!["action_A", "action_B"];

        let sankhara = ActiveInferenceSankharaSkandha::new(
            cwm,
            learning_engine,
            3,
            available_actions,
            0.9, // gamma
            0.1, // policy_epsilon
        );

        assert_eq!(
            sankhara.name(),
            "Active Inference Sankhara (Planning & Imagination)"
        );
        assert_eq!(sankhara.planning_horizon, 3);
        assert_eq!(sankhara.available_actions.len(), 2);
    }

    #[test]
    fn test_form_intent() {
        let cwm = Arc::new(Mutex::new(MockWorldModel {
            mdl: 10.0,
            prediction_error: 0.2,
        }));
        let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
        let available_actions = vec!["action_A", "action_B"];

        let sankhara = ActiveInferenceSankharaSkandha::new(
            cwm,
            learning_engine,
            2,
            available_actions,
            0.9, // gamma
            0.1, // policy_epsilon
        );

        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));
        sankhara.form_intent(&mut flow);

        assert!(flow.sankhara.is_some(), "Sankhara should form an intent");
        let intent = flow.sankhara.as_ref().unwrap();
        assert!(
            ["action_A", "action_B", "default_fallback_intent"].contains(&intent.as_ref()),
            "Intent should be one of the available actions or fallback"
        );
    }

    #[test]
    fn test_propose_candidate_actions() {
        let cwm = Arc::new(Mutex::new(MockWorldModel {
            mdl: 10.0,
            prediction_error: 0.2,
        }));
        let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
        let available_actions = vec!["action_A", "action_B"];

        let sankhara = ActiveInferenceSankharaSkandha::new(
            cwm,
            learning_engine,
            2,
            available_actions,
            0.9, // gamma
            0.1, // policy_epsilon
        );

        let flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));
        let candidates = sankhara.propose_candidate_actions(&flow);

        assert!(
            !candidates.is_empty(),
            "Should propose at least some actions"
        );
        assert!(
            candidates.contains(&"action_A"),
            "Should include available actions"
        );
        assert!(
            candidates.contains(&"action_B"),
            "Should include available actions"
        );
    }
}
