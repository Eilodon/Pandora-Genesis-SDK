//! Automatic Scientist Orchestrator
//!
//! This orchestrator implements the complete "Automatic Scientist" loop,
//! integrating MCG, CWM, and ActiveInferenceSankharaSkandha to enable
//! autonomous causal discovery and knowledge crystallization.

#[cfg(feature = "ml")]
use std::sync::{Arc, Mutex};
use std::time::Instant;
#[cfg(feature = "ml")]
use tracing::info;
#[cfg(feature = "ml")]
use pandora_core::ontology::EpistemologicalFlow;
#[cfg(feature = "ml")]
use pandora_core::world_model::DualIntrinsicReward;
#[cfg(feature = "ml")]
use pandora_core::interfaces::skandhas::SankharaSkandha;
#[cfg(feature = "ml")]
use pandora_cwm::model::InterdependentCausalModel;
#[cfg(feature = "ml")]
use pandora_learning_engine::{LearningEngine, ActiveInferenceSankharaSkandha};
#[cfg(feature = "ml")]
use pandora_learning_engine::active_inference_skandha::CausalHypothesis as LearningCausalHypothesis;
#[cfg(feature = "ml")]
use pandora_mcg::enhanced_mcg::{EnhancedMetaCognitiveGovernor, ActionTrigger};
#[cfg(feature = "ml")]
use pandora_error::PandoraError;

/// Enum representing the different states of the Automatic Scientist
#[derive(Debug, Clone, PartialEq)]
pub enum ScientistState {
    /// Observing the environment and monitoring for patterns
    Observing,
    /// Proposing a causal hypothesis based on observations
    Proposing { hypothesis: pandora_mcg::causal_discovery::CausalHypothesis },
    /// Conducting an experiment to test the hypothesis
    Experimenting { 
        hypothesis: pandora_mcg::causal_discovery::CausalHypothesis, 
        experiment_action: String, 
        start_time: Instant,
        steps_completed: usize,
    },
    /// Verifying the results of the experiment
    Verifying { 
        hypothesis: pandora_mcg::causal_discovery::CausalHypothesis, 
        experiment_results: Vec<ExperimentResult>,
    },
}

/// The Automatic Scientist Orchestrator coordinates the complete self-improvement loop.
///
/// This orchestrator manages the interaction between:
/// 1. Enhanced MCG for causal discovery
/// 2. CWM for world modeling and knowledge storage
/// 3. ActiveInferenceSankharaSkandha for experimental planning
/// 4. Learning Engine for reward calculation
#[cfg(feature = "ml")]
pub struct AutomaticScientistOrchestrator {
    /// The Causal World Model for storing and reasoning about causal relationships
    cwm: Arc<Mutex<InterdependentCausalModel>>,
    /// The Learning Engine for calculating rewards and learning
    learning_engine: Arc<LearningEngine>,
    /// The Active Inference Sankhara Skandha for planning experiments
    sankhara: Arc<Mutex<ActiveInferenceSankharaSkandha>>,
    /// The Enhanced Meta-Cognitive Governor for causal discovery
    mcg: Arc<Mutex<EnhancedMetaCognitiveGovernor>>,
    /// Current scientist state
    current_state: Arc<Mutex<ScientistState>>,
    /// Current experiment state (legacy, will be replaced by current_state)
    experiment_state: Arc<Mutex<ExperimentState>>,
}

/// Represents the current state of an ongoing experiment
#[derive(Debug, Clone)]
pub struct ExperimentState {
    /// Whether an experiment is currently active
    pub is_active: bool,
    /// The hypothesis being tested
    pub hypothesis: Option<pandora_mcg::causal_discovery::CausalHypothesis>,
    /// Number of experiment steps completed
    pub steps_completed: usize,
    /// Maximum number of steps for the experiment
    pub max_steps: usize,
    /// Results collected so far
    pub results: Vec<ExperimentResult>,
}

/// Represents the result of an experiment step
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentResult {
    pub step: usize,
    pub action_taken: String,
    pub observation: Vec<f32>,
    pub reward: f64,
    pub hypothesis_confirmed: bool,
}

#[cfg(feature = "ml")]
impl AutomaticScientistOrchestrator {
    /// Creates a new Automatic Scientist Orchestrator
    pub fn new(
        cwm: Arc<Mutex<InterdependentCausalModel>>,
        learning_engine: Arc<LearningEngine>,
        sankhara: Arc<Mutex<ActiveInferenceSankharaSkandha>>,
        mcg: Arc<Mutex<EnhancedMetaCognitiveGovernor>>,
    ) -> Self {
        Self {
            cwm,
            learning_engine,
            sankhara,
            mcg,
            current_state: Arc::new(Mutex::new(ScientistState::Observing)),
            experiment_state: Arc::new(Mutex::new(ExperimentState {
                is_active: false,
                hypothesis: None,
                steps_completed: 0,
                max_steps: 10,
                results: Vec::new(),
            })),
        }
    }

    /// Runs one cycle of the Automatic Scientist loop
    ///
    /// This method implements the complete self-improvement cycle using a state machine:
    /// 1. Observing: MCG monitors and discovers causal hypotheses
    /// 2. Proposing: If a hypothesis is found, transition to proposing state
    /// 3. Experimenting: Sankhara plans and executes experiments
    /// 4. Verifying: Results are analyzed and knowledge is crystallized
    pub async fn run_cycle(&self, current_flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        info!("=== Automatic Scientist Cycle Start ===");

        // Get current state
        let current_state = {
            let state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
            state.clone()
        };

        info!("Current Scientist State: {:?}", current_state);

        // State machine logic
        match current_state {
            ScientistState::Observing => {
                self.handle_observing_state(current_flow).await?;
            }
            ScientistState::Proposing { hypothesis } => {
                self.handle_proposing_state(hypothesis, current_flow).await?;
            }
            ScientistState::Experimenting { hypothesis, experiment_action, start_time, steps_completed } => {
                self.handle_experimenting_state(hypothesis, experiment_action, start_time, steps_completed, current_flow).await?;
            }
            ScientistState::Verifying { hypothesis, experiment_results } => {
                self.handle_verifying_state(hypothesis, experiment_results, current_flow).await?;
            }
        }

        info!("=== Automatic Scientist Cycle Complete ===");
        Ok(())
    }

    /// Handles the Observing state - MCG monitors and discovers causal hypotheses
    async fn handle_observing_state(&self, current_flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        info!("Scientist State: Observing - Monitoring for causal patterns");

        // MCG monitors and decides on meta-actions
        let action_trigger = {
            let mut mcg = self.mcg.lock().map_err(|_| PandoraError::config("Failed to acquire MCG lock"))?;
            let cwm = self.cwm.lock().map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;
            
            // Calculate current reward
            let reward = self.learning_engine.calculate_reward(&*cwm, &*cwm, current_flow);
            let dual_reward = DualIntrinsicReward {
                compression_reward: self.learning_engine.get_total_weighted_reward(&reward),
                prediction_reward: 0.5, // Placeholder
            };
            
            // Convert DualIntrinsicReward to SystemMetrics for MCG
            let metrics = pandora_mcg::enhanced_mcg::SystemMetrics {
                uncertainty: 0.5, // Derived from prediction uncertainty
                compression_reward: dual_reward.compression_reward,
                novelty_score: 0.3, // Placeholder - should be calculated
                performance: dual_reward.prediction_reward as f32,
                resource_usage: pandora_mcg::enhanced_mcg::ResourceMetrics {
                    cpu_usage: 0.0, // TODO: Collect from system
                    memory_usage: 0.0,
                    latency_ms: 0.0,
                },
            };
            mcg.monitor_comprehensive(&metrics).decision
        };

        // Handle the action trigger
        match action_trigger {
            ActionTrigger::TriggerSelfImprovementLevel1 { reason, target_component: _, confidence } => {
                info!("MCG triggered self-improvement Level 1: {} (confidence: {})", reason, confidence);
                // Create a hypothesis based on the trigger
                let hypothesis = pandora_mcg::causal_discovery::CausalHypothesis {
                    from_node_index: 0, // TODO: Map target_component to node index
                    to_node_index: 1,
                    strength: confidence,
                    confidence,
                    edge_type: pandora_mcg::causal_discovery::CausalEdgeType::Direct,
                };
                // Transition to Proposing state
                {
                    let mut state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
                    *state = ScientistState::Proposing { hypothesis };
                }
                info!("Transitioned to Proposing state with hypothesis");
            }
            ActionTrigger::NoAction => {
                info!("MCG: No action required, staying in Observing state");
            }
            _ => {
                info!("MCG: Other action triggered: {:?}", action_trigger);
            }
        }

        Ok(())
    }

    /// Handles the Proposing state - Set up experiment for the hypothesis
    async fn handle_proposing_state(&self, hypothesis: pandora_mcg::causal_discovery::CausalHypothesis, current_flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        info!("Scientist State: Proposing - Setting up experiment for hypothesis: {:?}", hypothesis);

        // Set the hypothesis in the Sankhara Skandha to enter experiment mode
        {
            let mut sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            // Convert MCG hypothesis to Learning hypothesis
            let learning_hypothesis = LearningCausalHypothesis {
                from_node_index: hypothesis.from_node_index,
                to_node_index: hypothesis.to_node_index,
                strength: hypothesis.strength,
                confidence: hypothesis.confidence,
                edge_type: match hypothesis.edge_type {
                    pandora_mcg::causal_discovery::CausalEdgeType::Direct => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Direct,
                    pandora_mcg::causal_discovery::CausalEdgeType::Indirect => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Indirect,
                    pandora_mcg::causal_discovery::CausalEdgeType::Conditional => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Conditional,
                    pandora_mcg::causal_discovery::CausalEdgeType::Inhibitory => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Inhibitory,
                },
            };
            sankhara.set_pending_hypothesis(Some(learning_hypothesis));
        }

        // Let the Sankhara Skandha form an intent (plan an experimental action)
        {
            let sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            sankhara.form_intent(current_flow);
        }

        // Get the planned action
        let experiment_action = current_flow.sankhara.as_ref()
            .map(|s| s.as_ref().to_string())
            .unwrap_or_else(|| "default_experimental_action".to_string());

        info!("Sankhara planned experimental action: {}", experiment_action);

        // Transition to Experimenting state
        {
            let mut state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
            *state = ScientistState::Experimenting {
                hypothesis: hypothesis.clone(),
                experiment_action,
                start_time: Instant::now(),
                steps_completed: 0,
            };
        }

        info!("Transitioned to Experimenting state");
        Ok(())
    }

    /// Handles the Experimenting state - Execute the experimental action
    async fn handle_experimenting_state(&self, hypothesis: pandora_mcg::causal_discovery::CausalHypothesis, experiment_action: String, start_time: Instant, mut steps_completed: usize, current_flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        info!("Scientist State: Experimenting - Executing action '{}' for hypothesis: {:?}", experiment_action, hypothesis);

        // Execute the experimental action (simplified)
        info!("Executing experimental action: {}", experiment_action);

        // Simulate the effect of the action
        let observation = self.simulate_action_effect(&experiment_action).await?;
        
        // Calculate reward
        let cwm = self.cwm.lock().map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;
        let reward = self.learning_engine.calculate_reward(&*cwm, &*cwm, current_flow);
        let reward_value = self.learning_engine.get_total_weighted_reward(&reward);

        // Check if hypothesis is confirmed
        let hypothesis_confirmed = self.check_hypothesis_confirmation(&observation, &hypothesis).await?;

        // Record the result
        let step = steps_completed;
        let experiment_result = ExperimentResult {
            step,
            action_taken: experiment_action.clone(),
            observation,
            reward: reward_value,
            hypothesis_confirmed,
        };

        steps_completed += 1;

        info!("Experiment step {} completed. Hypothesis confirmed: {}", 
              steps_completed, hypothesis_confirmed);

        // Check if experiment should continue or transition to verification
        let max_steps = 5; // Maximum experiment steps
        if steps_completed >= max_steps {
            info!("Experiment completed after {} steps, transitioning to verification", steps_completed);
            
            // Collect all experiment results (simplified - in real implementation, store them)
            let experiment_results = vec![experiment_result];
            
            // Transition to Verifying state
            {
                let mut state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
                *state = ScientistState::Verifying {
                    hypothesis: hypothesis.clone(),
                    experiment_results,
                };
            }
            info!("Transitioned to Verifying state");
        } else {
            // Continue experimenting - update state with new step count
            {
                let mut state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
                *state = ScientistState::Experimenting {
                    hypothesis: hypothesis.clone(),
                    experiment_action,
                    start_time,
                    steps_completed,
                };
            }
            info!("Continuing experiment - step {}/{}", steps_completed, max_steps);
        }

        Ok(())
    }

    /// Handles the Verifying state - Analyze results and crystallize knowledge if confirmed
    async fn handle_verifying_state(&self, hypothesis: pandora_mcg::causal_discovery::CausalHypothesis, experiment_results: Vec<ExperimentResult>, _current_flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        info!("Scientist State: Verifying - Analyzing experiment results for hypothesis: {:?}", hypothesis);

        // Analyze results to determine if hypothesis is confirmed
        let confirmed_steps = experiment_results.iter().filter(|r| r.hypothesis_confirmed).count();
        let confirmation_rate = confirmed_steps as f32 / experiment_results.len() as f32;
        
        info!("Experiment results: {}/{} steps confirmed hypothesis ({}%)", 
              confirmed_steps, experiment_results.len(), (confirmation_rate * 100.0) as u32);

        // If hypothesis is sufficiently confirmed, crystallize it
        if confirmation_rate > 0.6 { // 60% confirmation threshold
            info!("Hypothesis confirmed! Crystallizing knowledge...");
            
            // Convert MCG hypothesis to CWM hypothesis
            let cwm_hypothesis = pandora_cwm::model::CausalHypothesis {
                from_node_index: hypothesis.from_node_index,
                to_node_index: hypothesis.to_node_index,
                strength: hypothesis.strength,
                confidence: hypothesis.confidence,
                edge_type: match hypothesis.edge_type {
                    pandora_mcg::causal_discovery::CausalEdgeType::Direct => pandora_cwm::model::CausalEdgeType::Direct,
                    pandora_mcg::causal_discovery::CausalEdgeType::Indirect => pandora_cwm::model::CausalEdgeType::Indirect,
                    pandora_mcg::causal_discovery::CausalEdgeType::Conditional => pandora_cwm::model::CausalEdgeType::Conditional,
                    pandora_mcg::causal_discovery::CausalEdgeType::Inhibitory => pandora_cwm::model::CausalEdgeType::Inhibitory,
                },
            };

            // Crystallize the causal link in the CWM
            {
                let mut cwm = self.cwm.lock().map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;
                cwm.crystallize_causal_link(&cwm_hypothesis)?;
            }

            info!("Knowledge crystallized successfully!");
        } else {
            info!("Hypothesis not sufficiently confirmed. Discarding.");
        }

        // Clear hypothesis from Sankhara
        {
            let mut sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            sankhara.clear_pending_hypothesis();
        }

        // Transition back to Observing state
        {
            let mut state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
            *state = ScientistState::Observing;
        }

        info!("Transitioned back to Observing state");
        Ok(())
    }

    /// Starts a new experiment with the given hypothesis
    #[allow(dead_code)]
    async fn start_experiment(&self, hypothesis: pandora_mcg::causal_discovery::CausalHypothesis) -> Result<(), PandoraError> {
        info!("Starting experiment for hypothesis: {:?}", hypothesis);
        
        // Set experiment state
        {
            let mut state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
            state.is_active = true;
            state.hypothesis = Some(hypothesis.clone());
            state.steps_completed = 0;
            state.results.clear();
        }

        // Set the hypothesis in the Sankhara Skandha
        {
            let mut sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            // Convert MCG hypothesis to Learning hypothesis
            let learning_hypothesis = LearningCausalHypothesis {
                from_node_index: hypothesis.from_node_index,
                to_node_index: hypothesis.to_node_index,
                strength: hypothesis.strength,
                confidence: hypothesis.confidence,
                edge_type: match hypothesis.edge_type {
                    pandora_mcg::causal_discovery::CausalEdgeType::Direct => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Direct,
                    pandora_mcg::causal_discovery::CausalEdgeType::Indirect => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Indirect,
                    pandora_mcg::causal_discovery::CausalEdgeType::Conditional => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Conditional,
                    pandora_mcg::causal_discovery::CausalEdgeType::Inhibitory => pandora_learning_engine::active_inference_skandha::CausalEdgeType::Inhibitory,
                },
            };
            sankhara.set_pending_hypothesis(Some(learning_hypothesis));
        }

        info!("Experiment started successfully");
        Ok(())
    }

    /// Runs one step of the current experiment
    #[allow(dead_code)]
    async fn run_experiment_step(&self, flow: &mut EpistemologicalFlow) -> Result<(), PandoraError> {
        let mut state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
        
        if !state.is_active {
            return Ok(());
        }

        if state.steps_completed >= state.max_steps {
            info!("Experiment completed after {} steps", state.steps_completed);
            self.complete_experiment().await?;
            return Ok(());
        }

        // Let the Sankhara Skandha form an intent (plan an action)
        {
            let sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            sankhara.form_intent(flow);
        }

        // Execute the planned action (simplified)
        let action = flow.sankhara.as_ref().map(|s| s.as_ref()).unwrap_or("default_action");
        info!("Executing experimental action: {}", action);

        // Simulate the effect of the action
        let observation = self.simulate_action_effect(action).await?;
        
        // Calculate reward
        let cwm = self.cwm.lock().map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;
        let reward = self.learning_engine.calculate_reward(&*cwm, &*cwm, flow);
        let reward_value = self.learning_engine.get_total_weighted_reward(&reward);

        // Check if hypothesis is confirmed
        let hypothesis_confirmed = if let Some(ref hypothesis) = state.hypothesis {
            self.check_hypothesis_confirmation(&observation, hypothesis).await?
        } else {
            false
        };

        // Record the result
        let step = state.steps_completed;
        state.results.push(ExperimentResult {
            step,
            action_taken: action.to_string(),
            observation,
            reward: reward_value,
            hypothesis_confirmed,
        });

        state.steps_completed += 1;

        info!("Experiment step {} completed. Hypothesis confirmed: {}", 
              state.steps_completed, hypothesis_confirmed);

        Ok(())
    }

    /// Simulates the effect of an action (placeholder implementation)
    async fn simulate_action_effect(&self, action: &str) -> Result<Vec<f32>, PandoraError> {
        // In a real implementation, this would interact with the environment
        // For now, we'll simulate based on the action type
        let mut observation = vec![0.0; 64];
        
        match action {
            "MANIPULATE_CAUSE_VARIABLE" | "ACTIVATE_CAUSE_NODE" => {
                observation[0] = 1.0; // Simulate cause activation
            }
            "OBSERVE_EFFECT_VARIABLE" | "MEASURE_EFFECT_MAGNITUDE" => {
                observation[1] = 0.8; // Simulate effect observation
            }
            "CONDUCT_DEFINITIVE_EXPERIMENT" => {
                observation[0] = 1.0;
                observation[1] = 0.9; // High confidence result
            }
            _ => {
                observation[2] = 0.5; // Generic observation
            }
        }
        
        Ok(observation)
    }

    /// Checks if the current hypothesis is confirmed by the observations
    async fn check_hypothesis_confirmation(&self, observation: &[f32], hypothesis: &pandora_mcg::causal_discovery::CausalHypothesis) -> Result<bool, PandoraError> {
        // Simplified hypothesis confirmation logic
        // In a real implementation, this would be more sophisticated
        
        // Simple heuristic: if we see both cause and effect activation
        let cause_activated = observation.get(0).map(|&x| x > 0.5).unwrap_or(false);
        let effect_observed = observation.get(1).map(|&x| x > 0.5).unwrap_or(false);
        
        // For direct causality, both should be present
        if hypothesis.edge_type == pandora_mcg::causal_discovery::CausalEdgeType::Direct {
            return Ok(cause_activated && effect_observed);
        }
        
        // For indirect causality, we need to see some evidence of the causal chain
        if hypothesis.edge_type == pandora_mcg::causal_discovery::CausalEdgeType::Indirect {
            // Check for mediating variables (simplified)
            let mediator_observed = observation.get(2).map(|&x| x > 0.3).unwrap_or(false);
            return Ok((cause_activated || effect_observed) && mediator_observed);
        }
        
        // For conditional causality, we need to see the condition being met
        if hypothesis.edge_type == pandora_mcg::causal_discovery::CausalEdgeType::Conditional {
            let condition_met = observation.get(3).map(|&x| x > 0.5).unwrap_or(false);
            return Ok(cause_activated && condition_met && effect_observed);
        }
        
        // For inhibitory causality, we test by removing the inhibitor
        if hypothesis.edge_type == pandora_mcg::causal_discovery::CausalEdgeType::Inhibitory {
            // When inhibitor is removed, effect should be observed
            let inhibitor_removed = observation.get(0).map(|&x| x < 0.3).unwrap_or(false);
            return Ok(inhibitor_removed && effect_observed);
        }
        
        // Default: just check if we have some evidence
        Ok(cause_activated || effect_observed)
    }

    /// Completes the current experiment and crystallizes knowledge if confirmed
    #[allow(dead_code)]
    async fn complete_experiment(&self) -> Result<(), PandoraError> {
        info!("Completing experiment...");
        
        let (hypothesis, results) = {
            let state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
            (state.hypothesis.clone(), state.results.clone())
        };

        if let Some(hypothesis) = hypothesis {
            // Analyze results to determine if hypothesis is confirmed
            let confirmed_steps = results.iter().filter(|r| r.hypothesis_confirmed).count();
            let confirmation_rate = confirmed_steps as f32 / results.len() as f32;
            
            info!("Experiment results: {}/{} steps confirmed hypothesis ({}%)", 
                  confirmed_steps, results.len(), (confirmation_rate * 100.0) as u32);

            // If hypothesis is sufficiently confirmed, crystallize it
            if confirmation_rate > 0.6 { // 60% confirmation threshold
                info!("Hypothesis confirmed! Crystallizing knowledge...");
                
                // Convert MCG hypothesis to CWM hypothesis
                #[cfg(feature = "ml")]
                let cwm_hypothesis = pandora_cwm::model::CausalHypothesis {
                    from_node_index: hypothesis.from_node_index,
                    to_node_index: hypothesis.to_node_index,
                    strength: hypothesis.strength,
                    confidence: hypothesis.confidence,
                    edge_type: match hypothesis.edge_type {
                        pandora_mcg::causal_discovery::CausalEdgeType::Direct => {
                            #[cfg(feature = "ml")]
                            pandora_cwm::model::CausalEdgeType::Direct
                        },
                        pandora_mcg::causal_discovery::CausalEdgeType::Indirect => {
                            #[cfg(feature = "ml")]
                            pandora_cwm::model::CausalEdgeType::Indirect
                        },
                        pandora_mcg::causal_discovery::CausalEdgeType::Conditional => {
                            #[cfg(feature = "ml")]
                            pandora_cwm::model::CausalEdgeType::Conditional
                        },
                        pandora_mcg::causal_discovery::CausalEdgeType::Inhibitory => {
                            #[cfg(feature = "ml")]
                            pandora_cwm::model::CausalEdgeType::Inhibitory
                        },
                    },
                };

                // Crystallize the causal link in the CWM
                #[cfg(feature = "ml")]
                {
                    let mut cwm = self.cwm.lock().map_err(|_| PandoraError::config("Failed to acquire CWM lock"))?;
                    cwm.crystallize_causal_link(&cwm_hypothesis)?;
                }

                info!("Knowledge crystallized successfully!");
            } else {
                info!("Hypothesis not sufficiently confirmed. Discarding.");
            }
        }

        // Reset experiment state
        {
            let mut state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
            state.is_active = false;
            state.hypothesis = None;
            state.steps_completed = 0;
            state.results.clear();
        }

        // Clear hypothesis from Sankhara
        {
            let mut sankhara = self.sankhara.lock().map_err(|_| PandoraError::config("Failed to acquire Sankhara lock"))?;
            sankhara.clear_pending_hypothesis();
        }

        info!("Experiment completed and cleaned up");
        Ok(())
    }

    /// Checks if an experiment is currently active
    #[allow(dead_code)]
    async fn is_experiment_active(&self) -> Result<bool, PandoraError> {
        let state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
        Ok(state.is_active)
    }

    /// Gets the current scientist state
    pub fn get_current_state(&self) -> Result<ScientistState, PandoraError> {
        let state = self.current_state.lock().map_err(|_| PandoraError::config("Failed to acquire state lock"))?;
        Ok(state.clone())
    }

    /// Gets the current experiment state (legacy)
    pub fn get_experiment_state(&self) -> Result<ExperimentState, PandoraError> {
        let state = self.experiment_state.lock().map_err(|_| PandoraError::config("Failed to acquire experiment state lock"))?;
        Ok(state.clone())
    }
}
