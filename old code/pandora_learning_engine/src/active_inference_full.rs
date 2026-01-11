// Full Active Inference Implementation with dfdx
// EFE-based decision making với hierarchical planning

use async_trait::async_trait;
use dfdx::prelude::*;
use pandora_core::error::PandoraError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// ===== Enhanced Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intent {
    ExecuteTask { 
        task: String, 
        parameters: HashMap<String, String>,
        complexity: f32,
        priority: u8,
    },
    MaintainEquilibrium {
        current_state: HashMap<String, f32>,
        target_state: HashMap<String, f32>,
    },
    ExecutePlan { 
        plan: Vec<String>,
        estimated_duration: u64,
        resource_requirements: ResourceRequirements,
    },
    LearnSkill {
        skill_type: String,
        training_data: Vec<f32>,
        target_performance: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u8,
    pub memory_mb: u32,
    pub gpu_memory_mb: u32,
    pub network_bandwidth_mbps: u32,
    pub storage_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vedana {
    pub sensory_input: HashMap<String, f32>,
    pub emotional_state: EmotionalState,
    pub attention_weights: Vec<f32>,
    pub salience: f32,
    pub valence: f32,
    pub arousal: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub joy: f32,
    pub fear: f32,
    pub anger: f32,
    pub sadness: f32,
    pub surprise: f32,
    pub disgust: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sanna {
    pub perception: HashMap<String, f32>,
    pub confidence: f32,
    pub uncertainty: f32,
    pub attention_focus: Vec<String>,
    pub memory_traces: Vec<MemoryTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    pub content: String,
    pub strength: f32,
    pub recency: f32,
    pub emotional_valence: f32,
}

#[async_trait]
pub trait SankharaSkandha {
    async fn process(&self, vedana: &Vedana, sanna: &Sanna) -> Result<Intent, PandoraError>;
    fn get_confidence(&self) -> f32;
    fn get_complexity(&self) -> f32;
    fn get_resource_requirements(&self) -> ResourceRequirements;
}

// ===== EFE Calculator with dfdx =====

pub struct EFECalculator {
    device: Cpu,
    precision: f32,
    temperature: f32,
    kl_weight: f32,
    entropy_weight: f32,
    info_gain_weight: f32,
}

impl EFECalculator {
    pub fn new(precision: f32, temperature: f32) -> Self {
        Self {
            device: Cpu::default(),
            precision,
            temperature,
            kl_weight: 1.0,
            entropy_weight: 0.5,
            info_gain_weight: 0.3,
        }
    }

    /// Calculate Expected Free Energy for a given policy
    pub fn calculate_efe(
        &self,
        prior_beliefs: &Tensor<f32, (usize, usize)>,
        posterior_beliefs: &Tensor<f32, (usize, usize)>,
        observations: &Tensor<f32, (usize, usize)>,
        rewards: &Tensor<f32, (usize,)>,
    ) -> Result<f32, PandoraError> {
        // Pragmatic risk (KL divergence between prior and posterior)
        let kl_divergence = self.kl_divergence(prior_beliefs, posterior_beliefs)?;
        
        // Epistemic value (entropy + information gain)
        let entropy = self.entropy(posterior_beliefs)?;
        let info_gain = self.information_gain(prior_beliefs, posterior_beliefs, observations)?;
        
        // Expected reward
        let expected_reward = self.expected_reward(posterior_beliefs, rewards)?;
        
        // EFE = Pragmatic risk - Epistemic value + Expected reward
        let efe = self.kl_weight * kl_divergence 
                - self.entropy_weight * entropy 
                - self.info_gain_weight * info_gain
                + expected_reward;
        
        Ok(efe)
    }

    fn kl_divergence(
        &self,
        prior: &Tensor<f32, (usize, usize)>,
        posterior: &Tensor<f32, (usize, usize)>,
    ) -> Result<f32, PandoraError> {
        // KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        let log_ratio = posterior.log()? - prior.log()?;
        let kl = (posterior * log_ratio)?.sum()?;
        Ok(kl)
    }

    fn entropy(&self, distribution: &Tensor<f32, (usize, usize)>) -> Result<f32, PandoraError> {
        // H(X) = -Σ P(x) * log(P(x))
        let log_prob = distribution.log()?;
        let entropy = -(distribution * log_prob)?.sum()?;
        Ok(entropy)
    }

    fn information_gain(
        &self,
        prior: &Tensor<f32, (usize, usize)>,
        posterior: &Tensor<f32, (usize, usize)>,
        observations: &Tensor<f32, (usize, usize)>,
    ) -> Result<f32, PandoraError> {
        // Information gain = H(prior) - H(posterior|observations)
        let prior_entropy = self.entropy(prior)?;
        let posterior_entropy = self.entropy(posterior)?;
        
        // Mutual information between observations and beliefs
        let mutual_info = self.mutual_information(posterior, observations)?;
        
        Ok(prior_entropy - posterior_entropy + mutual_info)
    }

    fn mutual_information(
        &self,
        beliefs: &Tensor<f32, (usize, usize)>,
        observations: &Tensor<f32, (usize, usize)>,
    ) -> Result<f32, PandoraError> {
        // I(X;Y) = H(X) + H(Y) - H(X,Y)
        let h_x = self.entropy(beliefs)?;
        let h_y = self.entropy(observations)?;
        
        // Joint entropy (simplified)
        let joint = beliefs.matmul(observations)?;
        let h_xy = self.entropy(&joint)?;
        
        Ok(h_x + h_y - h_xy)
    }

    fn expected_reward(
        &self,
        beliefs: &Tensor<f32, (usize, usize)>,
        rewards: &Tensor<f32, (usize,)>,
    ) -> Result<f32, PandoraError> {
        // Expected reward = Σ P(a) * R(a)
        let belief_sum = beliefs.sum_axis::<1>()?;
        let expected = (belief_sum * rewards)?.sum()?;
        Ok(expected)
    }

    /// Calculate EFE for multiple policies
    pub fn calculate_policy_efe(
        &self,
        policies: &[Policy],
        world_model: &HierarchicalWorldModel,
        current_state: &WorldState,
    ) -> Result<Vec<(usize, f32)>, PandoraError> {
        let mut policy_efe = Vec::new();
        
        for (i, policy) in policies.iter().enumerate() {
            let efe = self.evaluate_policy(policy, world_model, current_state)?;
            policy_efe.push((i, efe));
        }
        
        // Sort by EFE (lower is better)
        policy_efe.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(policy_efe)
    }

    fn evaluate_policy(
        &self,
        policy: &Policy,
        world_model: &HierarchicalWorldModel,
        current_state: &WorldState,
    ) -> Result<f32, PandoraError> {
        // Simulate policy execution
        let (predicted_states, predicted_observations, predicted_rewards) = 
            world_model.simulate_policy(policy, current_state)?;
        
        // Calculate EFE for each time step
        let mut total_efe = 0.0;
        let time_steps = predicted_states.len();
        
        for t in 0..time_steps {
            let prior = if t == 0 {
                world_model.get_prior_beliefs(current_state)?
            } else {
                predicted_states[t-1].clone()
            };
            
            let posterior = predicted_states[t].clone();
            let observations = predicted_observations[t].clone();
            let rewards = predicted_rewards[t].clone();
            
            let efe = self.calculate_efe(&prior, &posterior, &observations, &rewards)?;
            total_efe += efe;
        }
        
        Ok(total_efe / time_steps as f32)
    }
}

// ===== Hierarchical World Model =====

pub struct HierarchicalWorldModel {
    device: Cpu,
    levels: Vec<WorldModelLevel>,
    transition_matrices: HashMap<String, Tensor<f32, (usize, usize)>>,
    observation_matrices: HashMap<String, Tensor<f32, (usize, usize)>>,
    reward_functions: HashMap<String, RewardFunction>,
}

#[derive(Debug, Clone)]
pub struct WorldModelLevel {
    pub name: String,
    pub state_dim: usize,
    pub action_dim: usize,
    pub observation_dim: usize,
    pub abstraction_level: f32,
    pub temporal_scale: u32,
}

#[derive(Debug, Clone)]
pub struct WorldState {
    pub level: String,
    pub state_vector: Vec<f32>,
    pub timestamp: u64,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct Policy {
    pub name: String,
    pub actions: Vec<Action>,
    pub probability: f32,
    pub expected_duration: u64,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub parameters: HashMap<String, f32>,
    pub probability: f32,
    pub expected_outcome: String,
}

#[derive(Debug, Clone)]
pub struct RewardFunction {
    pub name: String,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub nonlinear_terms: Vec<NonlinearTerm>,
}

#[derive(Debug, Clone)]
pub struct NonlinearTerm {
    pub feature_indices: Vec<usize>,
    pub weight: f32,
    pub degree: u32,
}

impl HierarchicalWorldModel {
    pub fn new(levels: Vec<WorldModelLevel>) -> Self {
        Self {
            device: Cpu::default(),
            levels,
            transition_matrices: HashMap::new(),
            observation_matrices: HashMap::new(),
            reward_functions: HashMap::new(),
        }
    }

    pub fn add_transition_matrix(&mut self, level: &str, matrix: Tensor<f32, (usize, usize)>) {
        self.transition_matrices.insert(level.to_string(), matrix);
    }

    pub fn add_observation_matrix(&mut self, level: &str, matrix: Tensor<f32, (usize, usize)>) {
        self.observation_matrices.insert(level.to_string(), matrix);
    }

    pub fn add_reward_function(&mut self, level: &str, function: RewardFunction) {
        self.reward_functions.insert(level.to_string(), function);
    }

    pub fn simulate_policy(
        &self,
        policy: &Policy,
        current_state: &WorldState,
    ) -> Result<(Vec<Tensor<f32, (usize, usize)>>, Vec<Tensor<f32, (usize, usize)>>, Vec<Tensor<f32, (usize,)>>), PandoraError> {
        let mut predicted_states = Vec::new();
        let mut predicted_observations = Vec::new();
        let mut predicted_rewards = Vec::new();
        
        let mut current = current_state.clone();
        
        for action in &policy.actions {
            // Predict next state
            let next_state = self.predict_next_state(&current, action)?;
            predicted_states.push(next_state.clone());
            
            // Predict observations
            let observations = self.predict_observations(&next_state)?;
            predicted_observations.push(observations);
            
            // Predict rewards
            let reward = self.predict_reward(&next_state, action)?;
            predicted_rewards.push(reward);
            
            current = next_state;
        }
        
        Ok((predicted_states, predicted_observations, predicted_rewards))
    }

    fn predict_next_state(
        &self,
        current_state: &WorldState,
        action: &Action,
    ) -> Result<WorldState, PandoraError> {
        let transition_matrix = self.transition_matrices.get(&current_state.level)
            .ok_or_else(|| PandoraError::PredictionFailed("Transition matrix not found".to_string()))?;
        
        let state_tensor = Tensor::new(&current_state.state_vector, &self.device)?
            .reshape((1, current_state.state_vector.len()))?;
        
        let next_state_tensor = state_tensor.matmul(transition_matrix)?;
        let next_state_vector: Vec<f32> = next_state_tensor.into_iter().collect();
        
        Ok(WorldState {
            level: current_state.level.clone(),
            state_vector: next_state_vector,
            timestamp: current_state.timestamp + 1,
            confidence: current_state.confidence * 0.95, // Decay confidence
        })
    }

    fn predict_observations(&self, state: &WorldState) -> Result<Tensor<f32, (usize, usize)>, PandoraError> {
        let observation_matrix = self.observation_matrices.get(&state.level)
            .ok_or_else(|| PandoraError::PredictionFailed("Observation matrix not found".to_string()))?;
        
        let state_tensor = Tensor::new(&state.state_vector, &self.device)?
            .reshape((1, state.state_vector.len()))?;
        
        state_tensor.matmul(observation_matrix)
    }

    fn predict_reward(&self, state: &WorldState, action: &Action) -> Result<Tensor<f32, (usize,)>, PandoraError> {
        let reward_function = self.reward_functions.get(&state.level)
            .ok_or_else(|| PandoraError::PredictionFailed("Reward function not found".to_string()))?;
        
        // Calculate reward based on state and action
        let mut reward = reward_function.bias;
        
        for (i, &weight) in reward_function.weights.iter().enumerate() {
            if i < state.state_vector.len() {
                reward += weight * state.state_vector[i];
            }
        }
        
        // Add nonlinear terms
        for term in &reward_function.nonlinear_terms {
            let mut product = 1.0;
            for &idx in &term.feature_indices {
                if idx < state.state_vector.len() {
                    product *= state.state_vector[idx].powi(term.degree as i32);
                }
            }
            reward += term.weight * product;
        }
        
        Ok(Tensor::new(&[reward], &self.device)?)
    }

    pub fn get_prior_beliefs(&self, state: &WorldState) -> Result<Tensor<f32, (usize, usize)>, PandoraError> {
        // Convert state to belief distribution
        let state_tensor = Tensor::new(&state.state_vector, &self.device)?
            .reshape((1, state.state_vector.len()))?;
        
        // Apply softmax to create probability distribution
        let beliefs = state_tensor.softmax()?;
        Ok(beliefs)
    }

    /// Compress world model to reduce computational complexity
    pub fn compress_model(&mut self, compression_ratio: f32) -> Result<(), PandoraError> {
        for level in &mut self.levels {
            level.state_dim = (level.state_dim as f32 * compression_ratio) as usize;
            level.action_dim = (level.action_dim as f32 * compression_ratio) as usize;
            level.observation_dim = (level.observation_dim as f32 * compression_ratio) as usize;
        }
        
        // Compress transition matrices
        for (level_name, matrix) in &mut self.transition_matrices {
            let compressed = self.compress_matrix(matrix, compression_ratio)?;
            *matrix = compressed;
        }
        
        // Compress observation matrices
        for (level_name, matrix) in &mut self.observation_matrices {
            let compressed = self.compress_matrix(matrix, compression_ratio)?;
            *matrix = compressed;
        }
        
        Ok(())
    }

    fn compress_matrix(
        &self,
        matrix: &Tensor<f32, (usize, usize)>,
        ratio: f32,
    ) -> Result<Tensor<f32, (usize, usize)>, PandoraError> {
        let (rows, cols) = matrix.shape();
        let new_rows = (rows as f32 * ratio) as usize;
        let new_cols = (cols as f32 * ratio) as usize;
        
        // Simple compression by taking top-left submatrix
        let compressed = matrix.slice((0..new_rows, 0..new_cols))?;
        Ok(compressed)
    }
}

// ===== Active Inference Sankhara =====

pub struct ActiveInferenceSankhara {
    efe_calculator: EFECalculator,
    world_model: HierarchicalWorldModel,
    performance_metrics: PerformanceMetrics,
    decision_threshold: f32,
    exploration_rate: f32,
    learning_rate: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub decision_accuracy: f32,
    pub exploration_efficiency: f32,
    pub computational_cost: f32,
    pub memory_usage: f32,
    pub energy_efficiency: f32,
}

impl ActiveInferenceSankhara {
    pub fn new(
        world_model: HierarchicalWorldModel,
        decision_threshold: f32,
        exploration_rate: f32,
        learning_rate: f32,
    ) -> Self {
        Self {
            efe_calculator: EFECalculator::new(1e-6, 1.0),
            world_model,
            performance_metrics: PerformanceMetrics::default(),
            decision_threshold,
            exploration_rate,
            learning_rate,
        }
    }

    /// Process vedana and sanna to generate intent
    pub async fn process_cognitive_input(
        &mut self,
        vedana: &Vedana,
        sanna: &Sanna,
    ) -> Result<Intent, PandoraError> {
        // Generate candidate policies
        let policies = self.generate_policies(vedana, sanna).await?;
        
        // Calculate EFE for each policy
        let current_state = self.vedana_sanna_to_world_state(vedana, sanna)?;
        let policy_efe = self.efe_calculator.calculate_policy_efe(&policies, &self.world_model, &current_state)?;
        
        // Select best policy
        let best_policy = self.select_policy(&policies, &policy_efe)?;
        
        // Convert policy to intent
        let intent = self.policy_to_intent(&best_policy, vedana, sanna)?;
        
        // Update performance metrics
        self.update_performance_metrics(&best_policy, &policy_efe);
        
        Ok(intent)
    }

    async fn generate_policies(
        &self,
        vedana: &Vedana,
        sanna: &Sanna,
    ) -> Result<Vec<Policy>, PandoraError> {
        let mut policies = Vec::new();
        
        // Policy 1: Maintain equilibrium
        if self.should_maintain_equilibrium(vedana, sanna) {
            policies.push(Policy {
                name: "maintain_equilibrium".to_string(),
                actions: vec![Action {
                    name: "balance_state".to_string(),
                    parameters: self.calculate_balance_parameters(vedana, sanna),
                    probability: 0.8,
                    expected_outcome: "stable_state".to_string(),
                }],
                probability: 0.7,
                expected_duration: 1000,
                resource_requirements: ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 50,
                    gpu_memory_mb: 0,
                    network_bandwidth_mbps: 10,
                    storage_mb: 20,
                },
            });
        }
        
        // Policy 2: Execute task
        if let Some(task) = self.extract_task_from_sanna(sanna) {
            policies.push(Policy {
                name: "execute_task".to_string(),
                actions: vec![Action {
                    name: "execute".to_string(),
                    parameters: self.calculate_task_parameters(&task, vedana, sanna),
                    probability: 0.9,
                    expected_outcome: "task_completed".to_string(),
                }],
                probability: 0.8,
                expected_duration: 5000,
                resource_requirements: ResourceRequirements {
                    cpu_cores: 2,
                    memory_mb: 200,
                    gpu_memory_mb: 0,
                    network_bandwidth_mbps: 50,
                    storage_mb: 100,
                },
            });
        }
        
        // Policy 3: Learn new skill
        if self.should_learn_skill(vedana, sanna) {
            policies.push(Policy {
                name: "learn_skill".to_string(),
                actions: vec![Action {
                    name: "learn".to_string(),
                    parameters: self.calculate_learning_parameters(vedana, sanna),
                    probability: 0.6,
                    expected_outcome: "skill_learned".to_string(),
                }],
                probability: 0.5,
                expected_duration: 10000,
                resource_requirements: ResourceRequirements {
                    cpu_cores: 4,
                    memory_mb: 500,
                    gpu_memory_mb: 1024,
                    network_bandwidth_mbps: 100,
                    storage_mb: 1000,
                },
            });
        }
        
        // Policy 4: Explore (if exploration rate allows)
        if self.should_explore() {
            policies.push(Policy {
                name: "explore".to_string(),
                actions: vec![Action {
                    name: "random_action".to_string(),
                    parameters: self.generate_random_parameters(),
                    probability: 0.3,
                    expected_outcome: "new_information".to_string(),
                }],
                probability: self.exploration_rate,
                expected_duration: 2000,
                resource_requirements: ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 100,
                    gpu_memory_mb: 0,
                    network_bandwidth_mbps: 25,
                    storage_mb: 50,
                },
            });
        }
        
        Ok(policies)
    }

    fn select_policy(
        &self,
        policies: &[Policy],
        policy_efe: &[(usize, f32)],
    ) -> Result<&Policy, PandoraError> {
        if policy_efe.is_empty() {
            return Err(PandoraError::PredictionFailed("No policies available".to_string()));
        }
        
        // Select policy with lowest EFE (best expected outcome)
        let (best_idx, _best_efe) = policy_efe[0];
        policies.get(best_idx)
            .ok_or_else(|| PandoraError::PredictionFailed("Policy index out of bounds".to_string()))
    }

    fn policy_to_intent(
        &self,
        policy: &Policy,
        vedana: &Vedana,
        sanna: &Sanna,
    ) -> Result<Intent, PandoraError> {
        match policy.name.as_str() {
            "maintain_equilibrium" => {
                Ok(Intent::MaintainEquilibrium {
                    current_state: vedana.sensory_input.clone(),
                    target_state: self.calculate_target_state(vedana, sanna),
                })
            }
            "execute_task" => {
                let task = self.extract_task_from_sanna(sanna)
                    .unwrap_or_else(|| "unknown_task".to_string());
                Ok(Intent::ExecuteTask {
                    task,
                    parameters: HashMap::new(),
                    complexity: self.calculate_task_complexity(vedana, sanna),
                    priority: self.calculate_task_priority(vedana, sanna),
                })
            }
            "learn_skill" => {
                Ok(Intent::LearnSkill {
                    skill_type: "adaptive_skill".to_string(),
                    training_data: self.extract_training_data(sanna),
                    target_performance: 0.8,
                })
            }
            "explore" => {
                Ok(Intent::ExecuteTask {
                    task: "explore_environment".to_string(),
                    parameters: HashMap::new(),
                    complexity: 0.5,
                    priority: 1,
                })
            }
            _ => Err(PandoraError::PredictionFailed("Unknown policy type".to_string())),
        }
    }

    fn vedana_sanna_to_world_state(
        &self,
        vedana: &Vedana,
        sanna: &Sanna,
    ) -> Result<WorldState, PandoraError> {
        let mut state_vector = Vec::new();
        
        // Add sensory input
        for value in vedana.sensory_input.values() {
            state_vector.push(*value);
        }
        
        // Add emotional state
        state_vector.push(vedana.emotional_state.joy);
        state_vector.push(vedana.emotional_state.fear);
        state_vector.push(vedana.emotional_state.anger);
        state_vector.push(vedana.emotional_state.sadness);
        state_vector.push(vedana.emotional_state.surprise);
        state_vector.push(vedana.emotional_state.disgust);
        
        // Add attention weights
        state_vector.extend_from_slice(&vedana.attention_weights);
        
        // Add perception
        for value in sanna.perception.values() {
            state_vector.push(*value);
        }
        
        // Add confidence and uncertainty
        state_vector.push(sanna.confidence);
        state_vector.push(sanna.uncertainty);
        
        Ok(WorldState {
            level: "sensory".to_string(),
            state_vector,
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            confidence: sanna.confidence,
        })
    }

    // Helper methods
    fn should_maintain_equilibrium(&self, vedana: &Vedana, sanna: &Sanna) -> bool {
        vedana.valence.abs() > 0.5 || sanna.uncertainty > 0.7
    }

    fn should_learn_skill(&self, vedana: &Vedana, sanna: &Sanna) -> bool {
        sanna.uncertainty > 0.8 && vedana.salience > 0.6
    }

    fn should_explore(&self) -> bool {
        rand::random::<f32>() < self.exploration_rate
    }

    fn calculate_balance_parameters(&self, vedana: &Vedana, sanna: &Sanna) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("target_valence".to_string(), 0.0);
        params.insert("target_arousal".to_string(), 0.5);
        params.insert("confidence_threshold".to_string(), 0.8);
        params
    }

    fn calculate_task_parameters(&self, task: &str, vedana: &Vedana, sanna: &Sanna) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("task_name".to_string(), task.len() as f32);
        params.insert("urgency".to_string(), vedana.salience);
        params.insert("confidence".to_string(), sanna.confidence);
        params
    }

    fn calculate_learning_parameters(&self, vedana: &Vedana, sanna: &Sanna) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("learning_rate".to_string(), self.learning_rate);
        params.insert("attention_weight".to_string(), vedana.salience);
        params.insert("uncertainty_threshold".to_string(), sanna.uncertainty);
        params
    }

    fn generate_random_parameters(&self) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert("random_factor".to_string(), rand::random::<f32>());
        params.insert("exploration_strength".to_string(), self.exploration_rate);
        params
    }

    fn extract_task_from_sanna(&self, sanna: &Sanna) -> Option<String> {
        // Simplified task extraction
        if sanna.confidence > 0.7 {
            Some("high_confidence_task".to_string())
        } else {
            None
        }
    }

    fn calculate_target_state(&self, vedana: &Vedana, sanna: &Sanna) -> HashMap<String, f32> {
        let mut target = HashMap::new();
        for (key, value) in &vedana.sensory_input {
            target.insert(key.clone(), value * 0.8); // Slightly reduce intensity
        }
        target
    }

    fn calculate_task_complexity(&self, vedana: &Vedana, sanna: &Sanna) -> f32 {
        sanna.uncertainty + vedana.salience
    }

    fn calculate_task_priority(&self, vedana: &Vedana, sanna: &Sanna) -> u8 {
        ((vedana.salience * 10.0) as u8).min(10)
    }

    fn extract_training_data(&self, sanna: &Sanna) -> Vec<f32> {
        let mut data = Vec::new();
        for value in sanna.perception.values() {
            data.push(*value);
        }
        data
    }

    fn update_performance_metrics(&mut self, policy: &Policy, policy_efe: &[(usize, f32)]) {
        // Update decision accuracy based on EFE values
        if let Some((_, best_efe)) = policy_efe.first() {
            self.performance_metrics.decision_accuracy = 1.0 / (1.0 + best_efe.abs());
        }
        
        // Update computational cost
        self.performance_metrics.computational_cost = policy.actions.len() as f32 * 0.1;
        
        // Update memory usage
        self.performance_metrics.memory_usage = policy.resource_requirements.memory_mb as f32;
    }

    /// Compress world model for edge devices
    pub fn compress_for_edge(&mut self, compression_ratio: f32) -> Result<(), PandoraError> {
        self.world_model.compress_model(compression_ratio)?;
        
        // Adjust exploration rate for edge devices
        self.exploration_rate *= compression_ratio;
        
        // Adjust learning rate
        self.learning_rate *= compression_ratio;
        
        Ok(())
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Update world model based on new observations
    pub fn update_world_model(&mut self, observation: &Observation) -> Result<(), PandoraError> {
        // Simplified world model update
        // In practice, this would use more sophisticated learning algorithms
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub sensory_data: HashMap<String, f32>,
    pub reward: f32,
    pub timestamp: u64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            decision_accuracy: 0.0,
            exploration_efficiency: 0.0,
            computational_cost: 0.0,
            memory_usage: 0.0,
            energy_efficiency: 0.0,
        }
    }
}

// ===== Error Extensions =====

impl From<dfdx::tensor::TensorError> for PandoraError {
    fn from(err: dfdx::tensor::TensorError) -> Self {
        PandoraError::PredictionFailed(format!("Tensor error: {}", err))
    }
}
