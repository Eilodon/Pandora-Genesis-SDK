use std::collections::HashMap;
use ndarray::{Array2, Array1};
use pandora_core::error::PandoraError;
use serde::{Deserialize, Serialize};

/// Lightweight Active Inference implementation using ndarray instead of dfdx
/// This provides the same functionality but with simpler tensor operations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    pub id: String,
    pub description: String,
    pub priority: u8,
    pub context: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vedana {
    pub emotional_state: EmotionalState,
    pub satisfaction_level: f32,
    pub stress_level: f32,
    pub energy_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sanna {
    pub perception_quality: f32,
    pub attention_focus: f32,
    pub memory_accessibility: f32,
    pub cognitive_load: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalState {
    pub valence: f32,    // -1.0 to 1.0
    pub arousal: f32,    // 0.0 to 1.0
    pub dominance: f32,  // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub state_vector: Array1<f32>,
    pub timestamp: u64,
    pub context: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    pub id: String,
    pub actions: Vec<Action>,
    pub expected_reward: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Action {
    pub id: String,
    pub description: String,
    pub parameters: HashMap<String, f32>,
}

impl Action {
    /// Create a new Action with custom parameters
    pub fn new(id: String, description: String) -> Self {
        Self {
            id,
            description,
            parameters: HashMap::new(),
        }
    }

    /// Factory method for Noop action (compatibility with enum-style usage)
    pub fn noop() -> Self {
        Self {
            id: "noop".to_string(),
            description: "No operation".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Factory method for Explore action
    pub fn explore() -> Self {
        Self {
            id: "explore".to_string(),
            description: "Explore environment".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Factory method for Exploit action
    pub fn exploit() -> Self {
        Self {
            id: "exploit".to_string(),
            description: "Exploit known strategy".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Factory method for UnlockDoor action
    pub fn unlock_door() -> Self {
        Self {
            id: "unlock_door".to_string(),
            description: "Unlock door".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Factory method for PickUpKey action
    pub fn pick_up_key() -> Self {
        Self {
            id: "pick_up_key".to_string(),
            description: "Pick up key".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Factory method for MoveForward action
    pub fn move_forward() -> Self {
        Self {
            id: "move_forward".to_string(),
            description: "Move forward".to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Get action ID as string (compatibility with old as_str() method)
    pub fn as_str(&self) -> &str {
        &self.id
    }

    /// Create Action from string ID
    pub fn from_str(id: &str) -> Self {
        match id {
            "noop" => Self::noop(),
            "explore" => Self::explore(),
            "exploit" => Self::exploit(),
            "unlock_door" => Self::unlock_door(),
            "pick_up_key" => Self::pick_up_key(),
            "move_forward" => Self::move_forward(),
            _ => Self::noop(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardFunction {
    pub weights: Array1<f32>,
    pub bias: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelLevel {
    pub name: String,
    pub state_size: usize,
    pub action_size: usize,
    pub transition_matrix: Array2<f32>,
    pub observation_matrix: Array2<f32>,
    pub reward_vector: Array1<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    pub state: WorldState,
    pub action: Action,
    pub reward: f32,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub data: Array1<f32>,
    pub timestamp: u64,
    pub source: String,
}

/// Expected Free Energy Calculator using ndarray
pub struct EFECalculator {
    pub precision_weight: f32,
    pub complexity_weight: f32,
}

impl EFECalculator {
    pub fn new(precision_weight: f32, complexity_weight: f32) -> Self {
        Self {
            precision_weight,
            complexity_weight,
        }
    }

    /// Calculate Expected Free Energy for a policy
    pub fn calculate_efe(
        &self,
        prior_beliefs: &Array2<f32>,
        posterior_beliefs: &Array2<f32>,
        observations: &Array2<f32>,
        rewards: &Array1<f32>,
    ) -> Result<f32, PandoraError> {
        // Pragmatic value (expected reward)
        let pragmatic_value = self.calculate_pragmatic_value(posterior_beliefs, rewards)?;
        
        // Epistemic value (information gain)
        let epistemic_value = self.calculate_epistemic_value(prior_beliefs, posterior_beliefs, observations)?;
        
        // EFE = -pragmatic_value - epistemic_value
        Ok(-pragmatic_value - epistemic_value)
    }

    fn calculate_pragmatic_value(&self, beliefs: &Array2<f32>, rewards: &Array1<f32>) -> Result<f32, PandoraError> {
        // Expected reward based on belief distribution
        let expected_reward = beliefs.dot(rewards).sum();
        Ok(expected_reward)
    }

    fn calculate_epistemic_value(
        &self,
        prior: &Array2<f32>,
        posterior: &Array2<f32>,
        observations: &Array2<f32>,
    ) -> Result<f32, PandoraError> {
        // KL divergence between prior and posterior
        let kl_divergence = self.kl_divergence(prior, posterior)?;
        
        // Entropy of observations
        let entropy = self.entropy(observations)?;
        
        // Information gain
        let information_gain = kl_divergence + entropy;
        
        Ok(self.precision_weight * information_gain)
    }

    fn kl_divergence(&self, prior: &Array2<f32>, posterior: &Array2<f32>) -> Result<f32, PandoraError> {
        // Ensure arrays have the same shape
        if prior.shape() != posterior.shape() {
            return Err(PandoraError::PredictionFailed("Array shape mismatch".to_string()));
        }

        let mut kl_sum = 0.0;
        for (p, q) in prior.iter().zip(posterior.iter()) {
            if *q > 0.0 && *p > 0.0 {
                kl_sum += q * (q / p).ln();
            }
        }
        Ok(kl_sum)
    }

    fn entropy(&self, distribution: &Array2<f32>) -> Result<f32, PandoraError> {
        let mut entropy = 0.0;
        for &prob in distribution.iter() {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        Ok(entropy)
    }

    #[allow(dead_code)]
    fn information_gain(
        &self,
        prior: &Array2<f32>,
        posterior: &Array2<f32>,
        observations: &Array2<f32>,
    ) -> Result<f32, PandoraError> {
        // Mutual information between beliefs and observations
        let kl_div = self.kl_divergence(prior, posterior)?;
        let entropy_obs = self.entropy(observations)?;
        Ok(kl_div + entropy_obs)
    }

    #[allow(dead_code)]
    fn pragmatic_risk(&self, beliefs: &Array2<f32>, rewards: &Array1<f32>) -> Result<f32, PandoraError> {
        // Variance of expected rewards
        let expected_rewards = beliefs.dot(rewards);
        let mean_reward = expected_rewards.mean().unwrap_or(0.0);
        let variance = expected_rewards.mapv(|x| (x - mean_reward).powi(2)).mean().unwrap_or(0.0);
        Ok(variance.sqrt())
    }
}

/// Hierarchical World Model using ndarray
pub struct HierarchicalWorldModel {
    pub levels: HashMap<String, WorldModelLevel>,
    pub compression_ratio: f32,
}

impl HierarchicalWorldModel {
    pub fn new(compression_ratio: f32) -> Self {
        Self {
            levels: HashMap::new(),
            compression_ratio,
        }
    }

    pub fn add_level(&mut self, level: WorldModelLevel) {
        self.levels.insert(level.name.clone(), level);
    }

    pub fn add_transition_matrix(&mut self, level: &str, matrix: Array2<f32>) {
        if let Some(level_data) = self.levels.get_mut(level) {
            level_data.transition_matrix = matrix;
        }
    }

    pub fn add_observation_matrix(&mut self, level: &str, matrix: Array2<f32>) {
        if let Some(level_data) = self.levels.get_mut(level) {
            level_data.observation_matrix = matrix;
        }
    }

    /// Compress world model for edge deployment
    pub fn compress_for_edge(
        &mut self,
        target_compression: f32,
    ) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array1<f32>>), PandoraError> {
        let mut compressed_transitions = Vec::new();
        let mut compressed_observations = Vec::new();
        let mut compressed_rewards = Vec::new();

        for (_level_name, level) in self.levels.iter() {
            let compression_ratio = (target_compression * self.compression_ratio).min(1.0);
            
            // Compress transition matrix
            let compressed_transition = self.compress_matrix(&level.transition_matrix, compression_ratio)?;
            compressed_transitions.push(compressed_transition);
            
            // Compress observation matrix
            let compressed_observation = self.compress_matrix(&level.observation_matrix, compression_ratio)?;
            compressed_observations.push(compressed_observation);
            
            // Compress reward vector
            let compressed_reward = self.compress_vector(&level.reward_vector, compression_ratio)?;
            compressed_rewards.push(compressed_reward);
        }

        Ok((compressed_transitions, compressed_observations, compressed_rewards))
    }

    fn compress_matrix(&self, matrix: &Array2<f32>, ratio: f32) -> Result<Array2<f32>, PandoraError> {
        let (rows, cols) = matrix.dim();
        let new_rows = ((rows as f32) * ratio) as usize;
        let new_cols = ((cols as f32) * ratio) as usize;
        
        if new_rows == 0 || new_cols == 0 {
            return Err(PandoraError::PredictionFailed("Compression ratio too small".to_string()));
        }

        // Simple compression by taking every nth element
        let step_row = rows / new_rows;
        let step_col = cols / new_cols;
        
        let mut compressed = Array2::zeros((new_rows, new_cols));
        for i in 0..new_rows {
            for j in 0..new_cols {
                let orig_i = (i * step_row).min(rows - 1);
                let orig_j = (j * step_col).min(cols - 1);
                compressed[[i, j]] = matrix[[orig_i, orig_j]];
            }
        }
        
        Ok(compressed)
    }

    fn compress_vector(&self, vector: &Array1<f32>, ratio: f32) -> Result<Array1<f32>, PandoraError> {
        let len = vector.len();
        let new_len = ((len as f32) * ratio) as usize;
        
        if new_len == 0 {
            return Err(PandoraError::PredictionFailed("Compression ratio too small".to_string()));
        }

        let step = len / new_len;
        let mut compressed = Array1::zeros(new_len);
        for i in 0..new_len {
            let orig_i = (i * step).min(len - 1);
            compressed[i] = vector[orig_i];
        }
        
        Ok(compressed)
    }

    /// Update world model with new observation
    pub fn update_beliefs(&self, state: &WorldState) -> Result<Array2<f32>, PandoraError> {
        // Simple belief update using matrix multiplication
        let state_vector = &state.state_vector;
        let beliefs = Array2::from_shape_vec((1, state_vector.len()), state_vector.to_vec())
            .map_err(|e| PandoraError::PredictionFailed(format!("Failed to create belief matrix: {}", e)))?;
        
        Ok(beliefs)
    }
}

/// Performance Metrics for Active Inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_efe: f32,
    pub pragmatic_value: f32,
    pub epistemic_value: f32,
    pub policy_confidence: f32,
    pub world_model_accuracy: f32,
    pub computation_time_ms: u64,
}

/// Active Inference Sankhara - Main orchestrator
pub struct ActiveInferenceSankhara {
    pub efe_calculator: EFECalculator,
    pub world_model: HierarchicalWorldModel,
    pub performance_metrics: PerformanceMetrics,
    pub memory_traces: Vec<MemoryTrace>,
    pub max_memory_size: usize,
}

impl ActiveInferenceSankhara {
    pub fn new(
        precision_weight: f32,
        complexity_weight: f32,
        compression_ratio: f32,
        max_memory_size: usize,
    ) -> Self {
        Self {
            efe_calculator: EFECalculator::new(precision_weight, complexity_weight),
            world_model: HierarchicalWorldModel::new(compression_ratio),
            performance_metrics: PerformanceMetrics {
                total_efe: 0.0,
                pragmatic_value: 0.0,
                epistemic_value: 0.0,
                policy_confidence: 0.0,
                world_model_accuracy: 0.0,
                computation_time_ms: 0,
            },
            memory_traces: Vec::new(),
            max_memory_size,
        }
    }

    /// Process intent through Active Inference
    pub async fn process_intent(&mut self, intent: &Intent, vedana: &Vedana, sanna: &Sanna) -> Result<Policy, PandoraError> {
        let start_time = std::time::Instant::now();
        
        // Generate candidate policies
        let policies = self.generate_policies(intent, vedana, sanna).await?;
        
        // Calculate EFE for each policy
        let mut policy_efe = Vec::new();
        for (i, policy) in policies.iter().enumerate() {
            let efe = self.calculate_policy_efe(policy, intent, vedana, sanna).await?;
            policy_efe.push((i, efe));
        }
        
        // Select best policy
        let best_policy = self.select_policy(&policies, &policy_efe)?;
        
        // Update performance metrics
        let computation_time = start_time.elapsed().as_millis() as u64;
        let best_policy_clone = best_policy.clone();
        self.update_performance_metrics(&best_policy_clone, &policy_efe, computation_time);
        
        Ok(best_policy_clone)
    }

    async fn generate_policies(&self, intent: &Intent, _vedana: &Vedana, _sanna: &Sanna) -> Result<Vec<Policy>, PandoraError> {
        // Generate candidate policies based on intent
        let mut policies = Vec::new();
        
        // Simple policy generation based on intent priority
        for i in 0..3 {
            let policy = Policy {
                id: format!("policy_{}_{}", intent.id, i),
                actions: vec![Action {
                    id: format!("action_{}", i),
                    description: format!("Action for intent: {}", intent.description),
                    parameters: intent.context.clone(),
                }],
                expected_reward: (intent.priority as f32) * 0.1 + (i as f32) * 0.05,
                confidence: 0.7 + (i as f32) * 0.1,
            };
            policies.push(policy);
        }
        
        Ok(policies)
    }

    async fn calculate_policy_efe(&self, policy: &Policy, _intent: &Intent, _vedana: &Vedana, _sanna: &Sanna) -> Result<f32, PandoraError> {
        // Simplified EFE calculation
        let pragmatic_value = policy.expected_reward;
        let epistemic_value = policy.confidence * 0.5; // Simplified epistemic value
        Ok(-pragmatic_value - epistemic_value)
    }

    fn select_policy<'a>(&self, policies: &'a [Policy], policy_efe: &[(usize, f32)]) -> Result<&'a Policy, PandoraError> {
        // Find policy with minimum EFE (best policy)
        let (best_idx, _) = policy_efe
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PandoraError::PredictionFailed("No policies available".to_string()))?;
        
        policies.get(*best_idx)
            .ok_or_else(|| PandoraError::PredictionFailed("Policy index out of bounds".to_string()))
    }

    fn update_performance_metrics(&mut self, policy: &Policy, policy_efe: &[(usize, f32)], computation_time: u64) {
        self.performance_metrics.total_efe = policy_efe.iter().map(|(_, efe)| efe).sum::<f32>() / policy_efe.len() as f32;
        self.performance_metrics.pragmatic_value = policy.expected_reward;
        self.performance_metrics.epistemic_value = policy.confidence * 0.5;
        self.performance_metrics.policy_confidence = policy.confidence;
        self.performance_metrics.computation_time_ms = computation_time;
    }

    /// Update world model with new observation
    pub fn update_world_model(&mut self, _observation: &Observation) -> Result<(), PandoraError> {
        // Simplified world model update
        // In a full implementation, this would update the hierarchical world model
        Ok(())
    }

    /// Add memory trace for learning
    pub fn add_memory_trace(&mut self, trace: MemoryTrace) {
        self.memory_traces.push(trace);
        
        // Maintain memory size limit
        if self.memory_traces.len() > self.max_memory_size {
            self.memory_traces.remove(0);
        }
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Compress world model for edge deployment
    pub fn compress_for_edge(&mut self, target_compression: f32) -> Result<(), PandoraError> {
        self.world_model.compress_for_edge(target_compression)?;
        Ok(())
    }
}

/// Trait for Sankhara Skandha (mental formations)
pub trait SankharaSkandha {
    fn process(&self, input: &Intent) -> Result<Policy, PandoraError>;
    fn update(&mut self, feedback: &MemoryTrace) -> Result<(), PandoraError>;
}

impl SankharaSkandha for ActiveInferenceSankhara {
    fn process(&self, intent: &Intent) -> Result<Policy, PandoraError> {
        // Simplified processing without async
        let policies = futures::executor::block_on(self.generate_policies(intent, &Vedana {
            emotional_state: EmotionalState {
                valence: 0.0,
                arousal: 0.5,
                dominance: 0.5,
            },
            satisfaction_level: 0.5,
            stress_level: 0.3,
            energy_level: 0.7,
        }, &Sanna {
            perception_quality: 0.8,
            attention_focus: 0.6,
            memory_accessibility: 0.7,
            cognitive_load: 0.4,
        }))?;
        
        let policy_efe = futures::executor::block_on(async {
            let mut efe_values = Vec::new();
            for (i, policy) in policies.iter().enumerate() {
                let efe = self.calculate_policy_efe(policy, intent, &Vedana {
                    emotional_state: EmotionalState {
                        valence: 0.0,
                        arousal: 0.5,
                        dominance: 0.5,
                    },
                    satisfaction_level: 0.5,
                    stress_level: 0.3,
                    energy_level: 0.7,
                }, &Sanna {
                    perception_quality: 0.8,
                    attention_focus: 0.6,
                    memory_accessibility: 0.7,
                    cognitive_load: 0.4,
                }).await.unwrap_or(0.0);
                efe_values.push((i, efe));
            }
            efe_values
        });
        
        self.select_policy(&policies, &policy_efe).cloned()
    }

    fn update(&mut self, feedback: &MemoryTrace) -> Result<(), PandoraError> {
        self.add_memory_trace(feedback.clone());
        Ok(())
    }
}
