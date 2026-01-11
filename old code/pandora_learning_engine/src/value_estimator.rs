use pandora_core::ontology::EpistemologicalFlow;
use std::collections::HashMap;

pub trait ValueEstimator {
    fn estimate(&self, flow: &EpistemologicalFlow) -> f64;
}

/// Trait for Q-value estimation, extending ValueEstimator with action-specific values
pub trait QValueEstimator: ValueEstimator {
    /// Get Q-values for all possible actions given a flow state
    fn get_q_values(
        &self,
        flow: &EpistemologicalFlow,
    ) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>>;

    /// Update Q-value for a specific state-action pair
    fn update_q_value(
        &mut self,
        flow: &EpistemologicalFlow,
        action: &str,
        reward: f64,
        next_flow: &EpistemologicalFlow,
    );

    /// Get visit count for a state-action pair (for UCB1 exploration)
    fn get_visit_count(&self, flow: &EpistemologicalFlow, action: &str) -> u32;
}

#[derive(Default)]
pub struct MeanRewardEstimator;

impl ValueEstimator for MeanRewardEstimator {
    fn estimate(&self, _flow: &EpistemologicalFlow) -> f64 {
        0.0
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialMovingAverageEstimator {
    alpha: f64,
    state: std::collections::HashMap<u64, f64>, // keyed by hash of flow
}

impl ExponentialMovingAverageEstimator {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            state: std::collections::HashMap::new(),
        }
    }

    fn key(flow: &EpistemologicalFlow) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        // Rely on EpistemologicalFlow implementing Debug/Hash-like via bytes
        format!("{:?}", flow).hash(&mut hasher);
        hasher.finish()
    }

    pub fn update(&mut self, flow: &EpistemologicalFlow, reward: f64) {
        let k = Self::key(flow);
        let prev = *self.state.get(&k).unwrap_or(&reward);
        let ema = self.alpha * reward + (1.0 - self.alpha) * prev;
        self.state.insert(k, ema);
    }
}

impl ValueEstimator for ExponentialMovingAverageEstimator {
    fn estimate(&self, flow: &EpistemologicalFlow) -> f64 {
        let k = Self::key(flow);
        *self.state.get(&k).unwrap_or(&0.0)
    }
}

/// Neural network-based Q-value estimator for deep reinforcement learning
#[derive(Debug, Clone)]
pub struct NeuralQValueEstimator {
    /// Q-values for state-action pairs
    q_values: HashMap<String, f64>,
    /// Visit counts for state-action pairs (for UCB1)
    visit_counts: HashMap<String, u32>,
    /// Learning rate for Q-value updates
    learning_rate: f64,
    /// Discount factor for future rewards
    discount_factor: f64,
    /// Available actions
    actions: Vec<&'static str>,
}

impl NeuralQValueEstimator {
    /// Creates a new neural Q-value estimator
    pub fn new(learning_rate: f64, discount_factor: f64) -> Self {
        Self {
            q_values: HashMap::new(),
            visit_counts: HashMap::new(),
            learning_rate,
            discount_factor,
            actions: vec![
                "unlock_door",
                "pick_up_key",
                "move_forward",
                "explore",
                "noop",
            ],
        }
    }

    /// Creates a state-action key for hashing
    fn state_action_key(flow: &EpistemologicalFlow, action: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}_{}", flow, action).hash(&mut hasher);
        hasher.finish().to_string()
    }

    /// Simple neural network forward pass (placeholder for actual NN)
    fn forward_pass(&self, flow: &EpistemologicalFlow, action: &str) -> f64 {
        // This is a simplified implementation
        // In a real system, this would use a trained neural network

        // Extract features from the flow
        let mut features = Vec::new();

        // Add intent-based features
        if let Some(ref sankhara) = flow.sankhara {
            let intent_str = sankhara.as_ref();
            features.push(if intent_str == action { 1.0 } else { 0.0 });
        } else {
            features.push(0.0);
        }

        // Add sanna-based features
        if let Some(ref sanna) = flow.sanna {
            features.push(sanna.active_indices.len() as f64 / 32.0);
        } else {
            features.push(0.0);
        }

        // Add related_eidos features
        if let Some(ref related) = flow.related_eidos {
            features.push(related.len() as f64 / 4.0);
        } else {
            features.push(0.0);
        }

        // Simple weighted sum (placeholder for neural network)
        let weights = vec![0.4, 0.3, 0.3];
        let mut q_value = 0.0;
        for (feature, weight) in features.iter().zip(weights.iter()) {
            q_value += feature * weight;
        }

        // Add some noise to simulate neural network behavior
        use rand::Rng;
        let mut rng = rand::thread_rng();
        q_value += rng.gen_range(-0.1..0.1);

        q_value
    }
}

impl ValueEstimator for NeuralQValueEstimator {
    fn estimate(&self, flow: &EpistemologicalFlow) -> f64 {
        // Return the maximum Q-value across all actions
        self.actions
            .iter()
            .map(|action| {
                let key = Self::state_action_key(flow, action);
                self.q_values.get(&key).copied().unwrap_or_else(|| {
                    // If no stored Q-value, compute it using forward pass
                    self.forward_pass(flow, action)
                })
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

impl QValueEstimator for NeuralQValueEstimator {
    fn get_q_values(
        &self,
        flow: &EpistemologicalFlow,
    ) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
        let mut q_values = Vec::new();

        for action in &self.actions {
            let key = Self::state_action_key(flow, action);
            let q_value = self.q_values.get(&key).copied().unwrap_or_else(|| {
                // If no stored Q-value, compute it using forward pass
                self.forward_pass(flow, action)
            });
            q_values.push((*action, q_value));
        }

        Ok(q_values)
    }

    fn update_q_value(
        &mut self,
        flow: &EpistemologicalFlow,
        action: &str,
        reward: f64,
        next_flow: &EpistemologicalFlow,
    ) {
        let key = Self::state_action_key(flow, action);

        // Get current Q-value
        let current_q = self
            .q_values
            .get(&key)
            .copied()
            .unwrap_or_else(|| self.forward_pass(flow, action));

        // Get maximum Q-value for next state
        let max_next_q = self.estimate(next_flow);

        // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
        let target = reward + self.discount_factor * max_next_q;
        let new_q = current_q + self.learning_rate * (target - current_q);

        self.q_values.insert(key.clone(), new_q);

        // Update visit count
        let count = self.visit_counts.get(&key).copied().unwrap_or(0);
        self.visit_counts.insert(key, count + 1);
    }

    fn get_visit_count(&self, flow: &EpistemologicalFlow, action: &str) -> u32 {
        let key = Self::state_action_key(flow, action);
        self.visit_counts.get(&key).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    #[test]
    fn default_estimator_returns_zero() {
        let est = MeanRewardEstimator::default();
        let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"x"));
        assert_eq!(est.estimate(&flow), 0.0);
    }

    #[test]
    fn ema_estimator_updates_state() {
        let mut est = ExponentialMovingAverageEstimator::new(0.5);
        let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"z"));
        assert_eq!(est.estimate(&flow), 0.0);
        est.update(&flow, 1.0);
        let v1 = est.estimate(&flow);
        assert!(v1 > 0.0);
        est.update(&flow, 0.0);
        let v2 = est.estimate(&flow);
        assert!(v2 >= 0.0);
    }
}
