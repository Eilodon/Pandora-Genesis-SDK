use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::domain::{Observation, CausalBeliefState};

/// Causal variable nodes representing observable and latent factors in the system.
/// These form the vertices of the Directed Acyclic Graph (DAG).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Variable {
    /// Notification pressure: rate of incoming notifications (digital stressor)
    NotificationPressure,
    /// Heart rate in BPM (physiological arousal indicator)
    HeartRate,
    /// Heart Rate Variability (parasympathetic tone, stress resilience)
    HeartRateVariability,
    /// Physical location type (environmental context)
    Location,
    /// Time of day (circadian influence)
    TimeOfDay,
    /// User action taken by the system (intervention)
    UserAction,
    /// Interaction intensity (digital engagement level)
    InteractionIntensity,
    /// Respiratory rate (breath pattern)
    RespiratoryRate,
    /// Noise level (environmental stressor)
    NoiseLevel,
}

impl Variable {
    /// Get all possible variables as a slice for iteration.
    pub fn all() -> &'static [Variable] {
        &[
            Variable::NotificationPressure,
            Variable::HeartRate,
            Variable::HeartRateVariability,
            Variable::Location,
            Variable::TimeOfDay,
            Variable::UserAction,
            Variable::InteractionIntensity,
            Variable::RespiratoryRate,
            Variable::NoiseLevel,
        ]
    }

    /// Get the index of this variable in the adjacency matrix.
    pub fn index(&self) -> usize {
        match self {
            Variable::NotificationPressure => 0,
            Variable::HeartRate => 1,
            Variable::HeartRateVariability => 2,
            Variable::Location => 3,
            Variable::TimeOfDay => 4,
            Variable::UserAction => 5,
            Variable::InteractionIntensity => 6,
            Variable::RespiratoryRate => 7,
            Variable::NoiseLevel => 8,
        }
    }

    /// Get variable from index.
    pub fn from_index(idx: usize) -> Option<Variable> {
        match idx {
            0 => Some(Variable::NotificationPressure),
            1 => Some(Variable::HeartRate),
            2 => Some(Variable::HeartRateVariability),
            3 => Some(Variable::Location),
            4 => Some(Variable::TimeOfDay),
            5 => Some(Variable::UserAction),
            6 => Some(Variable::InteractionIntensity),
            7 => Some(Variable::RespiratoryRate),
            8 => Some(Variable::NoiseLevel),
            _ => None,
        }
    }

    /// Total number of variables.
    pub const COUNT: usize = 9;
}

/// Directed Acyclic Graph (DAG) representing causal relationships.
/// Edge weights represent causal effect strengths: -1.0 (strong negative) to 1.0 (strong positive).
/// 
/// Example: weights[NotificationPressure][HeartRate] = 0.6 means
/// "Notification pressure causes a moderate positive increase in heart rate"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// Adjacency matrix: weights[cause][effect] = strength
    /// Shape: [Variable::COUNT x Variable::COUNT]
    /// weights[i][j] represents the causal effect of variable i on variable j
    weights: [[f32; Variable::COUNT]; Variable::COUNT],
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalGraph {
    /// Create a new causal graph with zero weights (no learned relationships yet).
    pub fn new() -> Self {
        Self {
            weights: [[0.0; Variable::COUNT]; Variable::COUNT],
        }
    }

    /// Create a causal graph with initial prior beliefs about relationships.
    /// These priors encode domain knowledge before any learning occurs.
    pub fn with_priors() -> Self {
        let mut graph = Self::new();
        
        // Domain knowledge priors (conservative estimates):
        // NotificationPressure -> HeartRate (stress response)
        graph.set_effect(Variable::NotificationPressure, Variable::HeartRate, 0.3);
        
        // NotificationPressure -> HeartRateVariability (reduced HRV under stress)
        graph.set_effect(Variable::NotificationPressure, Variable::HeartRateVariability, -0.2);
        
        // TimeOfDay -> HeartRate (circadian rhythm)
        graph.set_effect(Variable::TimeOfDay, Variable::HeartRate, 0.15);
        
        // NoiseLevel -> HeartRate (environmental stressor)
        graph.set_effect(Variable::NoiseLevel, Variable::HeartRate, 0.2);
        
        // InteractionIntensity -> NotificationPressure (engagement drives notifications)
        graph.set_effect(Variable::InteractionIntensity, Variable::NotificationPressure, 0.25);
        
        // UserAction -> RespiratoryRate (breath guidance intervention)
        graph.set_effect(Variable::UserAction, Variable::RespiratoryRate, 0.5);
        
        // RespiratoryRate -> HeartRate (respiratory sinus arrhythmia)
        graph.set_effect(Variable::RespiratoryRate, Variable::HeartRate, -0.3);
        
        // RespiratoryRate -> HeartRateVariability (breath coherence)
        graph.set_effect(Variable::RespiratoryRate, Variable::HeartRateVariability, 0.4);
        
        graph
    }

    /// Get the causal effect strength from cause to target.
    /// Returns 0.0 if no causal relationship exists.
    pub fn get_effect(&self, cause: Variable, target: Variable) -> f32 {
        self.weights[cause.index()][target.index()]
    }

    /// Set the causal effect strength from cause to target.
    /// Weight should be in range [-1.0, 1.0].
    pub fn set_effect(&mut self, cause: Variable, target: Variable, weight: f32) {
        self.weights[cause.index()][target.index()] = weight.clamp(-1.0, 1.0);
    }

    /// Get all incoming causal effects for a target variable.
    /// Returns a vector of (cause_variable, effect_strength) pairs.
    pub fn get_causes(&self, target: Variable) -> Vec<(Variable, f32)> {
        let target_idx = target.index();
        Variable::all()
            .iter()
            .filter_map(|&var| {
                let weight = self.weights[var.index()][target_idx];
                if weight.abs() > 1e-6 {
                    Some((var, weight))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all outgoing causal effects from a cause variable.
    /// Returns a vector of (effect_variable, effect_strength) pairs.
    pub fn get_effects(&self, cause: Variable) -> Vec<(Variable, f32)> {
        let cause_idx = cause.index();
        Variable::all()
            .iter()
            .filter_map(|&var| {
                let weight = self.weights[cause_idx][var.index()];
                if weight.abs() > 1e-6 {
                    Some((var, weight))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Predict the outcome of a proposed action on the current state.
    /// Uses simple linear projection: NextState = CurrentState + (Action * Weight)
    /// 
    /// This is a first-order approximation. Future versions will use:
    /// - Non-linear activation functions
    /// - Multi-step lookahead
    /// - Uncertainty quantification
    pub fn predict_outcome(
        &self,
        current_state: &crate::belief::BeliefState,
        proposed_action: &ActionPolicy,
    ) -> PredictedState {
        // Extract current variable values from belief state
        let mut state_values = self.extract_state_values(current_state);
        
        // Apply action effect
        let action_strength = proposed_action.intensity;
        state_values[Variable::UserAction.index()] = action_strength;
        
        // Propagate causal effects (single step)
        let mut predicted_values = state_values.clone();
        for target_var in Variable::all() {
            let target_idx = target_var.index();
            let mut delta = 0.0;
            
            // Sum all incoming causal effects
            for cause_var in Variable::all() {
                let cause_idx = cause_var.index();
                let weight = self.weights[cause_idx][target_idx];
                if weight.abs() > 1e-6 {
                    delta += state_values[cause_idx] * weight;
                }
            }
            
            // Apply delta with damping to prevent explosion
            predicted_values[target_idx] = (state_values[target_idx] + delta * 0.1).clamp(0.0, 1.0);
        }
        
        PredictedState {
            predicted_hr: predicted_values[Variable::HeartRate.index()] * 200.0, // denormalize
            predicted_hrv: predicted_values[Variable::HeartRateVariability.index()] * 100.0,
            predicted_rr: predicted_values[Variable::RespiratoryRate.index()] * 20.0,
            confidence: 0.5, // placeholder: will be computed from graph uncertainty later
        }
    }

    /// Extract normalized variable values from canonical belief state.
    /// Converts the 5-mode belief::BeliefState to causal variable space.
    /// Returns a vector of size Variable::COUNT with values in [0, 1].
    pub fn extract_state_values(&self, belief_state: &crate::belief::BeliefState) -> Vec<f32> {
        let mut values = vec![0.0; Variable::COUNT];
        
        // Map canonical belief state (5-mode) to causal variables (normalized to [0, 1])
        // belief_state.p = [Calm, Stress, Focus, Sleepy, Energize]
        // These mappings are heuristic and will be refined with learning
        
        // Bio state: Stress mode -> high HR, low HRV
        let stress_level = belief_state.p[1]; // Stress probability
        values[Variable::HeartRate.index()] = stress_level;
        values[Variable::HeartRateVariability.index()] = 1.0 - stress_level;
        
        // Cognitive state: Focus mode affects notification pressure inversely
        let focus_level = belief_state.p[2]; // Focus probability
        values[Variable::NotificationPressure.index()] = 1.0 - focus_level;
        values[Variable::InteractionIntensity.index()] = 1.0 - focus_level;
        
        // Placeholder for other variables (will be populated from Observation)
        values[Variable::Location.index()] = 0.5;
        values[Variable::TimeOfDay.index()] = 0.5;
        values[Variable::RespiratoryRate.index()] = 0.5;
        values[Variable::NoiseLevel.index()] = 0.5;
        values[Variable::UserAction.index()] = 0.0;
        
        values
    }

    /// Predict success probability of an action given the current context.
    /// 
    /// This method consults the learned causal graph to estimate how likely
    /// an action is to succeed in the current context. It looks at the edge
    /// weights between context variables and the UserAction node.
    /// 
    /// # Arguments
    /// * `context_state` - Current normalized state values [0, 1] for each variable
    /// * `action` - The proposed action policy
    /// 
    /// # Returns
    /// Success probability in [0.0, 1.0]. Returns 0.5 (neutral) if graph is empty (cold start).
    /// 
    /// # Logic
    /// - Computes weighted sum of context variables -> UserAction edge weights
    /// - Normalizes to [0, 1] probability range
    /// - Returns 0.5 for cold start (no learned relationships)
    pub fn predict_success_probability(&self, context_state: &[f32], action: &ActionPolicy) -> f32 {
        // Cold start: return neutral probability to allow exploration
        let has_learned_weights = Variable::all().iter().any(|&var| {
            self.weights[var.index()][Variable::UserAction.index()].abs() > 1e-6
        });
        
        if !has_learned_weights {
            return 0.5;
        }
        
        // Compute weighted sum of context influences on action success
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for var in Variable::all() {
            let var_idx = var.index();
            if var_idx >= context_state.len() {
                continue;
            }
            
            // Get the causal effect of this context variable on UserAction
            let weight = self.weights[var_idx][Variable::UserAction.index()];
            
            if weight.abs() > 1e-6 {
                // Positive weight = context supports action success
                // Negative weight = context inhibits action success
                weighted_sum += context_state[var_idx] * weight;
                total_weight += weight.abs();
            }
        }
        
        // Normalize to [0, 1] probability
        // If total_weight is 0, return neutral probability
        if total_weight < 1e-6 {
            return 0.5;
        }
        
        // Map weighted_sum to probability: [-1, 1] -> [0, 1]
        let normalized = weighted_sum / total_weight;
        ((normalized + 1.0) / 2.0).clamp(0.0, 1.0)
    }
    
    /// Update causal graph weights based on observed outcome.
    /// Uses simple gradient-based learning to reinforce or weaken causal links.
    /// 
    /// # Arguments
    /// * `context_state` - State values when action was taken
    /// * `action` - The action that was taken
    /// * `success` - Whether the action succeeded
    /// * `learning_rate` - How much to update weights (typically 0.01 - 0.1)
    /// 
    /// # Learning Rule
    /// If success: strengthen positive weights, weaken negative weights
    /// If failure: weaken positive weights, strengthen negative weights
    pub fn update_weights(
        &mut self,
        context_state: &[f32],
        action: &ActionPolicy,
        success: bool,
        learning_rate: f32,
    ) {
        let target_idx = Variable::UserAction.index();
        let reward = if success { 1.0 } else { -1.0 };
        
        for var in Variable::all() {
            let var_idx = var.index();
            if var_idx >= context_state.len() {
                continue;
            }
            
            let context_value = context_state[var_idx];
            if context_value.abs() < 1e-6 {
                continue; // Skip if context variable is not active
            }
            
            // Gradient update: weight += lr * context_value * reward
            let current_weight = self.weights[var_idx][target_idx];
            let delta = learning_rate * context_value * reward;
            let new_weight = (current_weight + delta).clamp(-1.0, 1.0);
            
            self.weights[var_idx][target_idx] = new_weight;
        }
    }

    /// Check if the graph is acyclic (DAG property).
    /// Returns true if no cycles exist.
    pub fn is_acyclic(&self) -> bool {
        // Simple cycle detection using DFS
        let mut visited = vec![false; Variable::COUNT];
        let mut rec_stack = vec![false; Variable::COUNT];
        
        for i in 0..Variable::COUNT {
            if !visited[i] {
                if self.has_cycle_util(i, &mut visited, &mut rec_stack) {
                    return false;
                }
            }
        }
        true
    }

    fn has_cycle_util(&self, v: usize, visited: &mut Vec<bool>, rec_stack: &mut Vec<bool>) -> bool {
        visited[v] = true;
        rec_stack[v] = true;
        
        // Check all neighbors
        for i in 0..Variable::COUNT {
            if self.weights[v][i].abs() > 1e-6 {
                if !visited[i] {
                    if self.has_cycle_util(i, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack[i] {
                    return true;
                }
            }
        }
        
        rec_stack[v] = false;
        false
    }
}

/// Action policy representing a proposed intervention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPolicy {
    /// Type of action (breath guidance, notification block, etc.)
    pub action_type: ActionType,
    /// Intensity of the action [0, 1]
    pub intensity: f32,
}

/// Types of interventions the system can take.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Breath guidance intervention
    BreathGuidance,
    /// Block or delay notifications
    NotificationBlock,
    /// Suggest app switch (e.g., from social to wellness)
    AppSuggestion,
    /// No action
    NoAction,
}

/// Predicted future state after applying an action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedState {
    /// Predicted heart rate (BPM)
    pub predicted_hr: f32,
    /// Predicted HRV (RMSSD in ms)
    pub predicted_hrv: f32,
    /// Predicted respiratory rate (BPM)
    pub predicted_rr: f32,
    /// Confidence in prediction [0, 1]
    pub confidence: f32,
}

/// Sliding window buffer for storing recent observations and actions.
/// This buffer will be used for batch learning of causal relationships (NOTEARS algorithm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalBuffer {
    /// Maximum buffer size (number of observation-action pairs)
    capacity: usize,
    /// Circular buffer of observations
    observations: Vec<ObservationSnapshot>,
    /// Current write position in the circular buffer
    write_pos: usize,
    /// Number of items currently in buffer (up to capacity)
    count: usize,
}

/// Snapshot of observation and action at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationSnapshot {
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// Observation data
    pub observation: Observation,
    /// Action taken (if any)
    pub action: Option<ActionPolicy>,
    /// Belief state at this time (using CausalBeliefState for 3-factor representation)
    pub belief_state: Option<CausalBeliefState>,
}

impl CausalBuffer {
    /// Create a new causal buffer with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            observations: Vec::with_capacity(capacity),
            write_pos: 0,
            count: 0,
        }
    }

    /// Create a buffer with default capacity of 1000.
    pub fn default_capacity() -> Self {
        Self::new(1000)
    }

    /// Push a new observation into the buffer.
    /// If buffer is full, overwrites the oldest entry (circular buffer).
    pub fn push(&mut self, snapshot: ObservationSnapshot) {
        if self.count < self.capacity {
            self.observations.push(snapshot);
            self.count += 1;
        } else {
            self.observations[self.write_pos] = snapshot;
        }
        
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    /// Get the current number of observations in the buffer.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.count >= self.capacity
    }

    /// Get all observations in chronological order.
    pub fn get_all(&self) -> Vec<&ObservationSnapshot> {
        if self.count < self.capacity {
            self.observations.iter().collect()
        } else {
            let mut result = Vec::with_capacity(self.count);
            for i in 0..self.count {
                let idx = (self.write_pos + i) % self.capacity;
                result.push(&self.observations[idx]);
            }
            result
        }
    }

    /// Get the most recent N observations.
    pub fn get_recent(&self, n: usize) -> Vec<&ObservationSnapshot> {
        let n = n.min(self.count);
        let mut result = Vec::with_capacity(n);
        
        for i in 0..n {
            let idx = if self.write_pos >= i + 1 {
                self.write_pos - i - 1
            } else {
                self.capacity + self.write_pos - i - 1
            };
            result.push(&self.observations[idx]);
        }
        
        result.reverse();
        result
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.observations.clear();
        self.write_pos = 0;
        self.count = 0;
    }

    /// Extract a data matrix for causal learning algorithms.
    /// Returns a matrix where each row is a time point and each column is a variable.
    /// Shape: [time_points x Variable::COUNT]
    pub fn to_data_matrix(&self) -> Vec<Vec<f32>> {
        let snapshots = self.get_all();
        let mut matrix = Vec::with_capacity(snapshots.len());
        
        for snapshot in snapshots {
            let mut row = vec![0.0; Variable::COUNT];
            
            // Extract variable values from observation
            if let Some(ref digital) = snapshot.observation.digital_context {
                if let Some(notif_pressure) = digital.notification_pressure {
                    row[Variable::NotificationPressure.index()] = notif_pressure;
                }
                if let Some(interaction) = digital.interaction_intensity {
                    row[Variable::InteractionIntensity.index()] = interaction;
                }
            }
            
            if let Some(ref bio) = snapshot.observation.bio_metrics {
                if let Some(hr) = bio.hr_bpm {
                    row[Variable::HeartRate.index()] = hr / 200.0; // normalize
                }
                if let Some(hrv) = bio.hrv_rmssd {
                    row[Variable::HeartRateVariability.index()] = hrv / 100.0; // normalize
                }
                if let Some(rr) = bio.respiratory_rate {
                    row[Variable::RespiratoryRate.index()] = rr / 20.0; // normalize
                }
            }
            
            if let Some(ref env) = snapshot.observation.environmental_context {
                if let Some(noise) = env.noise_level {
                    row[Variable::NoiseLevel.index()] = noise;
                }
                if let Some(loc) = env.location_type {
                    row[Variable::Location.index()] = match loc {
                        crate::domain::LocationType::Home => 0.0,
                        crate::domain::LocationType::Work => 0.5,
                        crate::domain::LocationType::Transit => 1.0,
                    };
                }
            }
            
            // Time of day (normalized to [0, 1])
            let hour = (snapshot.timestamp_us / 3_600_000_000) % 24;
            row[Variable::TimeOfDay.index()] = hour as f32 / 24.0;
            
            // Action
            if let Some(ref action) = snapshot.action {
                row[Variable::UserAction.index()] = action.intensity;
            }
            
            matrix.push(row);
        }
        
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_indexing() {
        assert_eq!(Variable::NotificationPressure.index(), 0);
        assert_eq!(Variable::from_index(0), Some(Variable::NotificationPressure));
        assert_eq!(Variable::all().len(), Variable::COUNT);
    }

    #[test]
    fn test_causal_graph_creation() {
        let graph = CausalGraph::new();
        assert_eq!(graph.get_effect(Variable::NotificationPressure, Variable::HeartRate), 0.0);
        
        let graph_with_priors = CausalGraph::with_priors();
        assert!(graph_with_priors.get_effect(Variable::NotificationPressure, Variable::HeartRate) > 0.0);
    }

    #[test]
    fn test_causal_graph_set_get() {
        let mut graph = CausalGraph::new();
        graph.set_effect(Variable::NotificationPressure, Variable::HeartRate, 0.5);
        assert_eq!(graph.get_effect(Variable::NotificationPressure, Variable::HeartRate), 0.5);
    }

    #[test]
    fn test_causal_graph_acyclic() {
        let graph = CausalGraph::with_priors();
        assert!(graph.is_acyclic());
        
        let mut cyclic_graph = CausalGraph::new();
        cyclic_graph.set_effect(Variable::HeartRate, Variable::NotificationPressure, 0.5);
        cyclic_graph.set_effect(Variable::NotificationPressure, Variable::HeartRate, 0.5);
        assert!(!cyclic_graph.is_acyclic());
    }

    #[test]
    fn test_causal_buffer_push() {
        let mut buffer = CausalBuffer::new(3);
        assert!(buffer.is_empty());
        
        let obs = Observation {
            timestamp_us: 0,
            bio_metrics: None,
            environmental_context: None,
            digital_context: None,
        };
        
        buffer.push(ObservationSnapshot {
            timestamp_us: 0,
            observation: obs.clone(),
            action: None,
            belief_state: None,
        });
        
        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_causal_buffer_circular() {
        let mut buffer = CausalBuffer::new(2);
        
        for i in 0..5 {
            let obs = Observation {
                timestamp_us: i,
                bio_metrics: None,
                environmental_context: None,
                digital_context: None,
            };
            buffer.push(ObservationSnapshot {
                timestamp_us: i,
                observation: obs,
                action: None,
                belief_state: None,
            });
        }
        
        assert_eq!(buffer.len(), 2);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_predict_outcome() {
        let graph = CausalGraph::with_priors();
        let belief_state = crate::belief::BeliefState::default();
        let action = ActionPolicy {
            action_type: ActionType::BreathGuidance,
            intensity: 0.8,
        };
        
        let prediction = graph.predict_outcome(&belief_state, &action);
        assert!(prediction.confidence > 0.0);
        assert!(prediction.predicted_hr >= 0.0);
    }
}
