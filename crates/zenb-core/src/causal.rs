use serde::{Deserialize, Serialize};
use crate::domain::{CausalBeliefState, Observation};

// Submodules
pub mod notears;
mod graph_change_detector;
pub use graph_change_detector::GraphChangeDetector;
pub use notears::{Notears, NotearsConfig};


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
    /// Cognitive load (semantic/task switching burden)
    CognitiveLoad,
    /// Emotional valence from voice/text (-1 to 1, normalized)
    EmotionalValence,
    /// Voice arousal (intensity/activation)
    VoiceArousal,
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
            Variable::CognitiveLoad,
            Variable::EmotionalValence,
            Variable::VoiceArousal,
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
            Variable::CognitiveLoad => 9,
            Variable::EmotionalValence => 10,
            Variable::VoiceArousal => 11,
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
            9 => Some(Variable::CognitiveLoad),
            10 => Some(Variable::EmotionalValence),
            11 => Some(Variable::VoiceArousal),
            _ => None,
        }
    }

    /// Total number of variables.
    pub const COUNT: usize = 12;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalSource {
    Prior(String), // e.g., "Meta-Analysis 2024: Stress-HRV"
    Learned {
        observation_count: u64,
        confidence_score: f32,
    },
    Heuristic(String), // Temporary placeholder (must be flagged)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub successes: u32,
    pub failures: u32,
    pub source: CausalSource,
}

impl CausalEdge {
    pub fn success_prob(&self) -> f32 {
        let total = self.successes + self.failures;
        if total > 0 {
            self.successes as f32 / total as f32
        } else {
            0.5
        }
    }

    pub fn zero() -> Self {
        Self {
            successes: 0,
            failures: 0,
            source: CausalSource::Heuristic("unset".to_string()),
        }
    }

    pub fn prior(successes: u32, failures: u32, note: &str) -> Self {
        Self {
            successes,
            failures,
            source: CausalSource::Prior(note.to_string()),
        }
    }
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
    #[serde(default)]
    weights: [[Option<CausalEdge>; Variable::COUNT]; Variable::COUNT],
    /// Pairwise interaction weights between context variables for non-linear effects.
    /// Only the upper-triangular part (i < j) is used.
    #[serde(default)]
    interaction_weights: [[Option<CausalEdge>; Variable::COUNT]; Variable::COUNT],
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
            weights: std::array::from_fn(|_| std::array::from_fn(|_| None)),
            interaction_weights: std::array::from_fn(|_| {
                std::array::from_fn(|_| None)
            }),
        }
    }

    /// Create a causal graph with initial prior beliefs about relationships.
    /// These priors encode domain knowledge before any learning occurs.
    pub fn with_priors() -> Self {
        let mut graph = Self::new();

        // Domain knowledge priors (conservative estimates):
        // NotificationPressure -> HeartRate (stress response)
        graph.set_link(
            Variable::NotificationPressure,
            Variable::HeartRate,
            CausalEdge::prior(30, 70, "Logic: stress response"), // 30/100 success
        );

        // NotificationPressure -> HeartRateVariability (reduced HRV under stress)
        graph.set_link(
            Variable::NotificationPressure,
            Variable::HeartRateVariability,
            CausalEdge::prior(20, 80, "Logic: stress lowers HRV"),
        );

        // TimeOfDay -> HeartRate (circadian rhythm)
        graph.set_link(
            Variable::TimeOfDay,
            Variable::HeartRate,
            CausalEdge::prior(15, 85, "Logic: circadian HR"),
        );

        // NoiseLevel -> HeartRate (environmental stressor)
        graph.set_link(
            Variable::NoiseLevel,
            Variable::HeartRate,
            CausalEdge::prior(20, 80, "Logic: noise raises HR"),
        );

        // InteractionIntensity -> NotificationPressure (engagement drives notifications)
        graph.set_link(
            Variable::InteractionIntensity,
            Variable::NotificationPressure,
            CausalEdge::prior(25, 75, "Logic: engagement drives notifications"),
        );

        // UserAction -> RespiratoryRate (breath guidance intervention)
        graph.set_link(
            Variable::UserAction,
            Variable::RespiratoryRate,
            CausalEdge::prior(50, 50, "Logic: intervention effect"),
        );

        // RespiratoryRate -> HeartRate (respiratory sinus arrhythmia)
        graph.set_link(
            Variable::RespiratoryRate,
            Variable::HeartRate,
            CausalEdge::prior(30, 70, "Logic: RSA HR"),
        );

        // RespiratoryRate -> HeartRateVariability (breath coherence)
        graph.set_link(
            Variable::RespiratoryRate,
            Variable::HeartRateVariability,
            CausalEdge::prior(40, 60, "Logic: RSA coherence"),
        );

        // Cognitive priors
        graph.set_link(
            Variable::CognitiveLoad,
            Variable::NotificationPressure,
            CausalEdge::prior(40, 60, "Logic: load drives notifications"),
        );
        // Positive valence lowers heart rate
        graph.set_link(
            Variable::EmotionalValence,
            Variable::HeartRate,
            CausalEdge::prior(30, 70, "Logic: positive lowers HR"),
        );
        // Higher arousal raises heart rate
        graph.set_link(
            Variable::VoiceArousal,
            Variable::HeartRate,
            CausalEdge::prior(20, 80, "Logic: arousal raises HR"),
        );

        graph
    }

    /// Get the causal effect strength from cause to target.
    /// Returns 0.0 if no causal relationship exists.
    pub fn get_effect(&self, cause: Variable, target: Variable) -> f32 {
        self.weights[cause.index()][target.index()]
            .as_ref()
            .map(|e| e.success_prob())
            .unwrap_or(0.0)
    }

    /// Set the causal effect strength from cause to target.
    pub fn set_link(&mut self, cause: Variable, target: Variable, edge: CausalEdge) {
        self.weights[cause.index()][target.index()] = Some(edge);
    }

    /// Get all incoming causal effects for a target variable.
    /// Returns a vector of (cause_variable, effect_strength) pairs.
    pub fn get_causes(&self, target: Variable) -> Vec<(Variable, f32)> {
        let target_idx = target.index();
        Variable::all()
            .iter()
            .filter_map(|&var| {
                self.weights[var.index()][target_idx]
                    .as_ref()
                    .and_then(|l| {
                        if l.success_prob() > 1e-6 {
                            Some((var, l.success_prob()))
                        } else {
                            None
                        }
                    })
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
                self.weights[cause_idx][var.index()]
                    .as_ref()
                    .and_then(|l| {
                        if l.success_prob() > 1e-6 {
                            Some((var, l.success_prob()))
                        } else {
                            None
                        }
                    })
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
                if let Some(link) = &self.weights[cause_idx][target_idx] {
                    let weight = link.success_prob();
                    if weight.abs() > 1e-6 {
                        delta += state_values[cause_idx] * weight;
                    }
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
        values[Variable::CognitiveLoad.index()] = 0.5;
        values[Variable::EmotionalValence.index()] = 0.5;
        values[Variable::VoiceArousal.index()] = 0.5;

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
    pub fn predict_success_probability(&self, context_state: &[f32], _action: &ActionPolicy) -> f32 {
        // Cold start: return neutral probability to allow exploration
        let has_learned_weights = Variable::all()
            .iter()
            .any(|&var| self.weights[var.index()][Variable::UserAction.index()].is_some());

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
            if let Some(link) = &self.weights[var_idx][Variable::UserAction.index()] {
                let weight = link.success_prob();
                // Axiomatic: Unlearned (zero weight) returns neutral 0.5, not veto
                if weight.abs() > 1e-6 {
                    weighted_sum += context_state[var_idx] * weight;
                    total_weight += weight.abs();
                }
            } else {
                // No link yet -> neutral contribution (exploration)
                total_weight += 0.5;
            }
        }

        // Add pairwise interaction contributions (upper triangular only)
        for i in 0..Variable::COUNT {
            if i >= context_state.len() { break; }
            for j in (i + 1)..Variable::COUNT {
                if j >= context_state.len() { break; }
                if let Some(link) = &self.interaction_weights[i][j] {
                    let w = link.success_prob();
                    if w.abs() > 1e-6 {
                        let interaction = context_state[i] * context_state[j];
                        weighted_sum += interaction * w;
                        total_weight += w.abs();
                    }
                }
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
        _action: &ActionPolicy,
        success: bool,
        learning_rate: f32,
    ) -> Result<(), String> {
        let target_idx = Variable::UserAction.index();
        let reward: f32 = if success { 1.0 } else { -1.0 };

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
            let updated = if let Some(existing) = &self.weights[var_idx][target_idx] {
                let mut edge = existing.clone();
                if reward > 0.0 {
                    edge.successes += 1;
                } else {
                    edge.failures += 1;
                }
                edge
            } else {
                let mut edge = CausalEdge::zero();
                if reward > 0.0 {
                    edge.successes = 1;
                } else {
                    edge.failures = 1;
                }
                edge.source = CausalSource::Learned {
                    observation_count: 1,
                    confidence_score: reward.abs(),
                };
                edge
            };
            self.weights[var_idx][target_idx] = Some(updated);
        }

        // Update pairwise interaction weights (upper triangular only)
        for i in 0..Variable::COUNT {
            if i >= context_state.len() { break; }
            let vi = context_state[i];
            if vi.abs() < 1e-6 { continue; }
            for j in (i + 1)..Variable::COUNT {
                if j >= context_state.len() { break; }
                let vj = context_state[j];
                if vj.abs() < 1e-6 { continue; }
                let updated = if let Some(existing) = &self.interaction_weights[i][j] {
                    let mut edge = existing.clone();
                    if reward > 0.0 {
                        edge.successes += 1;
                    } else {
                        edge.failures += 1;
                    }
                    edge
                } else {
                    let mut edge = CausalEdge::zero();
                    if reward > 0.0 {
                        edge.successes = 1;
                    } else {
                        edge.failures = 1;
                    }
                    edge.source = CausalSource::Learned {
                        observation_count: 1,
                        confidence_score: reward.abs(),
                    };
                    edge
                };
                self.interaction_weights[i][j] = Some(updated);
            }
        }
        
        // INVARIANT: Causal graph must remain acyclic after weight update
        // INVARIANT: Causal graph must remain acyclic after weight update
        if !self.is_acyclic() {
            return Err("INVARIANT VIOLATION: Causal graph has cycle after weight update".to_string());
        }
        Ok(())
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
            if let Some(link) = &self.weights[v][i] {
                if link.success_prob() > 1e-6 {
                    if !visited[i] {
                        if self.has_cycle_util(i, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack[i] {
                        return true;
                    }
                }
            }
        }

        rec_stack[v] = false;
        false
    }
}

// ============================================================================
// PC ALGORITHM FOR CAUSAL STRUCTURE LEARNING
// ============================================================================

/// PC Algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCConfig {
    /// Significance level for conditional independence tests (default 0.05)
    pub alpha: f32,
    /// Maximum conditioning set size (limits computational complexity)
    pub max_cond_set_size: usize,
    /// Minimum samples required before running PC
    pub min_samples: usize,
}

impl Default for PCConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 3,
            min_samples: 50,
        }
    }
}

/// PC Algorithm for causal structure learning from observational data.
/// 
/// The PC algorithm learns causal structure in two phases:
/// 1. Skeleton learning: Start with complete graph, remove edges based on
///    conditional independence tests
/// 2. Edge orientation: Orient edges using v-structures and orientation rules
#[derive(Debug)]
pub struct PCAlgorithm {
    config: PCConfig,
    /// Separation sets: sep_sets[i][j] contains variables that d-separate i and j
    sep_sets: [[Option<Vec<usize>>; Variable::COUNT]; Variable::COUNT],
}

impl PCAlgorithm {
    pub fn new(config: Option<PCConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
            sep_sets: std::array::from_fn(|_| std::array::from_fn(|_| None)),
        }
    }
    
    /// Learn causal structure from data matrix.
    /// Returns adjacency matrix where edges[i][j] = true means i -> j is possible.
    pub fn learn_structure(&mut self, data: &[Vec<f32>]) -> [[bool; Variable::COUNT]; Variable::COUNT] {
        let n_samples = data.len();
        let n_vars = Variable::COUNT;
        
        if n_samples < self.config.min_samples {
            log::warn!("PC: insufficient samples ({} < {}), returning prior graph",
                       n_samples, self.config.min_samples);
            return self.complete_graph();
        }
        
        // Phase 1: Learn skeleton
        let mut adj = self.complete_graph();
        self.learn_skeleton(&mut adj, data);
        
        // Phase 2: Orient edges using v-structures and Meek rules
        self.orient_edges(&mut adj);
        
        adj
    }
    
    /// Initialize with complete undirected graph
    fn complete_graph(&self) -> [[bool; Variable::COUNT]; Variable::COUNT] {
        let mut adj = [[false; Variable::COUNT]; Variable::COUNT];
        for i in 0..Variable::COUNT {
            for j in 0..Variable::COUNT {
                if i != j {
                    adj[i][j] = true;
                }
            }
        }
        adj
    }
    
    /// Phase 1: Skeleton learning using conditional independence tests
    fn learn_skeleton(&mut self, adj: &mut [[bool; Variable::COUNT]; Variable::COUNT], data: &[Vec<f32>]) {
        let n_vars = Variable::COUNT;
        
        // Compute correlation matrix once
        let corr = self.compute_correlation_matrix(data);
        let n = data.len();
        
        // For each conditioning set size 0, 1, 2, ...
        for cond_size in 0..=self.config.max_cond_set_size {
            let mut changed = false;
            
            for i in 0..n_vars {
                for j in (i+1)..n_vars {
                    if !adj[i][j] {
                        continue; // Edge already removed
                    }
                    
                    // Get neighbors of i excluding j
                    let neighbors: Vec<usize> = (0..n_vars)
                        .filter(|&k| k != i && k != j && adj[i][k])
                        .collect();
                    
                    if neighbors.len() < cond_size {
                        continue;
                    }
                    
                    // Try all conditioning sets of size cond_size
                    for cond_set in self.combinations(&neighbors, cond_size) {
                        if self.is_conditionally_independent(i, j, &cond_set, &corr, n) {
                            // Remove edge i -- j
                            adj[i][j] = false;
                            adj[j][i] = false;
                            
                            // Store separation set
                            self.sep_sets[i][j] = Some(cond_set.clone());
                            self.sep_sets[j][i] = Some(cond_set);
                            
                            changed = true;
                            break;
                        }
                    }
                }
            }
            
            if !changed {
                break; // No edges removed at this level, stop
            }
        }
    }
    
    /// Phase 2: Orient edges using v-structures
    fn orient_edges(&self, adj: &mut [[bool; Variable::COUNT]; Variable::COUNT]) {
        let n_vars = Variable::COUNT;
        
        // Rule 1: V-structure detection
        // If i - k - j and i not adjacent to j, and k not in sepset(i,j),
        // then orient as i -> k <- j
        for k in 0..n_vars {
            for i in 0..n_vars {
                if i == k || !adj[i][k] {
                    continue;
                }
                for j in (i+1)..n_vars {
                    if j == k {
                        continue;
                    }
                    
                    // Check: i - k - j structure exists, i and j not adjacent
                    if adj[j][k] && !adj[i][j] && !adj[j][i] {
                        // Check if k is in separation set
                        let in_sepset = self.sep_sets[i][j]
                            .as_ref()
                            .map(|s| s.contains(&k))
                            .unwrap_or(false);
                        
                        if !in_sepset {
                            // Orient as v-structure: i -> k <- j
                            adj[k][i] = false; // Remove k -> i
                            adj[k][j] = false; // Remove k -> j
                        }
                    }
                }
            }
        }
        
        // Additional Meek orientation rules could be applied here
        // (R1-R4 for DAG completion)
    }
    
    /// Conditional independence test using partial correlation and Fisher's z-transform
    fn is_conditionally_independent(
        &self,
        i: usize,
        j: usize,
        cond_set: &[usize],
        corr: &[[f32; Variable::COUNT]; Variable::COUNT],
        n: usize,
    ) -> bool {
        let partial_corr = if cond_set.is_empty() {
            corr[i][j]
        } else {
            self.partial_correlation(i, j, cond_set, corr)
        };
        
        // Fisher's z-transform
        let z = ((1.0 + partial_corr) / (1.0 - partial_corr + 1e-10)).ln() / 2.0;
        
        // Degrees of freedom
        let df = (n as f32) - (cond_set.len() as f32) - 3.0;
        if df <= 0.0 {
            return false; // Not enough samples
        }
        
        // Standard error
        let se = 1.0 / df.sqrt();
        
        // z-statistic
        let z_stat = z.abs() / se;
        
        // Critical value for alpha (two-tailed)
        // For alpha=0.05, critical z ≈ 1.96
        let critical_z = match (self.config.alpha * 100.0) as u32 {
            1 => 2.576,
            5 => 1.960,
            10 => 1.645,
            _ => 1.960,
        };
        
        z_stat < critical_z
    }
    
    /// Compute sample correlation matrix
    fn compute_correlation_matrix(&self, data: &[Vec<f32>]) -> [[f32; Variable::COUNT]; Variable::COUNT] {
        let mut corr = [[0.0f32; Variable::COUNT]; Variable::COUNT];
        let n_vars = Variable::COUNT;
        let n = data.len() as f32;
        
        if n < 2.0 {
            return corr;
        }
        
        // Compute means
        let mut means = [0.0f32; Variable::COUNT];
        for row in data {
            for j in 0..n_vars.min(row.len()) {
                means[j] += row[j];
            }
        }
        for m in means.iter_mut() {
            *m /= n;
        }
        
        // Compute variances and covariances
        let mut vars = [0.0f32; Variable::COUNT];
        let mut covs = [[0.0f32; Variable::COUNT]; Variable::COUNT];
        
        for row in data {
            for i in 0..n_vars.min(row.len()) {
                let di = row[i] - means[i];
                vars[i] += di * di;
                for j in i..n_vars.min(row.len()) {
                    let dj = row[j] - means[j];
                    covs[i][j] += di * dj;
                }
            }
        }
        
        // Convert to correlation
        for i in 0..n_vars {
            for j in i..n_vars {
                if i == j {
                    corr[i][j] = 1.0;
                } else {
                    let var_product = (vars[i] * vars[j]).sqrt();
                    if var_product > 1e-10 {
                        let c = covs[i][j] / var_product;
                        corr[i][j] = c.clamp(-1.0, 1.0);
                        corr[j][i] = corr[i][j];
                    }
                }
            }
        }
        
        corr
    }
    
    /// Compute partial correlation between i and j given conditioning set
    fn partial_correlation(
        &self,
        i: usize,
        j: usize,
        cond_set: &[usize],
        corr: &[[f32; Variable::COUNT]; Variable::COUNT],
    ) -> f32 {
        if cond_set.len() == 1 {
            // Special case: single conditioning variable
            let k = cond_set[0];
            let r_ij = corr[i][j];
            let r_ik = corr[i][k];
            let r_jk = corr[j][k];
            
            let denom = ((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk)).sqrt();
            if denom < 1e-10 {
                return 0.0;
            }
            
            (r_ij - r_ik * r_jk) / denom
        } else {
            // General case: recursive formula or matrix inversion
            // Simplified: use first-order approximation
            let k = cond_set[0];
            let r_ij = corr[i][j];
            let r_ik = corr[i][k];
            let r_jk = corr[j][k];
            
            let denom = ((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk)).sqrt();
            if denom < 1e-10 {
                return 0.0;
            }
            
            (r_ij - r_ik * r_jk) / denom
        }
    }
    
    /// Generate all combinations of size k from elements
    fn combinations(&self, elements: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k > elements.len() {
            return vec![];
        }
        
        let mut result = Vec::new();
        self.combinations_helper(elements, k, 0, &mut vec![], &mut result);
        result
    }
    
    fn combinations_helper(
        &self,
        elements: &[usize],
        k: usize,
        start: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        
        for i in start..elements.len() {
            current.push(elements[i]);
            self.combinations_helper(elements, k, i + 1, current, result);
            current.pop();
        }
    }
    
    /// Integrate learned structure into CausalGraph
    pub fn apply_to_graph(&self, adj: &[[bool; Variable::COUNT]; Variable::COUNT], graph: &mut CausalGraph) {
        // Update graph edges based on learned structure
        for i in 0..Variable::COUNT {
            for j in 0..Variable::COUNT {
                if i == j {
                    continue;
                }
                
                let cause = Variable::from_index(i).unwrap();
                let effect = Variable::from_index(j).unwrap();
                
                if adj[i][j] && !adj[j][i] {
                    // Directed edge i -> j
                    if graph.weights[i][j].is_none() {
                        graph.set_link(cause, effect, CausalEdge::prior(50, 50, "PC-learned"));
                    }
                } else if !adj[i][j] {
                    // No edge - could remove weak prior links
                    // (keeping this conservative for now)
                }
            }
        }
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

            if let Some(ref cognitive) = snapshot.observation.cognitive_context {
                if let Some(load) = cognitive.cognitive_load {
                    row[Variable::CognitiveLoad.index()] = load;
                }
                // Combine voice valence and screen text sentiment into emotional valence
                let combined_valence = match (cognitive.voice_valence, cognitive.screen_text_sentiment)
                {
                    (Some(v1), Some(v2)) => Some((v1 + v2) / 2.0),
                    (Some(v), None) | (None, Some(v)) => Some(v),
                    _ => None,
                };
                if let Some(valence) = combined_valence {
                    // Normalize from [-1, 1] to [0, 1] for matrix representation
                    row[Variable::EmotionalValence.index()] = (valence + 1.0) / 2.0;
                }
                if let Some(arousal) = cognitive.voice_arousal {
                    row[Variable::VoiceArousal.index()] = arousal;
                }
            }

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
        assert_eq!(
            Variable::from_index(0),
            Some(Variable::NotificationPressure)
        );
        assert_eq!(Variable::all().len(), Variable::COUNT);
    }

    #[test]
    fn test_causal_graph_creation() {
        let graph = CausalGraph::new();
        assert_eq!(
            graph.get_effect(Variable::NotificationPressure, Variable::HeartRate),
            0.0
        );

        let graph_with_priors = CausalGraph::with_priors();
        assert!(
            graph_with_priors.get_effect(Variable::NotificationPressure, Variable::HeartRate) > 0.0
        );
    }

    #[test]
    fn test_causal_graph_set_get() {
        let mut graph = CausalGraph::new();
        graph.set_link(
            Variable::NotificationPressure,
            Variable::HeartRate,
            CausalEdge::prior(50, 50, "test"),
        );
        assert_eq!(
            graph.get_effect(Variable::NotificationPressure, Variable::HeartRate),
            0.5
        );
    }

    #[test]
    fn test_causal_graph_acyclic() {
        let graph = CausalGraph::with_priors();
        assert!(graph.is_acyclic());

        let mut cyclic_graph = CausalGraph::new();
        cyclic_graph.set_link(
            Variable::HeartRate,
            Variable::NotificationPressure,
            CausalEdge::prior(50, 50, "test"),
        );
        cyclic_graph.set_link(
            Variable::NotificationPressure,
            Variable::HeartRate,
            CausalEdge::prior(50, 50, "test"),
        );
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
            cognitive_context: None,
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
                cognitive_context: None,
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
