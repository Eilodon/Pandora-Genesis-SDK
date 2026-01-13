//! Generic causal graph module for domain-agnostic causal reasoning.
//!
//! This module provides a `GenericCausalGraph<V>` that works with any
//! signal variable type implementing `SignalVariable`, enabling causal
//! inference across different domains.

use crate::core::SignalVariable;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

// =============================================================================
// CAUSAL EDGE
// =============================================================================

/// Source of causal knowledge for an edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalSource {
    /// Prior knowledge from domain expertise or literature.
    Prior(String),
    /// Learned from observational data.
    Learned {
        observation_count: u64,
        confidence_score: f32,
    },
    /// Heuristic placeholder (should be replaced with learned or prior).
    Heuristic(String),
}

/// An edge in the causal graph representing a causal relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Number of successful outcomes when this relationship was active.
    pub successes: u32,
    /// Number of failed outcomes.
    pub failures: u32,
    /// Source of this causal knowledge.
    pub source: CausalSource,
}

impl CausalEdge {
    /// Compute success probability (Bayesian estimate).
    pub fn success_prob(&self) -> f32 {
        let total = self.successes + self.failures;
        if total > 0 {
            self.successes as f32 / total as f32
        } else {
            0.5 // Prior: 50% if no observations
        }
    }

    /// Create an empty edge with no observations.
    pub fn zero() -> Self {
        Self {
            successes: 0,
            failures: 0,
            source: CausalSource::Heuristic("unset".to_string()),
        }
    }

    /// Create a prior edge with initial counts.
    pub fn prior(successes: u32, failures: u32, note: &str) -> Self {
        Self {
            successes,
            failures,
            source: CausalSource::Prior(note.to_string()),
        }
    }

    /// Create from a simple weight value [-1.0, 1.0].
    pub fn from_weight(weight: f32) -> Self {
        // Convert weight to success/failure counts
        // weight = 0.0 -> 50/50, weight = 1.0 -> 100/0, weight = -1.0 -> 0/100
        let normalized = (weight.clamp(-1.0, 1.0) + 1.0) / 2.0;
        let successes = (normalized * 100.0) as u32;
        let failures = 100 - successes;
        Self::prior(successes, failures, "from_weight")
    }
}

// =============================================================================
// GENERIC CAUSAL GRAPH
// =============================================================================

/// A generic causal graph that works with any signal variable type.
///
/// This is the domain-agnostic version of `CausalGraph` that uses dynamic
/// sizing based on the variable type's `count()` method.
///
/// # Type Parameters
/// - `V`: A type implementing `SignalVariable`
///
/// # Example
/// ```rust,ignore
/// use zenb_core::core::generic_causal::GenericCausalGraph;
/// use zenb_core::domains::biofeedback::BioVariable;
///
/// let mut graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::new();
/// graph.set_edge(BioVariable::NotificationPressure, BioVariable::HeartRate, 0.6);
/// ```
#[derive(Debug, Clone)]
pub struct GenericCausalGraph<V: SignalVariable> {
    /// Number of variables (cached for efficiency).
    size: usize,
    /// Adjacency matrix: weights[cause][effect] = edge.
    /// Stored as flat Vec for cache efficiency.
    weights: Vec<Option<CausalEdge>>,
    /// Pairwise interaction weights (upper triangular).
    interaction_weights: Vec<Option<CausalEdge>>,
    /// Marker for the variable type.
    _marker: PhantomData<V>,
}

impl<V: SignalVariable> Default for GenericCausalGraph<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: SignalVariable> GenericCausalGraph<V> {
    /// Create a new empty causal graph.
    pub fn new() -> Self {
        let size = V::count();
        Self {
            size,
            weights: vec![None; size * size],
            interaction_weights: vec![None; size * size],
            _marker: PhantomData,
        }
    }

    /// Create a graph with priors from a function.
    ///
    /// The prior function receives (cause_index, effect_index) and returns
    /// the prior weight [-1.0, 1.0] or 0.0 for no prior.
    pub fn with_priors(prior_fn: fn(usize, usize) -> f32) -> Self {
        let size = V::count();
        let mut weights = vec![None; size * size];

        for i in 0..size {
            for j in 0..size {
                let weight = prior_fn(i, j);
                if weight.abs() > 1e-6 {
                    weights[i * size + j] = Some(CausalEdge::from_weight(weight));
                }
            }
        }

        Self {
            size,
            weights,
            interaction_weights: vec![None; size * size],
            _marker: PhantomData,
        }
    }

    /// Get index into flat array.
    #[inline]
    fn idx(&self, cause: usize, effect: usize) -> usize {
        cause * self.size + effect
    }

    /// Get the causal effect strength from cause to effect.
    pub fn get_effect(&self, cause: V, effect: V) -> f32 {
        let idx = self.idx(cause.index(), effect.index());
        self.weights
            .get(idx)
            .and_then(|e| e.as_ref())
            .map(|e| e.success_prob())
            .unwrap_or(0.0)
    }

    /// Get the raw edge if it exists.
    pub fn get_edge(&self, cause: V, effect: V) -> Option<&CausalEdge> {
        let idx = self.idx(cause.index(), effect.index());
        self.weights.get(idx).and_then(|e| e.as_ref())
    }

    /// Set a causal edge from cause to effect.
    pub fn set_edge(&mut self, cause: V, effect: V, edge: CausalEdge) {
        let idx = self.idx(cause.index(), effect.index());
        if idx < self.weights.len() {
            self.weights[idx] = Some(edge);
        }
    }

    /// Set a simple weight [-1.0, 1.0] from cause to effect.
    pub fn set_weight(&mut self, cause: V, effect: V, weight: f32) {
        self.set_edge(cause, effect, CausalEdge::from_weight(weight));
    }

    /// Get all incoming causal effects for a target variable.
    pub fn get_causes(&self, target: V) -> Vec<(V, f32)> {
        let target_idx = target.index();
        V::all()
            .iter()
            .filter_map(|&var| {
                let idx = self.idx(var.index(), target_idx);
                self.weights
                    .get(idx)
                    .and_then(|e| e.as_ref())
                    .filter(|e| e.success_prob() > 1e-6)
                    .map(|e| (var, e.success_prob()))
            })
            .collect()
    }

    /// Get all outgoing causal effects from a cause variable.
    pub fn get_effects(&self, cause: V) -> Vec<(V, f32)> {
        let cause_idx = cause.index();
        V::all()
            .iter()
            .filter_map(|&var| {
                let idx = self.idx(cause_idx, var.index());
                self.weights
                    .get(idx)
                    .and_then(|e| e.as_ref())
                    .filter(|e| e.success_prob() > 1e-6)
                    .map(|e| (var, e.success_prob()))
            })
            .collect()
    }

    /// Update weights based on observed outcome (learning).
    ///
    /// # Arguments
    /// * `context_state` - Normalized state values [0, 1] for each variable
    /// * `success` - Whether the action succeeded
    pub fn learn_from_outcome(&mut self, context_state: &[f32], success: bool) {
        let reward = if success { 1 } else { 0 };

        // Update all edges where the cause variable was active
        for (i, &cause_value) in context_state.iter().enumerate() {
            if cause_value.abs() < 0.1 {
                continue; // Skip inactive variables
            }

            for j in 0..self.size {
                if i == j {
                    continue; // Skip self-loops
                }

                let idx = self.idx(i, j);
                if idx >= self.weights.len() {
                    continue;
                }

                // Update or create edge
                let edge = self.weights[idx].get_or_insert_with(|| CausalEdge {
                    successes: 0,
                    failures: 0,
                    source: CausalSource::Learned {
                        observation_count: 0,
                        confidence_score: 0.0,
                    },
                });

                if reward > 0 {
                    edge.successes += 1;
                } else {
                    edge.failures += 1;
                }

                // Update source metadata
                if let CausalSource::Learned {
                    observation_count,
                    confidence_score,
                } = &mut edge.source
                {
                    *observation_count += 1;
                    *confidence_score = (*observation_count as f32 / 100.0).min(1.0);
                }
            }
        }
    }

    /// Check if the graph is acyclic (DAG property).
    pub fn is_acyclic(&self) -> bool {
        let mut visited = vec![false; self.size];
        let mut rec_stack = vec![false; self.size];

        for i in 0..self.size {
            if !visited[i] && self.has_cycle_dfs(i, &mut visited, &mut rec_stack) {
                return false;
            }
        }
        true
    }

    fn has_cycle_dfs(&self, v: usize, visited: &mut [bool], rec_stack: &mut [bool]) -> bool {
        visited[v] = true;
        rec_stack[v] = true;

        for j in 0..self.size {
            let idx = self.idx(v, j);
            if let Some(Some(edge)) = self.weights.get(idx) {
                if edge.success_prob() > 1e-6 {
                    if !visited[j] {
                        if self.has_cycle_dfs(j, visited, rec_stack) {
                            return true;
                        }
                    } else if rec_stack[j] {
                        return true;
                    }
                }
            }
        }

        rec_stack[v] = false;
        false
    }

    /// Number of variables in this graph.
    pub fn variable_count(&self) -> usize {
        self.size
    }

    /// Total number of edges with non-zero weight.
    pub fn edge_count(&self) -> usize {
        self.weights
            .iter()
            .filter(|e| e.as_ref().map(|e| e.success_prob() > 1e-6).unwrap_or(false))
            .count()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::biofeedback::BioVariable;

    #[test]
    fn test_new_graph() {
        let graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::new();
        assert_eq!(graph.variable_count(), BioVariable::count());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_set_and_get_weight() {
        let mut graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::new();

        graph.set_weight(
            BioVariable::NotificationPressure,
            BioVariable::HeartRate,
            0.6,
        );

        let effect = graph.get_effect(BioVariable::NotificationPressure, BioVariable::HeartRate);

        assert!(
            effect > 0.5,
            "Effect should be approximately 0.6, got {}",
            effect
        );
    }

    #[test]
    fn test_with_priors() {
        let graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::with_priors(|c, e| {
            if c == 0 && e == 1 {
                0.6 // NotificationPressure -> HeartRate
            } else {
                0.0
            }
        });

        let effect = graph.get_effect(BioVariable::NotificationPressure, BioVariable::HeartRate);

        assert!(effect > 0.5);
    }

    #[test]
    fn test_acyclic() {
        let mut graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::new();

        // A -> B -> C (no cycle)
        graph.set_weight(
            BioVariable::NotificationPressure,
            BioVariable::HeartRate,
            0.5,
        );
        graph.set_weight(BioVariable::HeartRate, BioVariable::RespiratoryRate, 0.5);

        assert!(graph.is_acyclic());
    }

    #[test]
    fn test_get_causes() {
        let mut graph: GenericCausalGraph<BioVariable> = GenericCausalGraph::new();

        graph.set_weight(
            BioVariable::NotificationPressure,
            BioVariable::HeartRate,
            0.6,
        );
        graph.set_weight(BioVariable::NoiseLevel, BioVariable::HeartRate, 0.3);

        let causes = graph.get_causes(BioVariable::HeartRate);
        assert_eq!(causes.len(), 2);
    }
}
