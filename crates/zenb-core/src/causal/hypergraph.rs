//! Causal Hypergraph for Higher-Order Relationships
//!
//! Extends pairwise causal graphs to support hyperedges - relationships
//! involving multiple variables simultaneously. Essential for modeling
//! complex physiological interactions.

use super::{CausalEdge, Variable};
use serde::{Deserialize, Serialize};

/// Hyperedge: A causal relationship involving multiple source variables
///
/// # Example
/// Heart rate variability (HRV) depends on the interaction of:
/// - Heart rate (HR)
/// - Respiratory rate (RR)
/// - Emotional valence
///
/// This is a 3-way interaction that cannot be captured by pairwise edges alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperEdge {
    /// Source variables (e.g., [HR, RR, Valence])
    pub sources: Vec<Variable>,

    /// Target variable affected by this interaction
    pub target: Variable,

    /// Causal strength and statistics
    pub weight: CausalEdge,

    /// Order of interaction (number of sources)
    /// 2 = pairwise, 3 = three-way, etc.
    pub order: usize,
}

impl HyperEdge {
    /// Create a new hyperedge
    pub fn new(sources: Vec<Variable>, target: Variable, weight: CausalEdge) -> Self {
        let order = sources.len();
        Self {
            sources,
            target,
            weight,
            order,
        }
    }

    /// Compute interaction term from state values
    ///
    /// For sources [A, B, C], computes: state[A] * state[B] * state[C]
    pub fn compute_interaction(&self, state: &[f32]) -> f32 {
        self.sources
            .iter()
            .map(|var| {
                let idx = var.index();
                if idx < state.len() {
                    state[idx]
                } else {
                    0.0
                }
            })
            .product()
    }

    /// Check if this hyperedge involves a specific variable
    pub fn involves(&self, variable: Variable) -> bool {
        self.sources.contains(&variable) || self.target == variable
    }
}

/// Causal Hypergraph for complex multi-variable relationships
///
/// # Design Philosophy
/// Pairwise interactions (A→B, B→C) are insufficient for modeling:
/// - Synergistic effects (A+B together → C, but neither alone)
/// - Inhibitory interactions (A blocks B→C)
/// - Context-dependent causality (A→B only when C is present)
///
/// Hypergraphs capture these higher-order patterns.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CausalHypergraph {
    /// Collection of hyperedges
    hyperedges: Vec<HyperEdge>,

    /// Maximum order of interactions tracked
    max_order: usize,
}

impl CausalHypergraph {
    /// Create a new empty hypergraph
    pub fn new() -> Self {
        Self {
            hyperedges: Vec::new(),
            max_order: 2, // Start with pairwise
        }
    }

    /// Create with specific maximum order
    pub fn with_max_order(max_order: usize) -> Self {
        Self {
            hyperedges: Vec::new(),
            max_order,
        }
    }

    /// Add a hyperedge to the graph
    pub fn add_hyperedge(&mut self, edge: HyperEdge) {
        if edge.order > self.max_order {
            self.max_order = edge.order;
        }
        self.hyperedges.push(edge);
    }

    /// Add a hyperedge from components
    pub fn add_interaction(
        &mut self,
        sources: Vec<Variable>,
        target: Variable,
        weight: CausalEdge,
    ) {
        let edge = HyperEdge::new(sources, target, weight);
        self.add_hyperedge(edge);
    }

    /// Get all hyperedges affecting a target variable
    pub fn edges_to(&self, target: Variable) -> Vec<&HyperEdge> {
        self.hyperedges
            .iter()
            .filter(|edge| edge.target == target)
            .collect()
    }

    /// Get all hyperedges involving a variable (as source or target)
    pub fn edges_involving(&self, variable: Variable) -> Vec<&HyperEdge> {
        self.hyperedges
            .iter()
            .filter(|edge| edge.involves(variable))
            .collect()
    }

    /// Predict effect on target variable using hyperedges
    ///
    /// Computes: Σ (interaction_term * weight) for all hyperedges to target
    pub fn predict_effect(&self, state: &[f32], target: Variable) -> f32 {
        self.edges_to(target)
            .iter()
            .map(|edge| {
                let interaction = edge.compute_interaction(state);
                let weight = edge.weight.success_prob();
                interaction * weight
            })
            .sum()
    }

    /// Predict effect with uncertainty
    pub fn predict_effect_uncertain(
        &self,
        state: &[f32],
        target: Variable,
    ) -> crate::uncertain::Uncertain<f32> {
        let edges = self.edges_to(target);

        if edges.is_empty() {
            return crate::uncertain::Uncertain::new(0.0, 0.1, "Hypergraph (no edges)");
        }

        let mut total_effect = 0.0;
        let mut total_observations = 0u64;

        for edge in &edges {
            let interaction = edge.compute_interaction(state);
            let weight = edge.weight.success_prob();
            total_effect += interaction * weight;
            total_observations += (edge.weight.successes + edge.weight.failures) as u64;
        }

        // Confidence based on number of observations
        let confidence = (total_observations as f32 / 50.0).min(0.95);

        crate::uncertain::Uncertain::new(
            total_effect,
            confidence,
            format!("Hypergraph ({} edges)", edges.len()),
        )
    }

    /// Get number of hyperedges
    pub fn edge_count(&self) -> usize {
        self.hyperedges.len()
    }

    /// Get maximum interaction order
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Clear all hyperedges
    pub fn clear(&mut self) {
        self.hyperedges.clear();
        self.max_order = 2;
    }

    // =========================================================================
    // Cycle Detection (Phase 1 Enhancement)
    // =========================================================================

    /// Detect if the hypergraph contains any cycles.
    ///
    /// Returns the cycle path if found, or None if acyclic.
    ///
    /// # Algorithm
    /// Uses DFS with color marking:
    /// - White (unvisited)
    /// - Gray (in current path)
    /// - Black (finished)
    ///
    /// A back edge (to a Gray node) indicates a cycle.
    pub fn detect_cycle(&self) -> Option<CyclePath> {
        use std::collections::{HashMap, HashSet};

        // Build adjacency: target -> set of sources (reverse for cycle detection)
        // For hyperedges, we consider: each source can "reach" the target
        let mut out_edges: HashMap<Variable, HashSet<Variable>> = HashMap::new();
        
        for edge in &self.hyperedges {
            for source in &edge.sources {
                out_edges
                    .entry(*source)
                    .or_default()
                    .insert(edge.target);
            }
        }

        // DFS state
        #[derive(Clone, Copy, PartialEq)]
        enum Color { White, Gray, Black }
        
        let mut color: HashMap<Variable, Color> = HashMap::new();
        let mut parent: HashMap<Variable, Variable> = HashMap::new();

        // Get all unique variables
        let mut all_vars: HashSet<Variable> = HashSet::new();
        for edge in &self.hyperedges {
            all_vars.insert(edge.target);
            for s in &edge.sources {
                all_vars.insert(*s);
            }
        }

        // DFS
        fn dfs(
            v: Variable,
            out_edges: &HashMap<Variable, HashSet<Variable>>,
            color: &mut HashMap<Variable, Color>,
            parent: &mut HashMap<Variable, Variable>,
            path: &mut Vec<Variable>,
        ) -> Option<Vec<Variable>> {
            color.insert(v, Color::Gray);
            path.push(v);

            if let Some(neighbors) = out_edges.get(&v) {
                for &u in neighbors {
                    match color.get(&u).unwrap_or(&Color::White) {
                        Color::White => {
                            parent.insert(u, v);
                            if let Some(cycle) = dfs(u, out_edges, color, parent, path) {
                                return Some(cycle);
                            }
                        }
                        Color::Gray => {
                            // Found cycle! Reconstruct path
                            let cycle_start_idx = path.iter().position(|&x| x == u).unwrap();
                            let mut cycle_path = path[cycle_start_idx..].to_vec();
                            cycle_path.push(u); // Close the cycle
                            return Some(cycle_path);
                        }
                        Color::Black => {} // Already finished, no cycle through here
                    }
                }
            }

            color.insert(v, Color::Black);
            path.pop();
            None
        }

        // Run DFS from each unvisited node
        for start in all_vars {
            if color.get(&start).unwrap_or(&Color::White) == &Color::White {
                let mut path = Vec::new();
                if let Some(cycle) = dfs(start, &out_edges, &mut color, &mut parent, &mut path) {
                    return Some(CyclePath { path: cycle });
                }
            }
        }

        None
    }

    /// Check if the hypergraph is acyclic.
    pub fn is_acyclic(&self) -> bool {
        self.detect_cycle().is_none()
    }

    /// Add hyperedge with cycle check.
    ///
    /// Returns Err with the cycle path if adding this edge would create a cycle.
    pub fn try_add_hyperedge(&mut self, edge: HyperEdge) -> Result<(), CycleError> {
        // Temporarily add
        self.add_hyperedge(edge.clone());
        
        if let Some(cycle) = self.detect_cycle() {
            // Remove the edge we just added
            self.hyperedges.pop();
            Err(CycleError {
                message: format!(
                    "Adding edge {:?} -> {:?} would create cycle",
                    edge.sources, edge.target
                ),
                cycle_path: cycle,
            })
        } else {
            Ok(())
        }
    }
}

/// A cycle path in the hypergraph
#[derive(Debug, Clone)]
pub struct CyclePath {
    /// Sequence of variables forming the cycle (last = first for closed loop)
    pub path: Vec<Variable>,
}

impl std::fmt::Display for CyclePath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<String> = self.path.iter().map(|v| format!("{:?}", v)).collect();
        write!(f, "{}", names.join(" → "))
    }
}

/// Error when a cycle would be created
#[derive(Debug, Clone)]
pub struct CycleError {
    pub message: String,
    pub cycle_path: CyclePath,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.message, self.cycle_path)
    }
}

impl std::error::Error for CycleError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperedge_creation() {
        let edge = HyperEdge::new(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        assert_eq!(edge.order, 2);
        assert_eq!(edge.target, Variable::HeartRateVariability);
    }

    #[test]
    fn test_interaction_computation() {
        let edge = HyperEdge::new(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        let mut state = vec![0.0; Variable::COUNT];
        state[Variable::HeartRate.index()] = 0.8;
        state[Variable::RespiratoryRate.index()] = 0.6;

        let interaction = edge.compute_interaction(&state);
        assert!((interaction - 0.48).abs() < 0.01); // 0.8 * 0.6
    }

    #[test]
    fn test_hypergraph_creation() {
        let graph = CausalHypergraph::new();
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.max_order(), 2);
    }

    #[test]
    fn test_add_hyperedge() {
        let mut graph = CausalHypergraph::new();

        let edge = HyperEdge::new(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        graph.add_hyperedge(edge);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_edges_to_target() {
        let mut graph = CausalHypergraph::new();

        // Add edge to HRV
        graph.add_interaction(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        // Add edge to different target
        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::CognitiveLoad,
            CausalEdge::zero(),
        );

        let hrv_edges = graph.edges_to(Variable::HeartRateVariability);
        assert_eq!(hrv_edges.len(), 1);

        let load_edges = graph.edges_to(Variable::CognitiveLoad);
        assert_eq!(load_edges.len(), 1);
    }

    #[test]
    fn test_predict_effect() {
        let mut graph = CausalHypergraph::new();

        let mut weight = CausalEdge::zero();
        weight.successes = 8;
        weight.failures = 2;

        graph.add_interaction(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            weight,
        );

        let mut state = vec![0.0; Variable::COUNT];
        state[Variable::HeartRate.index()] = 0.8;
        state[Variable::RespiratoryRate.index()] = 0.5;

        let effect = graph.predict_effect(&state, Variable::HeartRateVariability);

        // Effect = 0.8 * 0.5 * (8/10) = 0.32
        assert!((effect - 0.32).abs() < 0.01);
    }

    #[test]
    fn test_predict_effect_uncertain() {
        let mut graph = CausalHypergraph::new();

        let mut weight = CausalEdge::zero();
        weight.successes = 40;
        weight.failures = 10;

        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::HeartRateVariability,
            weight,
        );

        let mut state = vec![0.0; Variable::COUNT];
        state[Variable::HeartRate.index()] = 0.6;

        let prediction = graph.predict_effect_uncertain(&state, Variable::HeartRateVariability);

        // Should have high confidence (50 observations)
        assert!(prediction.confidence > 0.8);
        assert_eq!(prediction.source, "Hypergraph (1 edges)");
    }

    #[test]
    fn test_three_way_interaction() {
        let mut graph = CausalHypergraph::new();

        // Three-way interaction: HR * RR * Valence → Stress
        graph.add_interaction(
            vec![
                Variable::HeartRate,
                Variable::RespiratoryRate,
                Variable::EmotionalValence,
            ],
            Variable::CognitiveLoad, // Using as proxy for stress
            CausalEdge::zero(),
        );

        assert_eq!(graph.max_order(), 3);

        let mut state = vec![0.0; Variable::COUNT];
        state[Variable::HeartRate.index()] = 0.8;
        state[Variable::RespiratoryRate.index()] = 0.7;
        state[Variable::EmotionalValence.index()] = 0.5;

        let edges = graph.edges_to(Variable::CognitiveLoad);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].order, 3);

        let interaction = edges[0].compute_interaction(&state);
        assert!((interaction - 0.28).abs() < 0.01); // 0.8 * 0.7 * 0.5
    }

    #[test]
    fn test_edges_involving() {
        let mut graph = CausalHypergraph::new();

        graph.add_interaction(
            vec![Variable::HeartRate, Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::CognitiveLoad,
            CausalEdge::zero(),
        );

        let hr_edges = graph.edges_involving(Variable::HeartRate);
        assert_eq!(hr_edges.len(), 2); // HR appears in both edges

        let rr_edges = graph.edges_involving(Variable::RespiratoryRate);
        assert_eq!(rr_edges.len(), 1); // RR only in first edge
    }

    // =========================================================================
    // Cycle Detection Tests
    // =========================================================================

    #[test]
    fn test_acyclic_graph() {
        let mut graph = CausalHypergraph::new();

        // Simple chain: HR -> RR -> HRV (no cycle)
        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::RespiratoryRate,
            CausalEdge::zero(),
        );
        graph.add_interaction(
            vec![Variable::RespiratoryRate],
            Variable::HeartRateVariability,
            CausalEdge::zero(),
        );

        assert!(graph.is_acyclic());
        assert!(graph.detect_cycle().is_none());
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = CausalHypergraph::new();

        // Create cycle: HR -> RR -> HR
        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::RespiratoryRate,
            CausalEdge::zero(),
        );
        graph.add_interaction(
            vec![Variable::RespiratoryRate],
            Variable::HeartRate,
            CausalEdge::zero(),
        );

        assert!(!graph.is_acyclic());
        let cycle = graph.detect_cycle();
        assert!(cycle.is_some());
        
        let path = cycle.unwrap();
        println!("Detected cycle: {}", path);
        assert!(path.path.len() >= 2); // At least HR -> RR -> HR
    }

    #[test]
    fn test_try_add_hyperedge_rejects_cycle() {
        let mut graph = CausalHypergraph::new();

        // Add first edge
        graph.add_interaction(
            vec![Variable::HeartRate],
            Variable::RespiratoryRate,
            CausalEdge::zero(),
        );

        // Try to add edge that creates cycle
        let cyclic_edge = HyperEdge::new(
            vec![Variable::RespiratoryRate],
            Variable::HeartRate,
            CausalEdge::zero(),
        );

        let result = graph.try_add_hyperedge(cyclic_edge);
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        println!("Cycle error: {}", err);
        assert!(err.message.contains("would create cycle"));
        
        // Graph should still have only 1 edge
        assert_eq!(graph.edge_count(), 1);
    }
}
