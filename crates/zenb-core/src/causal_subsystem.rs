//! Causal Subsystem: Encapsulated causal reasoning for the Engine.
//!
//! This module extracts causal-related concerns from the Engine god object,
//! providing a clean interface for causal graph management, observation
//! buffering, and automatic causal discovery.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           CausalSubsystem               │
//! │  ┌───────────────┐  ┌───────────────┐  │
//! │  │  CausalGraph  │  │  Scientist    │  │
//! │  │  (structure)  │  │  (discovery)  │  │
//! │  └───────────────┘  └───────────────┘  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Extracted from Engine
//! This subsystem replaces the following Engine fields:
//! - `causal_graph`
//! - `scientist`
//! - `observation_buffer_min_samples`

use crate::causal::{CausalEdge, CausalGraph, Variable};
use crate::config::ZenbConfig;
use crate::domain::Observation;
use crate::scientist::{AutomaticScientist, CausalHypothesis};

/// Encapsulated causal reasoning subsystem.
///
/// Provides a single point of control for all causal-related state and operations,
/// reducing the Engine's field count and improving cohesion.
#[derive(Debug)]
pub struct CausalSubsystem {
    /// The causal graph structure (learned and prior relationships)
    pub graph: CausalGraph,

    /// Automatic scientist for hypothesis generation
    scientist: AutomaticScientist,

    /// Latest observation for context queries
    pub last_observation: Option<Observation>,

    /// Discovery interval tracking
    last_discovery_ts: u64,
    discovery_interval_us: u64,
}

impl Default for CausalSubsystem {
    fn default() -> Self {
        Self::new(&ZenbConfig::default())
    }
}

impl CausalSubsystem {
    /// Create a new CausalSubsystem with the given configuration.
    pub fn new(_cfg: &ZenbConfig) -> Self {
        Self {
            graph: CausalGraph::with_priors(),
            scientist: AutomaticScientist::new(),
            last_observation: None,
            last_discovery_ts: 0,
            discovery_interval_us: 60_000_000, // 60 seconds
        }
    }

    // =========================================================================
    // Observation and Discovery
    // =========================================================================

    /// Add an observation with features.
    ///
    /// - `features`: [hr, hrv, rr, quality, motion] (for scientist)
    /// - `snapshot`: Full observation for context/storage
    pub fn observe(&mut self, features: [f32; 5], snapshot: Option<Observation>) {
        self.last_observation = snapshot;
        self.scientist.observe(features);
    }

    /// Run discovery tick (call periodically, e.g., every second).
    ///
    /// Returns true if state machine transitioned.
    pub fn tick(&mut self) -> bool {
        self.scientist.tick()
    }

    /// Drain pending discoveries (moves metadata to caller).
    /// Drain pending discoveries (moves metadata to caller).
    pub fn drain_discoveries(&mut self) -> Vec<CausalHypothesis> {
        let discoveries = self.scientist.drain_pending_discoveries();

        // Wire discoveries to graph automatically
        for h in &discoveries {
            self.wire_hypothesis(h);
        }

        discoveries
    }

    /// Access internal scientist for diagnostics/tests
    pub fn scientist(&self) -> &AutomaticScientist {
        &self.scientist
    }

    // Internal helper to wire hypothesis to graph
    fn wire_hypothesis(&mut self, hypo: &CausalHypothesis) {
        if hypo.confidence > 0.5 {
            if let (Some(cause), Some(effect)) = (
                Self::index_to_variable(hypo.from_variable),
                Self::index_to_variable(hypo.to_variable),
            ) {
                let weight = (hypo.strength * 100.0) as u32;
                let edge = CausalEdge::prior(weight, 100 - weight, "Auto-Scientist Discovery");
                self.graph.set_link(cause, effect, edge);
            }
        }
    }

    /// Map observation index to Variable enum.
    /// Format: [hr, hrv, rr, quality, motion]
    fn index_to_variable(idx: u8) -> Option<Variable> {
        match idx {
            0 => Some(Variable::HeartRate),
            1 => Some(Variable::HeartRateVariability),
            2 => Some(Variable::RespiratoryRate),
            // 3 (Quality) and 4 (Motion) are not yet in CausalGraph
            _ => None,
        }
    }

    // =========================================================================
    // Graph Queries
    // =========================================================================

    /// Get effect of one variable on another.
    #[inline]
    pub fn effect(&self, cause: Variable, effect: Variable) -> f32 {
        self.graph.get_effect(cause, effect)
    }

    /// Get all causes of a variable.
    #[inline]
    pub fn causes_of(&self, var: Variable) -> Vec<(Variable, f32)> {
        self.graph.get_causes(var)
    }

    /// Get all effects of a variable.
    #[inline]
    pub fn effects_of(&self, var: Variable) -> Vec<(Variable, f32)> {
        self.graph.get_effects(var)
    }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// Get scientist state name.
    pub fn scientist_state(&self) -> &'static str {
        self.scientist.state_name()
    }

    /// Get number of crystallized hypotheses.
    pub fn crystallized_count(&self) -> usize {
        self.scientist.crystallized.len()
    }

    /// Get crystallized hypotheses.
    pub fn crystallized(&self) -> &[CausalHypothesis] {
        self.scientist.get_crystallized()
    }

    /// Get total cycles run.
    pub fn total_cycles(&self) -> u64 {
        self.scientist.total_cycles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_subsystem_default() {
        let cs = CausalSubsystem::default();
        assert_eq!(cs.scientist_state(), "Observing");
        assert_eq!(cs.crystallized_count(), 0);
    }

    #[test]
    fn test_observe_and_tick() {
        let mut cs = CausalSubsystem::default();

        // Add some observations
        for i in 0..5 {
            cs.observe([i as f32 / 5.0; 5], None);
        }

        // Tick should return false (not enough observations)
        assert!(!cs.tick());
        assert_eq!(cs.scientist_state(), "Observing");
    }

    #[test]
    fn test_graph_queries() {
        let cs = CausalSubsystem::default();

        // Should have priors from with_priors()
        let effect = cs.effect(Variable::HeartRate, Variable::RespiratoryRate);
        // Default priors vary, just check it returns a value
        assert!(effect.is_finite());
    }
}
