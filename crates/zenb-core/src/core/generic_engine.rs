//! Generic Engine: Domain-agnostic adaptive control engine.
//!
//! This module provides a generic engine wrapper that works with any domain
//! implementing the `Domain` trait. It provides the core control loop
//! abstraction while delegating domain-specific behavior to trait implementations.

use crate::core::{
    ActionKind, BeliefMode, Domain, GenericBeliefState, GenericCausalGraph, OscillatorConfig,
    SignalVariable,
};
use std::marker::PhantomData;

/// Generic observation trait for domain inputs.
///
/// Observations are the raw sensor/data inputs that the engine processes.
/// Each domain defines its own observation type.
pub trait Observation: Clone + Send + Sync + 'static {
    /// Extract normalized variable values [0, 1] from this observation.
    fn to_variable_values<V: SignalVariable>(&self) -> Vec<f32>;

    /// Observation timestamp in microseconds.
    fn timestamp_us(&self) -> i64;

    /// Quality/confidence of this observation [0, 1].
    fn quality(&self) -> f32 {
        1.0
    }
}

/// Generic control decision output.
#[derive(Debug, Clone)]
pub struct GenericControlDecision<A: ActionKind> {
    /// Selected action.
    pub action: A,
    /// Confidence in this decision [0, 1].
    pub confidence: f32,
    /// Whether the decision changed from the previous tick.
    pub changed: bool,
    /// Reason for denial if action was rejected by safety.
    pub deny_reason: Option<String>,
}

/// Generic Engine that works with any domain.
///
/// This is a lightweight wrapper providing domain-agnostic control loop
/// functionality. Domain-specific behavior is handled through trait bounds.
///
/// # Type Parameters
/// - `D`: Domain type implementing the `Domain` trait
///
/// # Example
/// ```rust,ignore
/// use zenb_core::core::{GenericEngine, GenericCausalGraph};
/// use zenb_core::domains::TradingDomain;
///
/// let engine: GenericEngine<TradingDomain> = GenericEngine::new();
/// ```
pub struct GenericEngine<D: Domain> {
    /// Causal graph for this domain.
    pub causal_graph: GenericCausalGraph<D::Variable>,

    /// Current belief state.
    pub belief_state: GenericBeliefState<D::Mode>,

    /// Domain configuration.
    pub config: D::Config,

    /// Last selected action.
    pub last_action: Option<D::Action>,

    /// Total ticks processed.
    pub tick_count: u64,

    /// Last observation timestamp.
    pub last_timestamp_us: i64,

    /// Domain marker.
    _domain: PhantomData<D>,
}

impl<D: Domain> Default for GenericEngine<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: Domain> GenericEngine<D> {
    /// Create a new generic engine with default configuration.
    pub fn new() -> Self {
        Self {
            causal_graph: GenericCausalGraph::with_priors(D::default_priors()),
            belief_state: GenericBeliefState::default(),
            config: D::Config::default(),
            last_action: None,
            tick_count: 0,
            last_timestamp_us: 0,
            _domain: PhantomData,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: D::Config) -> Self {
        Self {
            causal_graph: GenericCausalGraph::with_priors(D::default_priors()),
            belief_state: GenericBeliefState::default(),
            config,
            last_action: None,
            tick_count: 0,
            last_timestamp_us: 0,
            _domain: PhantomData,
        }
    }

    /// Get the domain name.
    pub fn domain_name(&self) -> &'static str {
        D::name()
    }

    /// Get current belief mode.
    pub fn current_mode(&self) -> D::Mode {
        self.belief_state.mode
    }

    /// Get belief distribution.
    pub fn belief_distribution(&self) -> &[f32] {
        &self.belief_state.distribution
    }

    /// Get confidence in current belief.
    pub fn confidence(&self) -> f32 {
        self.belief_state.confidence
    }

    /// Process an observation and update internal state.
    ///
    /// This is the main input method. Call this with each new observation.
    pub fn observe<O: Observation>(&mut self, obs: &O) {
        let vars = obs.to_variable_values::<D::Variable>();
        self.last_timestamp_us = obs.timestamp_us();
        self.tick_count += 1;

        // Update belief state based on observation
        // (Simple update - real impl would use Active Inference)
        self.update_belief(&vars, obs.quality());
    }

    /// Update belief state based on variable values.
    fn update_belief(&mut self, vars: &[f32], quality: f32) {
        // Simple belief update: adjust confidence based on observation quality
        self.belief_state.confidence = self.belief_state.confidence * 0.9 + quality * 0.1;

        // Update mode based on dominant signals (domain-specific logic would go here)
        self.belief_state.update_mode();
    }

    /// Select an action based on current belief state.
    ///
    /// This implements the basic action selection. Override via domain
    /// for more sophisticated EFE-based selection.
    pub fn select_action(&mut self) -> GenericControlDecision<D::Action> {
        let mode = self.belief_state.mode;

        // Try domain-specific mode-to-action mapping
        let action = D::mode_to_default_action(mode);

        let (action, deny_reason) = match action {
            Some(a) => (a, None),
            None => {
                // No action available for this mode
                return GenericControlDecision {
                    action: D::mode_to_default_action(D::Mode::default_mode())
                        .expect("Domain must provide default action"),
                    confidence: 0.0,
                    changed: false,
                    deny_reason: Some("No action mapping for mode".to_string()),
                };
            }
        };

        let changed = self.last_action.is_none()
            || self.last_action.as_ref().map(|a| a.type_id()) != Some(action.type_id());

        self.last_action = Some(action.clone());

        GenericControlDecision {
            action,
            confidence: self.belief_state.confidence,
            changed,
            deny_reason,
        }
    }

    /// Learn from the outcome of an action.
    ///
    /// Call this after executing an action and observing the result.
    pub fn learn_from_outcome<O: Observation>(&mut self, obs: &O, success: bool) {
        let vars = obs.to_variable_values::<D::Variable>();
        self.causal_graph.learn_from_outcome(&vars, success);

        // Adjust belief confidence based on outcome
        if success {
            self.belief_state.confidence = (self.belief_state.confidence + 0.1).min(1.0);
        } else {
            self.belief_state.confidence = (self.belief_state.confidence - 0.1).max(0.0);
        }
    }

    /// Get causal effect between two variables.
    pub fn get_causal_effect(&self, cause: D::Variable, effect: D::Variable) -> f32 {
        self.causal_graph.get_effect(cause, effect)
    }

    /// Get all causes for a variable.
    pub fn get_causes(&self, target: D::Variable) -> Vec<(D::Variable, f32)> {
        self.causal_graph.get_causes(target)
    }

    /// Diagnostic info.
    pub fn diagnostics(&self) -> GenericEngineDiagnostics<D> {
        GenericEngineDiagnostics {
            domain_name: D::name(),
            tick_count: self.tick_count,
            current_mode: self.belief_state.mode,
            confidence: self.belief_state.confidence,
            causal_edge_count: self.causal_graph.edge_count(),
            variable_count: D::Variable::count(),
            mode_count: D::Mode::count(),
        }
    }
}

/// Diagnostic information for a generic engine.
#[derive(Debug, Clone)]
pub struct GenericEngineDiagnostics<D: Domain> {
    pub domain_name: &'static str,
    pub tick_count: u64,
    pub current_mode: D::Mode,
    pub confidence: f32,
    pub causal_edge_count: usize,
    pub variable_count: usize,
    pub mode_count: usize,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::trading::{MarketMode, TradingAction, TradingDomain, TradingVariable};

    // Simple test observation
    #[derive(Clone)]
    struct TestObs {
        values: Vec<f32>,
        ts: i64,
    }

    impl Observation for TestObs {
        fn to_variable_values<V: SignalVariable>(&self) -> Vec<f32> {
            self.values.clone()
        }
        fn timestamp_us(&self) -> i64 {
            self.ts
        }
    }

    #[test]
    fn test_generic_engine_trading() {
        let mut engine: GenericEngine<TradingDomain> = GenericEngine::new();

        assert_eq!(engine.domain_name(), "trading");
        assert_eq!(engine.current_mode(), MarketMode::Sideways);
    }

    #[test]
    fn test_observe_and_select() {
        let mut engine: GenericEngine<TradingDomain> = GenericEngine::new();

        let obs = TestObs {
            values: vec![0.5; TradingVariable::count()],
            ts: 1000,
        };

        engine.observe(&obs);
        let decision = engine.select_action();

        assert!(decision.confidence >= 0.0);
        assert!(matches!(
            decision.action,
            TradingAction::Hold | TradingAction::Hedge { .. }
        ));
    }

    #[test]
    fn test_diagnostics() {
        let engine: GenericEngine<TradingDomain> = GenericEngine::new();
        let diag = engine.diagnostics();

        assert_eq!(diag.domain_name, "trading");
        assert_eq!(diag.variable_count, 7);
        assert_eq!(diag.mode_count, 5);
    }
}
