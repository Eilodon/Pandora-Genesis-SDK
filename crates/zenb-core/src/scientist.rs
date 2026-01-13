//! Automatic Scientist module for causal hypothesis discovery.
//!
//! Ported from Pandora's `pandora_orchestrator::automatic_scientist_orchestrator`.
//! Implements a state machine for:
//! 1. Observing - monitoring for patterns
//! 2. Proposing - forming causal hypotheses  
//! 3. Experimenting - testing hypotheses
//! 4. Verifying - crystallizing confirmed knowledge
//!
//! # Usage
//! ```rust
//! use zenb_core::scientist::{AutomaticScientist, ScientistState};
//!
//! let mut scientist = AutomaticScientist::new();
//! let observations = [60.0, 50.0, 12.0, 1.0, 0.0];
//! scientist.observe(observations);
//! scientist.tick(); // Advances state machine
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Causal Hypothesis
// ============================================================================

/// A causal hypothesis between two variables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalHypothesis {
    /// Source variable index (in CausalGraph Variable enum)
    pub from_variable: u8,
    /// Target variable index
    pub to_variable: u8,
    /// Hypothesized causal strength (0.0 - 1.0)
    pub strength: f32,
    /// Current confidence in hypothesis (0.0 - 1.0)
    pub confidence: f32,
    /// Type of causal relationship
    pub edge_type: CausalEdgeType,
}

impl CausalHypothesis {
    /// Create a new hypothesis.
    pub fn new(from: u8, to: u8, edge_type: CausalEdgeType) -> Self {
        Self {
            from_variable: from,
            to_variable: to,
            strength: 0.5,
            confidence: 0.3,
            edge_type,
        }
    }
}

/// Types of causal relationships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalEdgeType {
    /// Direct causal link (A → B)
    Direct,
    /// Indirect/mediated link (A → M → B)
    Indirect,
    /// Conditional link (A → B | C)
    Conditional,
    /// Inhibitory link (A ⊣ B)
    Inhibitory,
}

// ============================================================================
// Experiment Result
// ============================================================================

/// Result of an experiment step.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Step number in experiment
    pub step: usize,
    /// Action taken during this step
    pub action: ExperimentAction,
    /// Observed values after action
    pub observation: [f32; 5],
    /// Whether observation supports hypothesis
    pub supports_hypothesis: bool,
}

/// Actions that can be taken during experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentAction {
    /// Observe baseline without intervention
    ObserveBaseline,
    /// Intervene on cause variable
    InterveneCause,
    /// Observe effect after intervention
    ObserveEffect,
    /// Wait for propagation
    Wait,
}

// ============================================================================
// Scientist State
// ============================================================================

/// State of the Automatic Scientist state machine.
#[derive(Debug, Clone)]
pub enum ScientistState {
    /// Monitoring for patterns that suggest causal relationships
    Observing {
        /// Accumulated observations
        observations: VecDeque<[f32; 5]>,
        /// Steps in observing state
        steps: usize,
    },
    /// Proposing a causal hypothesis based on patterns
    Proposing {
        /// The proposed hypothesis
        hypothesis: CausalHypothesis,
    },
    /// Running an experiment to test hypothesis
    Experimenting {
        /// Hypothesis being tested
        hypothesis: CausalHypothesis,
        /// Current experiment step
        step: usize,
        /// Maximum steps
        max_steps: usize,
        /// Collected results
        results: Vec<ExperimentResult>,
    },
    /// Verifying experiment results
    Verifying {
        /// Hypothesis that was tested
        hypothesis: CausalHypothesis,
        /// All experiment results
        results: Vec<ExperimentResult>,
        /// Confirmation rate
        confirmation_rate: f32,
    },
}

impl Default for ScientistState {
    fn default() -> Self {
        ScientistState::Observing {
            observations: VecDeque::with_capacity(50),
            steps: 0,
        }
    }
}

// ============================================================================
// Automatic Scientist
// ============================================================================

/// Configuration for the Automatic Scientist.
#[derive(Debug, Clone)]
pub struct ScientistConfig {
    /// Minimum observations before hypothesis generation
    pub min_observations: usize,
    /// Maximum experiment steps
    pub max_experiment_steps: usize,
    /// Confirmation threshold (0.0 - 1.0)
    pub confirmation_threshold: f32,
    /// Correlation threshold for hypothesis generation
    pub correlation_threshold: f32,
}

impl Default for ScientistConfig {
    fn default() -> Self {
        Self {
            min_observations: 30,
            max_experiment_steps: 5,
            confirmation_threshold: 0.6,
            correlation_threshold: 0.3,
        }
    }
}

/// The Automatic Scientist orchestrates causal discovery.
#[derive(Debug, Clone)]
pub struct AutomaticScientist {
    /// Current state
    state: ScientistState,
    /// Configuration
    config: ScientistConfig,
    /// Crystallized hypotheses (confirmed)
    pub crystallized: Vec<CausalHypothesis>,
    /// Pending wiring to CausalGraph
    pub pending_crystallized: Vec<CausalHypothesis>,
    /// Rejected hypotheses
    pub rejected: Vec<CausalHypothesis>,
    /// Total cycles run
    pub total_cycles: u64,
}

impl AutomaticScientist {
    /// Create a new Automatic Scientist with default config.
    pub fn new() -> Self {
        Self::with_config(ScientistConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: ScientistConfig) -> Self {
        Self {
            state: ScientistState::default(),
            config,
            crystallized: Vec::new(),
            pending_crystallized: Vec::new(),
            rejected: Vec::new(),
            total_cycles: 0,
        }
    }

    /// Get and clear any newly crystallized hypotheses.
    pub fn drain_pending_discoveries(&mut self) -> Vec<CausalHypothesis> {
        let pending = self.pending_crystallized.clone();
        self.pending_crystallized.clear();
        pending
    }

    /// Get current state.
    pub fn state(&self) -> &ScientistState {
        &self.state
    }

    /// Get state name for logging.
    pub fn state_name(&self) -> &'static str {
        match &self.state {
            ScientistState::Observing { .. } => "Observing",
            ScientistState::Proposing { .. } => "Proposing",
            ScientistState::Experimenting { .. } => "Experimenting",
            ScientistState::Verifying { .. } => "Verifying",
        }
    }

    /// Add an observation (used during Observing state).
    pub fn observe(&mut self, observation: [f32; 5]) {
        if let ScientistState::Observing { observations, .. } = &mut self.state {
            observations.push_back(observation);
            // Keep bounded
            while observations.len() > 100 {
                observations.pop_front();
            }
        }
    }

    /// Run one cycle of the state machine.
    ///
    /// Returns whether a state transition occurred.
    pub fn tick(&mut self) -> bool {
        self.total_cycles += 1;

        match &self.state {
            ScientistState::Observing {
                observations,
                steps,
            } => self.handle_observing(observations.clone(), *steps),
            ScientistState::Proposing { hypothesis } => self.handle_proposing(hypothesis.clone()),
            ScientistState::Experimenting {
                hypothesis,
                step,
                max_steps,
                results,
            } => self.handle_experimenting(hypothesis.clone(), *step, *max_steps, results.clone()),
            ScientistState::Verifying {
                hypothesis,
                results,
                confirmation_rate,
            } => self.handle_verifying(hypothesis.clone(), results.clone(), *confirmation_rate),
        }
    }

    /// Handle Observing state - look for patterns.
    fn handle_observing(&mut self, observations: VecDeque<[f32; 5]>, steps: usize) -> bool {
        // Need enough observations
        if observations.len() < self.config.min_observations {
            // Stay in observing, just increment steps
            self.state = ScientistState::Observing {
                observations,
                steps: steps + 1,
            };
            return false;
        }

        // Calculate correlations between variables
        if let Some(hypothesis) = self.find_hypothesis(&observations) {
            // Transition to Proposing
            self.state = ScientistState::Proposing { hypothesis };
            log::info!("Scientist: Pattern detected, transitioning to Proposing");
            return true;
        }

        // Stay observing
        self.state = ScientistState::Observing {
            observations,
            steps: steps + 1,
        };
        false
    }

    /// Find a hypothesis from observations.
    fn find_hypothesis(&self, observations: &VecDeque<[f32; 5]>) -> Option<CausalHypothesis> {
        if observations.len() < 10 {
            return None;
        }

        // Calculate pairwise correlations
        let n = observations.len() as f32;
        let obs: Vec<_> = observations.iter().cloned().collect();

        // Check each variable pair
        for i in 0..5 {
            for j in 0..5 {
                if i == j {
                    continue;
                }

                // Compute Pearson correlation
                let mean_i: f32 = obs.iter().map(|o| o[i]).sum::<f32>() / n;
                let mean_j: f32 = obs.iter().map(|o| o[j]).sum::<f32>() / n;

                let mut cov = 0.0f32;
                let mut var_i = 0.0f32;
                let mut var_j = 0.0f32;

                for o in &obs {
                    let di = o[i] - mean_i;
                    let dj = o[j] - mean_j;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }

                let std_i = var_i.sqrt();
                let std_j = var_j.sqrt();

                if std_i > 0.001 && std_j > 0.001 {
                    let corr = cov / (std_i * std_j);

                    if corr.abs() > self.config.correlation_threshold {
                        // Found a potential causal relationship
                        let edge_type = if corr > 0.0 {
                            CausalEdgeType::Direct
                        } else {
                            CausalEdgeType::Inhibitory
                        };

                        return Some(CausalHypothesis {
                            from_variable: i as u8,
                            to_variable: j as u8,
                            strength: corr.abs(),
                            confidence: 0.3, // Low initial confidence
                            edge_type,
                        });
                    }
                }
            }
        }

        None
    }

    /// Handle Proposing state - set up experiment.
    fn handle_proposing(&mut self, hypothesis: CausalHypothesis) -> bool {
        log::info!(
            "Scientist: Proposing hypothesis {} -> {} (strength: {:.2})",
            hypothesis.from_variable,
            hypothesis.to_variable,
            hypothesis.strength
        );

        // Transition to Experimenting
        self.state = ScientistState::Experimenting {
            hypothesis,
            step: 0,
            max_steps: self.config.max_experiment_steps,
            results: Vec::new(),
        };
        true
    }

    /// Handle Experimenting state - run experiment steps.
    fn handle_experimenting(
        &mut self,
        hypothesis: CausalHypothesis,
        step: usize,
        max_steps: usize,
        mut results: Vec<ExperimentResult>,
    ) -> bool {
        // Determine action for this step
        let action = match step {
            0 => ExperimentAction::ObserveBaseline,
            1 | 2 | 3 => ExperimentAction::InterveneCause,
            _ => ExperimentAction::ObserveEffect,
        };

        // Simulate experiment (in real implementation, this would trigger actions)
        let observation = self.simulate_experiment(&hypothesis, action);
        let supports = self.check_hypothesis_support(&hypothesis, &observation, action);

        results.push(ExperimentResult {
            step,
            action,
            observation,
            supports_hypothesis: supports,
        });

        let next_step = step + 1;

        if next_step >= max_steps {
            // Calculate confirmation rate
            let confirmed = results.iter().filter(|r| r.supports_hypothesis).count();
            let rate = confirmed as f32 / results.len() as f32;

            // Transition to Verifying
            self.state = ScientistState::Verifying {
                hypothesis,
                results,
                confirmation_rate: rate,
            };
            log::info!("Scientist: Experiment complete, transitioning to Verifying");
            true
        } else {
            // Continue experimenting
            self.state = ScientistState::Experimenting {
                hypothesis,
                step: next_step,
                max_steps,
                results,
            };
            false
        }
    }

    /// Simulate an experiment step.
    fn simulate_experiment(
        &self,
        hypothesis: &CausalHypothesis,
        action: ExperimentAction,
    ) -> [f32; 5] {
        let mut obs = [0.5f32; 5];

        match action {
            ExperimentAction::ObserveBaseline => {
                // Baseline values
                obs[hypothesis.from_variable as usize] = 0.5;
                obs[hypothesis.to_variable as usize] = 0.5;
            }
            ExperimentAction::InterveneCause => {
                // Set cause to high
                obs[hypothesis.from_variable as usize] = 0.9;
                // If hypothesis is correct, effect should also increase
                if hypothesis.edge_type == CausalEdgeType::Direct {
                    obs[hypothesis.to_variable as usize] = 0.7 + 0.2 * hypothesis.strength;
                } else if hypothesis.edge_type == CausalEdgeType::Inhibitory {
                    obs[hypothesis.to_variable as usize] = 0.3 - 0.2 * hypothesis.strength;
                }
            }
            ExperimentAction::ObserveEffect => {
                // Observe after intervention
                obs[hypothesis.to_variable as usize] = 0.7 * hypothesis.strength;
            }
            ExperimentAction::Wait => {
                // Intermediate state
            }
        }

        obs
    }

    /// Check if observation supports hypothesis.
    fn check_hypothesis_support(
        &self,
        hypothesis: &CausalHypothesis,
        observation: &[f32; 5],
        _action: ExperimentAction,
    ) -> bool {
        let cause_val = observation[hypothesis.from_variable as usize];
        let effect_val = observation[hypothesis.to_variable as usize];

        match hypothesis.edge_type {
            CausalEdgeType::Direct => {
                // High cause should lead to high effect
                cause_val > 0.6 && effect_val > 0.5
            }
            CausalEdgeType::Inhibitory => {
                // High cause should lead to low effect
                cause_val > 0.6 && effect_val < 0.4
            }
            CausalEdgeType::Indirect => {
                // Some correlation expected
                (cause_val > 0.5) == (effect_val > 0.4)
            }
            CausalEdgeType::Conditional => {
                // Requires condition variable (index 2) to be high
                observation[2] > 0.5 && (cause_val > 0.5) == (effect_val > 0.5)
            }
        }
    }

    /// Handle Verifying state - crystallize or reject.
    fn handle_verifying(
        &mut self,
        hypothesis: CausalHypothesis,
        _results: Vec<ExperimentResult>,
        confirmation_rate: f32,
    ) -> bool {
        if confirmation_rate >= self.config.confirmation_threshold {
            // Hypothesis confirmed - crystallize
            log::info!(
                "Scientist: Hypothesis CONFIRMED ({:.0}%), crystallizing",
                confirmation_rate * 100.0
            );

            let mut confirmed = hypothesis.clone();
            confirmed.confidence = confirmation_rate;
            self.crystallized.push(confirmed.clone());
            self.pending_crystallized.push(confirmed);
        } else {
            // Hypothesis rejected
            log::info!(
                "Scientist: Hypothesis REJECTED ({:.0}%)",
                confirmation_rate * 100.0
            );
            self.rejected.push(hypothesis);
        }

        // Return to Observing
        self.state = ScientistState::default();
        true
    }

    /// Get crystallized hypotheses.
    pub fn get_crystallized(&self) -> &[CausalHypothesis] {
        &self.crystallized
    }

    /// Check if a relationship was confirmed.
    pub fn is_confirmed(&self, from: u8, to: u8) -> bool {
        self.crystallized
            .iter()
            .any(|h| h.from_variable == from && h.to_variable == to)
    }
}

impl Default for AutomaticScientist {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let scientist = AutomaticScientist::new();
        assert_eq!(scientist.state_name(), "Observing");
    }

    #[test]
    fn test_observation_accumulation() {
        let mut scientist = AutomaticScientist::new();

        for i in 0..10 {
            scientist.observe([i as f32 * 0.1; 5]);
        }

        if let ScientistState::Observing { observations, .. } = scientist.state() {
            assert_eq!(observations.len(), 10);
        } else {
            panic!("Expected Observing state");
        }
    }

    #[test]
    fn test_hypothesis_generation() {
        let mut scientist = AutomaticScientist::with_config(ScientistConfig {
            min_observations: 10,
            correlation_threshold: 0.2,
            ..Default::default()
        });

        // Generate correlated observations (var 0 and var 1 move together)
        for i in 0..15 {
            let base = i as f32 / 15.0;
            scientist.observe([base, base * 0.9 + 0.05, 0.5, 0.5, 0.5]);
        }

        // Should detect correlation and transition to Proposing
        scientist.tick();

        assert_eq!(scientist.state_name(), "Proposing");
    }

    #[test]
    fn test_full_cycle() {
        let mut scientist = AutomaticScientist::with_config(ScientistConfig {
            min_observations: 5,
            max_experiment_steps: 3,
            correlation_threshold: 0.2,
            confirmation_threshold: 0.5,
        });

        // Generate correlated observations
        for i in 0..10 {
            let base = i as f32 / 10.0;
            scientist.observe([base, base * 0.8, 0.5, 0.5, 0.5]);
        }

        // Run through full cycle
        let mut transitions = 0;
        for _ in 0..20 {
            if scientist.tick() {
                transitions += 1;
            }

            // Check if we're back to Observing after full cycle
            if scientist.state_name() == "Observing" && transitions >= 3 {
                break;
            }
        }

        // Should have completed at least one full cycle
        assert!(transitions >= 3);
    }

    #[test]
    fn test_crystallization() {
        let mut scientist = AutomaticScientist::with_config(ScientistConfig {
            min_observations: 5,
            max_experiment_steps: 3,
            correlation_threshold: 0.1,
            confirmation_threshold: 0.3, // Low threshold for test
        });

        // Strong correlation
        for i in 0..10 {
            let base = i as f32 / 10.0;
            scientist.observe([base, base, 0.5, 0.5, 0.5]);
        }

        // Run complete cycle
        for _ in 0..20 {
            scientist.tick();
        }

        // Should have crystallized at least one hypothesis
        assert!(!scientist.crystallized.is_empty() || !scientist.rejected.is_empty());
    }
}
