//! Decision Tree module for context-aware routing.
//!
//! Ported from Pandora's `pandora_orchestrator::decision_tree`.
//! Provides a simple decision tree for routing between different
//! processing strategies based on context.
//!
//! # Usage
//! ```rust
//! use zenb_core::decision_tree::{DecisionTree, DecisionContext, RouteAction};
//!
//! let tree = DecisionTree::default_for_zenb();
//! let context = DecisionContext::default();
//! let action = tree.decide(&context);
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Route Actions
// ============================================================================

/// Actions that can be taken based on decision tree evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RouteAction {
    /// Use full computation path (EFE, Vajra, etc.)
    #[default]
    FullComputation,
    /// Use simplified computation for low resources
    SimplifiedComputation,
    /// Skip processing and use cached result
    UseCached,
    /// Defer processing to later
    Defer,
    /// Fallback to safe defaults
    SafeFallback,
}

// ============================================================================
// Decision Context
// ============================================================================

/// Context for decision tree evaluation.
#[derive(Debug, Clone, Default)]
pub struct DecisionContext {
    /// Battery level (0.0 - 1.0), None if plugged in
    pub battery_level: Option<f32>,
    /// Available RAM in MB
    pub available_ram_mb: f32,
    /// Current CPU load (0.0 - 1.0)
    pub cpu_load: f32,
    /// Time since last computation in milliseconds
    pub time_since_last_compute_ms: u64,
    /// Current belief confidence (0.0 - 1.0)
    pub belief_confidence: f32,
    /// Is thermal throttling active?
    pub is_thermal_throttled: bool,
    /// Hour of day (0-23)
    pub hour_of_day: u8,
    /// Is high-priority context (user actively engaged)?
    pub is_high_priority: bool,
    /// Consecutive failures count
    pub failure_count: u32,
}

impl DecisionContext {
    /// Create context from minimal parameters.
    pub fn new(battery: Option<f32>, ram_mb: f32, confidence: f32) -> Self {
        Self {
            battery_level: battery,
            available_ram_mb: ram_mb,
            belief_confidence: confidence,
            cpu_load: 0.3,
            time_since_last_compute_ms: 100,
            is_thermal_throttled: false,
            hour_of_day: 12,
            is_high_priority: false,
            failure_count: 0,
        }
    }

    /// Is battery critically low (< 10%)?
    pub fn is_battery_critical(&self) -> bool {
        self.battery_level.map(|b| b < 0.1).unwrap_or(false)
    }

    /// Is battery low (< 20%)?
    pub fn is_battery_low(&self) -> bool {
        self.battery_level.map(|b| b < 0.2).unwrap_or(false)
    }

    /// Is RAM constrained (< 256 MB)?
    pub fn is_ram_constrained(&self) -> bool {
        self.available_ram_mb < 256.0
    }

    /// Should we use power-saving mode?
    pub fn should_save_power(&self) -> bool {
        self.is_battery_low() || self.is_thermal_throttled
    }
}

// ============================================================================
// Decision Node
// ============================================================================

/// A node in the decision tree.
#[derive(Debug, Clone)]
pub enum DecisionNode {
    /// Internal node with condition
    Branch {
        /// Condition to evaluate
        condition: Condition,
        /// Node if condition is true
        if_true: Box<DecisionNode>,
        /// Node if condition is false
        if_false: Box<DecisionNode>,
    },
    /// Leaf node with final action
    Leaf {
        action: RouteAction,
        confidence: f32,
    },
}

/// Conditions for branching.
#[derive(Debug, Clone)]
pub enum Condition {
    /// Battery below threshold
    BatteryBelow(f32),
    /// RAM below threshold (MB)
    RamBelow(f32),
    /// CPU load above threshold
    CpuAbove(f32),
    /// Confidence below threshold
    ConfidenceBelow(f32),
    /// Thermal throttling active
    ThermalThrottled,
    /// Is high priority context
    HighPriority,
    /// Time since last compute above threshold (ms)
    TimeSinceComputeAbove(u64),
    /// Failure count above threshold
    FailureCountAbove(u32),
    /// Night time (22:00 - 06:00)
    IsNightTime,
}

impl Condition {
    /// Evaluate condition against context.
    pub fn evaluate(&self, ctx: &DecisionContext) -> bool {
        match self {
            Condition::BatteryBelow(threshold) => {
                ctx.battery_level.map(|b| b < *threshold).unwrap_or(false)
            }
            Condition::RamBelow(threshold) => ctx.available_ram_mb < *threshold,
            Condition::CpuAbove(threshold) => ctx.cpu_load > *threshold,
            Condition::ConfidenceBelow(threshold) => ctx.belief_confidence < *threshold,
            Condition::ThermalThrottled => ctx.is_thermal_throttled,
            Condition::HighPriority => ctx.is_high_priority,
            Condition::TimeSinceComputeAbove(threshold) => {
                ctx.time_since_last_compute_ms > *threshold
            }
            Condition::FailureCountAbove(threshold) => ctx.failure_count > *threshold,
            Condition::IsNightTime => ctx.hour_of_day >= 22 || ctx.hour_of_day < 6,
        }
    }
}

// ============================================================================
// Decision Tree
// ============================================================================

/// Decision tree for routing computation strategies.
#[derive(Debug, Clone)]
pub struct DecisionTree {
    root: DecisionNode,
}

impl DecisionTree {
    /// Create a new decision tree with given root node.
    pub fn new(root: DecisionNode) -> Self {
        Self { root }
    }

    /// Create default decision tree for ZenB.
    ///
    /// Tree structure:
    /// ```text
    /// [FailureCount > 3?]
    ///     YES → SafeFallback
    ///     NO → [Battery < 10%?]
    ///         YES → SimplifiedComputation
    ///         NO → [ThermalThrottled?]
    ///             YES → SimplifiedComputation
    ///             NO → [RAM < 256MB?]
    ///                 YES → SimplifiedComputation
    ///                 NO → [HighPriority?]
    ///                     YES → FullComputation
    ///                     NO → [Confidence < 0.3?]
    ///                         YES → FullComputation (needs recalc)
    ///                         NO → [TimeSinceCompute > 5000ms?]
    ///                             YES → FullComputation
    ///                             NO → UseCached
    /// ```
    pub fn default_for_zenb() -> Self {
        use Condition::*;
        use DecisionNode::*;
        use RouteAction::*;

        Self::new(Branch {
            condition: FailureCountAbove(3),
            if_true: Box::new(Leaf {
                action: SafeFallback,
                confidence: 0.9,
            }),
            if_false: Box::new(Branch {
                condition: BatteryBelow(0.1),
                if_true: Box::new(Leaf {
                    action: SimplifiedComputation,
                    confidence: 0.85,
                }),
                if_false: Box::new(Branch {
                    condition: ThermalThrottled,
                    if_true: Box::new(Leaf {
                        action: SimplifiedComputation,
                        confidence: 0.8,
                    }),
                    if_false: Box::new(Branch {
                        condition: RamBelow(256.0),
                        if_true: Box::new(Leaf {
                            action: SimplifiedComputation,
                            confidence: 0.85,
                        }),
                        if_false: Box::new(Branch {
                            condition: HighPriority,
                            if_true: Box::new(Leaf {
                                action: FullComputation,
                                confidence: 0.95,
                            }),
                            if_false: Box::new(Branch {
                                condition: ConfidenceBelow(0.3),
                                if_true: Box::new(Leaf {
                                    action: FullComputation,
                                    confidence: 0.9,
                                }),
                                if_false: Box::new(Branch {
                                    condition: TimeSinceComputeAbove(5000),
                                    if_true: Box::new(Leaf {
                                        action: FullComputation,
                                        confidence: 0.8,
                                    }),
                                    if_false: Box::new(Leaf {
                                        action: UseCached,
                                        confidence: 0.7,
                                    }),
                                }),
                            }),
                        }),
                    }),
                }),
            }),
        })
    }

    /// Evaluate decision tree and return action.
    pub fn decide(&self, ctx: &DecisionContext) -> DecisionResult {
        self.traverse(&self.root, ctx, Vec::new())
    }

    /// Traverse tree recursively.
    fn traverse(
        &self,
        node: &DecisionNode,
        ctx: &DecisionContext,
        mut path: Vec<String>,
    ) -> DecisionResult {
        match node {
            DecisionNode::Branch {
                condition,
                if_true,
                if_false,
            } => {
                let result = condition.evaluate(ctx);
                path.push(format!("{:?} = {}", condition, result));

                if result {
                    self.traverse(if_true, ctx, path)
                } else {
                    self.traverse(if_false, ctx, path)
                }
            }
            DecisionNode::Leaf { action, confidence } => DecisionResult {
                action: action.clone(),
                confidence: *confidence,
                path,
            },
        }
    }
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::default_for_zenb()
    }
}

// ============================================================================
// Decision Result
// ============================================================================

/// Result of decision tree evaluation.
#[derive(Debug, Clone)]
pub struct DecisionResult {
    /// Recommended action
    pub action: RouteAction,
    /// Confidence in decision (0.0 - 1.0)
    pub confidence: f32,
    /// Path taken through tree (for debugging)
    pub path: Vec<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_tree_full_resources() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.8),
            available_ram_mb: 2048.0,
            cpu_load: 0.3,
            time_since_last_compute_ms: 100,
            belief_confidence: 0.9,
            is_thermal_throttled: false,
            hour_of_day: 14,
            is_high_priority: false,
            failure_count: 0,
        };

        let result = tree.decide(&ctx);
        // High confidence, recent compute → UseCached
        assert_eq!(result.action, RouteAction::UseCached);
    }

    #[test]
    fn test_low_battery_simplified() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.05), // Critical
            available_ram_mb: 2048.0,
            cpu_load: 0.3,
            time_since_last_compute_ms: 100,
            belief_confidence: 0.9,
            is_thermal_throttled: false,
            hour_of_day: 14,
            is_high_priority: true,
            failure_count: 0,
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, RouteAction::SimplifiedComputation);
    }

    #[test]
    fn test_high_failures_fallback() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.9),
            available_ram_mb: 4096.0,
            cpu_load: 0.1,
            time_since_last_compute_ms: 100,
            belief_confidence: 0.95,
            is_thermal_throttled: false,
            hour_of_day: 12,
            is_high_priority: true,
            failure_count: 5, // High failures
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, RouteAction::SafeFallback);
    }

    #[test]
    fn test_low_confidence_recompute() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.8),
            available_ram_mb: 2048.0,
            cpu_load: 0.3,
            time_since_last_compute_ms: 100,
            belief_confidence: 0.2, // Low confidence
            is_thermal_throttled: false,
            hour_of_day: 14,
            is_high_priority: false,
            failure_count: 0,
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, RouteAction::FullComputation);
    }

    #[test]
    fn test_high_priority_full_compute() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.6),
            available_ram_mb: 1024.0,
            cpu_load: 0.5,
            time_since_last_compute_ms: 1000,
            belief_confidence: 0.7,
            is_thermal_throttled: false,
            hour_of_day: 10,
            is_high_priority: true, // High priority
            failure_count: 0,
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, RouteAction::FullComputation);
    }

    #[test]
    fn test_stale_data_recompute() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext {
            battery_level: Some(0.8),
            available_ram_mb: 2048.0,
            cpu_load: 0.3,
            time_since_last_compute_ms: 10_000, // Old data
            belief_confidence: 0.9,
            is_thermal_throttled: false,
            hour_of_day: 14,
            is_high_priority: false,
            failure_count: 0,
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, RouteAction::FullComputation);
    }

    #[test]
    fn test_decision_path() {
        let tree = DecisionTree::default_for_zenb();
        let ctx = DecisionContext::default();
        let result = tree.decide(&ctx);

        // Path should contain multiple entries
        assert!(!result.path.is_empty());
    }
}
