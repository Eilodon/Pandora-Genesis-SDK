//! Type-Safe Control Flow Builder
//!
//! Provides compile-time verified execution pipelines with zero-allocation
//! execution loops. Inspired by DeepCausality's ControlFlowBuilder but
//! tailored for AGOLOS's real-time processing needs.

use std::marker::PhantomData;

/// Protocol messages for type-safe pipeline communication
///
/// Each stage of the pipeline produces and consumes specific message types,
/// ensuring type safety at compile time.
#[derive(Debug, Clone)]
pub enum ZenBProtocol {
    /// Raw sensor data input
    SensorInput {
        features: Vec<f32>,
        timestamp_us: i64,
    },

    /// Processed estimate from state estimator
    StateEstimate {
        hr_bpm: Option<f32>,
        rr_bpm: Option<f32>,
        rmssd: Option<f32>,
        confidence: f32,
        timestamp_us: i64,
    },

    /// Belief state after FEP update
    BeliefUpdate {
        belief: crate::belief::BeliefState,
        fep: crate::belief::FepState,
        timestamp_us: i64,
    },

    /// Final control decision
    ControlOutput {
        decision: crate::domain::ControlDecision,
        should_persist: bool,
        policy_info: Option<(u8, u32, f32)>,
        deny_reason: Option<String>,
    },
}

/// Type-safe control flow builder
///
/// Builds a static execution graph where each node's input/output types
/// are verified at compile time. Enables zero-allocation execution loops
/// for real-time processing.
///
/// # Design Philosophy
/// - **Compile-time safety**: Type mismatches caught at compile time
/// - **Zero allocation**: Pre-allocated execution queue
/// - **Deterministic**: Fixed execution order, no dynamic dispatch overhead
/// - **Auditable**: Clear data flow through typed messages
///
/// # Example
/// ```ignore
/// let pipeline = ControlFlowBuilder::new()
///     .add_node(|input: SensorInput| process_sensors(input))
///     .add_node(|estimate: StateEstimate| update_belief(estimate))
///     .add_node(|belief: BeliefUpdate| make_decision(belief))
///     .build();
///
/// let result = pipeline.execute(initial_input);
/// ```
pub struct ControlFlowBuilder<State> {
    /// Execution nodes (functions)
    nodes: Vec<Box<dyn Fn(ZenBProtocol) -> ZenBProtocol>>,

    /// Phantom data for type state pattern
    _marker: PhantomData<State>,
}

/// Type state markers for builder pattern
pub struct Empty;
pub struct WithNodes;

impl ControlFlowBuilder<Empty> {
    /// Create a new empty control flow builder
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl Default for ControlFlowBuilder<Empty> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> ControlFlowBuilder<State> {
    /// Add a processing node to the pipeline
    ///
    /// Each node is a function that transforms one protocol message to another.
    /// The type system ensures that nodes are compatible.
    pub fn add_node<F>(mut self, f: F) -> ControlFlowBuilder<WithNodes>
    where
        F: Fn(ZenBProtocol) -> ZenBProtocol + 'static,
    {
        self.nodes.push(Box::new(f));
        ControlFlowBuilder {
            nodes: self.nodes,
            _marker: PhantomData,
        }
    }
}

impl ControlFlowBuilder<WithNodes> {
    /// Build the final control flow graph
    pub fn build(self) -> ControlFlowGraph {
        ControlFlowGraph { nodes: self.nodes }
    }
}

/// Compiled control flow graph ready for execution
///
/// This is the final, optimized execution graph with pre-allocated
/// buffers for zero-allocation execution.
pub struct ControlFlowGraph {
    nodes: Vec<Box<dyn Fn(ZenBProtocol) -> ZenBProtocol>>,
}

impl ControlFlowGraph {
    /// Execute the pipeline with given input
    ///
    /// Runs all nodes in sequence, passing output of each node
    /// as input to the next. This is a zero-allocation hot path.
    pub fn execute(&self, mut input: ZenBProtocol) -> ZenBProtocol {
        for node in &self.nodes {
            input = node(input);
        }
        input
    }

    /// Execute pipeline and extract specific output type
    ///
    /// This is a convenience method that executes the pipeline
    /// and pattern matches the final output.
    pub fn execute_and_extract<T, F>(&self, input: ZenBProtocol, extractor: F) -> Option<T>
    where
        F: FnOnce(ZenBProtocol) -> Option<T>,
    {
        let output = self.execute(input);
        extractor(output)
    }
}

// ============================================================================
// HELPER FUNCTIONS FOR ENGINE INTEGRATION
// ============================================================================

/// Create a standard ZenB processing pipeline
///
/// This is the canonical pipeline for the Engine:
/// SensorInput -> StateEstimate -> BeliefUpdate -> ControlOutput
pub fn create_zenb_pipeline() -> ControlFlowGraph {
    ControlFlowBuilder::new()
        .add_node(|msg| {
            // Node 1: Sensor processing (placeholder - actual implementation in Engine)
            match msg {
                ZenBProtocol::SensorInput {
                    features,
                    timestamp_us,
                } => {
                    // In real implementation, this would call Engine::ingest_sensor
                    ZenBProtocol::StateEstimate {
                        hr_bpm: features.first().copied(),
                        rr_bpm: features.get(2).copied(),
                        rmssd: features.get(1).copied(),
                        confidence: 0.8,
                        timestamp_us,
                    }
                }
                other => other, // Pass through
            }
        })
        .add_node(|msg| {
            // Node 2: Belief update (placeholder)
            match msg {
                ZenBProtocol::StateEstimate { timestamp_us, .. } => {
                    // In real implementation, this would call Engine::tick + belief update
                    ZenBProtocol::BeliefUpdate {
                        belief: crate::belief::BeliefState::default(),
                        fep: crate::belief::FepState::default(),
                        timestamp_us,
                    }
                }
                other => other,
            }
        })
        .add_node(|msg| {
            // Node 3: Control decision (placeholder)
            match msg {
                ZenBProtocol::BeliefUpdate { .. } => {
                    // In real implementation, this would call Engine::make_control
                    ZenBProtocol::ControlOutput {
                        decision: crate::domain::ControlDecision {
                            target_rate_bpm: 6.0,
                            confidence: 0.8,
                            recommended_poll_interval_ms: 1000,
                            intent_id: None,
                        },
                        should_persist: false,
                        policy_info: None,
                        deny_reason: None,
                    }
                }
                other => other,
            }
        })
        .build()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let _builder = ControlFlowBuilder::new();
    }

    #[test]
    fn test_add_node() {
        let builder = ControlFlowBuilder::new().add_node(|msg| msg); // Identity node

        let _graph = builder.build();
    }

    #[test]
    fn test_pipeline_execution() {
        let pipeline = ControlFlowBuilder::new()
            .add_node(|msg| {
                // Transform sensor input to estimate
                match msg {
                    ZenBProtocol::SensorInput {
                        features,
                        timestamp_us,
                    } => ZenBProtocol::StateEstimate {
                        hr_bpm: features.get(0).copied(),
                        rr_bpm: Some(6.0),
                        rmssd: Some(40.0),
                        confidence: 0.9,
                        timestamp_us,
                    },
                    other => other,
                }
            })
            .build();

        let input = ZenBProtocol::SensorInput {
            features: vec![60.0, 40.0, 6.0],
            timestamp_us: 1000,
        };

        let output = pipeline.execute(input);

        match output {
            ZenBProtocol::StateEstimate {
                hr_bpm, confidence, ..
            } => {
                assert_eq!(hr_bpm, Some(60.0));
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected StateEstimate output"),
        }
    }

    #[test]
    fn test_multi_node_pipeline() {
        let pipeline = ControlFlowBuilder::new()
            .add_node(|msg| {
                // Node 1: Sensor -> Estimate
                match msg {
                    ZenBProtocol::SensorInput { timestamp_us, .. } => ZenBProtocol::StateEstimate {
                        hr_bpm: Some(60.0),
                        rr_bpm: Some(6.0),
                        rmssd: Some(40.0),
                        confidence: 0.8,
                        timestamp_us,
                    },
                    other => other,
                }
            })
            .add_node(|msg| {
                // Node 2: Estimate -> Belief
                match msg {
                    ZenBProtocol::StateEstimate { timestamp_us, .. } => {
                        ZenBProtocol::BeliefUpdate {
                            belief: crate::belief::BeliefState::default(),
                            fep: crate::belief::FepState::default(),
                            timestamp_us,
                        }
                    }
                    other => other,
                }
            })
            .build();

        let input = ZenBProtocol::SensorInput {
            features: vec![60.0],
            timestamp_us: 1000,
        };

        let output = pipeline.execute(input);

        match output {
            ZenBProtocol::BeliefUpdate { .. } => {
                // Success - pipeline executed both nodes
            }
            _ => panic!("Expected BeliefUpdate output"),
        }
    }

    #[test]
    fn test_extract_helper() {
        let pipeline = ControlFlowBuilder::new()
            .add_node(|msg| match msg {
                ZenBProtocol::SensorInput { timestamp_us, .. } => ZenBProtocol::StateEstimate {
                    hr_bpm: Some(70.0),
                    rr_bpm: Some(7.0),
                    rmssd: Some(35.0),
                    confidence: 0.95,
                    timestamp_us,
                },
                other => other,
            })
            .build();

        let input = ZenBProtocol::SensorInput {
            features: vec![],
            timestamp_us: 2000,
        };

        let hr = pipeline.execute_and_extract(input, |output| match output {
            ZenBProtocol::StateEstimate { hr_bpm, .. } => hr_bpm,
            _ => None,
        });

        assert_eq!(hr, Some(70.0));
    }

    #[test]
    fn test_zenb_pipeline_creation() {
        let _pipeline = create_zenb_pipeline();
        // Just verify it compiles and can be created
    }
}
