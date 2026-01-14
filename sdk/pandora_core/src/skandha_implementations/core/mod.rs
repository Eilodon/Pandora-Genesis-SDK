//! Core abstractions for the Skandha system.
//!
//! This module provides the foundational traits, state types, and control
//! primitives that enable both stateless and stateful skandha implementations.
//!
//! # Module Organization
//!
//! - `traits`: Trait hierarchy (Skandha, StatefulSkandha, etc.)
//! - `state_management`: State types (MoodState, PatternMemoryState)
//! - `flow_control`: Control flow (CognitiveFlowStatus, EnergyBudget)
//!
//! # Architecture Principles
//!
//! 1. **Opt-in Complexity**: Base traits remain simple, advanced features are optional
//! 2. **Type Safety**: Compiler enforces contracts at trait boundaries
//! 3. **Zero Cost**: Trait dispatch via monomorphization (no vtable overhead when possible)
//! 4. **Composability**: Mix stateful and stateless implementations freely

pub mod traits;
pub mod state_management;
pub mod flow_control;

// Re-export commonly used types
pub use traits::{
    Skandha, StatefulSkandha, DecayableState,
    RupaSkandha,
    VedanaSkandha, StatefulVedanaSkandha,
    SannaSkandha, StatefulSannaSkandha,
    SankharaSkandha,
    VinnanaSkandha,
    ObservableSkandha, SkandhMetrics,
};

pub use state_management::{
    MoodState, PatternMemoryState, PatternEntry, AutoDecayState,
};

pub use flow_control::{
    CognitiveFlowStatus, SkandhaStage, EnergyBudget, CycleResult, TerminationReason,
};
