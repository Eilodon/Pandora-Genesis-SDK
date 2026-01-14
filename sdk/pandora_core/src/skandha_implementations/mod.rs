//! Skandha implementations and processors.
//!
//! # Module Organization
//!
//! - `core/`: Foundational traits and primitives
//! - `basic/`: V1 stateless implementations
//! - `stateful/`: V2 stateful implementations (Phase 1)
//! - `processors/`: Pipeline processors
//! - `factory/`: Pre-configured processor factories

pub mod core;
pub mod basic;
pub mod stateful;
pub mod processors;
pub mod factory;

// Re-export commonly used types
pub use core::{
    Skandha, StatefulSkandha, DecayableState,
    RupaSkandha, VedanaSkandha, StatefulVedanaSkandha,
    SannaSkandha, StatefulSannaSkandha,
    SankharaSkandha, VinnanaSkandha,
    MoodState, PatternMemoryState,
    CognitiveFlowStatus, SkandhaStage, EnergyBudget, CycleResult,
};

pub use basic::{
    BasicRupaSkandha,
    BasicVedanaSkandha,
    BasicSannaSkandha,
    BasicSankharaSkandha,
    BasicVinnanaSkandha,
};

pub use processors::{LinearProcessor, RecurrentProcessor};
pub use factory::{ProcessorFactory, ProcessorPreset};

pub use stateful::{StatefulVedana, StatefulSanna, StatelessAdapter};
