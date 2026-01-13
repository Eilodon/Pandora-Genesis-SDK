//! Core module: Domain-agnostic abstractions for the adaptive control engine.
//!
//! This module provides traits and types that are independent of any specific
//! application domain (e.g., biofeedback, IoT control, trading).

pub mod generic_causal;
pub mod generic_engine;
mod traits;

pub use generic_causal::{CausalEdge, CausalSource, GenericCausalGraph};
pub use generic_engine::{
    GenericControlDecision, GenericEngine, GenericEngineDiagnostics, Observation,
};
pub use traits::*;
