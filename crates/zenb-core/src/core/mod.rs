//! Core module: Domain-agnostic abstractions for the adaptive control engine.
//!
//! This module provides traits and types that are independent of any specific
//! application domain (e.g., biofeedback, IoT control, trading).

mod traits;

pub use traits::*;
