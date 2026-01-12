//! Safety Monitor module
//!
//! LTL runtime verification and Dharma-based ethical filtering

pub mod dharma;
pub mod monitor;
pub mod consciousness; // TIER 5: 11D Consciousness Vector

pub use dharma::{AlignmentCategory, ComplexDecision, DharmaFilter};
pub use monitor::{RuntimeState, SafetyMonitor, SafetyProperty, SafetyViolation, Severity};
pub use consciousness::ConsciousnessVector;
