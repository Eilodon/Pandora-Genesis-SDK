//! Safety Monitor module
//!
//! LTL runtime verification and Dharma-based ethical filtering

pub mod consciousness;
pub mod dharma;
pub mod monitor; // TIER 5: 11D Consciousness Vector

pub use consciousness::ConsciousnessVector;
pub use dharma::{AlignmentCategory, ComplexDecision, DharmaFilter};
pub use monitor::{RuntimeState, SafetyMonitor, SafetyProperty, SafetyViolation, Severity};
