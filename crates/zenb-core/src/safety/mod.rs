//! Safety Monitor module
//!
//! LTL runtime verification, Dharma-based ethical filtering, and Triple Guardians

pub mod consciousness;
pub mod dharma;
pub mod guardians; // VAJRA V5: Triple Guardians safety layers
pub mod monitor;   // TIER 5: 11D Consciousness Vector

pub use consciousness::ConsciousnessVector;
pub use dharma::{AlignmentCategory, ComplexDecision, DharmaFilter};
pub use guardians::{
    FepConfig, FepMonitor, GuardianDecision, GuardianDiagnostics, HamiltonianConfig,
    HamiltonianGuard, PhysicalState, TripleGuardians,
};
pub use monitor::{RuntimeState, SafetyMonitor, SafetyProperty, SafetyViolation, Severity};
