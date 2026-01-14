//! Safety Monitor module
//!
//! LTL runtime verification, Dharma-based ethical filtering, and Triple Guardians

pub mod consciousness;
pub mod dharma;
pub mod guardians; // VAJRA V5: Triple Guardians safety layers
pub mod monitor;   // TIER 5: 11D Consciousness Vector
pub mod contracts; // VAJRA-VOID: Formal verification contracts for Aeneas/Lean4

pub use consciousness::{ConsciousnessVector, PhaseConsciousnessVector};
pub use dharma::{AlignmentCategory, ComplexDecision, DharmaFilter};
pub use guardians::{
    FepConfig, FepMonitor, GuardianDecision, GuardianDiagnostics, HamiltonianConfig,
    HamiltonianGuard, PhysicalState, TripleGuardians,
};
pub use monitor::{RuntimeState, SafetyMonitor, SafetyProperty, SafetyViolation, Severity};
