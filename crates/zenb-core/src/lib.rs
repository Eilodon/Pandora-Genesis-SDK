//! ZenB core domain: deterministic domain types, replay, and state hashing.
//!
//! # PR1 EXPORT SURFACE CLEANUP
//! This module now uses curated exports instead of wildcard re-exports.
//! The canonical BeliefState is crate::belief::BeliefState (5-mode collapsed).
//! CausalBeliefState (3-factor) is exported for causal layer use only.

pub mod adaptive; // PANDORA PORT: Adaptive thresholds and anomaly detection
pub mod agent_container;
pub mod ai; // NEW: AI Tools
pub mod belief;
pub mod belief_subsystem; // Phase 2: Extracted from Engine god-object
pub mod breath_engine;
pub mod causal;
pub mod circuit_breaker; // PANDORA PORT: Resilient operation execution
pub mod config;
pub mod control_flow; // NEW: Type-safe pipeline builder
pub mod controller;
pub mod core; // V2.0: Domain-agnostic traits
pub mod decision_tree; // PANDORA PORT: Context-aware routing
pub mod domain;
pub mod domains; // V2.0: Pluggable domain implementations
pub mod edge; // PANDORA PORT: Device-aware optimization
pub mod engine;
pub mod estimator;
pub mod estimators; // NEW: UKF and advanced estimators
pub mod memory; // VAJRA-001: Holographic Memory
pub mod perception; // VAJRA-001: Sheaf Perception
pub mod perception_subsystem; // Phase 2: Extracted from Engine god-object
pub mod phase_machine;
pub mod policy;
pub mod replay;
pub mod resonance;
pub mod safety; // NEW: LTL Safety Monitor + DharmaFilter
pub mod safety_subsystem; // Phase 2: Extracted from Engine god-object
pub mod safety_swarm;
pub mod scientist; // PANDORA PORT: Automatic causal discovery
pub mod sensory; // NEW: Binaural, Soundscape, Haptics
pub mod skandha; // PANDORA PORT: Five Skandhas cognitive pipeline
pub mod thermo_logic; // TIER 3: Thermodynamic Logic (GENERIC framework)
pub mod timestamp;
pub mod trauma_cache;
pub mod uncertain; // NEW: Uncertainty quantification
pub mod validation; // Phase 2.4: Consolidated timestamp tracking

#[cfg(test)]
pub mod tests_config;
#[cfg(test)]
pub mod tests_determinism;
#[cfg(test)]
pub mod tests_estimator;
#[cfg(test)]
pub mod tests_proptest; // EIDOLON FIX 4.1: Property-based testing

// ============================================================================
// V2.0: DOMAIN-AGNOSTIC ABSTRACTION LAYER
// ============================================================================

// Core traits for custom domain implementations
pub use core::{ActionKind, Domain, OscillatorConfig, SignalVariable};

// Built-in domains
pub use domains::biofeedback::{BioAction, BioVariable, BreathConfig as DomainBreathConfig};
pub use domains::BiofeedbackDomain;

// ============================================================================
// CURATED PUBLIC API EXPORTS (PR1: No more wildcard exports)
// ============================================================================

// Domain types (event sourcing infrastructure)
pub use domain::{
    dt_sec, // PR4: Time delta helpers (prevent wraparound)
    dt_us,
    AppCategory,
    BioMetrics,
    BioState,
    BreathState,
    CausalBeliefState, // 3-factor representation for causal layer ONLY
    CognitiveState,
    ControlDecision,
    DigitalContext,
    DomainError,
    Envelope,
    EnvironmentalContext,
    Event,
    EventPriority,
    LocationType,
    Observation,
    SessionId,
    SocialState,
};

// Configuration
pub use config::{
    BeliefConfig, BreathConfig, FepConfig, ResonanceConfig, SafetyConfig, ZenbConfig,
};

// Estimator
pub use estimator::{Estimate, Estimator};

// Safety swarm (trauma system)
pub use safety_swarm::{
    decide, trauma_sig_hash, BreathBoundsGuard, Clamp, ComfortGuard, ConfidenceGuard, Guard,
    PatternPatch, RateLimitGuard, ResourceGuard, TraumaGuard, TraumaHit, TraumaRegistry,
    TraumaSource,
};

// Trauma cache
pub use trauma_cache::TraumaCache;

// Controller
pub use controller::{compute_poll_interval, AdaptiveController, ControllerConfig};

// Phase machine
pub use phase_machine::{PhaseDurations, PhaseMachine};

// Breath engine
pub use breath_engine::{BreathEngine, BreathMode};

// Belief engine (CANONICAL BeliefState is here)
pub use belief::{
    AgentStrategy,  // Data-oriented agent enum
    AgentVote,      // Agent vote output
    BeliefBasis,    // Enum: Calm, Stress, Focus, Sleepy, Energize
    BeliefDebug,    // Debug info
    BeliefEngine,   // Belief update engine
    BeliefState,    // CANONICAL: 5-mode collapsed belief state
    Context,        // Contextual info (hour, charging, sessions)
    FepState,       // Free Energy Principle state
    FepUpdateOut,   // FEP update output
    PhysioState,    // Physiological state
    SensorFeatures, // Sensor input
};

// Resonance tracker
pub use resonance::{ResonanceFeatures, ResonanceTracker};

// Engine (high-level orchestrator)
pub use engine::Engine;

// Causal reasoning
pub use causal::{
    ActionPolicy, ActionType, CausalBuffer, CausalGraph, ObservationSnapshot, PredictedState,
    Variable,
};

// Causal interventions (Pearl's do-calculus)
pub use causal::intervenable::{Intervenable, InterventionLog};

// Monadic causal effects
pub use causal::propagating_effect::PropagatingEffect;

// Uncertainty quantification
pub use uncertain::{MaybeUncertain, Uncertain};

// Type-safe pipeline
pub use control_flow::{ControlFlowBuilder, ControlFlowGraph, ZenBProtocol};

// VAJRA-001: Holographic Memory
pub use memory::HolographicMemory;

// VAJRA-001: Sheaf Perception
pub use perception::SheafPerception;

// VAJRA-001: Dharma Filter (exported from safety module)
pub use safety::{AlignmentCategory, ComplexDecision, DharmaFilter};

// PANDORA PORT: Circuit Breaker for resilient operations
pub use circuit_breaker::{
    CircuitBreakerConfig, CircuitBreakerManager, CircuitState, CircuitStats,
};

// PANDORA PORT: Adaptive thresholds and anomaly detection
pub use adaptive::{AdaptiveThreshold, AnomalyDetector, ConfidenceTracker};

// PANDORA PORT: Edge device optimization
pub use edge::{
    EdgeDeviceSpecs, EdgeDeviceType, EdgeOptimizer, OptimizationConfig as EdgeOptimizationConfig,
};

// PANDORA PORT: Context-aware routing
pub use decision_tree::{Condition, DecisionContext, DecisionResult, DecisionTree, RouteAction};

// TIER 3: Thermodynamic Logic (GENERIC framework)
pub use thermo_logic::{ThermoConfig, ThermodynamicEngine};
