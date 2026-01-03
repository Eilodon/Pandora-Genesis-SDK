//! ZenB core domain: deterministic domain types, replay, and state hashing.
//!
//! # PR1 EXPORT SURFACE CLEANUP
//! This module now uses curated exports instead of wildcard re-exports.
//! The canonical BeliefState is crate::belief::BeliefState (5-mode collapsed).
//! CausalBeliefState (3-factor) is exported for causal layer use only.

pub mod domain;
pub mod replay;
pub mod policy;
pub mod config;
pub mod estimator;
pub mod safety;
pub mod safety_swarm;
pub mod trauma_cache;
pub mod controller;
pub mod phase_machine;
pub mod breath_engine;
pub mod belief;
pub mod resonance;
pub mod engine;
pub mod causal;

// ============================================================================
// CURATED PUBLIC API EXPORTS (PR1: No more wildcard exports)
// ============================================================================

// Domain types (event sourcing infrastructure)
pub use domain::{
    SessionId, Envelope, Event, EventPriority, ControlDecision, DomainError, BreathState,
    Observation, BioMetrics, EnvironmentalContext, DigitalContext,
    LocationType, AppCategory,
    BioState, CognitiveState, SocialState,
    CausalBeliefState, // 3-factor representation for causal layer ONLY
    dt_us, dt_sec, // PR4: Time delta helpers (prevent wraparound)
};

// Replay infrastructure
pub use replay::{Replayer, ReplayError};

// Policy types
pub use policy::{PolicyMode, PolicyConfig};

// Configuration
pub use config::{ZenbConfig, BreathConfig, BeliefConfig, FepConfig, ResonanceConfig, SafetyConfig};

// Estimator
pub use estimator::{Estimator, Estimate};

// Safety envelope
pub use safety::SafetyEnvelope;

// Safety swarm (trauma system)
pub use safety_swarm::{
    TraumaHit, TraumaSource, TraumaRegistry, Guard, TraumaGuard, ConfidenceGuard,
    BreathBoundsGuard, RateLimitGuard, ComfortGuard, ResourceGuard,
    PatternPatch, Clamp, decide, trauma_sig_hash,
};

// Trauma cache
pub use trauma_cache::TraumaCache;

// Controller
pub use controller::{AdaptiveController, ControllerConfig, compute_poll_interval};

// Phase machine
pub use phase_machine::{PhaseMachine, BreathPhase, PhaseDurations, PhaseTransition};

// Breath engine
pub use breath_engine::{BreathEngine, BreathMode};

// Belief engine (CANONICAL BeliefState is here)
pub use belief::{
    BeliefState,      // CANONICAL: 5-mode collapsed belief state
    BeliefBasis,      // Enum: Calm, Stress, Focus, Sleepy, Energize
    FepState,         // Free Energy Principle state
    FepUpdateOut,     // FEP update output
    BeliefEngine,     // Belief update engine
    BeliefDebug,      // Debug info
    Context,          // Contextual info (hour, charging, sessions)
    SensorFeatures,   // Sensor input
    PhysioState,      // Physiological state
    PathwayOut,       // Pathway output
    Pathway,          // Pathway trait
};

// Resonance tracker
pub use resonance::{ResonanceTracker, ResonanceFeatures};

// Engine (high-level orchestrator)
pub use engine::Engine;

// Causal reasoning
pub use causal::{
    Variable, CausalGraph, ActionType, ActionPolicy, PredictedState,
    ObservationSnapshot, CausalBuffer,
};

#[cfg(test)]
mod tests_determinism;
#[cfg(test)]
mod tests_estimator;
#[cfg(test)]
mod tests_config;
