//! ZenB core domain: deterministic domain types, replay, and state hashing.
//!
//! # PR1 EXPORT SURFACE CLEANUP
//! This module now uses curated exports instead of wildcard re-exports.
//! The canonical BeliefState is crate::belief::BeliefState (5-mode collapsed).
//! CausalBeliefState (3-factor) is exported for causal layer use only.

pub mod agent_container;
pub mod ai;          // NEW: AI Tools
pub mod belief;
pub mod breath_engine;
pub mod causal;
pub mod config;
pub mod controller;
pub mod domain;
pub mod engine;
pub mod estimator;
pub mod estimators;  // NEW: UKF and advanced estimators
pub mod phase_machine;
pub mod policy;
pub mod replay;
pub mod resonance;
pub mod safety;      // NEW: LTL Safety Monitor
pub mod safety_swarm;
pub mod sensory;     // NEW: Binaural, Soundscape, Haptics
pub mod trauma_cache;
pub mod validation;

#[cfg(test)]
pub mod tests_config;
#[cfg(test)]
pub mod tests_determinism;
#[cfg(test)]
pub mod tests_estimator;

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
    BeliefBasis,    // Enum: Calm, Stress, Focus, Sleepy, Energize
    BeliefDebug,    // Debug info
    BeliefEngine,   // Belief update engine
    BeliefState,    // CANONICAL: 5-mode collapsed belief state
    Context,        // Contextual info (hour, charging, sessions)
    FepState,       // Free Energy Principle state
    FepUpdateOut,   // FEP update output
    AgentStrategy,  // Data-oriented agent enum
    AgentVote,      // Agent vote output
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
