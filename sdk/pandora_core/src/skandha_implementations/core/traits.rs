//! Core trait hierarchy for Skandha system
//!
//! This module defines the foundational traits that enable:
//! - Type-safe composition of stateful and stateless skandhas
//! - Opt-in state management capabilities
//! - Compiler-enforced processor contracts
//!
//! # Architecture
//!
//! ```text
//! Skandha (base)
//!   ├─→ VedanaSkandha (capability: feeling)
//!   │    └─→ StatefulVedanaSkandha (opt-in: state)
//!   └─→ StatefulSkandha (capability: state mgmt)
//! ```

use crate::ontology::EpistemologicalFlow;
use std::time::Duration;

// ============================================================================
// BASE TRAITS (Unchanged from original)
// ============================================================================

/// Base trait for all Skandhas (aggregates) in the cognitive architecture.
///
/// All skandhas must be thread-safe (`Send + Sync`) to support concurrent
/// processing in future phases.
pub trait Skandha: Send + Sync {
    /// Human-readable name for this skandha instance.
    ///
    /// # Examples
    /// ```
    /// // assert_eq!(vedana.name(), "BasicVedana");
    /// // assert_eq!(stateful_vedana.name(), "StatefulVedana-Mood");
    /// ```
    fn name(&self) -> &'static str;
}

// ============================================================================
// CAPABILITY TRAITS - State Management
// ============================================================================

/// Capability trait for skandhas that maintain internal state.
///
/// This trait enables:
/// - Persistent state across processing cycles
/// - State inspection for debugging/monitoring
/// - State mutation for learning/adaptation
///
/// # Design Rationale
///
/// Using an associated type `State` instead of generic parameter allows:
/// - Type inference without explicit turbofish (`::< >`)
/// - Easier trait object usage in factories
/// - Clearer API boundaries
///
/// # Examples
///
/// ```rust
/// # use std::fmt::Debug;
/// # use pandora_core::skandha_implementations::core::traits::{Skandha, StatefulSkandha};
/// struct MoodState {
///     valence: f32,
///     arousal: f32,
/// }
///
/// # impl Default for MoodState { fn default() -> Self { Self { valence: 0.0, arousal: 0.0 } } }
/// # impl Clone for MoodState { fn clone(&self) -> Self { Self { valence: self.valence, arousal: self.arousal } } }
/// # impl Debug for MoodState { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("MoodState").finish() } }
///
/// struct MyVedana { mood: MoodState }
/// # impl Skandha for MyVedana { fn name(&self) -> &'static str { "MyVedana" } }
///
/// impl StatefulSkandha for MyVedana {
///     type State = MoodState;
///
///     fn state(&self) -> &Self::State { &self.mood }
///     fn state_mut(&mut self) -> &mut Self::State { &mut self.mood }
/// }
/// ```
pub trait StatefulSkandha: Skandha {
    /// The type of state maintained by this skandha.
    ///
    /// Must implement:
    /// - `Default`: For initialization
    /// - `Clone`: For snapshotting/checkpointing
    /// - `Debug`: For introspection
    type State: Default + Clone + std::fmt::Debug;

    /// Immutable access to internal state.
    fn state(&self) -> &Self::State;

    /// Mutable access to internal state.
    ///
    /// # Safety Note
    ///
    /// Implementors must ensure state mutations are:
    /// - Bounded (no unbounded growth)
    /// - Deterministic (same inputs → same state transitions)
    /// - Reversible (via `reset()` if needed)
    fn state_mut(&mut self) -> &mut Self::State;

    /// Reset state to default (optional override).
    ///
    /// Default implementation creates new default state.
    fn reset(&mut self) {
        *self.state_mut() = Self::State::default();
    }
}

/// Capability trait for states that decay over time.
///
/// Implements temporal decay to prevent:
/// - Infinite state accumulation
/// - Stale state pollution
/// - Hysteresis in decision-making
///
/// # Decay Strategies
///
/// Common implementations:
/// - **Exponential decay**: `state *= decay_factor.powf(delta_time.as_secs_f32())`
/// - **Linear decay**: `state -= decay_rate * delta_time.as_secs_f32()`
/// - **Threshold decay**: Decay only if `abs(state) > threshold`
///
/// # Examples
///
/// ```rust
/// # use std::time::Duration;
/// # use pandora_core::skandha_implementations::core::traits::DecayableState;
/// # struct MoodState { valence: f32, arousal: f32 }
/// impl DecayableState for MoodState {
///     fn decay(&mut self, delta_time: Duration) {
///         let factor = 0.95_f32.powf(delta_time.as_secs_f32());
///         self.valence *= factor;
///         self.arousal *= factor;
///     }
///     fn reset(&mut self) {
///         self.valence = 0.0;
///         self.arousal = 0.0;
///     }
///     fn decay_factor(&self) -> f32 { 0.95 }
/// }
/// ```
pub trait DecayableState: Sized {
    /// Apply time-based decay to state.
    ///
    /// # Arguments
    ///
    /// * `delta_time` - Time elapsed since last decay
    fn decay(&mut self, delta_time: Duration);

    /// Reset state to neutral/zero value.
    ///
    /// Different from `Default::default()` - this returns to
    /// neutral state, not initial state.
    fn reset(&mut self);

    /// Decay factor per second (0.0 = instant decay, 1.0 = no decay).
    ///
    /// Default: 0.95 (5% decay per second, half-life ~13.5 sec)
    fn decay_factor(&self) -> f32 {
        0.95
    }

    /// Half-life in seconds (derived from decay factor).
    ///
    /// Formula: `t_half = ln(2) / ln(1/decay_factor)`
    fn half_life(&self) -> f32 {
        let factor = self.decay_factor();
        if factor >= 1.0 {
            f32::INFINITY
        } else {
            0.693_147_18 / (1.0 / factor).ln() // ln(2) ≈ 0.693
        }
    }
}

// ============================================================================
// SKANDHA-SPECIFIC TRAITS
// ============================================================================

/// 1. Rūpa-skandha (Form): Raw sensory input processing.
///
/// Converts raw event data into structured epistemological flow.
/// This is the ONLY stateless-only skandha (perception is memoryless).
pub trait RupaSkandha: Skandha {
    /// Process raw event bytes into epistemological flow.
    ///
    /// # Contract
    ///
    /// - MUST populate `flow.rupa` with parsed data
    /// - SHOULD add metadata (timestamps, source, etc.)
    /// - MUST NOT mutate other flow fields
    ///
    /// # Performance
    ///
    /// Target: <5µs for events <1KB
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow;
}

/// 2. Vedanā-skandha (Feeling): Moral/emotional valence assignment.
///
/// Assigns hedonic tone (pleasant/unpleasant/neutral) to perceptions.
/// This is the PRIMARY candidate for stateful implementation (mood).
pub trait VedanaSkandha: Skandha {
    /// Assign feeling/valence to the epistemological flow.
    ///
    /// # Contract
    ///
    /// - MUST read `flow.rupa`
    /// - MUST write `flow.vedana`
    /// - MAY read previous `flow.vedana` for context
    ///
    /// # Performance
    ///
    /// Target: <3µs (simple keyword matching)
    fn feel(&self, flow: &mut EpistemologicalFlow);
}

/// Stateful variant of VedanaSkandha with mood/emotion tracking.
///
/// Enables:
/// - Mood inertia (pleasant events feel better when mood is high)
/// - Emotional contagion (feelings influence subsequent feelings)
/// - Affect regulation (mood decays toward neutral over time)
///
/// # Design Pattern
///
/// This trait EXTENDS VedanaSkandha, meaning:
/// - All StatefulVedanaSkandha ARE VedanaSkandha (Liskov substitution)
/// - Can be used anywhere VedanaSkandha is expected
/// - Adds opt-in mutation capability
#[async_trait::async_trait]
pub trait StatefulVedanaSkandha: VedanaSkandha + StatefulSkandha {
    /// Stateful version with mood influence and long-term memory (Ālaya).
    ///
    /// Typical implementation:
    /// 1. Call `self.feel(flow)` for base feeling
    /// 2. Query Ālaya for similar past experiences (async)
    /// 3. Read `self.state()` for current mood
    /// 4. Adjust `flow.vedana` based on mood and past experiences
    /// 5. Update `self.state_mut()` based on new feeling
    /// 6. Store new experience in Ālaya for future (async)
    async fn feel_with_state(&mut self, flow: &mut EpistemologicalFlow);
}

/// 3. Saññā-skandha (Perception): Pattern recognition.
///
/// Identifies patterns, retrieves related concepts from world model.
pub trait SannaSkandha: Skandha {
    /// Perceive patterns and retrieve related knowledge.
    ///
    /// # Contract
    ///
    /// - MUST read `flow.rupa` and `flow.vedana`
    /// - MUST write `flow.sanna` (primary pattern)
    /// - MAY write `flow.related_eidos` (associated patterns)
    ///
    /// # Performance
    ///
    /// Target: <10µs (hash-based pattern matching)
    fn perceive(&self, flow: &mut EpistemologicalFlow);
}

/// Stateful variant with pattern memory/reinforcement.
///
/// Enables:
/// - Pattern frequency tracking
/// - Associative memory strengthening
/// - Contextual pattern priming
pub trait StatefulSannaSkandha: SannaSkandha + StatefulSkandha {
    /// Stateful perception with pattern reinforcement.
    ///
    /// Typical implementation:
    /// 1. Call `self.perceive(flow)` for base perception
    /// 2. Check `self.state()` for pattern history
    /// 3. Strengthen detected patterns in `self.state_mut()`
    /// 4. Adjust `flow.sanna` confidence based on frequency
    fn perceive_with_state(&mut self, flow: &mut EpistemologicalFlow);
}

/// 4. Saṅkhāra-skandha (Formations): Volitional intent formation.
///
/// Forms actionable intent based on all prior skandha processing.
pub trait SankharaSkandha: Skandha {
    /// Form volitional intent based on complete flow.
    ///
    /// # Contract
    ///
    /// - MUST read entire flow (rupa, vedana, sanna)
    /// - MUST write `flow.sankhara` (intent)
    /// - SHOULD use intent constants from `crate::intents`
    ///
    /// # Performance
    ///
    /// Target: <5µs (decision tree traversal)
    fn form_intent(&self, flow: &mut EpistemologicalFlow);
}

/// 5. Viññāṇa-skandha (Consciousness): Synthesis and rebirth.
///
/// Final integration, decides if flow spawns new events.
pub trait VinnanaSkandha: Skandha {
    /// Synthesize complete epistemological flow.
    ///
    /// # Contract
    ///
    /// - MUST read entire flow
    /// - Returns `Some(event)` to trigger rebirth cycle
    /// - Returns `None` to terminate processing
    ///
    /// # Performance
    ///
    /// Target: <5µs (synthesis decision)
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>>;
}

// ============================================================================
// UTILITY TRAITS
// ============================================================================

/// Trait for skandhas that support metrics collection.
///
/// Optional trait for monitoring/observability.
pub trait ObservableSkandha: Skandha {
    /// Get processing metrics since last reset.
    fn metrics(&self) -> SkandhMetrics;

    /// Reset metrics counters.
    fn reset_metrics(&mut self);
}

/// Metrics collected during skandha processing.
#[derive(Debug, Clone, Default)]
pub struct SkandhMetrics {
    /// Total number of processing cycles
    pub cycles: u64,

    /// Total processing time (microseconds)
    pub total_time_us: u64,

    /// Average processing time (microseconds)
    pub avg_time_us: f64,

    /// Peak processing time (microseconds)
    pub peak_time_us: u64,
}

impl SkandhMetrics {
    pub fn record_cycle(&mut self, elapsed_us: u64) {
        self.cycles += 1;
        self.total_time_us += elapsed_us;
        self.avg_time_us = self.total_time_us as f64 / self.cycles as f64;
        self.peak_time_us = self.peak_time_us.max(elapsed_us);
    }
}
