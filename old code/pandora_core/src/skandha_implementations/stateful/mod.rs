//! Stateful skandha implementations (Phase 1).
//!
//! These implementations maintain internal state across processing cycles,
//! enabling mood tracking, pattern reinforcement, and contextual decision-making.
//!
//! # Performance Considerations
//!
//! - Latency: ~50-70µs per cycle (vs 30-40µs for stateless)
//! - Memory: ~8KB per skandha instance (state overhead)
//! - State decay: Automatic cleanup prevents unbounded growth
//!
//! # Use Cases
//!
//! - Systems requiring contextual awareness
//! - Adaptive error handling (mood-based severity)
//! - Pattern learning and reinforcement
//! - Cognitive debugging (state introspection)

pub mod vedana;
pub mod sanna;
pub mod adapters;

pub use vedana::StatefulVedana;
pub use sanna::StatefulSanna;
pub use adapters::StatelessAdapter;
