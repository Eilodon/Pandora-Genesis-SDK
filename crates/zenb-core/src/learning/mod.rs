//! Learning module for Active Inference and behavior prediction.
//!
//! # Components
//! - `PriorityExperienceBuffer`: Prioritized replay buffer for faster learning
//! - `TemporalPrefixSpanEngine`: Sequential pattern mining for action prediction

mod experience_buffer;
mod pattern_mining;

pub use experience_buffer::{ExperienceSample, PriorityExperienceBuffer};
pub use pattern_mining::{
    ActionPrediction, Event, PatternError, Sequence, TemporalPattern, TemporalPrefixSpanEngine,
};
