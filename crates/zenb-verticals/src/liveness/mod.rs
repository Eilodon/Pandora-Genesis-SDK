//! Liveness Detection Module
//!
//! Multi-modal liveness detection for authentication.

pub mod detector;
pub mod texture_analyzer;
pub mod challenge_response;
pub mod temporal_consistency;

pub use detector::{LivenessDetector, LivenessConfig, LivenessResult, SpoofingType};
pub use texture_analyzer::{TextureAnalyzer, TextureResult};
pub use challenge_response::{ChallengeGenerator, Challenge, ChallengeType, ChallengeResult};
pub use temporal_consistency::{TemporalConsistencyChecker, ConsistencyResult};
