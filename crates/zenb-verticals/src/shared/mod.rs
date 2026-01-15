//! Shared components reused across verticals.

pub mod eye_metrics;
pub mod gaze_estimator;
pub mod micro_expression;
pub mod attention_tracker;
pub mod stress_profiler;
pub mod safety_guard;
pub mod privacy;

pub use eye_metrics::{
    EyeMetricsAnalyzer, EyeMetricsConfig, EarResult, BlinkResult, PerclosResult,
};
pub use gaze_estimator::{GazeEstimator, GazeConfig, GazeDirection, GazeResult, GazeTarget};
pub use micro_expression::{
    MicroExpressionAnalyzer, EmotionResult, BasicEmotion, ActionUnits,
};
pub use attention_tracker::{AttentionTracker, AttentionMetrics};
pub use stress_profiler::{StressProfiler, StressConfig, StressProfile};
pub use safety_guard::{SafetyGuard, SafetyConfig, SafeDefault, SafetyError};
pub use privacy::{RetentionPolicy, ConsentStatus, PrivacyWrapped, AggregateAnalytics};
