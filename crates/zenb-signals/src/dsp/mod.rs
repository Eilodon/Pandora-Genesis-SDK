//! DSP (Digital Signal Processing) module
//!
//! Provides signal processing utilities for rPPG and physiological signal analysis.
//!
//! ## Legacy API (backward compatible)
//! - `DspProcessor` - FFT-based heart rate computation
//! - `FilterConfig` - Bandpass filter configuration
//!
//! ## New SOTA Components
//! - `AdaptiveFilter` - SNR-aware adaptive filtering
//! - `SignalQualityAnalyzer` - Signal quality assessment
//! - `HrTracker` - Kalman filter for BPM stabilization
//! - `QualityScorer` - Multi-metric quality scoring
//! - `TemporalNorm` - TD/TN detrending methods (ME-rPPG 2025)

mod filters;
mod hr_tracker;
mod legacy;
pub mod motion_detector;
mod quality_score;
mod signal_quality;
pub mod temporal_norm;

// Re-export legacy API for backward compatibility
pub use legacy::{DspProcessor, FilterConfig};

// Export new SOTA components
pub use filters::{adaptive_bandpass_filter, AdaptiveFilter, AdaptiveFilterConfig};
pub use signal_quality::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};

// Tracking & composite quality exports
pub use hr_tracker::{HrTracker, HrTrackerConfig, HrTrackedValue};
pub use quality_score::{ExternalQuality, QualityScore, QualityScorer, QualityScorerConfig};

// Temporal normalization exports
pub use temporal_norm::{
    combined_detrending, temporal_difference, temporal_normalization,
    TdConfig, TnConfig,
};

// Motion detection exports
pub use motion_detector::{
    compute_frame_motion, compute_landmark_motion,
    MotionDetector, MotionDetectorConfig, MotionState, MotionStatus,
};
