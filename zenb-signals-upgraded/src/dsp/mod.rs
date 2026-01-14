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

mod filters;
mod legacy;
mod signal_quality;
mod hr_tracker;
mod quality_score;

// Re-export legacy API for backward compatibility
pub use legacy::{DspProcessor, FilterConfig};

// Export new SOTA components
pub use filters::{adaptive_bandpass_filter, AdaptiveFilter, AdaptiveFilterConfig};
pub use signal_quality::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};

// Tracking & composite quality exports
pub use hr_tracker::{HrTracker, HrTrackerConfig, HrTrackedValue};
pub use quality_score::{ExternalQuality, QualityScore, QualityScorer, QualityScorerConfig};
