//! # zenb-signals
//!
//! Biometric signal processing for ZenB.
//!
//! This crate provides:
//! - **rPPG algorithms**: CHROM, POS, PRISM (2025 SOTA), Multi-ROI, APON
//! - **DSP functions**: FFT, filtering, signal quality, temporal normalization
//! - **Wavelet transforms**: Morlet CWT, Fast CWT, ALDTF denoising
//! - **Physiological estimators**: HRV and respiration from pulse waveforms
//! - **Motion detection**: Adaptive mode switching for mobile rPPG
//! - **Vision**: Face detection interface and ROI extraction
//!
//! ## Example
//!
//! ```ignore
//! use zenb_signals::{RppgProcessor, RppgMethod, PrismProcessor, HrTracker};
//!
//! // Legacy API (CHROM/POS)
//! let mut processor = RppgProcessor::new(RppgMethod::Pos, 90, 30.0);
//! for frame in video_frames {
//!     processor.add_sample(frame.r, frame.g, frame.b);
//! }
//! if let Some((bpm, confidence)) = processor.process() {
//!     println!("Heart rate: {:.1} BPM", bpm);
//! }
//!
//! // SOTA PRISM algorithm with Kalman tracking
//! let mut prism = PrismProcessor::new();
//! let mut tracker = HrTracker::new();
//! if let Some(result) = prism.process(&r_array, &g_array, &b_array) {
//!     let tracked = tracker.update(result.bpm, result.confidence, dt);
//!     println!("Tracked: {:.1} BPM", tracked.bpm);
//! }
//! ```

pub mod dsp;
pub mod physio;
pub mod rppg;
pub mod vision;
pub mod wavelet;
pub mod beauty;

// Legacy DSP exports
pub use dsp::{DspProcessor, FilterConfig};

// New SOTA DSP exports
pub use dsp::{adaptive_bandpass_filter, AdaptiveFilter, AdaptiveFilterConfig};
pub use dsp::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};
pub use dsp::{HrTracker, HrTrackerConfig, HrTrackedValue};
pub use dsp::{ExternalQuality, QualityScore, QualityScorer, QualityScorerConfig};

// Temporal normalization & motion detection (SOTA 2026)
pub use dsp::{combined_detrending, temporal_difference, temporal_normalization, TdConfig, TnConfig};
pub use dsp::{compute_frame_motion, compute_landmark_motion, MotionDetector, MotionDetectorConfig, MotionState, MotionStatus};

// Legacy rPPG exports
pub use rppg::{RppgMethod, RppgProcessor, RppgResult};

// New SOTA rPPG exports
pub use rppg::{AponConfig, AponNoiseEstimator, AponResult};
pub use rppg::{EnsembleConfig, EnsembleProcessor, EnsembleResult};
pub use rppg::{MultiRoiConfig, MultiRoiProcessor, MultiRoiResult, RoiResult, RoiSignal};
pub use rppg::{PrismConfig, PrismProcessor, PrismResult};

// Vision exports
pub use vision::{
    FaceDetection, FaceDetector, ExternalLandmarkDetector,
    extract_roi_grid, extract_roi_mean_rgb,
    Polygon, forehead_roi, left_cheek_roi, right_cheek_roi,
    compute_polygon_mean_rgb, FOREHEAD_LANDMARKS, CHEEK_LEFT_LANDMARKS, CHEEK_RIGHT_LANDMARKS,
};

// Wavelet exports
pub use wavelet::{BandType, MorletConfig, MorletWavelet, WaveletBands};

// New SOTA wavelet exports
pub use wavelet::{aldtf_denoise, AldtfConfig, AldtfDenoiser};
pub use wavelet::{FastCWT, FastCwtConfig};

// Physiological estimators
pub use physio::{HrvConfig, HrvEstimator, HrvMetrics, HrvResult};
pub use physio::{MethodResults, RespirationConfig, RespirationEstimator, RespirationResult};

// Beauty module exports (Mắt Thần Hình)
pub use beauty::{
    BeautyAnalyzer, BeautyConfig, BeautyInput, BeautyFrameResult, RoiColors,
    BeautyQuality, QualityConfig, FaceMeasurements,
    FaceShape, FaceShapeResult, ShapeClassifier,
    MakeupPlan, MakeupZone, ZoneType, ShadeHint, MakeupStyle,
    SkinAnalysis, Undertone, SkinDepth,
    CanonicalLandmarks, normalize_to_canonical, landmark_indices,
};
