//! # zenb-signals
//!
//! Biometric signal processing for ZenB.
//!
//! This crate provides:
//! - **rPPG algorithms**: CHROM, POS, and PRISM (2025 SOTA) for heart rate extraction from RGB video
//! - **DSP functions**: FFT, filtering, signal quality analysis
//! - **Wavelet transforms**: Morlet CWT for frequency-band analysis
//! - **Physiological estimators**: HRV and respiration from pulse waveforms
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
pub mod wavelet;

// Legacy DSP exports
pub use dsp::{DspProcessor, FilterConfig};

// New SOTA DSP exports
pub use dsp::{adaptive_bandpass_filter, AdaptiveFilter, AdaptiveFilterConfig};
pub use dsp::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};
pub use dsp::{HrTracker, HrTrackerConfig, HrTrackedValue};
pub use dsp::{ExternalQuality, QualityScore, QualityScorer, QualityScorerConfig};

// Legacy rPPG exports
pub use rppg::{RppgMethod, RppgProcessor, RppgResult};

// New SOTA rPPG exports
pub use rppg::{EnsembleConfig, EnsembleProcessor, EnsembleResult};
pub use rppg::{PrismConfig, PrismProcessor, PrismResult};

// Wavelet exports
pub use wavelet::{BandType, MorletConfig, MorletWavelet, WaveletBands};

// New SOTA wavelet exports
pub use wavelet::{aldtf_denoise, AldtfConfig, AldtfDenoiser};
pub use wavelet::{FastCWT, FastCwtConfig};

// Physiological estimators
pub use physio::{HrvConfig, HrvEstimator, HrvMetrics, HrvResult};
pub use physio::{RespirationConfig, RespirationEstimator, RespirationResult};

