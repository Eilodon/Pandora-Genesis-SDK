//! # zenb-signals
//!
//! Biometric signal processing for ZenB.
//!
//! This crate provides:
//! - **rPPG algorithms**: CHROM, POS, and PRISM (2025 SOTA) for heart rate extraction from RGB video
//! - **DSP functions**: FFT, filtering, signal quality analysis
//! - **Wavelet transforms**: Morlet CWT for frequency-band analysis
//!
//! ## Example
//!
//! ```ignore
//! use zenb_signals::{RppgProcessor, RppgMethod, PrismProcessor};
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
//! // SOTA PRISM algorithm
//! let mut prism = PrismProcessor::new();
//! if let Some(result) = prism.process(&r_array, &g_array, &b_array) {
//!     println!("PRISM: {:.1} BPM (SNR: {:.1} dB)", result.bpm, result.snr);
//! }
//! ```

pub mod dsp;
pub mod rppg;
pub mod wavelet;

// Legacy DSP exports
pub use dsp::{DspProcessor, FilterConfig};

// New SOTA DSP exports
pub use dsp::{adaptive_bandpass_filter, AdaptiveFilter, AdaptiveFilterConfig};
pub use dsp::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};

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

