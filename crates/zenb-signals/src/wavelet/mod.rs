//! Wavelet Transform module
//!
//! Provides continuous wavelet transform (CWT) implementations for frequency analysis.
//!
//! ## Legacy API (backward compatible)
//! - `MorletWavelet` - Original FFT-based CWT
//! - `WaveletBands` - Frequency band extraction
//!
//! ## New SOTA Components
//! - `FastCWT` - 100x+ speedup with precomputed signal FFT
//! - `AldtfDenoiser` - Adaptive wavelet denoising (+3-10 dB SNR)

mod denoising;
mod fcwt;
mod legacy;

// Re-export legacy API for backward compatibility
pub use legacy::{BandType, MorletConfig, MorletWavelet, WaveletBands};

// Export new SOTA components
pub use denoising::{aldtf_denoise, AldtfConfig, AldtfDenoiser};
pub use fcwt::{FastCWT, FastCwtConfig};
