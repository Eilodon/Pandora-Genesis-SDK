//! # zenb-signals
//!
//! Biometric signal processing for ZenB.
//!
//! This crate provides:
//! - **rPPG algorithms**: CHROM and POS for heart rate extraction from RGB video
//! - **DSP functions**: FFT, filtering, and signal analysis
//!
//! ## Example
//!
//! ```ignore
//! use zenb_signals::{RppgProcessor, RppgMethod};
//!
//! let mut processor = RppgProcessor::new(RppgMethod::Pos, 90, 30.0);
//!
//! // Add RGB samples from camera
//! for frame in video_frames {
//!     processor.add_sample(frame.r, frame.g, frame.b);
//! }
//!
//! // Extract heart rate
//! if let Some((bpm, confidence)) = processor.process() {
//!     println!("Heart rate: {:.1} BPM (confidence: {:.2})", bpm, confidence);
//! }
//! ```

pub mod dsp;
pub mod rppg;

pub use dsp::{DspProcessor, FilterConfig};
pub use rppg::{RppgMethod, RppgProcessor, RppgResult};
