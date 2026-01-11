//! Sensory systems module
//!
//! Binaural, soundscape, and haptics

pub mod binaural;
pub mod haptics;
pub mod soundscape;

pub use binaural::{BinauralEngine, BrainWaveState};
pub use haptics::{HapticEngine, HapticPattern};
pub use soundscape::{LayerConfig, SoundscapeEngine};
