//! Machine Learning module for Pandora CWM
//! 
//! This module provides ML capabilities for the Causal World Model,
//! including prediction, training, and inference components.

#[cfg(feature = "ml")]
pub mod predictor;

#[cfg(feature = "ml")]
pub mod trainer;

// Re-export main components
#[cfg(feature = "ml")]
pub use predictor::*;

#[cfg(feature = "ml")]
pub use trainer::*;