//! Remote Photoplethysmography (rPPG) module
//!
//! Provides multiple algorithms for heart rate extraction from RGB video.
//!
//! ## Legacy API (backward compatible)
//! - `RppgProcessor` - CHROM and POS algorithms
//! - `RppgMethod` - Selection enum
//! - `RppgResult` - Processing result
//!
//! ## New SOTA Components  
//! - `PrismProcessor` - PRISM adaptive algorithm (2025 SOTA)
//! - `EnsembleProcessor` - Multi-algorithm voting for robustness
//! - `AponNoiseEstimator` - APON noise direction for PRISM warm-start
//! - `MultiRoiProcessor` - Grid-based processing with weighted voting

pub mod apon;
mod ensemble;
mod legacy;
pub mod multi_roi;
mod prism;

// Re-export legacy API for backward compatibility
pub use legacy::{RppgMethod, RppgProcessor, RppgResult};

// Export new SOTA components
pub use apon::{AponConfig, AponNoiseEstimator, AponResult};
pub use ensemble::{EnsembleConfig, EnsembleProcessor, EnsembleResult};
pub use multi_roi::{MultiRoiConfig, MultiRoiProcessor, MultiRoiResult, RoiResult, RoiSignal};
pub use prism::{PrismConfig, PrismProcessor, PrismResult};
