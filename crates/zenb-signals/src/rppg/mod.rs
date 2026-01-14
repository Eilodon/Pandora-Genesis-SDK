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

mod ensemble;
mod legacy;
mod prism;

// Re-export legacy API for backward compatibility
pub use legacy::{RppgMethod, RppgProcessor, RppgResult};

// Export new SOTA components
pub use ensemble::{EnsembleConfig, EnsembleProcessor, EnsembleResult};
pub use prism::{PrismConfig, PrismProcessor, PrismResult};
