
//! Physiological estimators built on top of rPPG/BVP waveforms.
//!
//! Notes:
//! - HR is typically more reliable than HRV when using camera-based rPPG.
//! - These estimators implement *baseline* (deployable) approaches with validity gating.

mod hrv;
mod hrv_trend;
mod respiration;

pub use hrv::{HrvConfig, HrvMetrics, HrvResult, HrvEstimator};
pub use hrv_trend::{
    HrvBaseline, HrvTrendConfig, HrvTrendResult, HrvTrendTracker, TrendDirection,
};
pub use respiration::{MethodResults, RespirationConfig, RespirationResult, RespirationEstimator};
