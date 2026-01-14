//! Perception Subsystem
//!
//! # EIDOLON FIX: Engine Decomposition (Phase 2)
//!
//! Extracts perception-related fields from the Engine struct to reduce
//! cognitive load and improve maintainability. This subsystem handles:
//!
//! - Resonance tracking and scoring
//! - Sheaf energy monitoring
//! - Sensor anomaly detection
//!
//! # Invariants
//! - `resonance_score_ema` is always in [0.0, 1.0]
//! - `last_sheaf_energy` is non-negative
//! - Anomaly detector maintains rolling statistics

use crate::adaptive::AnomalyDetector;
use crate::config::ZenbConfig;
use crate::resonance::{ResonanceFeatures, ResonanceTracker};

/// Perception Subsystem - unified perception state management.
///
/// # EIDOLON FIX: Engine Decomposition
/// Extracts resonance and anomaly detection from Engine.
///
/// Note: This struct intentionally does not derive Serialize/Deserialize
/// because some contained types (AnomalyDetector) don't implement serde.
/// Use specific getters/setters for state persistence if needed.
#[derive(Debug, Clone)]
pub struct PerceptionSubsystem {
    /// Resonance tracker for coherence detection
    tracker: ResonanceTracker,
    /// Last computed resonance score
    last_resonance_score: f32,
    /// Exponential moving average of resonance score
    score_ema: f32,
    /// EMA smoothing factor
    ema_alpha: f32,
    /// Last sheaf energy value (from SheafPerception)
    sheaf_energy: f32,
    /// Anomaly detector for sensor readings
    anomaly_detector: AnomalyDetector,
}

impl PerceptionSubsystem {
    /// Create new perception subsystem with default settings.
    pub fn new() -> Self {
        Self {
            tracker: ResonanceTracker::default(),
            last_resonance_score: 0.0,
            score_ema: 0.5,
            ema_alpha: 0.1,
            sheaf_energy: 0.0,
            anomaly_detector: AnomalyDetector::default(),
        }
    }
    
    /// Create with custom EMA smoothing factor.
    pub fn with_ema_alpha(alpha: f32) -> Self {
        Self {
            ema_alpha: alpha.clamp(0.01, 0.99),
            ..Self::new()
        }
    }
    
    /// Update resonance from sensor features.
    /// 
    /// # Arguments
    /// * `ts_us` - Timestamp in microseconds
    /// * `guide_phase_norm` - Normalized guide phase [0, 1]
    /// * `guide_bpm` - Guide breath rate in BPM
    /// * `rr_bpm` - Optional respiration rate in BPM
    /// * `cfg` - ZenB configuration
    /// 
    /// # Returns
    /// Current resonance features
    pub fn update_resonance(
        &mut self,
        ts_us: i64,
        guide_phase_norm: f32,
        guide_bpm: f32,
        rr_bpm: Option<f32>,
        cfg: &ZenbConfig,
    ) -> ResonanceFeatures {
        let features = self.tracker.update(ts_us, guide_phase_norm, guide_bpm, rr_bpm, cfg);
        self.last_resonance_score = features.resonance_score;
        self.score_ema = self.score_ema * (1.0 - self.ema_alpha) 
            + features.resonance_score * self.ema_alpha;
        features
    }
    
    /// Get current resonance score.
    pub fn resonance_score(&self) -> f32 {
        self.last_resonance_score
    }
    
    /// Get smoothed resonance EMA.
    pub fn resonance_ema(&self) -> f32 {
        self.score_ema
    }
    
    /// Update sheaf energy (from SheafPerception).
    pub fn set_sheaf_energy(&mut self, energy: f32) {
        self.sheaf_energy = energy.max(0.0);
    }
    
    /// Get last sheaf energy.
    pub fn sheaf_energy(&self) -> f32 {
        self.sheaf_energy
    }
    
    /// Score a sensor reading for anomaly.
    /// Returns anomaly score (0 = normal, >0 = anomalous).
    pub fn score_anomaly(&mut self, value: f32) -> f32 {
        self.anomaly_detector.score(value)
    }
    
    /// Check if sensor reading is anomalous (score > 0).
    pub fn is_anomaly(&mut self, value: f32) -> bool {
        self.anomaly_detector.score(value) > 0.0
    }
    
    /// Get anomaly detector reference.
    pub fn anomaly_detector(&self) -> &AnomalyDetector {
        &self.anomaly_detector
    }
    
    /// Get mutable anomaly detector reference.
    pub fn anomaly_detector_mut(&mut self) -> &mut AnomalyDetector {
        &mut self.anomaly_detector
    }
    
    /// Get resonance tracker reference.
    pub fn tracker(&self) -> &ResonanceTracker {
        &self.tracker
    }
    
    /// Get mutable resonance tracker reference.
    pub fn tracker_mut(&mut self) -> &mut ResonanceTracker {
        &mut self.tracker
    }
    
    /// Get diagnostics tuple: (resonance_score, ema, sheaf_energy)
    pub fn diagnostics(&self) -> (f32, f32, f32) {
        (self.last_resonance_score, self.score_ema, self.sheaf_energy)
    }
}

impl Default for PerceptionSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perception_subsystem_creation() {
        let ps = PerceptionSubsystem::new();
        assert_eq!(ps.resonance_score(), 0.0);
        assert_eq!(ps.sheaf_energy(), 0.0);
    }
    
    #[test]
    fn test_sheaf_energy_clamped() {
        let mut ps = PerceptionSubsystem::new();
        ps.set_sheaf_energy(-5.0);
        assert_eq!(ps.sheaf_energy(), 0.0);
    }
    
    #[test]
    fn test_ema_alpha_clamped() {
        let ps = PerceptionSubsystem::with_ema_alpha(2.0);
        assert!(ps.ema_alpha <= 0.99);
        
        let ps2 = PerceptionSubsystem::with_ema_alpha(-0.5);
        assert!(ps2.ema_alpha >= 0.01);
    }
}
