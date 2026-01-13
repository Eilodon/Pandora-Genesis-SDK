//! Observation types for the biofeedback domain.

use crate::core::DomainObservation;
use serde::{Deserialize, Serialize};

/// Biofeedback sensor observation.
///
/// This struct captures a single point-in-time reading from
/// physiological sensors (heart rate monitor, HRV sensor, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioObservation {
    /// Heart rate in BPM (beats per minute).
    pub hr_bpm: Option<f32>,
    
    /// Heart rate variability (RMSSD in milliseconds).
    pub hrv_rmssd: Option<f32>,
    
    /// Respiratory rate in BPM.
    pub rr_bpm: Option<f32>,
    
    /// Signal quality [0, 1].
    pub signal_quality: f32,
    
    /// Motion intensity [0, 1].
    pub motion: f32,
    
    /// Timestamp in microseconds.
    pub timestamp_us: i64,
}

impl Default for BioObservation {
    fn default() -> Self {
        Self {
            hr_bpm: None,
            hrv_rmssd: None,
            rr_bpm: None,
            signal_quality: 1.0,
            motion: 0.0,
            timestamp_us: 0,
        }
    }
}

impl BioObservation {
    /// Create a new observation with all values.
    pub fn new(
        hr_bpm: Option<f32>,
        hrv_rmssd: Option<f32>,
        rr_bpm: Option<f32>,
        signal_quality: f32,
        motion: f32,
        timestamp_us: i64,
    ) -> Self {
        Self {
            hr_bpm,
            hrv_rmssd,
            rr_bpm,
            signal_quality,
            motion,
            timestamp_us,
        }
    }
    
    /// Create from a feature array (legacy compatibility).
    ///
    /// Expected format: [HR, HRV, RR, Quality, Motion]
    pub fn from_features(features: &[f32], timestamp_us: i64) -> Self {
        Self {
            hr_bpm: features.first().copied(),
            hrv_rmssd: features.get(1).copied(),
            rr_bpm: features.get(2).copied(),
            signal_quality: features.get(3).copied().unwrap_or(1.0),
            motion: features.get(4).copied().unwrap_or(0.0),
            timestamp_us,
        }
    }
}

impl DomainObservation for BioObservation {
    fn timestamp_us(&self) -> i64 {
        self.timestamp_us
    }
    
    fn quality(&self) -> f32 {
        self.signal_quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_features() {
        let features = vec![72.0, 45.0, 6.5, 0.9, 0.1];
        let obs = BioObservation::from_features(&features, 1000);
        
        assert_eq!(obs.hr_bpm, Some(72.0));
        assert_eq!(obs.hrv_rmssd, Some(45.0));
        assert_eq!(obs.rr_bpm, Some(6.5));
        assert_eq!(obs.signal_quality, 0.9);
        assert_eq!(obs.motion, 0.1);
        assert_eq!(obs.timestamp_us, 1000);
    }
    
    #[test]
    fn test_domain_observation_trait() {
        let obs = BioObservation::new(Some(70.0), Some(40.0), Some(6.0), 0.85, 0.05, 5000);
        assert_eq!(obs.timestamp_us(), 5000);
        assert_eq!(obs.quality(), 0.85);
    }
}
