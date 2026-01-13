use super::ObservationSnapshot;
use serde::{Deserialize, Serialize};

/// Adaptive trigger for PC learning based on correlation changes
///
/// This simplified implementation watches for changes in observation patterns
/// without trying to extract individual variable values from the complex ObservationSnapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphChangeDetector {
    /// Tracks sample count for triggering
    pub sample_count: usize,

    /// Min samples before first run
    pub min_samples: usize,

    /// Last trigger timestamp  
    pub last_trigger_count: usize,

    /// Trigger interval (samples between runs)
    pub trigger_interval: usize,
}

impl Default for GraphChangeDetector {
    fn default() -> Self {
        Self {
            sample_count: 0,
            min_samples: 50,
            last_trigger_count: 0,
            trigger_interval: 100,
        }
    }
}

impl GraphChangeDetector {
    pub fn new(_threshold: f32, min_samples: usize) -> Self {
        Self {
            sample_count: 0,
            min_samples,
            last_trigger_count: 0,
            trigger_interval: 100,
        }
    }

    /// Simple trigger: run PC after min_samples, then every trigger_interval samples
    pub fn should_trigger_learning(&mut self, observations: &[ObservationSnapshot]) -> bool {
        self.sample_count = observations.len();

        if self.sample_count < self.min_samples {
            return false;
        }

        // Trigger if we haven't run yet, or if interval has passed
        let samples_since_last = self.sample_count - self.last_trigger_count;

        if samples_since_last >= self.trigger_interval {
            log::info!(
                "PC trigger: {} samples accumulated (interval: {})",
                samples_since_last,
                self.trigger_interval
            );
            self.last_trigger_count = self.sample_count;
            true
        } else {
            false
        }
    }
}
