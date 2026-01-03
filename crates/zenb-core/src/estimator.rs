use crate::domain::ControlDecision;

/// Sensor estimate result
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    pub ts_us: i64,
    pub hr_bpm: Option<f32>,
    pub rr_bpm: Option<f32>,
    pub rmssd: Option<f32>,
    pub confidence: f32,
}

/// Minimum time between updates to filter sensor bursts (10ms)
const MIN_UPDATE_INTERVAL_US: i64 = 10_000;

/// Simple EMA estimator for HR, RR, RMSSD with dt-aware alpha.
#[derive(Debug, Clone)]
pub struct Estimator {
    hr_ema: Option<f32>,
    rr_ema: Option<f32>,
    rmssd_ema: Option<f32>,
    last_ts_us: Option<i64>,
    last_estimate: Option<Estimate>,
}

impl Default for Estimator {
    fn default() -> Self {
        Self {
            hr_ema: None,
            rr_ema: None,
            rmssd_ema: None,
            last_ts_us: None,
            last_estimate: None,
        }
    }
}

impl Estimator {
    /// Ingest a feature vector. Layout (optional): [hr_bpm, rmssd, rr_bpm]
    pub fn ingest(&mut self, features: &[f32], ts_us: i64) -> Estimate {
        // Calculate dt_us for burst detection
        let dt_us = match self.last_ts_us {
            Some(last) => (ts_us - last).max(0),
            None => 0,
        };

        // Skip near-duplicate updates (sensor burst protection)
        // Return cached estimate if available, otherwise continue with initialization
        if dt_us > 0 && dt_us < MIN_UPDATE_INTERVAL_US {
            if let Some(ref cached) = self.last_estimate {
                return cached.clone();
            }
            // If no cached estimate, continue to initialize
        }

        // extract inputs
        let hr = features.get(0).cloned();
        let rmssd = features.get(1).cloned();
        let rr = features.get(2).cloned();

        // dt in seconds
        let dt_s = (dt_us as f32) / 1_000_000f32;

        // First sample (dt_us == 0) uses alpha=1.0 for direct initialization
        // Subsequent samples use exponential decay based on dt
        let alpha = if dt_us == 0 {
            1.0
        } else {
            (1.0 - (-dt_s).exp()).clamp(0.01, 0.9)
        };

        if let Some(v) = hr {
            self.hr_ema = Some(match self.hr_ema {
                Some(prev) => prev * (1.0 - alpha) + v * alpha,
                None => v,
            });
        }
        if let Some(v) = rmssd {
            self.rmssd_ema = Some(match self.rmssd_ema {
                Some(prev) => prev * (1.0 - alpha) + v * alpha,
                None => v,
            });
        }
        if let Some(v) = rr {
            self.rr_ema = Some(match self.rr_ema {
                Some(prev) => prev * (1.0 - alpha) + v * alpha,
                None => v,
            });
        }

        self.last_ts_us = Some(ts_us);

        // confidence heuristic: presence of rr + hr and rmssd, and magnitude of rmssd
        let mut conf = 0.0f32;
        if self.hr_ema.is_some() {
            conf += 0.35;
        }
        if self.rr_ema.is_some() {
            conf += 0.35;
        }
        if self.rmssd_ema.is_some() {
            conf += 0.3;
        }
        // more rmssd implies better signal (not perfect, but simple)
        if let Some(rm) = self.rmssd_ema {
            conf *= (rm / (rm + 20.0)).clamp(0.0, 1.0);
        }
        let conf = conf.clamp(0.0, 1.0);

        let estimate = Estimate {
            ts_us,
            hr_bpm: self.hr_ema,
            rr_bpm: self.rr_ema,
            rmssd: self.rmssd_ema,
            confidence: conf,
        };

        // Cache estimate for burst protection
        self.last_estimate = Some(estimate.clone());

        estimate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_basics() {
        let mut est = Estimator::default();
        let e1 = est.ingest(&[60.0, 50.0, 6.0], 0);
        assert!(e1.confidence > 0.0);
        let e2 = est.ingest(&[62.0, 48.0, 6.1], 1_000_000);
        assert!(e2.hr_bpm.is_some());
        assert!(e2.rr_bpm.is_some());
        assert!(e2.confidence >= e1.confidence - 0.5);
    }
}
