//! Stress profiling helper
//!
//! Simple stress score from HR/HRV trends.

#[derive(Debug, Clone, Default)]
pub struct StressProfile {
    pub stress_score: f32,
    pub hr_bpm: Option<f32>,
    pub hrv_rmssd: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct StressConfig {
    pub hrv_low_threshold_ms: f32,
    pub hr_high_threshold_bpm: f32,
    pub smoothing_alpha: f32,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            hrv_low_threshold_ms: 20.0,
            hr_high_threshold_bpm: 100.0,
            smoothing_alpha: 0.2,
        }
    }
}

pub struct StressProfiler {
    config: StressConfig,
    smoothed_score: f32,
}

impl StressProfiler {
    pub fn new() -> Self {
        Self::with_config(StressConfig::default())
    }

    pub fn with_config(config: StressConfig) -> Self {
        Self {
            config,
            smoothed_score: 0.0,
        }
    }

    pub fn update(&mut self, hr_bpm: Option<f32>, hrv_rmssd: Option<f32>) -> StressProfile {
        let mut score = 0.0;

        if let Some(hr) = hr_bpm {
            if hr > self.config.hr_high_threshold_bpm {
                score += ((hr - self.config.hr_high_threshold_bpm) / 40.0).clamp(0.0, 1.0) * 0.5;
            }
        }

        if let Some(hrv) = hrv_rmssd {
            if hrv < self.config.hrv_low_threshold_ms {
                score += (1.0 - (hrv / self.config.hrv_low_threshold_ms)).clamp(0.0, 1.0) * 0.5;
            }
        }

        self.smoothed_score = self.smoothed_score * self.config.smoothing_alpha
            + score * (1.0 - self.config.smoothing_alpha);

        StressProfile {
            stress_score: self.smoothed_score.clamp(0.0, 1.0),
            hr_bpm,
            hrv_rmssd,
        }
    }

    pub fn reset(&mut self) {
        self.smoothed_score = 0.0;
    }
}

impl Default for StressProfiler {
    fn default() -> Self {
        Self::new()
    }
}
