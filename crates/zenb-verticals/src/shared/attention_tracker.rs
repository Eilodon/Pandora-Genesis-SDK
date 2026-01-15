//! Attention tracking utility
//!
//! Light-weight attention scoring from gaze + blink metrics.

use super::{EyeMetricsAnalyzer, GazeEstimator, GazeResult};
use zenb_signals::beauty::CanonicalLandmarks;

#[derive(Debug, Clone, Default)]
pub struct AttentionMetrics {
    pub attention_score: f32,
    pub on_target: bool,
    pub blink_rate: f32,
    pub gaze: GazeResult,
}

pub struct AttentionTracker {
    gaze_estimator: GazeEstimator,
    eye_metrics: EyeMetricsAnalyzer,
    smoothing_alpha: f32,
    smoothed_score: f32,
}

impl AttentionTracker {
    pub fn new() -> Self {
        Self {
            gaze_estimator: GazeEstimator::new(),
            eye_metrics: EyeMetricsAnalyzer::new(),
            smoothing_alpha: 0.2,
            smoothed_score: 0.0,
        }
    }

    pub fn update(
        &mut self,
        landmarks: &CanonicalLandmarks,
        head_pose: Option<[f32; 3]>,
        timestamp_us: i64,
    ) -> AttentionMetrics {
        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);
        let (_ear, blink, _perclos) = self.eye_metrics.analyze(landmarks, timestamp_us);

        let raw_score = if gaze.on_screen { 1.0 } else { 0.3 };
        self.smoothed_score = self.smoothed_score * self.smoothing_alpha
            + raw_score * (1.0 - self.smoothing_alpha);

        AttentionMetrics {
            attention_score: self.smoothed_score.clamp(0.0, 1.0),
            on_target: gaze.on_screen,
            blink_rate: blink.blinks_per_minute,
            gaze,
        }
    }

    pub fn reset(&mut self) {
        self.gaze_estimator.reset();
        self.eye_metrics.reset();
        self.smoothed_score = 0.0;
    }
}

impl Default for AttentionTracker {
    fn default() -> Self {
        Self::new()
    }
}
