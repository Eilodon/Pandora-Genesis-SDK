//! Eye Metrics Analysis
//!
//! Computes eye-related metrics from MediaPipe 468 landmarks:
//! - Eye Aspect Ratio (EAR) for blink detection
//! - PERCLOS for drowsiness
//! - Blink rate and duration

use zenb_signals::beauty::{CanonicalLandmarks, landmark_distance};

/// MediaPipe eye landmark indices
pub mod eye_indices {
    // Left eye contour (6 points for EAR)
    pub const LEFT_EYE_TOP_1: usize = 159;
    pub const LEFT_EYE_TOP_2: usize = 158;
    pub const LEFT_EYE_BOTTOM_1: usize = 145;
    pub const LEFT_EYE_BOTTOM_2: usize = 153;
    pub const LEFT_EYE_INNER: usize = 133;
    pub const LEFT_EYE_OUTER: usize = 33;

    // Right eye contour
    pub const RIGHT_EYE_TOP_1: usize = 386;
    pub const RIGHT_EYE_TOP_2: usize = 385;
    pub const RIGHT_EYE_BOTTOM_1: usize = 374;
    pub const RIGHT_EYE_BOTTOM_2: usize = 380;
    pub const RIGHT_EYE_INNER: usize = 362;
    pub const RIGHT_EYE_OUTER: usize = 263;
}

/// Eye Aspect Ratio result
#[derive(Debug, Clone, Default)]
pub struct EarResult {
    /// Left eye EAR (0 = closed, ~0.3 = open)
    pub left_ear: f32,
    /// Right eye EAR
    pub right_ear: f32,
    /// Average EAR
    pub avg_ear: f32,
    /// Is eye closed (EAR < threshold)
    pub is_closed: bool,
}

/// Blink detection result
#[derive(Debug, Clone, Default)]
pub struct BlinkResult {
    /// Blink detected in this frame
    pub blink_detected: bool,
    /// Blink duration in ms (if blink just ended)
    pub blink_duration_ms: Option<f32>,
    /// Blinks per minute (rolling average)
    pub blinks_per_minute: f32,
}

/// PERCLOS result (Percentage of Eye Closure)
#[derive(Debug, Clone, Default)]
pub struct PerclosResult {
    /// PERCLOS value (0-1, >0.4 = drowsy)
    pub perclos: f32,
    /// Drowsiness level (0-1)
    pub drowsiness_level: f32,
    /// Is drowsy flag
    pub is_drowsy: bool,
}

/// Eye Metrics Analyzer configuration
#[derive(Debug, Clone)]
pub struct EyeMetricsConfig {
    /// EAR threshold for closed eye
    pub ear_threshold: f32,
    /// PERCLOS window in seconds
    pub perclos_window_sec: f32,
    /// PERCLOS threshold for drowsiness
    pub perclos_threshold: f32,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Minimum blink duration in ms
    pub min_blink_duration_ms: f32,
    /// Maximum blink duration in ms
    pub max_blink_duration_ms: f32,
}

impl Default for EyeMetricsConfig {
    fn default() -> Self {
        Self {
            ear_threshold: 0.21,
            perclos_window_sec: 60.0,
            perclos_threshold: 0.4,
            sample_rate: 30.0,
            min_blink_duration_ms: 100.0,
            max_blink_duration_ms: 400.0,
        }
    }
}

/// Eye Metrics Analyzer
pub struct EyeMetricsAnalyzer {
    config: EyeMetricsConfig,
    // Blink detection state
    eye_was_closed: bool,
    close_start_frame: usize,
    current_frame: usize,
    blink_history: Vec<i64>,
    // PERCLOS state
    ear_history: Vec<f32>,
}

impl EyeMetricsAnalyzer {
    pub fn new() -> Self {
        Self::with_config(EyeMetricsConfig::default())
    }

    pub fn with_config(config: EyeMetricsConfig) -> Self {
        let window_frames = (config.perclos_window_sec * config.sample_rate) as usize;
        Self {
            config,
            eye_was_closed: false,
            close_start_frame: 0,
            current_frame: 0,
            blink_history: Vec::with_capacity(100),
            ear_history: Vec::with_capacity(window_frames),
        }
    }

    /// Compute Eye Aspect Ratio from landmarks
    pub fn compute_ear(&self, landmarks: &CanonicalLandmarks) -> EarResult {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return EarResult::default();
        }

        let left_ear = self.compute_single_ear(
            landmarks,
            eye_indices::LEFT_EYE_TOP_1,
            eye_indices::LEFT_EYE_TOP_2,
            eye_indices::LEFT_EYE_BOTTOM_1,
            eye_indices::LEFT_EYE_BOTTOM_2,
            eye_indices::LEFT_EYE_INNER,
            eye_indices::LEFT_EYE_OUTER,
        );

        let right_ear = self.compute_single_ear(
            landmarks,
            eye_indices::RIGHT_EYE_TOP_1,
            eye_indices::RIGHT_EYE_TOP_2,
            eye_indices::RIGHT_EYE_BOTTOM_1,
            eye_indices::RIGHT_EYE_BOTTOM_2,
            eye_indices::RIGHT_EYE_INNER,
            eye_indices::RIGHT_EYE_OUTER,
        );

        let avg_ear = (left_ear + right_ear) / 2.0;
        let is_closed = avg_ear < self.config.ear_threshold;

        EarResult {
            left_ear,
            right_ear,
            avg_ear,
            is_closed,
        }
    }

    fn compute_single_ear(
        &self,
        landmarks: &CanonicalLandmarks,
        top1: usize,
        top2: usize,
        bottom1: usize,
        bottom2: usize,
        inner: usize,
        outer: usize,
    ) -> f32 {
        let v1 = landmark_distance(landmarks, top1, bottom1);
        let v2 = landmark_distance(landmarks, top2, bottom2);
        let h = landmark_distance(landmarks, inner, outer);

        if h < 0.001 {
            return 0.0;
        }

        (v1 + v2) / (2.0 * h)
    }

    /// Update blink detection with new EAR
    pub fn update_blink(&mut self, ear: &EarResult, timestamp_us: i64) -> BlinkResult {
        self.current_frame += 1;

        let mut result = BlinkResult::default();

        if ear.is_closed && !self.eye_was_closed {
            self.close_start_frame = self.current_frame;
        } else if !ear.is_closed && self.eye_was_closed {
            let duration_frames = self.current_frame - self.close_start_frame;
            let duration_ms = duration_frames as f32 / self.config.sample_rate * 1000.0;

            if duration_ms >= self.config.min_blink_duration_ms
                && duration_ms <= self.config.max_blink_duration_ms
            {
                result.blink_detected = true;
                result.blink_duration_ms = Some(duration_ms);
                self.blink_history.push(timestamp_us);

                let cutoff = timestamp_us - 60_000_000;
                self.blink_history.retain(|&t| t > cutoff);
            }
        }

        self.eye_was_closed = ear.is_closed;

        if !self.blink_history.is_empty() {
            let window_sec = 60.0f32
                .min((timestamp_us - self.blink_history[0]) as f32 / 1_000_000.0)
                .max(1.0);
            result.blinks_per_minute = self.blink_history.len() as f32 / window_sec * 60.0;
        }

        result
    }

    /// Update PERCLOS calculation
    pub fn update_perclos(&mut self, ear: &EarResult) -> PerclosResult {
        let window_frames = (self.config.perclos_window_sec * self.config.sample_rate) as usize;

        self.ear_history.push(ear.avg_ear);
        if self.ear_history.len() > window_frames {
            self.ear_history.remove(0);
        }

        if self.ear_history.len() < 30 {
            return PerclosResult::default();
        }

        let closed_count = self
            .ear_history
            .iter()
            .filter(|&&e| e < self.config.ear_threshold)
            .count();

        let perclos = closed_count as f32 / self.ear_history.len() as f32;

        let drowsiness_level = if perclos < 0.15 {
            0.0
        } else if perclos < self.config.perclos_threshold {
            (perclos - 0.15) / (self.config.perclos_threshold - 0.15) * 0.5
        } else {
            0.5 + (perclos - self.config.perclos_threshold)
                / (1.0 - self.config.perclos_threshold)
                * 0.5
        };

        PerclosResult {
            perclos,
            drowsiness_level: drowsiness_level.clamp(0.0, 1.0),
            is_drowsy: perclos > self.config.perclos_threshold,
        }
    }

    /// Full analysis pipeline
    pub fn analyze(
        &mut self,
        landmarks: &CanonicalLandmarks,
        timestamp_us: i64,
    ) -> (EarResult, BlinkResult, PerclosResult) {
        let ear = self.compute_ear(landmarks);
        let blink = self.update_blink(&ear, timestamp_us);
        let perclos = self.update_perclos(&ear);
        (ear, blink, perclos)
    }

    /// Reset analyzer state
    pub fn reset(&mut self) {
        self.eye_was_closed = false;
        self.close_start_frame = 0;
        self.current_frame = 0;
        self.blink_history.clear();
        self.ear_history.clear();
    }
}

impl Default for EyeMetricsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ear_threshold() {
        let config = EyeMetricsConfig::default();
        assert!((config.ear_threshold - 0.21).abs() < 0.01);
    }

    #[test]
    fn test_perclos_threshold() {
        let config = EyeMetricsConfig::default();
        assert!((config.perclos_threshold - 0.4).abs() < 0.01);
    }
}
