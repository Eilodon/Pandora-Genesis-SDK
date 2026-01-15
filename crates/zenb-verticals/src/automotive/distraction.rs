//! Distraction Detection
//!
//! Monitors driver attention based on gaze direction.

use crate::shared::{GazeResult, GazeTarget};

/// Distraction level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistractionLevel {
    Focused,
    Mild,
    Moderate,
    Severe,
}

/// Distraction detection result
#[derive(Debug, Clone)]
pub struct DistractionResult {
    /// Distraction level (0-1)
    pub level: f32,
    /// Classification
    pub classification: DistractionLevel,
    /// Time looking away (seconds)
    pub off_road_duration_sec: f32,
    /// Current gaze target
    pub gaze_target: GazeTarget,
}

/// Distraction Detector
pub struct DistractionDetector {
    off_road_threshold_sec: f32,

    off_road_start_us: Option<i64>,
    total_off_road_us: i64,
    window_start_us: i64,

    smoothed_level: f32,
    alpha: f32,
}

impl DistractionDetector {
    pub fn new() -> Self {
        Self::with_threshold(2.0)
    }

    pub fn with_threshold(threshold_sec: f32) -> Self {
        Self {
            off_road_threshold_sec: threshold_sec,
            off_road_start_us: None,
            total_off_road_us: 0,
            window_start_us: 0,
            smoothed_level: 0.0,
            alpha: 0.1,
        }
    }

    /// Update distraction detection
    pub fn update(&mut self, gaze: &GazeResult, timestamp_us: i64) -> DistractionResult {
        if self.window_start_us == 0 {
            self.window_start_us = timestamp_us;
        }

        if !gaze.on_screen {
            if self.off_road_start_us.is_none() {
                self.off_road_start_us = Some(timestamp_us);
            }
        } else if let Some(start) = self.off_road_start_us {
            self.total_off_road_us += timestamp_us - start;
            self.off_road_start_us = None;
        }

        let current_off_road_us = if let Some(start) = self.off_road_start_us {
            timestamp_us - start
        } else {
            0
        };

        let off_road_duration_sec = current_off_road_us as f32 / 1_000_000.0;

        let raw_level = (off_road_duration_sec / self.off_road_threshold_sec).min(1.0);

        self.smoothed_level = self.smoothed_level * (1.0 - self.alpha) + raw_level * self.alpha;

        let classification = if self.smoothed_level < 0.25 {
            DistractionLevel::Focused
        } else if self.smoothed_level < 0.5 {
            DistractionLevel::Mild
        } else if self.smoothed_level < 0.75 {
            DistractionLevel::Moderate
        } else {
            DistractionLevel::Severe
        };

        if timestamp_us - self.window_start_us > 30_000_000 {
            self.total_off_road_us = 0;
            self.window_start_us = timestamp_us;
        }

        DistractionResult {
            level: self.smoothed_level,
            classification,
            off_road_duration_sec,
            gaze_target: gaze.target,
        }
    }

    pub fn reset(&mut self) {
        self.off_road_start_us = None;
        self.total_off_road_us = 0;
        self.window_start_us = 0;
        self.smoothed_level = 0.0;
    }
}

impl Default for DistractionDetector {
    fn default() -> Self {
        Self::new()
    }
}
