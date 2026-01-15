//! Drowsiness Detection
//!
//! Combines multiple indicators:
//! - PERCLOS (primary)
//! - Blink rate changes

use crate::shared::{PerclosResult, BlinkResult};

/// Drowsiness level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrowsinessLevel {
    Alert,
    Mild,
    Moderate,
    Severe,
}

/// Drowsiness detection result
#[derive(Debug, Clone)]
pub struct DrowsinessResult {
    /// Drowsiness level (0-1)
    pub level: f32,
    /// Classification
    pub classification: DrowsinessLevel,
    /// PERCLOS contribution
    pub perclos_score: f32,
    /// Blink rate contribution
    pub blink_score: f32,
}

/// Drowsiness Detector
pub struct DrowsinessDetector {
    perclos_mild: f32,
    perclos_moderate: f32,
    perclos_severe: f32,

    baseline_blink_rate: Option<f32>,
    blink_history: Vec<f32>,

    smoothed_level: f32,
    alpha: f32,
}

impl DrowsinessDetector {
    pub fn new() -> Self {
        Self {
            perclos_mild: 0.15,
            perclos_moderate: 0.30,
            perclos_severe: 0.50,
            baseline_blink_rate: None,
            blink_history: Vec::with_capacity(60),
            smoothed_level: 0.0,
            alpha: 0.1,
        }
    }

    /// Update drowsiness detection
    pub fn update(&mut self, perclos: &PerclosResult, blink: &BlinkResult) -> DrowsinessResult {
        self.blink_history.push(blink.blinks_per_minute);
        if self.blink_history.len() > 60 {
            self.blink_history.remove(0);
        }

        if self.baseline_blink_rate.is_none() && self.blink_history.len() >= 30 {
            self.baseline_blink_rate = Some(
                self.blink_history.iter().sum::<f32>() / self.blink_history.len() as f32,
            );
        }

        let perclos_score = perclos.perclos;

        let blink_score = if let Some(baseline) = self.baseline_blink_rate {
            let _deviation = (blink.blinks_per_minute - baseline).abs() / baseline.max(1.0);
            if blink.blinks_per_minute < baseline * 0.5 {
                0.5
            } else if blink.blinks_per_minute > baseline * 1.5 {
                0.3
            } else {
                0.0
            }
        } else {
            0.0
        };

        let raw_level = perclos_score * 0.7 + blink_score * 0.3;

        self.smoothed_level = self.smoothed_level * (1.0 - self.alpha) + raw_level * self.alpha;

        let classification = if self.smoothed_level < self.perclos_mild {
            DrowsinessLevel::Alert
        } else if self.smoothed_level < self.perclos_moderate {
            DrowsinessLevel::Mild
        } else if self.smoothed_level < self.perclos_severe {
            DrowsinessLevel::Moderate
        } else {
            DrowsinessLevel::Severe
        };

        DrowsinessResult {
            level: self.smoothed_level,
            classification,
            perclos_score,
            blink_score,
        }
    }

    pub fn reset(&mut self) {
        self.baseline_blink_rate = None;
        self.blink_history.clear();
        self.smoothed_level = 0.0;
    }
}

impl Default for DrowsinessDetector {
    fn default() -> Self {
        Self::new()
    }
}
