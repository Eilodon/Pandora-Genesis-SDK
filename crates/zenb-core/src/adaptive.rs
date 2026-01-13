//! Adaptive thresholds and anomaly detection for self-tuning systems.
//!
//! Ported from Pandora's `pandora_mcg::enhanced_mcg` module.
//! Provides three key components:
//! 1. `AdaptiveThreshold` - Self-adjusting thresholds based on performance feedback
//! 2. `AnomalyDetector` - Z-score based anomaly detection with sliding window
//! 3. `ConfidenceTracker` - Decision confidence based on historical success rate
//!
//! # Performance Requirements
//! - AdaptiveThreshold::adapt() < 100ns
//! - AnomalyDetector::score() < 500ns
//! - ConfidenceTracker::compute() < 100ns
//!
//! # Example
//! ```rust
//! use zenb_core::adaptive::{AdaptiveThreshold, AnomalyDetector, ConfidenceTracker};
//!
//! // Self-adjusting threshold
//! let mut threshold = AdaptiveThreshold::new(0.5, 0.2, 0.8, 0.05);
//! let value = 0.6;
//! if value > threshold.get() {
//!     // Take action...
//! }
//! // After observing outcome, adapt
//! let performance_delta = 0.1;
//! threshold.adapt(performance_delta);
//! ```

use std::collections::VecDeque;

// ============================================================================
// Adaptive Threshold
// ============================================================================

/// Self-adjusting threshold that adapts based on performance feedback.
///
/// When performance is good (positive delta), the threshold becomes more aggressive (lower).
/// When performance is bad (negative delta), the threshold becomes more conservative (higher).
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    /// Base value (starting point)
    base_value: f32,
    /// Current adjusted value
    current_value: f32,
    /// Rate of adjustment (0.01 - 0.1 typical)
    learning_rate: f32,
    /// Minimum allowed value
    min_value: f32,
    /// Maximum allowed value
    max_value: f32,
}

impl AdaptiveThreshold {
    /// Creates a new adaptive threshold.
    ///
    /// # Arguments
    /// * `base` - Initial threshold value
    /// * `min` - Minimum allowed value
    /// * `max` - Maximum allowed value
    /// * `learning_rate` - How quickly to adapt (0.01 - 0.1)
    pub fn new(base: f32, min: f32, max: f32, learning_rate: f32) -> Self {
        debug_assert!(
            min <= base && base <= max,
            "Invalid bounds: min <= base <= max"
        );
        debug_assert!(
            learning_rate > 0.0 && learning_rate < 1.0,
            "Learning rate must be in (0, 1)"
        );

        Self {
            base_value: base,
            current_value: base,
            learning_rate,
            min_value: min,
            max_value: max,
        }
    }

    /// Gets the current threshold value.
    #[inline]
    pub fn get(&self) -> f32 {
        self.current_value
    }

    /// Gets the base (initial) threshold value.
    #[inline]
    pub fn base(&self) -> f32 {
        self.base_value
    }

    /// Adapts threshold based on performance feedback.
    ///
    /// Positive `performance_delta` → threshold decreases (more aggressive)
    /// Negative `performance_delta` → threshold increases (more conservative)
    pub fn adapt(&mut self, performance_delta: f32) {
        // Invert: good performance (positive) should lower threshold
        let adjustment = -performance_delta * self.learning_rate;
        self.current_value =
            (self.current_value + adjustment).clamp(self.min_value, self.max_value);

        log::trace!(
            "Adapted threshold: {:.4} → {:.4} (delta: {:.4})",
            self.base_value,
            self.current_value,
            adjustment
        );
    }

    /// Resets threshold to base value.
    pub fn reset(&mut self) {
        self.current_value = self.base_value;
    }

    /// Returns how far the current value has drifted from base (in %).
    pub fn drift_percent(&self) -> f32 {
        if self.base_value.abs() < 1e-6 {
            return 0.0;
        }
        ((self.current_value - self.base_value) / self.base_value * 100.0).abs()
    }
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new(0.5, 0.1, 0.9, 0.05)
    }
}

// ============================================================================
// Anomaly Detector
// ============================================================================

/// Z-score based anomaly detection with sliding window.
///
/// Maintains a window of recent observations and flags values that are
/// more than `threshold_std` standard deviations from the mean.
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Size of the sliding window
    window_size: usize,
    /// Historical values
    history: VecDeque<f32>,
    /// Number of std deviations to consider anomalous
    threshold_std: f32,
}

impl AnomalyDetector {
    /// Creates a new anomaly detector.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent observations to track
    /// * `threshold_std` - Z-score threshold for anomaly detection (2.0 - 3.0 typical)
    pub fn new(window_size: usize, threshold_std: f32) -> Self {
        debug_assert!(window_size >= 3, "Window size must be at least 3");
        debug_assert!(threshold_std > 0.0, "Threshold must be positive");

        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
            threshold_std,
        }
    }

    /// Adds a new observation and returns anomaly score.
    ///
    /// Returns 0.0 if not anomalous, or normalized score > 0 if anomalous.
    /// Score of 1.0 means exactly at the threshold; higher means more anomalous.
    pub fn score(&mut self, value: f32) -> f32 {
        // Maintain window size
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(value);

        // Need minimum data for statistics
        if self.history.len() < 3 {
            return 0.0;
        }

        // Calculate mean and std
        let mean: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance: f32 = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / self.history.len() as f32;
        let std = variance.sqrt();

        // Avoid division by zero
        if std < 1e-6 {
            return 0.0;
        }

        // Z-score: how many standard deviations away from mean
        let z_score = ((value - mean) / std).abs();

        // Return normalized anomaly score if beyond threshold
        if z_score > self.threshold_std {
            log::debug!(
                "Anomaly detected: value={:.4}, mean={:.4}, std={:.4}, z={:.4}",
                value,
                mean,
                std,
                z_score
            );
            z_score / self.threshold_std // Normalized: 1.0 = at threshold
        } else {
            0.0
        }
    }

    /// Returns current window statistics (mean, std).
    pub fn stats(&self) -> (f32, f32) {
        if self.history.len() < 2 {
            return (0.0, 0.0);
        }

        let mean: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance: f32 = self.history.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / self.history.len() as f32;

        (mean, variance.sqrt())
    }

    /// Clears history.
    pub fn reset(&mut self) {
        self.history.clear();
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new(50, 2.5)
    }
}

// ============================================================================
// Confidence Tracker
// ============================================================================

/// Tracks decision confidence based on historical success rate.
///
/// Maintains a sliding window of recent decision outcomes and computes
/// confidence that can be used to weight decisions or trigger conservative modes.
#[derive(Debug, Clone)]
pub struct ConfidenceTracker {
    /// Total successes in window
    success_count: usize,
    /// Total decisions in window
    total_count: usize,
    /// Sliding window of outcomes
    recent_outcomes: VecDeque<bool>,
    /// Window size
    window_size: usize,
}

impl ConfidenceTracker {
    /// Creates a new confidence tracker.
    ///
    /// # Arguments
    /// * `window_size` - Number of recent outcomes to track
    pub fn new(window_size: usize) -> Self {
        debug_assert!(window_size > 0, "Window size must be positive");

        Self {
            success_count: 0,
            total_count: 0,
            recent_outcomes: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Computes confidence, optionally penalized by anomaly score.
    ///
    /// Returns value in [0, 1] where 1.0 is highest confidence.
    pub fn compute(&self, anomaly_score: f32) -> f32 {
        let base_confidence = if self.total_count > 0 {
            self.success_count as f32 / self.total_count as f32
        } else {
            0.5 // Default moderate confidence
        };

        // Reduce confidence if anomaly detected
        let anomaly_penalty = anomaly_score * 0.3;
        (base_confidence - anomaly_penalty).clamp(0.0, 1.0)
    }

    /// Records outcome of a decision.
    pub fn update(&mut self, success: bool) {
        // Evict oldest if at capacity
        if self.recent_outcomes.len() >= self.window_size {
            if let Some(old) = self.recent_outcomes.pop_front() {
                if old {
                    self.success_count = self.success_count.saturating_sub(1);
                }
                self.total_count = self.total_count.saturating_sub(1);
            }
        }

        // Add new outcome
        self.recent_outcomes.push_back(success);
        if success {
            self.success_count += 1;
        }
        self.total_count += 1;
    }

    /// Returns current success rate.
    #[inline]
    pub fn success_rate(&self) -> f32 {
        if self.total_count > 0 {
            self.success_count as f32 / self.total_count as f32
        } else {
            0.5
        }
    }

    /// Returns total number of tracked outcomes.
    #[inline]
    pub fn sample_count(&self) -> usize {
        self.total_count
    }

    /// Resets tracker.
    pub fn reset(&mut self) {
        self.success_count = 0;
        self.total_count = 0;
        self.recent_outcomes.clear();
    }
}

impl Default for ConfidenceTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_adapt() {
        let mut threshold = AdaptiveThreshold::new(0.5, 0.2, 0.8, 0.1);

        // Good performance should lower threshold
        threshold.adapt(0.1);
        assert!(threshold.get() < 0.5);

        // Bad performance should raise threshold
        let old_value = threshold.get();
        threshold.adapt(-0.1);
        assert!(threshold.get() > old_value);
    }

    #[test]
    fn test_adaptive_threshold_clamp() {
        let mut threshold = AdaptiveThreshold::new(0.5, 0.2, 0.8, 0.5);

        // Extreme good performance - should clamp to min
        threshold.adapt(10.0);
        assert_eq!(threshold.get(), 0.2);

        // Reset and test max
        threshold.reset();
        threshold.adapt(-10.0);
        assert_eq!(threshold.get(), 0.8);
    }

    #[test]
    fn test_anomaly_detector_normal() {
        let mut detector = AnomalyDetector::new(10, 2.0);

        // Normal values - no anomaly
        for _ in 0..10 {
            assert_eq!(detector.score(5.0), 0.0);
        }
    }

    #[test]
    fn test_anomaly_detector_spike() {
        let mut detector = AnomalyDetector::new(10, 2.0);

        // Build up normal history
        for _ in 0..10 {
            detector.score(5.0);
        }

        // Anomalous spike
        let score = detector.score(50.0);
        assert!(score > 0.0, "Should detect anomaly");
    }

    #[test]
    fn test_confidence_tracker_success_rate() {
        let mut tracker = ConfidenceTracker::new(10);

        // 7 successes, 3 failures
        for _ in 0..7 {
            tracker.update(true);
        }
        for _ in 0..3 {
            tracker.update(false);
        }

        assert!((tracker.success_rate() - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_confidence_tracker_windowing() {
        let mut tracker = ConfidenceTracker::new(5);

        // Fill with failures
        for _ in 0..5 {
            tracker.update(false);
        }
        assert_eq!(tracker.success_rate(), 0.0);

        // Replace with successes
        for _ in 0..5 {
            tracker.update(true);
        }
        assert_eq!(tracker.success_rate(), 1.0);
    }

    #[test]
    fn test_confidence_with_anomaly_penalty() {
        let mut tracker = ConfidenceTracker::new(10);

        // 100% success rate
        for _ in 0..10 {
            tracker.update(true);
        }

        // No anomaly - full confidence
        assert_eq!(tracker.compute(0.0), 1.0);

        // High anomaly - reduced confidence
        let conf = tracker.compute(1.0);
        assert!(conf < 1.0);
    }
}
