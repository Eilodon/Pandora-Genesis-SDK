//! Motion Detection for Adaptive rPPG Processing
//!
//! Tracks motion level over time and signals when to switch between
//! stationary (PRISM) and dynamic processing modes.
//!
//! # Usage
//!
//! ```ignore
//! let mut detector = MotionDetector::new();
//! 
//! // Feed motion signals (from optical flow, IMU, or landmark velocity)
//! detector.update(frame_motion_level);
//!
//! if detector.is_motion_active() {
//!     // Use full ensemble or dynamic mode
//! } else {
//!     // Use PRISM alone (stationary mode)
//! }
//! ```

/// Motion detector configuration
#[derive(Debug, Clone)]
pub struct MotionDetectorConfig {
    /// Motion threshold (0-1 scale) for triggering dynamic mode
    pub threshold: f32,
    /// History window size for smoothing
    pub history_size: usize,
    /// Hysteresis: require motion below threshold * factor before deactivating
    pub hysteresis_factor: f32,
    /// Minimum consecutive frames below threshold to deactivate
    pub deactivation_frames: usize,
}

impl Default for MotionDetectorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            history_size: 10,
            hysteresis_factor: 0.7,      // Deactivate at 0.3 * 0.7 = 0.21
            deactivation_frames: 5,
        }
    }
}

/// Motion detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionState {
    /// Low motion - stationary mode
    Stationary,
    /// High motion - dynamic mode
    Dynamic,
}

/// Motion detector result
#[derive(Debug, Clone)]
pub struct MotionStatus {
    /// Current motion state
    pub state: MotionState,
    /// Current motion level (smoothed)
    pub motion_level: f32,
    /// Raw (unsmoothed) motion level
    pub raw_motion: f32,
    /// Frames since last state change
    pub frames_in_state: usize,
}

/// Motion Detector for adaptive ensemble switching
pub struct MotionDetector {
    config: MotionDetectorConfig,
    /// Motion history for smoothing
    history: Vec<f32>,
    /// Current state
    state: MotionState,
    /// Frames in current state
    frames_in_state: usize,
    /// Consecutive frames below threshold (for hysteresis)
    frames_below_threshold: usize,
}

impl MotionDetector {
    /// Create with default config
    pub fn new() -> Self {
        Self::with_config(MotionDetectorConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: MotionDetectorConfig) -> Self {
        Self {
            history: Vec::with_capacity(config.history_size),
            config,
            state: MotionState::Stationary,
            frames_in_state: 0,
            frames_below_threshold: 0,
        }
    }

    /// Update with new motion measurement
    ///
    /// # Arguments
    /// * `motion` - Motion level (0-1 scale, where 0 = no motion, 1 = extreme motion)
    ///
    /// # Returns
    /// Current motion status
    pub fn update(&mut self, motion: f32) -> MotionStatus {
        let motion_clamped = motion.clamp(0.0, 1.0);
        
        // Update history
        self.history.push(motion_clamped);
        if self.history.len() > self.config.history_size {
            self.history.remove(0);
        }
        
        // Compute smoothed motion
        let smoothed = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f32>() / self.history.len() as f32
        };
        
        // State machine update
        let old_state = self.state;
        
        match self.state {
            MotionState::Stationary => {
                if smoothed >= self.config.threshold {
                    self.state = MotionState::Dynamic;
                    self.frames_in_state = 0;
                    self.frames_below_threshold = 0;
                } else {
                    self.frames_in_state += 1;
                }
            }
            MotionState::Dynamic => {
                let deactivation_threshold = self.config.threshold * self.config.hysteresis_factor;
                
                if smoothed < deactivation_threshold {
                    self.frames_below_threshold += 1;
                    
                    if self.frames_below_threshold >= self.config.deactivation_frames {
                        self.state = MotionState::Stationary;
                        self.frames_in_state = 0;
                        self.frames_below_threshold = 0;
                    }
                } else {
                    self.frames_below_threshold = 0;
                    self.frames_in_state += 1;
                }
            }
        }
        
        if old_state != self.state {
            self.frames_in_state = 0;
        }
        
        MotionStatus {
            state: self.state,
            motion_level: smoothed,
            raw_motion: motion_clamped,
            frames_in_state: self.frames_in_state,
        }
    }

    /// Check if motion is currently active (dynamic mode)
    pub fn is_motion_active(&self) -> bool {
        self.state == MotionState::Dynamic
    }

    /// Check if stationary (low motion mode)
    pub fn is_stationary(&self) -> bool {
        self.state == MotionState::Stationary
    }

    /// Get current motion level (smoothed)
    pub fn motion_level(&self) -> f32 {
        if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().sum::<f32>() / self.history.len() as f32
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.history.clear();
        self.state = MotionState::Stationary;
        self.frames_in_state = 0;
        self.frames_below_threshold = 0;
    }

    /// Get configuration
    pub fn config(&self) -> &MotionDetectorConfig {
        &self.config
    }
}

impl Default for MotionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute motion metric from frame difference (helper)
///
/// # Arguments
/// * `prev_frame_mean` - Mean pixel value of previous frame ROI
/// * `curr_frame_mean` - Mean pixel value of current frame ROI
/// * `baseline` - Expected baseline intensity (e.g., 128 for 8-bit)
///
/// # Returns
/// Normalized motion level (0-1)
pub fn compute_frame_motion(prev_frame_mean: f32, curr_frame_mean: f32, baseline: f32) -> f32 {
    let diff = (curr_frame_mean - prev_frame_mean).abs();
    // Normalize: 5% of baseline = moderate motion, 15% = high motion
    let motion = diff / (baseline * 0.15);
    motion.clamp(0.0, 1.0)
}

/// Compute motion from landmark velocity (helper)
///
/// # Arguments  
/// * `velocity` - Average landmark displacement in pixels
/// * `max_velocity` - Maximum expected velocity for normalization
pub fn compute_landmark_motion(velocity: f32, max_velocity: f32) -> f32 {
    (velocity / max_velocity.max(1e-6)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_detector_stationary_start() {
        let detector = MotionDetector::new();
        assert!(detector.is_stationary());
        assert!(!detector.is_motion_active());
    }

    #[test]
    fn test_motion_detector_transition_to_dynamic() {
        let mut detector = MotionDetector::new();
        
        // Feed low motion
        for _ in 0..5 {
            detector.update(0.1);
        }
        assert!(detector.is_stationary());
        
        // Feed high motion
        for _ in 0..10 {
            let status = detector.update(0.5);
            if status.motion_level >= 0.3 {
                break;
            }
        }
        
        // Should transition to dynamic
        assert!(detector.is_motion_active());
    }

    #[test]
    fn test_motion_detector_hysteresis() {
        let mut detector = MotionDetector::new();
        
        // Go to dynamic mode
        for _ in 0..10 {
            detector.update(0.5);
        }
        assert!(detector.is_motion_active());
        
        // Feed motion just below threshold (but above hysteresis)
        for _ in 0..3 {
            detector.update(0.25); // Above 0.21 hysteresis threshold
        }
        // Should still be in dynamic mode due to hysteresis
        assert!(detector.is_motion_active());
        
        // Feed very low motion for deactivation frames
        for _ in 0..10 {
            detector.update(0.1); // Below 0.21
        }
        // Should now be stationary
        assert!(detector.is_stationary());
    }

    #[test]
    fn test_compute_frame_motion() {
        // No motion
        let motion = compute_frame_motion(128.0, 128.0, 128.0);
        assert!(motion == 0.0);
        
        // Moderate motion
        let motion = compute_frame_motion(128.0, 138.0, 128.0);
        assert!(motion > 0.0 && motion < 1.0);
        
        // High motion
        let motion = compute_frame_motion(100.0, 150.0, 128.0);
        assert!(motion >= 0.5);
    }

    #[test]
    fn test_reset() {
        let mut detector = MotionDetector::new();
        
        for _ in 0..10 {
            detector.update(0.5);
        }
        assert!(detector.is_motion_active());
        
        detector.reset();
        assert!(detector.is_stationary());
        assert!(detector.motion_level() == 0.0);
    }
}
