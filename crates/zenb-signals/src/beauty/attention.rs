//! Attention Metrics Module
//!
//! Provides attention-related metrics from face landmarks:
//! - Eye openness (Eye Aspect Ratio - EAR)
//! - Blink detection and rate
//! - Head pose direction (proxy for gaze)
//! - Focus/attention score

use super::landmarks::{get_point, CanonicalLandmarks};

/// Eye landmark indices for EAR calculation (MediaPipe)
mod eye_indices {
    // Left eye vertical points
    pub const LEFT_EYE_TOP: usize = 159;
    pub const LEFT_EYE_BOTTOM: usize = 145;
    // Left eye vertical points (alternate)
    pub const LEFT_EYE_TOP2: usize = 158;
    pub const LEFT_EYE_BOTTOM2: usize = 153;
    // Left eye horizontal
    pub const LEFT_EYE_LEFT: usize = 33;   // Outer corner
    pub const LEFT_EYE_RIGHT: usize = 133;  // Inner corner
    
    // Right eye vertical points
    pub const RIGHT_EYE_TOP: usize = 386;
    pub const RIGHT_EYE_BOTTOM: usize = 374;
    // Right eye vertical (alternate)
    pub const RIGHT_EYE_TOP2: usize = 385;
    pub const RIGHT_EYE_BOTTOM2: usize = 380;
    // Right eye horizontal
    pub const RIGHT_EYE_LEFT: usize = 362;  // Inner corner
    pub const RIGHT_EYE_RIGHT: usize = 263; // Outer corner
}

/// Configuration for attention tracking
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// EAR threshold for blink detection (eyes closed)
    pub ear_blink_threshold: f32,
    /// Minimum frames for blink confirmation
    pub min_blink_frames: usize,
    /// Window size for blink rate calculation (frames)
    pub blink_rate_window: usize,
    /// Yaw threshold (degrees) for off-screen gaze
    pub yaw_attention_threshold: f32,
    /// Pitch threshold for looking away
    pub pitch_attention_threshold: f32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            ear_blink_threshold: 0.21,
            min_blink_frames: 2,
            blink_rate_window: 300, // ~10 seconds at 30fps
            yaw_attention_threshold: 20.0, // degrees
            pitch_attention_threshold: 15.0,
        }
    }
}

/// Eye openness result
#[derive(Debug, Clone, Copy)]
pub struct EyeOpenness {
    /// Left eye aspect ratio (0 = closed, ~0.3 = open)
    pub left_ear: f32,
    /// Right eye aspect ratio
    pub right_ear: f32,
    /// Average EAR
    pub avg_ear: f32,
    /// Whether eyes are currently closed (below threshold)
    pub eyes_closed: bool,
}

/// Blink detection state
#[derive(Debug, Clone)]
pub struct BlinkState {
    /// Current blink in progress
    pub blinking: bool,
    /// Total blinks detected in window
    pub blink_count: usize,
    /// Blink rate (blinks per minute)
    pub blink_rate: f32,
    /// Frames since last blink
    pub frames_since_blink: usize,
}

impl Default for BlinkState {
    fn default() -> Self {
        Self {
            blinking: false,
            blink_count: 0,
            blink_rate: 0.0,
            frames_since_blink: 0,
        }
    }
}

/// Head pose direction (simplified)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GazeDirection {
    Center,
    Left,
    Right,
    Up,
    Down,
    Away,
}

/// Attention metrics result
#[derive(Debug, Clone)]
pub struct AttentionMetrics {
    /// Eye openness data
    pub eye_openness: EyeOpenness,
    /// Blink state
    pub blink_state: BlinkState,
    /// Gaze direction (from head pose)
    pub gaze_direction: GazeDirection,
    /// Attention score (0-1)
    pub attention_score: f32,
    /// Drowsiness indicator (0-1)
    pub drowsiness: f32,
    /// Head pose [yaw, pitch, roll] in degrees
    pub head_pose: Option<[f32; 3]>,
}

/// Attention Tracker
#[derive(Debug, Clone)]
pub struct AttentionTracker {
    config: AttentionConfig,
    /// Recent EAR values for smoothing
    ear_history: Vec<f32>,
    /// Blink detection state
    blink_in_progress: bool,
    blink_frame_count: usize,
    /// Blink timestamps (frame indices)
    blink_timestamps: Vec<usize>,
    /// Frame counter
    frame_count: usize,
    /// Recent attention scores for smoothing
    attention_history: Vec<f32>,
}

impl AttentionTracker {
    pub fn new() -> Self {
        Self::with_config(AttentionConfig::default())
    }
    
    pub fn with_config(config: AttentionConfig) -> Self {
        Self {
            config,
            ear_history: Vec::with_capacity(10),
            blink_in_progress: false,
            blink_frame_count: 0,
            blink_timestamps: Vec::new(),
            frame_count: 0,
            attention_history: Vec::with_capacity(30),
        }
    }
    
    /// Compute attention metrics from landmarks
    pub fn update(
        &mut self,
        landmarks: &CanonicalLandmarks,
        pose: Option<[f32; 3]>,
        sample_rate: f32,
    ) -> AttentionMetrics {
        self.frame_count += 1;
        
        // 1. Compute eye openness (EAR)
        let eye_openness = self.compute_eye_openness(landmarks);
        
        // 2. Update blink detection
        let blink_state = self.update_blink_state(&eye_openness, sample_rate);
        
        // 3. Compute gaze direction from pose
        let gaze_direction = self.compute_gaze_direction(pose);
        
        // 4. Compute attention score
        let attention_score = self.compute_attention_score(&eye_openness, &blink_state, pose);
        
        // 5. Compute drowsiness
        let drowsiness = self.compute_drowsiness(&eye_openness, &blink_state);
        
        AttentionMetrics {
            eye_openness,
            blink_state,
            gaze_direction,
            attention_score,
            drowsiness,
            head_pose: pose,
        }
    }
    
    /// Reset tracker state
    pub fn reset(&mut self) {
        self.ear_history.clear();
        self.blink_in_progress = false;
        self.blink_frame_count = 0;
        self.blink_timestamps.clear();
        self.frame_count = 0;
        self.attention_history.clear();
    }
    
    // --- Private ---
    
    fn compute_eye_openness(&mut self, landmarks: &CanonicalLandmarks) -> EyeOpenness {
        use eye_indices::*;
        
        // Left eye EAR
        let left_ear = self.compute_ear(
            landmarks,
            LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
            LEFT_EYE_TOP2, LEFT_EYE_BOTTOM2,
            LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
        );
        
        // Right eye EAR
        let right_ear = self.compute_ear(
            landmarks,
            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
            RIGHT_EYE_TOP2, RIGHT_EYE_BOTTOM2,
            RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT,
        );
        
        let avg_ear = (left_ear + right_ear) / 2.0;
        
        // Update history for smoothing
        self.ear_history.push(avg_ear);
        if self.ear_history.len() > 10 {
            self.ear_history.remove(0);
        }
        
        let eyes_closed = avg_ear < self.config.ear_blink_threshold;
        
        EyeOpenness {
            left_ear,
            right_ear,
            avg_ear,
            eyes_closed,
        }
    }
    
    fn compute_ear(
        &self,
        landmarks: &CanonicalLandmarks,
        top1: usize, bottom1: usize,
        top2: usize, bottom2: usize,
        left: usize, right: usize,
    ) -> f32 {
        // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        // Vertical distances
        let v1 = self.point_distance(landmarks, top1, bottom1);
        let v2 = self.point_distance(landmarks, top2, bottom2);
        // Horizontal distance
        let h = self.point_distance(landmarks, left, right).max(0.001);
        
        (v1 + v2) / (2.0 * h)
    }
    
    fn point_distance(&self, landmarks: &CanonicalLandmarks, i1: usize, i2: usize) -> f32 {
        let p1 = get_point(landmarks, i1);
        let p2 = get_point(landmarks, i2);
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        (dx * dx + dy * dy).sqrt()
    }
    
    fn update_blink_state(&mut self, eye: &EyeOpenness, sample_rate: f32) -> BlinkState {
        // Detect blink start
        if eye.eyes_closed && !self.blink_in_progress {
            self.blink_in_progress = true;
            self.blink_frame_count = 1;
        } else if eye.eyes_closed && self.blink_in_progress {
            self.blink_frame_count += 1;
        } else if !eye.eyes_closed && self.blink_in_progress {
            // Blink ended - check if valid
            if self.blink_frame_count >= self.config.min_blink_frames {
                self.blink_timestamps.push(self.frame_count);
            }
            self.blink_in_progress = false;
            self.blink_frame_count = 0;
        }
        
        // Clean old blinks from window
        let window_start = self.frame_count.saturating_sub(self.config.blink_rate_window);
        self.blink_timestamps.retain(|&t| t > window_start);
        
        // Compute blink rate
        let blink_count = self.blink_timestamps.len();
        let window_seconds = self.config.blink_rate_window as f32 / sample_rate;
        let blink_rate = (blink_count as f32 / window_seconds) * 60.0; // blinks/min
        
        let frames_since_blink = if let Some(&last) = self.blink_timestamps.last() {
            self.frame_count.saturating_sub(last)
        } else {
            self.frame_count
        };
        
        BlinkState {
            blinking: self.blink_in_progress,
            blink_count,
            blink_rate,
            frames_since_blink,
        }
    }
    
    fn compute_gaze_direction(&self, pose: Option<[f32; 3]>) -> GazeDirection {
        let Some([yaw, pitch, _roll]) = pose else {
            return GazeDirection::Center;
        };
        
        let yaw_abs = yaw.abs();
        let pitch_abs = pitch.abs();
        
        // Check if looking significantly away
        if yaw_abs > self.config.yaw_attention_threshold * 1.5 
            || pitch_abs > self.config.pitch_attention_threshold * 1.5 {
            return GazeDirection::Away;
        }
        
        // Determine primary direction
        if yaw_abs > self.config.yaw_attention_threshold {
            if yaw > 0.0 { GazeDirection::Right } else { GazeDirection::Left }
        } else if pitch_abs > self.config.pitch_attention_threshold {
            if pitch > 0.0 { GazeDirection::Down } else { GazeDirection::Up }
        } else {
            GazeDirection::Center
        }
    }
    
    fn compute_attention_score(
        &mut self,
        eye: &EyeOpenness,
        blink: &BlinkState,
        pose: Option<[f32; 3]>,
    ) -> f32 {
        // Components of attention:
        // 1. Eyes open (0-1)
        let eye_open_score = (eye.avg_ear / 0.3).clamp(0.0, 1.0);
        
        // 2. Head facing forward (0-1)
        let pose_score = if let Some([yaw, pitch, _]) = pose {
            let yaw_factor = (1.0 - yaw.abs() / 45.0).clamp(0.0, 1.0);
            let pitch_factor = (1.0 - pitch.abs() / 30.0).clamp(0.0, 1.0);
            yaw_factor * pitch_factor
        } else {
            0.5 // Unknown
        };
        
        // 3. Normal blink rate (not too high = distracted, not too low = staring)
        let blink_score = if blink.blink_rate > 5.0 && blink.blink_rate < 25.0 {
            1.0
        } else if blink.blink_rate < 3.0 || blink.blink_rate > 35.0 {
            0.5
        } else {
            0.8
        };
        
        let raw_score = eye_open_score * 0.4 + pose_score * 0.4 + blink_score * 0.2;
        
        // Smooth with EMA
        self.attention_history.push(raw_score);
        if self.attention_history.len() > 30 {
            self.attention_history.remove(0);
        }
        
        let smoothed = self.attention_history.iter().sum::<f32>() 
            / self.attention_history.len() as f32;
        
        smoothed.clamp(0.0, 1.0)
    }
    
    fn compute_drowsiness(&self, eye: &EyeOpenness, blink: &BlinkState) -> f32 {
        // Drowsiness indicators:
        // 1. Low EAR (droopy eyes)
        let ear_factor = if eye.avg_ear < 0.2 {
            (0.2 - eye.avg_ear) / 0.1
        } else {
            0.0
        };
        
        // 2. High blink rate
        let blink_factor = if blink.blink_rate > 20.0 {
            (blink.blink_rate - 20.0) / 20.0
        } else {
            0.0
        };
        
        // 3. Long blinks (extended eye closure)
        let long_blink_factor = if blink.blinking && self.blink_frame_count > 10 {
            0.5
        } else {
            0.0
        };
        
        (ear_factor * 0.4 + blink_factor * 0.3 + long_blink_factor * 0.3).clamp(0.0, 1.0)
    }
}

impl Default for AttentionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_open_eye_landmarks() -> CanonicalLandmarks {
        // Create landmarks with open eyes (high EAR)
        let mut points = vec![[0.0, 0.0]; 468];
        // Set eye points for ~0.3 EAR (eyes open)
        // Left eye
        points[eye_indices::LEFT_EYE_TOP] = [-0.2, -0.05];
        points[eye_indices::LEFT_EYE_BOTTOM] = [-0.2, 0.05];
        points[eye_indices::LEFT_EYE_TOP2] = [-0.22, -0.04];
        points[eye_indices::LEFT_EYE_BOTTOM2] = [-0.22, 0.04];
        points[eye_indices::LEFT_EYE_LEFT] = [-0.3, 0.0];
        points[eye_indices::LEFT_EYE_RIGHT] = [-0.1, 0.0];
        // Right eye (mirror)
        points[eye_indices::RIGHT_EYE_TOP] = [0.2, -0.05];
        points[eye_indices::RIGHT_EYE_BOTTOM] = [0.2, 0.05];
        points[eye_indices::RIGHT_EYE_TOP2] = [0.22, -0.04];
        points[eye_indices::RIGHT_EYE_BOTTOM2] = [0.22, 0.04];
        points[eye_indices::RIGHT_EYE_LEFT] = [0.1, 0.0];
        points[eye_indices::RIGHT_EYE_RIGHT] = [0.3, 0.0];
        
        CanonicalLandmarks {
            points,
            inter_ocular_px: 200.0,
            origin: [500.0, 400.0],
            valid: true,
        }
    }
    
    #[test]
    fn test_eye_openness() {
        let mut tracker = AttentionTracker::new();
        let landmarks = make_open_eye_landmarks();
        
        let metrics = tracker.update(&landmarks, None, 30.0);
        
        // Eyes should be detected as open
        assert!(!metrics.eye_openness.eyes_closed);
        assert!(metrics.eye_openness.avg_ear > 0.2);
    }
    
    #[test]
    fn test_gaze_center() {
        let tracker = AttentionTracker::new();
        
        let gaze = tracker.compute_gaze_direction(Some([0.0, 0.0, 0.0]));
        assert_eq!(gaze, GazeDirection::Center);
    }
    
    #[test]
    fn test_gaze_away() {
        let tracker = AttentionTracker::new();
        
        let gaze = tracker.compute_gaze_direction(Some([40.0, 0.0, 0.0]));
        assert_eq!(gaze, GazeDirection::Away);
    }
}
