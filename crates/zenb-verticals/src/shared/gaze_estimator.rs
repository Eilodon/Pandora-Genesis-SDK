//! Gaze Estimation
//!
//! Estimates gaze direction from:
//! - Head pose (yaw, pitch, roll)
//! - Eye landmark positions (iris center relative to eye corners)

use zenb_signals::beauty::{CanonicalLandmarks, get_point};

/// Gaze direction in 3D space
#[derive(Debug, Clone, Default)]
pub struct GazeDirection {
    /// Horizontal angle in degrees (negative = left, positive = right)
    pub yaw: f32,
    /// Vertical angle in degrees (negative = down, positive = up)
    pub pitch: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
}

/// Gaze target classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GazeTarget {
    /// Looking at screen/camera
    Screen,
    /// Looking left
    Left,
    /// Looking right
    Right,
    /// Looking up
    Up,
    /// Looking down
    Down,
    /// Unknown/uncertain
    Unknown,
}

/// Gaze estimation result
#[derive(Debug, Clone)]
pub struct GazeResult {
    /// Gaze direction
    pub direction: GazeDirection,
    /// Classified gaze target
    pub target: GazeTarget,
    /// Is looking at screen
    pub on_screen: bool,
    /// Gaze deviation from center (0-1)
    pub deviation: f32,
}

impl Default for GazeResult {
    fn default() -> Self {
        Self {
            direction: GazeDirection::default(),
            target: GazeTarget::Unknown,
            on_screen: false,
            deviation: 1.0,
        }
    }
}

/// Gaze estimator configuration
#[derive(Debug, Clone)]
pub struct GazeConfig {
    /// Maximum yaw for "on screen" classification (degrees)
    pub max_screen_yaw: f32,
    /// Maximum pitch for "on screen" classification (degrees)
    pub max_screen_pitch: f32,
    /// Smoothing factor (0 = no smoothing, 1 = max smoothing)
    pub smoothing_alpha: f32,
    /// Iris landmark indices (MediaPipe)
    pub left_iris_center: usize,
    pub right_iris_center: usize,
}

impl Default for GazeConfig {
    fn default() -> Self {
        Self {
            max_screen_yaw: 20.0,
            max_screen_pitch: 15.0,
            smoothing_alpha: 0.3,
            left_iris_center: 468,
            right_iris_center: 473,
        }
    }
}

/// Gaze Estimator
pub struct GazeEstimator {
    config: GazeConfig,
    prev_gaze: Option<GazeDirection>,
}

impl GazeEstimator {
    pub fn new() -> Self {
        Self::with_config(GazeConfig::default())
    }

    pub fn with_config(config: GazeConfig) -> Self {
        Self {
            config,
            prev_gaze: None,
        }
    }

    /// Estimate gaze from landmarks and optional head pose
    pub fn estimate(
        &mut self,
        landmarks: &CanonicalLandmarks,
        head_pose: Option<[f32; 3]>,
    ) -> GazeResult {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return GazeResult::default();
        }

        let (head_yaw, head_pitch) = if let Some([yaw, pitch, _roll]) = head_pose {
            (yaw, pitch)
        } else {
            self.estimate_head_pose_from_landmarks(landmarks)
        };

        let (eye_yaw, eye_pitch, eye_conf) = self.estimate_eye_gaze(landmarks);

        let total_yaw = head_yaw + eye_yaw * 0.5;
        let total_pitch = head_pitch + eye_pitch * 0.5;

        let confidence = eye_conf * 0.7 + 0.3;

        let mut direction = GazeDirection {
            yaw: total_yaw,
            pitch: total_pitch,
            confidence,
        };

        if let Some(ref prev) = self.prev_gaze {
            let alpha = self.config.smoothing_alpha;
            direction.yaw = prev.yaw * alpha + direction.yaw * (1.0 - alpha);
            direction.pitch = prev.pitch * alpha + direction.pitch * (1.0 - alpha);
        }
        self.prev_gaze = Some(direction.clone());

        let target = self.classify_target(&direction);
        let on_screen = target == GazeTarget::Screen;

        let deviation = ((direction.yaw / self.config.max_screen_yaw).powi(2)
            + (direction.pitch / self.config.max_screen_pitch).powi(2))
        .sqrt()
        .min(1.0);

        GazeResult {
            direction,
            target,
            on_screen,
            deviation,
        }
    }

    fn estimate_head_pose_from_landmarks(&self, landmarks: &CanonicalLandmarks) -> (f32, f32) {
        let nose = get_point(landmarks, 4);
        let left_cheek = get_point(landmarks, 234);
        let right_cheek = get_point(landmarks, 454);

        let left_dist = ((nose[0] - left_cheek[0]).powi(2)
            + (nose[1] - left_cheek[1]).powi(2))
        .sqrt();
        let right_dist = ((nose[0] - right_cheek[0]).powi(2)
            + (nose[1] - right_cheek[1]).powi(2))
        .sqrt();

        let yaw = if left_dist + right_dist > 0.01 {
            ((right_dist - left_dist) / (left_dist + right_dist)) * 45.0
        } else {
            0.0
        };

        let forehead = get_point(landmarks, 10);
        let chin = get_point(landmarks, 152);
        let nose_bridge = get_point(landmarks, 6);

        let upper = ((nose_bridge[0] - forehead[0]).powi(2)
            + (nose_bridge[1] - forehead[1]).powi(2))
        .sqrt();
        let lower = ((chin[0] - nose_bridge[0]).powi(2)
            + (chin[1] - nose_bridge[1]).powi(2))
        .sqrt();

        let pitch = if upper + lower > 0.01 {
            ((lower - upper) / (upper + lower)) * 30.0
        } else {
            0.0
        };

        (yaw, pitch)
    }

    fn estimate_eye_gaze(&self, landmarks: &CanonicalLandmarks) -> (f32, f32, f32) {
        let left_inner = get_point(landmarks, 133);
        let left_outer = get_point(landmarks, 33);
        let right_inner = get_point(landmarks, 362);
        let right_outer = get_point(landmarks, 263);

        let has_iris = landmarks.points.len() >= 478;
        if !has_iris {
            return (0.0, 0.0, 0.5);
        }

        let left_iris = get_point(landmarks, self.config.left_iris_center);
        let right_iris = get_point(landmarks, self.config.right_iris_center);

        let left_eye_width = left_outer[0] - left_inner[0];
        let right_eye_width = right_inner[0] - right_outer[0];

        if left_eye_width.abs() < 0.01 || right_eye_width.abs() < 0.01 {
            return (0.0, 0.0, 0.3);
        }

        let left_iris_pos = (left_iris[0] - left_inner[0]) / left_eye_width;
        let right_iris_pos = (right_iris[0] - right_outer[0]) / right_eye_width;

        let avg_iris_pos = (left_iris_pos + right_iris_pos) / 2.0;
        let eye_yaw = (avg_iris_pos - 0.5) * 30.0;

        let left_eye_center_y = (left_inner[1] + left_outer[1]) / 2.0;
        let right_eye_center_y = (right_inner[1] + right_outer[1]) / 2.0;
        let avg_eye_center_y = (left_eye_center_y + right_eye_center_y) / 2.0;
        let avg_iris_y = (left_iris[1] + right_iris[1]) / 2.0;

        let eye_pitch = (avg_iris_y - avg_eye_center_y) * 50.0;

        (eye_yaw, eye_pitch, 0.8)
    }

    fn classify_target(&self, direction: &GazeDirection) -> GazeTarget {
        let yaw_abs = direction.yaw.abs();
        let pitch_abs = direction.pitch.abs();

        if yaw_abs <= self.config.max_screen_yaw && pitch_abs <= self.config.max_screen_pitch {
            GazeTarget::Screen
        } else if direction.yaw < -self.config.max_screen_yaw {
            GazeTarget::Left
        } else if direction.yaw > self.config.max_screen_yaw {
            GazeTarget::Right
        } else if direction.pitch > self.config.max_screen_pitch {
            GazeTarget::Up
        } else if direction.pitch < -self.config.max_screen_pitch {
            GazeTarget::Down
        } else {
            GazeTarget::Unknown
        }
    }

    /// Reset estimator state
    pub fn reset(&mut self) {
        self.prev_gaze = None;
    }
}

impl Default for GazeEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_mock_landmarks(count: usize) -> CanonicalLandmarks {
        CanonicalLandmarks {
            points: vec![[0.5, 0.5]; count],
            inter_ocular_px: 100.0,
            origin: [500.0, 400.0],
            valid: true,
        }
    }

    #[test]
    fn test_gaze_config_defaults() {
        let config = GazeConfig::default();
        assert!((config.max_screen_yaw - 20.0).abs() < 0.01);
        assert!((config.max_screen_pitch - 15.0).abs() < 0.01);
        assert!((config.smoothing_alpha - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_gaze_target_screen() {
        let mut estimator = GazeEstimator::new();
        let landmarks = create_mock_landmarks(478);
        
        // With centered head pose, should be on screen
        let result = estimator.estimate(&landmarks, Some([0.0, 0.0, 0.0]));
        assert_eq!(result.target, GazeTarget::Screen);
        assert!(result.on_screen);
    }

    #[test]
    fn test_gaze_invalid_landmarks() {
        let mut estimator = GazeEstimator::new();
        let invalid = CanonicalLandmarks {
            points: vec![],
            inter_ocular_px: 0.0,
            origin: [0.0, 0.0],
            valid: false,
        };
        
        let result = estimator.estimate(&invalid, None);
        assert_eq!(result.target, GazeTarget::Unknown);
        assert!(!result.on_screen);
    }

    #[test]
    fn test_gaze_reset() {
        let mut estimator = GazeEstimator::new();
        let landmarks = create_mock_landmarks(478);
        
        estimator.estimate(&landmarks, Some([0.0, 0.0, 0.0]));
        assert!(estimator.prev_gaze.is_some());
        
        estimator.reset();
        assert!(estimator.prev_gaze.is_none());
    }
}
