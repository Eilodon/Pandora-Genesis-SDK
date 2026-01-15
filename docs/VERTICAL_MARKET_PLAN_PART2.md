# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET - PART 2

## 3.2 Gaze Estimator

**File:** `crates/zenb-verticals/src/shared/gaze_estimator.rs`

**Mục đích:** Ước tính hướng nhìn từ head pose + eye landmarks

```rust
//! Gaze Estimation
//!
//! Estimates gaze direction from:
//! - Head pose (yaw, pitch, roll)
//! - Eye landmark positions (iris center relative to eye corners)

use zenb_signals::beauty::landmarks::{CanonicalLandmarks, get_point, landmark_distance};
use std::f32::consts::PI;

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
            left_iris_center: 468,  // MediaPipe iris landmarks
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
        head_pose: Option<[f32; 3]>, // [yaw, pitch, roll] in degrees
    ) -> GazeResult {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return GazeResult::default();
        }
        
        // Method 1: Head pose contribution
        let (head_yaw, head_pitch) = if let Some([yaw, pitch, _roll]) = head_pose {
            (yaw, pitch)
        } else {
            // Estimate head pose from landmarks
            self.estimate_head_pose_from_landmarks(landmarks)
        };
        
        // Method 2: Eye gaze contribution (iris position relative to eye corners)
        let (eye_yaw, eye_pitch, eye_conf) = self.estimate_eye_gaze(landmarks);
        
        // Combine head pose and eye gaze
        // Eye gaze is relative to head, so we add them
        let total_yaw = head_yaw + eye_yaw * 0.5; // Eye contribution weighted
        let total_pitch = head_pitch + eye_pitch * 0.5;
        
        let confidence = eye_conf * 0.7 + 0.3; // Base confidence from landmarks
        
        let mut direction = GazeDirection {
            yaw: total_yaw,
            pitch: total_pitch,
            confidence,
        };
        
        // Apply temporal smoothing
        if let Some(ref prev) = self.prev_gaze {
            let alpha = self.config.smoothing_alpha;
            direction.yaw = prev.yaw * alpha + direction.yaw * (1.0 - alpha);
            direction.pitch = prev.pitch * alpha + direction.pitch * (1.0 - alpha);
        }
        self.prev_gaze = Some(direction.clone());
        
        // Classify gaze target
        let target = self.classify_target(&direction);
        let on_screen = target == GazeTarget::Screen;
        
        // Calculate deviation from center
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
    
    /// Estimate head pose from landmark asymmetry
    fn estimate_head_pose_from_landmarks(&self, landmarks: &CanonicalLandmarks) -> (f32, f32) {
        // Use nose and cheek landmarks to estimate yaw
        let nose = get_point(landmarks, 4);
        let left_cheek = get_point(landmarks, 234);
        let right_cheek = get_point(landmarks, 454);
        
        let left_dist = ((nose[0] - left_cheek[0]).powi(2) + (nose[1] - left_cheek[1]).powi(2)).sqrt();
        let right_dist = ((nose[0] - right_cheek[0]).powi(2) + (nose[1] - right_cheek[1]).powi(2)).sqrt();
        
        // Asymmetry indicates yaw
        let yaw = if left_dist + right_dist > 0.01 {
            ((right_dist - left_dist) / (left_dist + right_dist)) * 45.0
        } else {
            0.0
        };
        
        // Use forehead and chin for pitch
        let forehead = get_point(landmarks, 10);
        let chin = get_point(landmarks, 152);
        let nose_bridge = get_point(landmarks, 6);
        
        let upper = ((nose_bridge[0] - forehead[0]).powi(2) + (nose_bridge[1] - forehead[1]).powi(2)).sqrt();
        let lower = ((chin[0] - nose_bridge[0]).powi(2) + (chin[1] - nose_bridge[1]).powi(2)).sqrt();
        
        let pitch = if upper + lower > 0.01 {
            ((lower - upper) / (upper + lower)) * 30.0
        } else {
            0.0
        };
        
        (yaw, pitch)
    }
    
    /// Estimate eye gaze from iris position
    fn estimate_eye_gaze(&self, landmarks: &CanonicalLandmarks) -> (f32, f32, f32) {
        // Get eye corners and iris centers
        let left_inner = get_point(landmarks, 133);
        let left_outer = get_point(landmarks, 33);
        let right_inner = get_point(landmarks, 362);
        let right_outer = get_point(landmarks, 263);
        
        // Iris centers (if available in 478 landmark model)
        let has_iris = landmarks.points.len() >= 478;
        
        if !has_iris {
            return (0.0, 0.0, 0.5);
        }
        
        let left_iris = get_point(landmarks, self.config.left_iris_center);
        let right_iris = get_point(landmarks, self.config.right_iris_center);
        
        // Calculate iris position relative to eye corners (0 = inner, 1 = outer)
        let left_eye_width = left_outer[0] - left_inner[0];
        let right_eye_width = right_inner[0] - right_outer[0];
        
        if left_eye_width.abs() < 0.01 || right_eye_width.abs() < 0.01 {
            return (0.0, 0.0, 0.3);
        }
        
        let left_iris_pos = (left_iris[0] - left_inner[0]) / left_eye_width;
        let right_iris_pos = (right_iris[0] - right_outer[0]) / right_eye_width;
        
        // Average iris position (0.5 = center)
        let avg_iris_pos = (left_iris_pos + right_iris_pos) / 2.0;
        
        // Convert to yaw angle (0.5 = 0 degrees, 0 = -15 degrees, 1 = +15 degrees)
        let eye_yaw = (avg_iris_pos - 0.5) * 30.0;
        
        // Vertical gaze (simplified - use iris y relative to eye center)
        let left_eye_center_y = (left_inner[1] + left_outer[1]) / 2.0;
        let right_eye_center_y = (right_inner[1] + right_outer[1]) / 2.0;
        let avg_eye_center_y = (left_eye_center_y + right_eye_center_y) / 2.0;
        let avg_iris_y = (left_iris[1] + right_iris[1]) / 2.0;
        
        let eye_pitch = (avg_iris_y - avg_eye_center_y) * 50.0;
        
        (eye_yaw, eye_pitch, 0.8)
    }
    
    /// Classify gaze target
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
```

---

## 3.3 Micro Expression Analyzer

**File:** `crates/zenb-verticals/src/shared/micro_expression.rs`

```rust
//! Micro Expression Analysis
//!
//! Detects facial Action Units (AUs) from landmarks for emotion recognition.
//! Based on Facial Action Coding System (FACS).

use zenb_signals::beauty::landmarks::{CanonicalLandmarks, get_point, landmark_distance};

/// Action Unit intensities (0-1)
#[derive(Debug, Clone, Default)]
pub struct ActionUnits {
    // Upper face
    pub au1_inner_brow_raise: f32,
    pub au2_outer_brow_raise: f32,
    pub au4_brow_lowerer: f32,
    pub au5_upper_lid_raise: f32,
    pub au6_cheek_raise: f32,
    pub au7_lid_tightener: f32,
    
    // Lower face
    pub au9_nose_wrinkle: f32,
    pub au10_upper_lip_raise: f32,
    pub au12_lip_corner_pull: f32,  // Smile
    pub au14_dimpler: f32,
    pub au15_lip_corner_depress: f32,
    pub au17_chin_raise: f32,
    pub au20_lip_stretch: f32,
    pub au23_lip_tightener: f32,
    pub au24_lip_press: f32,
    pub au25_lips_part: f32,
    pub au26_jaw_drop: f32,
}

/// Basic emotion classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasicEmotion {
    Neutral,
    Happy,
    Sad,
    Angry,
    Fearful,
    Disgusted,
    Surprised,
}

/// Emotion result with confidence
#[derive(Debug, Clone)]
pub struct EmotionResult {
    pub primary_emotion: BasicEmotion,
    pub confidence: f32,
    pub valence: f32,  // -1 (negative) to +1 (positive)
    pub arousal: f32,  // 0 (calm) to 1 (excited)
    pub action_units: ActionUnits,
}

impl Default for EmotionResult {
    fn default() -> Self {
        Self {
            primary_emotion: BasicEmotion::Neutral,
            confidence: 0.0,
            valence: 0.0,
            arousal: 0.0,
            action_units: ActionUnits::default(),
        }
    }
}

/// Micro Expression Analyzer
pub struct MicroExpressionAnalyzer {
    baseline_measurements: Option<BaselineMeasurements>,
    smoothing_alpha: f32,
    prev_aus: Option<ActionUnits>,
}

#[derive(Debug, Clone, Default)]
struct BaselineMeasurements {
    brow_height: f32,
    eye_openness: f32,
    mouth_width: f32,
    mouth_height: f32,
}

impl MicroExpressionAnalyzer {
    pub fn new() -> Self {
        Self {
            baseline_measurements: None,
            smoothing_alpha: 0.3,
            prev_aus: None,
        }
    }
    
    /// Calibrate baseline (neutral face)
    pub fn calibrate(&mut self, landmarks: &CanonicalLandmarks) {
        if !landmarks.valid {
            return;
        }
        
        self.baseline_measurements = Some(BaselineMeasurements {
            brow_height: self.measure_brow_height(landmarks),
            eye_openness: self.measure_eye_openness(landmarks),
            mouth_width: self.measure_mouth_width(landmarks),
            mouth_height: self.measure_mouth_height(landmarks),
        });
    }
    
    /// Analyze micro expressions
    pub fn analyze(&mut self, landmarks: &CanonicalLandmarks) -> EmotionResult {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return EmotionResult::default();
        }
        
        // Compute Action Units
        let mut aus = self.compute_action_units(landmarks);
        
        // Apply temporal smoothing
        if let Some(ref prev) = self.prev_aus {
            aus = self.smooth_aus(prev, &aus);
        }
        self.prev_aus = Some(aus.clone());
        
        // Classify emotion from AUs
        let (emotion, confidence) = self.classify_emotion(&aus);
        
        // Compute valence and arousal
        let valence = self.compute_valence(&aus);
        let arousal = self.compute_arousal(&aus);
        
        EmotionResult {
            primary_emotion: emotion,
            confidence,
            valence,
            arousal,
            action_units: aus,
        }
    }
    
    fn compute_action_units(&self, landmarks: &CanonicalLandmarks) -> ActionUnits {
        let baseline = self.baseline_measurements.as_ref();
        
        // Current measurements
        let brow_height = self.measure_brow_height(landmarks);
        let eye_openness = self.measure_eye_openness(landmarks);
        let mouth_width = self.measure_mouth_width(landmarks);
        let mouth_height = self.measure_mouth_height(landmarks);
        
        // Compute deviations from baseline
        let brow_delta = if let Some(b) = baseline {
            (brow_height - b.brow_height) / b.brow_height.max(0.01)
        } else {
            0.0
        };
        
        let eye_delta = if let Some(b) = baseline {
            (eye_openness - b.eye_openness) / b.eye_openness.max(0.01)
        } else {
            0.0
        };
        
        let mouth_w_delta = if let Some(b) = baseline {
            (mouth_width - b.mouth_width) / b.mouth_width.max(0.01)
        } else {
            0.0
        };
        
        let mouth_h_delta = if let Some(b) = baseline {
            (mouth_height - b.mouth_height) / b.mouth_height.max(0.01)
        } else {
            0.0
        };
        
        // Map to Action Units
        ActionUnits {
            au1_inner_brow_raise: brow_delta.max(0.0).min(1.0),
            au2_outer_brow_raise: brow_delta.max(0.0).min(1.0) * 0.8,
            au4_brow_lowerer: (-brow_delta).max(0.0).min(1.0),
            au5_upper_lid_raise: eye_delta.max(0.0).min(1.0),
            au6_cheek_raise: self.measure_cheek_raise(landmarks),
            au7_lid_tightener: (-eye_delta).max(0.0).min(1.0) * 0.5,
            au9_nose_wrinkle: 0.0, // Requires more landmarks
            au10_upper_lip_raise: 0.0,
            au12_lip_corner_pull: mouth_w_delta.max(0.0).min(1.0),
            au14_dimpler: 0.0,
            au15_lip_corner_depress: (-mouth_w_delta).max(0.0).min(1.0) * 0.5,
            au17_chin_raise: 0.0,
            au20_lip_stretch: mouth_w_delta.max(0.0).min(1.0) * 0.5,
            au23_lip_tightener: 0.0,
            au24_lip_press: (-mouth_h_delta).max(0.0).min(1.0) * 0.5,
            au25_lips_part: mouth_h_delta.max(0.0).min(1.0) * 0.5,
            au26_jaw_drop: mouth_h_delta.max(0.0).min(1.0),
        }
    }
    
    fn measure_brow_height(&self, landmarks: &CanonicalLandmarks) -> f32 {
        let left_brow = get_point(landmarks, 105);
        let left_eye = get_point(landmarks, 159);
        let right_brow = get_point(landmarks, 334);
        let right_eye = get_point(landmarks, 386);
        
        let left_dist = (left_brow[1] - left_eye[1]).abs();
        let right_dist = (right_brow[1] - right_eye[1]).abs();
        
        (left_dist + right_dist) / 2.0
    }
    
    fn measure_eye_openness(&self, landmarks: &CanonicalLandmarks) -> f32 {
        let left_top = get_point(landmarks, 159);
        let left_bottom = get_point(landmarks, 145);
        let right_top = get_point(landmarks, 386);
        let right_bottom = get_point(landmarks, 374);
        
        let left_open = (left_top[1] - left_bottom[1]).abs();
        let right_open = (right_top[1] - right_bottom[1]).abs();
        
        (left_open + right_open) / 2.0
    }
    
    fn measure_mouth_width(&self, landmarks: &CanonicalLandmarks) -> f32 {
        landmark_distance(landmarks, 61, 291)
    }
    
    fn measure_mouth_height(&self, landmarks: &CanonicalLandmarks) -> f32 {
        landmark_distance(landmarks, 0, 17)
    }
    
    fn measure_cheek_raise(&self, landmarks: &CanonicalLandmarks) -> f32 {
        // Cheek raise indicated by lower eyelid position
        let left_lower_lid = get_point(landmarks, 145);
        let right_lower_lid = get_point(landmarks, 374);
        let left_cheek = get_point(landmarks, 234);
        let right_cheek = get_point(landmarks, 454);
        
        let left_dist = (left_lower_lid[1] - left_cheek[1]).abs();
        let right_dist = (right_lower_lid[1] - right_cheek[1]).abs();
        
        // Normalize and invert (smaller distance = more cheek raise)
        let avg = (left_dist + right_dist) / 2.0;
        (0.3 - avg).max(0.0).min(1.0) * 3.0
    }
    
    fn smooth_aus(&self, prev: &ActionUnits, curr: &ActionUnits) -> ActionUnits {
        let a = self.smoothing_alpha;
        let b = 1.0 - a;
        
        ActionUnits {
            au1_inner_brow_raise: prev.au1_inner_brow_raise * a + curr.au1_inner_brow_raise * b,
            au2_outer_brow_raise: prev.au2_outer_brow_raise * a + curr.au2_outer_brow_raise * b,
            au4_brow_lowerer: prev.au4_brow_lowerer * a + curr.au4_brow_lowerer * b,
            au5_upper_lid_raise: prev.au5_upper_lid_raise * a + curr.au5_upper_lid_raise * b,
            au6_cheek_raise: prev.au6_cheek_raise * a + curr.au6_cheek_raise * b,
            au7_lid_tightener: prev.au7_lid_tightener * a + curr.au7_lid_tightener * b,
            au9_nose_wrinkle: prev.au9_nose_wrinkle * a + curr.au9_nose_wrinkle * b,
            au10_upper_lip_raise: prev.au10_upper_lip_raise * a + curr.au10_upper_lip_raise * b,
            au12_lip_corner_pull: prev.au12_lip_corner_pull * a + curr.au12_lip_corner_pull * b,
            au14_dimpler: prev.au14_dimpler * a + curr.au14_dimpler * b,
            au15_lip_corner_depress: prev.au15_lip_corner_depress * a + curr.au15_lip_corner_depress * b,
            au17_chin_raise: prev.au17_chin_raise * a + curr.au17_chin_raise * b,
            au20_lip_stretch: prev.au20_lip_stretch * a + curr.au20_lip_stretch * b,
            au23_lip_tightener: prev.au23_lip_tightener * a + curr.au23_lip_tightener * b,
            au24_lip_press: prev.au24_lip_press * a + curr.au24_lip_press * b,
            au25_lips_part: prev.au25_lips_part * a + curr.au25_lips_part * b,
            au26_jaw_drop: prev.au26_jaw_drop * a + curr.au26_jaw_drop * b,
        }
    }
    
    fn classify_emotion(&self, aus: &ActionUnits) -> (BasicEmotion, f32) {
        // FACS-based emotion classification rules
        let mut scores = [
            (BasicEmotion::Neutral, 0.3f32),
            (BasicEmotion::Happy, aus.au6_cheek_raise * 0.5 + aus.au12_lip_corner_pull * 0.5),
            (BasicEmotion::Sad, aus.au1_inner_brow_raise * 0.3 + aus.au4_brow_lowerer * 0.3 + aus.au15_lip_corner_depress * 0.4),
            (BasicEmotion::Angry, aus.au4_brow_lowerer * 0.4 + aus.au7_lid_tightener * 0.3 + aus.au24_lip_press * 0.3),
            (BasicEmotion::Fearful, aus.au1_inner_brow_raise * 0.3 + aus.au2_outer_brow_raise * 0.2 + aus.au5_upper_lid_raise * 0.3 + aus.au20_lip_stretch * 0.2),
            (BasicEmotion::Disgusted, aus.au9_nose_wrinkle * 0.5 + aus.au10_upper_lip_raise * 0.5),
            (BasicEmotion::Surprised, aus.au1_inner_brow_raise * 0.25 + aus.au2_outer_brow_raise * 0.25 + aus.au5_upper_lid_raise * 0.25 + aus.au26_jaw_drop * 0.25),
        ];
        
        // Find max score
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        (scores[0].0, scores[0].1.min(1.0))
    }
    
    fn compute_valence(&self, aus: &ActionUnits) -> f32 {
        // Positive: smile, cheek raise
        // Negative: brow lowerer, lip corner depress
        let positive = aus.au12_lip_corner_pull * 0.5 + aus.au6_cheek_raise * 0.5;
        let negative = aus.au4_brow_lowerer * 0.3 + aus.au15_lip_corner_depress * 0.4 + aus.au9_nose_wrinkle * 0.3;
        
        (positive - negative).clamp(-1.0, 1.0)
    }
    
    fn compute_arousal(&self, aus: &ActionUnits) -> f32 {
        // High arousal: wide eyes, open mouth, raised brows
        let arousal = aus.au5_upper_lid_raise * 0.3
            + aus.au26_jaw_drop * 0.2
            + aus.au1_inner_brow_raise * 0.2
            + aus.au2_outer_brow_raise * 0.15
            + aus.au20_lip_stretch * 0.15;
        
        arousal.clamp(0.0, 1.0)
    }
    
    pub fn reset(&mut self) {
        self.baseline_measurements = None;
        self.prev_aus = None;
    }
}

impl Default for MicroExpressionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## 3.4 Shared Module Entry Point

**File:** `crates/zenb-verticals/src/shared/mod.rs`

```rust
//! Shared Components for Vertical Markets
//!
//! Reusable analyzers used across multiple verticals.

pub mod eye_metrics;
pub mod gaze_estimator;
pub mod micro_expression;

pub use eye_metrics::{
    EyeMetricsAnalyzer, EyeMetricsConfig,
    EarResult, BlinkResult, PerclosResult,
};

pub use gaze_estimator::{
    GazeEstimator, GazeConfig,
    GazeResult, GazeDirection, GazeTarget,
};

pub use micro_expression::{
    MicroExpressionAnalyzer,
    ActionUnits, BasicEmotion, EmotionResult,
};
```

---

*Tiếp tục trong PART3 - Liveness Detection Module...*
