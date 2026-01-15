//! Micro Expression Analysis
//!
//! Detects facial Action Units (AUs) from landmarks for emotion recognition.
//! Based on Facial Action Coding System (FACS).

use zenb_signals::beauty::{CanonicalLandmarks, get_point, landmark_distance};

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
    pub au12_lip_corner_pull: f32,
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
    pub valence: f32,
    pub arousal: f32,
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

        let mut aus = self.compute_action_units(landmarks);

        if let Some(ref prev) = self.prev_aus {
            aus = self.smooth_aus(prev, &aus);
        }
        self.prev_aus = Some(aus.clone());

        let (emotion, confidence) = self.classify_emotion(&aus);
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

        let brow_height = self.measure_brow_height(landmarks);
        let eye_openness = self.measure_eye_openness(landmarks);
        let mouth_width = self.measure_mouth_width(landmarks);
        let mouth_height = self.measure_mouth_height(landmarks);

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

        ActionUnits {
            au1_inner_brow_raise: brow_delta.max(0.0).min(1.0),
            au2_outer_brow_raise: brow_delta.max(0.0).min(1.0) * 0.8,
            au4_brow_lowerer: (-brow_delta).max(0.0).min(1.0),
            au5_upper_lid_raise: eye_delta.max(0.0).min(1.0),
            au6_cheek_raise: self.measure_cheek_raise(landmarks),
            au7_lid_tightener: (-eye_delta).max(0.0).min(1.0) * 0.5,
            au9_nose_wrinkle: 0.0,
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
        let left_lower_lid = get_point(landmarks, 145);
        let left_cheek = get_point(landmarks, 123);
        let right_lower_lid = get_point(landmarks, 374);
        let right_cheek = get_point(landmarks, 352);

        let left = (left_cheek[1] - left_lower_lid[1]).abs();
        let right = (right_cheek[1] - right_lower_lid[1]).abs();

        let raise = (left + right) / 2.0;
        (raise * 10.0).clamp(0.0, 1.0)
    }

    fn smooth_aus(&self, prev: &ActionUnits, current: &ActionUnits) -> ActionUnits {
        let alpha = self.smoothing_alpha;
        ActionUnits {
            au1_inner_brow_raise: prev.au1_inner_brow_raise * alpha + current.au1_inner_brow_raise * (1.0 - alpha),
            au2_outer_brow_raise: prev.au2_outer_brow_raise * alpha + current.au2_outer_brow_raise * (1.0 - alpha),
            au4_brow_lowerer: prev.au4_brow_lowerer * alpha + current.au4_brow_lowerer * (1.0 - alpha),
            au5_upper_lid_raise: prev.au5_upper_lid_raise * alpha + current.au5_upper_lid_raise * (1.0 - alpha),
            au6_cheek_raise: prev.au6_cheek_raise * alpha + current.au6_cheek_raise * (1.0 - alpha),
            au7_lid_tightener: prev.au7_lid_tightener * alpha + current.au7_lid_tightener * (1.0 - alpha),
            au9_nose_wrinkle: prev.au9_nose_wrinkle * alpha + current.au9_nose_wrinkle * (1.0 - alpha),
            au10_upper_lip_raise: prev.au10_upper_lip_raise * alpha + current.au10_upper_lip_raise * (1.0 - alpha),
            au12_lip_corner_pull: prev.au12_lip_corner_pull * alpha + current.au12_lip_corner_pull * (1.0 - alpha),
            au14_dimpler: prev.au14_dimpler * alpha + current.au14_dimpler * (1.0 - alpha),
            au15_lip_corner_depress: prev.au15_lip_corner_depress * alpha + current.au15_lip_corner_depress * (1.0 - alpha),
            au17_chin_raise: prev.au17_chin_raise * alpha + current.au17_chin_raise * (1.0 - alpha),
            au20_lip_stretch: prev.au20_lip_stretch * alpha + current.au20_lip_stretch * (1.0 - alpha),
            au23_lip_tightener: prev.au23_lip_tightener * alpha + current.au23_lip_tightener * (1.0 - alpha),
            au24_lip_press: prev.au24_lip_press * alpha + current.au24_lip_press * (1.0 - alpha),
            au25_lips_part: prev.au25_lips_part * alpha + current.au25_lips_part * (1.0 - alpha),
            au26_jaw_drop: prev.au26_jaw_drop * alpha + current.au26_jaw_drop * (1.0 - alpha),
        }
    }

    fn classify_emotion(&self, aus: &ActionUnits) -> (BasicEmotion, f32) {
        if aus.au12_lip_corner_pull > 0.3 {
            (BasicEmotion::Happy, 0.7)
        } else if aus.au4_brow_lowerer > 0.3 && aus.au15_lip_corner_depress > 0.2 {
            (BasicEmotion::Sad, 0.6)
        } else if aus.au4_brow_lowerer > 0.4 && aus.au23_lip_tightener > 0.2 {
            (BasicEmotion::Angry, 0.6)
        } else if aus.au1_inner_brow_raise > 0.4 && aus.au26_jaw_drop > 0.2 {
            (BasicEmotion::Surprised, 0.6)
        } else {
            (BasicEmotion::Neutral, 0.3)
        }
    }

    fn compute_valence(&self, aus: &ActionUnits) -> f32 {
        let positive = aus.au12_lip_corner_pull;
        let negative = aus.au15_lip_corner_depress + aus.au4_brow_lowerer * 0.5;
        (positive - negative).clamp(-1.0, 1.0)
    }

    fn compute_arousal(&self, aus: &ActionUnits) -> f32 {
        let arousal = aus.au5_upper_lid_raise.max(aus.au26_jaw_drop);
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
