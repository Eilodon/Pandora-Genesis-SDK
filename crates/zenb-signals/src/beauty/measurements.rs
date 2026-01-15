//! Face Geometry Measurements
//!
//! Computes normalized measurements from canonical landmarks.

use super::landmarks::{CanonicalLandmarks, get_point, landmark_distance, landmark_angle, indices};

/// Normalized face measurements
///
/// All distances are normalized to inter-ocular distance = 1.0
#[derive(Debug, Clone, Default)]
pub struct FaceMeasurements {
    // === Widths ===
    /// Forehead width at temples
    pub forehead_width: f32,
    /// Temple width
    pub temple_width: f32,
    /// Cheekbone width (widest point)
    pub cheekbone_width: f32,
    /// Jaw width at jaw angle
    pub jaw_width: f32,
    /// Chin width
    pub chin_width: f32,
    
    // === Lengths ===
    /// Total face length (forehead to chin)
    pub face_length: f32,
    /// Upper face (hairline to nose bridge)
    pub upper_face: f32,
    /// Mid face (nose bridge to nose tip)
    pub mid_face: f32,
    /// Lower face (nose tip to chin)
    pub lower_face: f32,
    
    // === Ratios ===
    /// Face length to cheekbone width ratio
    pub length_to_cheek_ratio: f32,
    /// Jaw to cheekbone width ratio
    pub jaw_to_cheek_ratio: f32,
    /// Forehead to cheekbone width ratio
    pub forehead_to_cheek_ratio: f32,
    /// Chin to jaw width ratio
    pub chin_to_jaw_ratio: f32,
    
    // === Angles & Curves ===
    /// Jaw angle in degrees
    pub jaw_angle: f32,
    /// Face taper: (forehead - jaw) / forehead
    pub face_taper: f32,
    /// Roundness score (0 = angular, 1 = rounded)
    pub roundness_score: f32,
    
    // === Feature Measurements ===
    /// Eye spacing (inter-ocular distance, always 1.0 after normalization)
    pub eye_spacing: f32,
    /// Nose width
    pub nose_width: f32,
    /// Lip width
    pub lip_width: f32,
    /// Canthal tilt (eye angle in degrees, positive = upward)
    pub canthal_tilt: f32,
    /// Eyebrow arch height
    pub brow_arch_height: f32,
}

impl FaceMeasurements {
    /// Compute measurements from canonical landmarks
    pub fn from_landmarks(landmarks: &CanonicalLandmarks) -> Self {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return Self::default();
        }
        
        // === Widths ===
        let forehead_width = landmark_distance(landmarks, indices::LEFT_TEMPLE, indices::RIGHT_TEMPLE);
        let temple_width = forehead_width; // Same measurement
        let cheekbone_width = landmark_distance(landmarks, indices::LEFT_CHEEKBONE, indices::RIGHT_CHEEKBONE);
        let jaw_width = landmark_distance(landmarks, indices::LEFT_JAW_ANGLE, indices::RIGHT_JAW_ANGLE);
        let chin_width = landmark_distance(landmarks, indices::CHIN_LEFT, indices::CHIN_RIGHT);
        
        // === Lengths ===
        let face_length = landmark_distance(landmarks, indices::FOREHEAD_TOP, indices::CHIN);
        let upper_face = landmark_distance(landmarks, indices::FOREHEAD_TOP, indices::NOSE_BRIDGE);
        let mid_face = landmark_distance(landmarks, indices::NOSE_BRIDGE, indices::NOSE_TIP);
        let lower_face = landmark_distance(landmarks, indices::NOSE_TIP, indices::CHIN);
        
        // === Ratios ===
        let length_to_cheek_ratio = if cheekbone_width > 0.01 {
            face_length / cheekbone_width
        } else {
            1.0
        };
        
        let jaw_to_cheek_ratio = if cheekbone_width > 0.01 {
            jaw_width / cheekbone_width
        } else {
            1.0
        };
        
        let forehead_to_cheek_ratio = if cheekbone_width > 0.01 {
            forehead_width / cheekbone_width
        } else {
            1.0
        };
        
        let chin_to_jaw_ratio = if jaw_width > 0.01 {
            chin_width / jaw_width
        } else {
            1.0
        };
        
        // === Angles & Curves ===
        let jaw_angle = landmark_angle(
            landmarks,
            indices::LEFT_JAW,
            indices::LEFT_JAW_ANGLE,
            indices::CHIN,
        ).abs();
        
        let face_taper = if forehead_width > 0.01 {
            (forehead_width - jaw_width) / forehead_width
        } else {
            0.0
        };
        
        // Roundness: based on jaw angle and chin-to-jaw ratio
        let roundness_score = Self::compute_roundness(jaw_angle, chin_to_jaw_ratio);
        
        // === Feature Measurements ===
        let eye_spacing = 1.0; // By definition in canonical space
        
        let nose_width = landmark_distance(landmarks, indices::NOSE_LEFT_ALAR, indices::NOSE_RIGHT_ALAR);
        let lip_width = landmark_distance(landmarks, indices::LEFT_MOUTH_CORNER, indices::RIGHT_MOUTH_CORNER);
        
        // Canthal tilt: angle of eye line relative to horizontal
        let left_outer = get_point(landmarks, indices::LEFT_EYE_OUTER);
        let left_inner = get_point(landmarks, indices::LEFT_EYE_INNER);
        let eye_dy = left_inner[1] - left_outer[1];
        let eye_dx = left_inner[0] - left_outer[0];
        let canthal_tilt = eye_dy.atan2(eye_dx.abs()).to_degrees();
        
        // Brow arch height: distance from brow arch to eye center
        let brow_arch = get_point(landmarks, indices::LEFT_BROW_ARCH);
        let eye_outer = get_point(landmarks, indices::LEFT_EYE_OUTER);
        let brow_arch_height = (brow_arch[1] - eye_outer[1]).abs();
        
        Self {
            forehead_width,
            temple_width,
            cheekbone_width,
            jaw_width,
            chin_width,
            face_length,
            upper_face,
            mid_face,
            lower_face,
            length_to_cheek_ratio,
            jaw_to_cheek_ratio,
            forehead_to_cheek_ratio,
            chin_to_jaw_ratio,
            jaw_angle,
            face_taper,
            roundness_score,
            eye_spacing,
            nose_width,
            lip_width,
            canthal_tilt,
            brow_arch_height,
        }
    }
    
    /// Compute roundness score from jaw characteristics
    fn compute_roundness(jaw_angle: f32, chin_to_jaw_ratio: f32) -> f32 {
        // Higher jaw angle = more angular
        // Higher chin_to_jaw_ratio = rounder
        
        let angle_factor = 1.0 - (jaw_angle / 180.0).clamp(0.0, 1.0);
        let chin_factor = chin_to_jaw_ratio.clamp(0.0, 1.0);
        
        (angle_factor + chin_factor) / 2.0
    }
    
    /// Linear interpolation between two measurements
    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let t_inv = 1.0 - t;
        
        Self {
            forehead_width: a.forehead_width * t_inv + b.forehead_width * t,
            temple_width: a.temple_width * t_inv + b.temple_width * t,
            cheekbone_width: a.cheekbone_width * t_inv + b.cheekbone_width * t,
            jaw_width: a.jaw_width * t_inv + b.jaw_width * t,
            chin_width: a.chin_width * t_inv + b.chin_width * t,
            face_length: a.face_length * t_inv + b.face_length * t,
            upper_face: a.upper_face * t_inv + b.upper_face * t,
            mid_face: a.mid_face * t_inv + b.mid_face * t,
            lower_face: a.lower_face * t_inv + b.lower_face * t,
            length_to_cheek_ratio: a.length_to_cheek_ratio * t_inv + b.length_to_cheek_ratio * t,
            jaw_to_cheek_ratio: a.jaw_to_cheek_ratio * t_inv + b.jaw_to_cheek_ratio * t,
            forehead_to_cheek_ratio: a.forehead_to_cheek_ratio * t_inv + b.forehead_to_cheek_ratio * t,
            chin_to_jaw_ratio: a.chin_to_jaw_ratio * t_inv + b.chin_to_jaw_ratio * t,
            jaw_angle: a.jaw_angle * t_inv + b.jaw_angle * t,
            face_taper: a.face_taper * t_inv + b.face_taper * t,
            roundness_score: a.roundness_score * t_inv + b.roundness_score * t,
            eye_spacing: a.eye_spacing * t_inv + b.eye_spacing * t,
            nose_width: a.nose_width * t_inv + b.nose_width * t,
            lip_width: a.lip_width * t_inv + b.lip_width * t,
            canthal_tilt: a.canthal_tilt * t_inv + b.canthal_tilt * t,
            brow_arch_height: a.brow_arch_height * t_inv + b.brow_arch_height * t,
        }
    }
    
    /// Get feature vector for classification
    pub fn to_feature_vector(&self) -> [f32; 10] {
        [
            self.length_to_cheek_ratio,
            self.jaw_to_cheek_ratio,
            self.forehead_to_cheek_ratio,
            self.chin_to_jaw_ratio,
            self.jaw_angle / 180.0, // Normalize to 0-1
            self.face_taper,
            self.roundness_score,
            self.cheekbone_width / self.jaw_width.max(0.01),
            self.forehead_width / self.jaw_width.max(0.01),
            self.upper_face / self.lower_face.max(0.01),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_measurements() {
        let m = FaceMeasurements::default();
        assert_eq!(m.eye_spacing, 0.0);
    }
    
    #[test]
    fn test_lerp() {
        let a = FaceMeasurements {
            forehead_width: 1.0,
            ..Default::default()
        };
        let b = FaceMeasurements {
            forehead_width: 2.0,
            ..Default::default()
        };
        
        let mid = FaceMeasurements::lerp(&a, &b, 0.5);
        assert!((mid.forehead_width - 1.5).abs() < 0.001);
    }
    
    #[test]
    fn test_feature_vector_size() {
        let m = FaceMeasurements::default();
        let features = m.to_feature_vector();
        assert_eq!(features.len(), 10);
    }
}
