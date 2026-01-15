//! MediaPipe Landmark Indices and Normalization
//!
//! Provides canonical face space normalization using inter-ocular distance
//! as scale and midpoint between inner eye corners as origin.

/// MediaPipe Face Mesh 468 landmark indices
pub mod indices {
    // === Eyes ===
    /// Left eye inner corner
    pub const LEFT_EYE_INNER: usize = 133;
    /// Left eye outer corner
    pub const LEFT_EYE_OUTER: usize = 33;
    /// Right eye inner corner
    pub const RIGHT_EYE_INNER: usize = 362;
    /// Right eye outer corner
    pub const RIGHT_EYE_OUTER: usize = 263;
    /// Left eye center (pupil area)
    pub const LEFT_EYE_CENTER: usize = 468; // Not in 468, using approximation
    /// Right eye center
    pub const RIGHT_EYE_CENTER: usize = 473;
    
    // === Face Contour ===
    /// Top of forehead (hairline)
    pub const FOREHEAD_TOP: usize = 10;
    /// Center of forehead
    pub const FOREHEAD_CENTER: usize = 151;
    /// Left temple
    pub const LEFT_TEMPLE: usize = 54;
    /// Right temple
    pub const RIGHT_TEMPLE: usize = 284;
    /// Left cheekbone (widest point)
    pub const LEFT_CHEEKBONE: usize = 234;
    /// Right cheekbone
    pub const RIGHT_CHEEKBONE: usize = 454;
    /// Left jaw angle
    pub const LEFT_JAW_ANGLE: usize = 136;
    /// Right jaw angle
    pub const RIGHT_JAW_ANGLE: usize = 365;
    /// Left jaw edge
    pub const LEFT_JAW: usize = 172;
    /// Right jaw edge
    pub const RIGHT_JAW: usize = 397;
    /// Chin center
    pub const CHIN: usize = 152;
    /// Chin left
    pub const CHIN_LEFT: usize = 175;
    /// Chin right
    pub const CHIN_RIGHT: usize = 396;
    
    // === Nose ===
    /// Nose tip
    pub const NOSE_TIP: usize = 4;
    /// Nose bridge (between eyes)
    pub const NOSE_BRIDGE: usize = 6;
    /// Nose left alar (nostril)
    pub const NOSE_LEFT_ALAR: usize = 115;
    /// Nose right alar
    pub const NOSE_RIGHT_ALAR: usize = 344;
    
    // === Lips ===
    /// Upper lip center
    pub const UPPER_LIP_CENTER: usize = 0;
    /// Lower lip center
    pub const LOWER_LIP_CENTER: usize = 17;
    /// Left mouth corner
    pub const LEFT_MOUTH_CORNER: usize = 61;
    /// Right mouth corner
    pub const RIGHT_MOUTH_CORNER: usize = 291;
    
    // === Eyebrows ===
    /// Left eyebrow inner
    pub const LEFT_BROW_INNER: usize = 107;
    /// Left eyebrow outer
    pub const LEFT_BROW_OUTER: usize = 70;
    /// Left eyebrow arch (peak)
    pub const LEFT_BROW_ARCH: usize = 105;
    /// Right eyebrow inner
    pub const RIGHT_BROW_INNER: usize = 336;
    /// Right eyebrow outer
    pub const RIGHT_BROW_OUTER: usize = 300;
    /// Right eyebrow arch
    pub const RIGHT_BROW_ARCH: usize = 334;
}

/// Landmarks normalized to canonical face space
#[derive(Debug, Clone)]
pub struct CanonicalLandmarks {
    /// Normalized landmarks (origin = eye midpoint, scale = inter-ocular = 1.0)
    pub points: Vec<[f32; 2]>,
    /// Inter-ocular distance in original pixels (for reference)
    pub inter_ocular_px: f32,
    /// Origin offset in original coordinates
    pub origin: [f32; 2],
    /// Whether normalization was successful
    pub valid: bool,
}

impl Default for CanonicalLandmarks {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            inter_ocular_px: 1.0,
            origin: [0.0, 0.0],
            valid: false,
        }
    }
}

/// Normalize landmarks to canonical face space
///
/// - Origin: midpoint between inner eye corners
/// - Scale: inter-ocular distance = 1.0
/// - Optionally corrects for head rotation (if pose provided)
///
/// # Arguments
/// * `landmarks` - 468 MediaPipe landmarks
/// * `normalized` - Whether input is normalized (0..1) or pixel coords
/// * `width`, `height` - Frame dimensions (for denormalization if needed)
/// * `pose` - Optional [yaw, pitch, roll] in degrees
pub fn normalize_to_canonical(
    landmarks: &[[f32; 2]],
    normalized: bool,
    width: u32,
    height: u32,
    pose: Option<[f32; 3]>,
) -> CanonicalLandmarks {
    if landmarks.len() < 468 {
        return CanonicalLandmarks::default();
    }
    
    // Convert to pixel coordinates if normalized
    let pixels: Vec<[f32; 2]> = if normalized {
        landmarks.iter()
            .map(|[x, y]| [x * width as f32, y * height as f32])
            .collect()
    } else {
        landmarks.to_vec()
    };
    
    // Get eye inner corners
    let left_inner = pixels[indices::LEFT_EYE_INNER];
    let right_inner = pixels[indices::RIGHT_EYE_INNER];
    
    // Compute origin (midpoint between inner eye corners)
    let origin = [
        (left_inner[0] + right_inner[0]) / 2.0,
        (left_inner[1] + right_inner[1]) / 2.0,
    ];
    
    // Compute inter-ocular distance (scale factor)
    let dx = right_inner[0] - left_inner[0];
    let dy = right_inner[1] - left_inner[1];
    let inter_ocular = (dx * dx + dy * dy).sqrt().max(1.0);
    
    // Compute rotation angle from eye line (for 2D correction)
    let eye_angle = dy.atan2(dx);
    
    // Apply pose correction if available (reduce effect of yaw on measurements)
    let yaw_correction = if let Some([yaw, _pitch, _roll]) = pose {
        // Simple perspective correction: stretch x based on yaw
        let yaw_rad = yaw.to_radians();
        1.0 / yaw_rad.cos().max(0.7)
    } else {
        1.0
    };
    
    // Transform all points
    let points: Vec<[f32; 2]> = pixels.iter()
        .map(|[x, y]| {
            // Translate to origin
            let tx = x - origin[0];
            let ty = y - origin[1];
            
            // Rotate to align eyes horizontally
            let rx = tx * eye_angle.cos() + ty * eye_angle.sin();
            let ry = -tx * eye_angle.sin() + ty * eye_angle.cos();
            
            // Scale to inter-ocular = 1.0, apply yaw correction
            [rx / inter_ocular * yaw_correction, ry / inter_ocular]
        })
        .collect();
    
    CanonicalLandmarks {
        points,
        inter_ocular_px: inter_ocular,
        origin,
        valid: true,
    }
}

/// Get landmark point by index (with bounds check)
pub fn get_point(landmarks: &CanonicalLandmarks, idx: usize) -> [f32; 2] {
    if idx < landmarks.points.len() {
        landmarks.points[idx]
    } else {
        [0.0, 0.0]
    }
}

/// Compute distance between two landmarks
pub fn landmark_distance(landmarks: &CanonicalLandmarks, idx1: usize, idx2: usize) -> f32 {
    let p1 = get_point(landmarks, idx1);
    let p2 = get_point(landmarks, idx2);
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    (dx * dx + dy * dy).sqrt()
}

/// Compute angle between three landmarks (angle at middle point)
pub fn landmark_angle(landmarks: &CanonicalLandmarks, idx1: usize, idx2: usize, idx3: usize) -> f32 {
    let p1 = get_point(landmarks, idx1);
    let p2 = get_point(landmarks, idx2);
    let p3 = get_point(landmarks, idx3);
    
    let v1 = [p1[0] - p2[0], p1[1] - p2[1]];
    let v2 = [p3[0] - p2[0], p3[1] - p2[1]];
    
    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let cross = v1[0] * v2[1] - v1[1] * v2[0];
    
    cross.atan2(dot).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_dummy_landmarks() -> Vec<[f32; 2]> {
        let mut landmarks = vec![[0.5, 0.5]; 468];
        // Set eye corners for inter-ocular distance
        landmarks[indices::LEFT_EYE_INNER] = [0.4, 0.4];
        landmarks[indices::RIGHT_EYE_INNER] = [0.6, 0.4];
        landmarks
    }
    
    #[test]
    fn test_normalize_requires_468_landmarks() {
        let too_few = vec![[0.0, 0.0]; 100];
        let result = normalize_to_canonical(&too_few, true, 1920, 1080, None);
        assert!(!result.valid);
    }
    
    #[test]
    fn test_normalize_computes_origin() {
        let landmarks = make_dummy_landmarks();
        let result = normalize_to_canonical(&landmarks, true, 1000, 1000, None);
        assert!(result.valid);
        // Origin should be at midpoint (0.5, 0.4) * 1000 = (500, 400)
        assert!((result.origin[0] - 500.0).abs() < 1.0);
        assert!((result.origin[1] - 400.0).abs() < 1.0);
    }
    
    #[test]
    fn test_inter_ocular_distance() {
        let landmarks = make_dummy_landmarks();
        let result = normalize_to_canonical(&landmarks, true, 1000, 1000, None);
        // IOD = (0.6 - 0.4) * 1000 = 200px
        assert!((result.inter_ocular_px - 200.0).abs() < 1.0);
    }
}
