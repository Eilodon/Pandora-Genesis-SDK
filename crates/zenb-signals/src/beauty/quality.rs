//! Quality Gating for Beauty Analysis
//!
//! Determines whether frame quality is sufficient for accurate measurements.

use super::BeautyInput;

/// Configuration for quality gating
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Maximum acceptable yaw angle (degrees)
    pub max_yaw: f32,
    /// Maximum acceptable pitch angle (degrees)
    pub max_pitch: f32,
    /// Minimum ROI stability (0-1)
    pub min_stability: f32,
    /// Minimum face detection confidence (0-1)
    pub min_face_confidence: f32,
    /// Maximum landmark jitter (normalized to face size)
    pub max_jitter: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            max_yaw: 25.0,
            max_pitch: 20.0,
            min_stability: 0.7,
            min_face_confidence: 0.8,
            max_jitter: 0.05,
        }
    }
}

/// Quality assessment result
#[derive(Debug, Clone)]
pub struct BeautyQuality {
    /// Face detection is valid
    pub detection_ok: bool,
    /// Pose is within acceptable range
    pub pose_ok: bool,
    /// Face might be occluded (glasses, hand, mask)
    pub occlusion_suspected: bool,
    /// Lighting is acceptable for color analysis
    pub lighting_ok: bool,
    /// ROI stability (0 = unstable, 1 = stable)
    pub roi_stability: f32,
    /// Overall confidence (0-1)
    pub overall_confidence: f32,
    /// Reason codes for debugging
    pub issues: Vec<QualityIssue>,
}

impl Default for BeautyQuality {
    fn default() -> Self {
        Self {
            detection_ok: false,
            pose_ok: false,
            occlusion_suspected: false,
            lighting_ok: true,
            roi_stability: 0.0,
            overall_confidence: 0.0,
            issues: Vec::new(),
        }
    }
}

/// Quality issue codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityIssue {
    LowDetectionConfidence,
    InsufficientLandmarks,
    ExcessiveYaw,
    ExcessivePitch,
    HighJitter,
    SuspectedOcclusion,
    PoorLighting,
}

/// Compute quality metrics for a frame
pub fn compute_quality(
    input: &BeautyInput,
    prev_landmarks: Option<&[[f32; 2]]>,
    config: &QualityConfig,
) -> BeautyQuality {
    let mut quality = BeautyQuality::default();
    
    // 1. Detection confidence check
    if input.landmarks_2d.len() < 468 {
        quality.issues.push(QualityIssue::InsufficientLandmarks);
        return quality;
    }
    
    if input.face_confidence < config.min_face_confidence {
        quality.issues.push(QualityIssue::LowDetectionConfidence);
    } else {
        quality.detection_ok = true;
    }
    
    // 2. Pose check
    if let Some([yaw, pitch, _roll]) = input.pose {
        if yaw.abs() > config.max_yaw {
            quality.issues.push(QualityIssue::ExcessiveYaw);
        }
        if pitch.abs() > config.max_pitch {
            quality.issues.push(QualityIssue::ExcessivePitch);
        }
        
        quality.pose_ok = yaw.abs() <= config.max_yaw && pitch.abs() <= config.max_pitch;
    } else {
        // No pose data - estimate from landmark asymmetry
        quality.pose_ok = estimate_pose_from_landmarks(&input.landmarks_2d, config);
    }
    
    // 3. Stability check (jitter calculation)
    if let Some(prev) = prev_landmarks {
        let jitter = compute_landmark_jitter(&input.landmarks_2d, prev);
        quality.roi_stability = (1.0 - jitter / config.max_jitter).clamp(0.0, 1.0);
        
        if jitter > config.max_jitter {
            quality.issues.push(QualityIssue::HighJitter);
        }
    } else {
        quality.roi_stability = 1.0; // First frame, assume stable
    }
    
    // 4. Occlusion heuristics
    quality.occlusion_suspected = detect_occlusion(&input.landmarks_2d);
    if quality.occlusion_suspected {
        quality.issues.push(QualityIssue::SuspectedOcclusion);
    }
    
    // 5. Lighting check (if ROI colors available)
    if let Some(ref colors) = input.roi_colors {
        quality.lighting_ok = check_lighting(colors);
        if !quality.lighting_ok {
            quality.issues.push(QualityIssue::PoorLighting);
        }
    }
    
    // 6. Compute overall confidence
    let mut confidence = input.face_confidence;
    
    if !quality.pose_ok {
        confidence *= 0.5;
    }
    if quality.occlusion_suspected {
        confidence *= 0.7;
    }
    if quality.roi_stability < config.min_stability {
        confidence *= quality.roi_stability / config.min_stability;
    }
    
    quality.overall_confidence = confidence.clamp(0.0, 1.0);
    
    quality
}

/// Estimate if pose is acceptable from landmark asymmetry
fn estimate_pose_from_landmarks(landmarks: &[[f32; 2]], _config: &QualityConfig) -> bool {
    if landmarks.len() < 468 {
        return false;
    }
    
    // Check left/right symmetry of key landmarks
    // Left eye inner vs right eye inner y-coordinate difference indicates pitch
    // Width ratio of left vs right face indicates yaw
    
    let left_cheek = landmarks[234];
    let right_cheek = landmarks[454];
    let nose = landmarks[4];
    
    // Compute distance from nose to each cheek
    let left_dist = ((nose[0] - left_cheek[0]).powi(2) + (nose[1] - left_cheek[1]).powi(2)).sqrt();
    let right_dist = ((nose[0] - right_cheek[0]).powi(2) + (nose[1] - right_cheek[1]).powi(2)).sqrt();
    
    // Asymmetry ratio
    let ratio = if left_dist > right_dist {
        right_dist / left_dist.max(0.001)
    } else {
        left_dist / right_dist.max(0.001)
    };
    
    // If ratio < 0.7, face is turned too much
    ratio > 0.7
}

/// Compute landmark jitter (normalized movement from previous frame)
fn compute_landmark_jitter(curr: &[[f32; 2]], prev: &[[f32; 2]]) -> f32 {
    if curr.len() != prev.len() || curr.is_empty() {
        return 0.0;
    }
    
    // Use anchor landmarks (less affected by expressions)
    let anchor_indices = [4, 133, 362, 152]; // nose tip, eye inners, chin
    
    let mut total_dist = 0.0;
    let mut count = 0;
    
    for &idx in &anchor_indices {
        if idx < curr.len() && idx < prev.len() {
            let dx = curr[idx][0] - prev[idx][0];
            let dy = curr[idx][1] - prev[idx][1];
            total_dist += (dx * dx + dy * dy).sqrt();
            count += 1;
        }
    }
    
    if count > 0 {
        total_dist / count as f32
    } else {
        0.0
    }
}

/// Detect potential face occlusion
fn detect_occlusion(landmarks: &[[f32; 2]]) -> bool {
    if landmarks.len() < 468 {
        return true;
    }
    
    // Check for abnormal landmark configurations
    // 1. Left/right face area should be roughly similar
    // 2. Face contour should be convex (mostly)
    
    // Simple heuristic: check if any key landmark is at origin (failed detection)
    let key_indices = [4, 152, 234, 454, 10]; // nose, chin, cheeks, forehead
    
    for &idx in &key_indices {
        let [x, y] = landmarks[idx];
        if x.abs() < 0.001 && y.abs() < 0.001 {
            return true; // Landmark at origin = detection failure
        }
    }
    
    false
}

/// Check if lighting is acceptable for color analysis
fn check_lighting(colors: &super::RoiColors) -> bool {
    // Check if ROI colors have reasonable values and aren't too dark/bright
    
    let regions = [
        colors.forehead,
        colors.left_cheek,
        colors.right_cheek,
        colors.chin,
    ];
    
    for rgb in &regions {
        let brightness = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        
        // Too dark or too bright = poor lighting
        if brightness < 30.0 || brightness > 240.0 {
            return false;
        }
    }
    
    // Check for even lighting (cheeks should be similar)
    let left_brightness = (colors.left_cheek[0] + colors.left_cheek[1] + colors.left_cheek[2]) / 3.0;
    let right_brightness = (colors.right_cheek[0] + colors.right_cheek[1] + colors.right_cheek[2]) / 3.0;
    
    let brightness_diff = (left_brightness - right_brightness).abs();
    
    // More than 30 points difference = uneven lighting
    brightness_diff < 30.0
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quality_config_defaults() {
        let config = QualityConfig::default();
        assert_eq!(config.max_yaw, 25.0);
        assert_eq!(config.max_pitch, 20.0);
    }
    
    #[test]
    fn test_insufficient_landmarks() {
        let input = BeautyInput {
            landmarks_2d: vec![[0.0, 0.0]; 100],
            face_confidence: 0.9,
            ..Default::default()
        };
        
        let quality = compute_quality(&input, None, &QualityConfig::default());
        assert!(!quality.detection_ok);
        assert!(quality.issues.contains(&QualityIssue::InsufficientLandmarks));
    }
    
    #[test]
    fn test_pose_check() {
        let input = BeautyInput {
            landmarks_2d: vec![[0.5, 0.5]; 468],
            face_confidence: 0.95,
            pose: Some([30.0, 5.0, 0.0]), // Excessive yaw
            ..Default::default()
        };
        
        let quality = compute_quality(&input, None, &QualityConfig::default());
        assert!(!quality.pose_ok);
        assert!(quality.issues.contains(&QualityIssue::ExcessiveYaw));
    }
    
    #[test]
    fn test_good_quality() {
        let input = BeautyInput {
            landmarks_2d: vec![[0.5, 0.5]; 468],
            face_confidence: 0.95,
            pose: Some([5.0, 3.0, 1.0]), // Good pose
            ..Default::default()
        };
        
        let quality = compute_quality(&input, None, &QualityConfig::default());
        assert!(quality.detection_ok);
        assert!(quality.pose_ok);
        assert!(quality.overall_confidence > 0.8);
    }
}
