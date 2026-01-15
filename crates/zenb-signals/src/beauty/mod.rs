//! Beauty Analysis Module (Mắt Thần Hình)
//!
//! Face geometry analysis, shape classification, and makeup recommendations.
//!
//! # Overview
//!
//! ```text
//! BeautyInput (landmarks + ROI colors)
//!     │
//!     ├──► Quality Gating (pose, stability, lighting)
//!     │
//!     ├──► Landmark Normalization (canonical face space)
//!     │
//!     ├──► Face Measurements (22 geometric features)
//!     │
//!     ├──► Face Shape Classification (6 categories)
//!     │
//!     ├──► Makeup Plan Generation (zones + polygons)
//!     │
//!     └──► Skin Tone Analysis (optional)
//!
//! Output: BeautyFrameResult
//! ```
//!
//! # Example
//!
//! ```ignore
//! use zenb_signals::beauty::{BeautyAnalyzer, BeautyInput};
//!
//! let mut analyzer = BeautyAnalyzer::new();
//!
//! // Each frame from video
//! let input = BeautyInput {
//!     timestamp_us: 0,
//!     frame_width: 1920,
//!     frame_height: 1080,
//!     landmarks_2d: landmarks_from_mediapipe,
//!     landmarks_normalized: true,
//!     face_confidence: 0.95,
//!     pose: Some([5.0, -3.0, 1.0]), // yaw, pitch, roll
//!     roi_colors: None,
//! };
//!
//! let result = analyzer.analyze(&input);
//! println!("Face shape: {:?} ({:.0}%)", result.face_shape.shape, result.face_shape.confidence * 100.0);
//! ```

mod landmarks;
mod measurements;
mod face_shape;
mod makeup_plan;
mod skin_tone;
mod quality;

pub use landmarks::{
    CanonicalLandmarks, normalize_to_canonical,
    indices as landmark_indices,
};
pub use measurements::FaceMeasurements;
pub use face_shape::{FaceShape, FaceShapeResult, ShapeClassifier};
pub use makeup_plan::{MakeupPlan, MakeupZone, ZoneType, ShadeHint, MakeupStyle};
pub use skin_tone::{SkinAnalysis, Undertone, SkinDepth};
pub use quality::{BeautyQuality, QualityConfig, compute_quality};



// ============================================================================
// Input/Output Contracts
// ============================================================================

/// Pre-computed ROI colors from client
#[derive(Debug, Clone, Default)]
pub struct RoiColors {
    /// Forehead mean RGB (0-255)
    pub forehead: [f32; 3],
    /// Left cheek mean RGB
    pub left_cheek: [f32; 3],
    /// Right cheek mean RGB
    pub right_cheek: [f32; 3],
    /// Chin mean RGB
    pub chin: [f32; 3],
}

/// Input for beauty analysis (FFI-friendly)
#[derive(Debug, Clone)]
pub struct BeautyInput {
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// Frame width in pixels
    pub frame_width: u32,
    /// Frame height in pixels
    pub frame_height: u32,
    /// 468 MediaPipe landmarks [(x, y), ...]
    pub landmarks_2d: Vec<[f32; 2]>,
    /// Whether landmarks are normalized (0..1) or pixel coordinates
    pub landmarks_normalized: bool,
    /// Face detection confidence (0-1)
    pub face_confidence: f32,
    /// Optional: Pose from ARKit/MediaPipe [yaw, pitch, roll] in degrees
    pub pose: Option<[f32; 3]>,
    /// Optional: ROI mean colors pre-computed by client
    pub roi_colors: Option<RoiColors>,
}

impl Default for BeautyInput {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            frame_width: 0,
            frame_height: 0,
            landmarks_2d: Vec::new(),
            landmarks_normalized: true,
            face_confidence: 0.0,
            pose: None,
            roi_colors: None,
        }
    }
}

/// Result of beauty frame analysis
#[derive(Debug, Clone)]
pub struct BeautyFrameResult {
    /// Quality metrics
    pub quality: BeautyQuality,
    /// Face geometry measurements (normalized)
    pub measurements: FaceMeasurements,
    /// Face shape classification
    pub face_shape: FaceShapeResult,
    /// Makeup recommendations
    pub makeup_plan: MakeupPlan,
    /// Skin tone analysis (if available)
    pub skin: Option<SkinAnalysis>,
}

// ============================================================================
// BeautyAnalyzer (Stateful)
// ============================================================================

/// Configuration for beauty analysis
#[derive(Debug, Clone)]
pub struct BeautyConfig {
    /// Quality gating configuration
    pub quality: QualityConfig,
    /// Makeup style preference
    pub style: MakeupStyle,
    /// Smoothing factor for measurements (0 = no smoothing, 1 = max smoothing)
    pub smoothing_alpha: f32,
}

impl Default for BeautyConfig {
    fn default() -> Self {
        Self {
            quality: QualityConfig::default(),
            style: MakeupStyle::Natural,
            smoothing_alpha: 0.3,
        }
    }
}

/// Stateful beauty analyzer with temporal smoothing
pub struct BeautyAnalyzer {
    config: BeautyConfig,
    classifier: ShapeClassifier,
    prev_landmarks: Option<Vec<[f32; 2]>>,
    smoothed_measurements: Option<FaceMeasurements>,
    shape_history: Vec<FaceShape>,
}

impl BeautyAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self::with_config(BeautyConfig::default())
    }
    
    /// Create analyzer with custom config
    pub fn with_config(config: BeautyConfig) -> Self {
        Self {
            config,
            classifier: ShapeClassifier::new(),
            prev_landmarks: None,
            smoothed_measurements: None,
            shape_history: Vec::with_capacity(30),
        }
    }
    
    /// Analyze single frame
    pub fn analyze(&mut self, input: &BeautyInput) -> BeautyFrameResult {
        // 1. Quality gating
        let quality = compute_quality(
            input,
            self.prev_landmarks.as_deref(),
            &self.config.quality,
        );
        
        // 2. Normalize landmarks to canonical space
        let canonical = normalize_to_canonical(
            &input.landmarks_2d,
            input.landmarks_normalized,
            input.frame_width,
            input.frame_height,
            input.pose,
        );
        
        // 3. Compute measurements
        let raw_measurements = FaceMeasurements::from_landmarks(&canonical);
        let measurements = self.smooth_measurements(&raw_measurements);
        
        // 4. Classify face shape
        let face_shape = self.classifier.classify(&measurements, &quality);
        self.update_shape_history(face_shape.shape);
        
        // 5. Generate makeup plan
        let makeup_plan = makeup_plan::generate_makeup_plan(
            face_shape.shape,
            &measurements,
            &canonical,
            self.config.style,
        );
        
        // 6. Skin analysis (if ROI colors available)
        let skin = input.roi_colors.as_ref()
            .and_then(|colors| skin_tone::analyze_skin(colors, &quality));
        
        // Update state
        self.prev_landmarks = Some(input.landmarks_2d.clone());
        
        BeautyFrameResult {
            quality,
            measurements,
            face_shape,
            makeup_plan,
            skin,
        }
    }
    
    /// Reset analyzer state
    pub fn reset(&mut self) {
        self.prev_landmarks = None;
        self.smoothed_measurements = None;
        self.shape_history.clear();
    }
    
    /// Get current config
    pub fn config(&self) -> &BeautyConfig {
        &self.config
    }
    
    /// Update config
    pub fn set_config(&mut self, config: BeautyConfig) {
        self.config = config;
    }
    
    /// Set makeup style
    pub fn set_style(&mut self, style: MakeupStyle) {
        self.config.style = style;
    }
    
    // --- Private ---
    
    fn smooth_measurements(&mut self, raw: &FaceMeasurements) -> FaceMeasurements {
        let alpha = self.config.smoothing_alpha;
        
        if alpha < 0.01 {
            // No smoothing
            self.smoothed_measurements = Some(raw.clone());
            return raw.clone();
        }
        
        match &self.smoothed_measurements {
            Some(prev) => {
                let smoothed = FaceMeasurements::lerp(prev, raw, 1.0 - alpha);
                self.smoothed_measurements = Some(smoothed.clone());
                smoothed
            }
            None => {
                self.smoothed_measurements = Some(raw.clone());
                raw.clone()
            }
        }
    }
    
    fn update_shape_history(&mut self, shape: FaceShape) {
        self.shape_history.push(shape);
        if self.shape_history.len() > 30 {
            self.shape_history.remove(0);
        }
    }
}

impl Default for BeautyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_beauty_input_default() {
        let input = BeautyInput::default();
        assert_eq!(input.landmarks_2d.len(), 0);
        assert!(input.landmarks_normalized);
    }
    
    #[test]
    fn test_analyzer_creation() {
        let analyzer = BeautyAnalyzer::new();
        assert!(analyzer.prev_landmarks.is_none());
    }
}
