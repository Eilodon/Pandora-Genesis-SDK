//! Face Shape Classification
//!
//! Rule-based classification of face shapes based on measurements.

use super::measurements::FaceMeasurements;
use super::quality::BeautyQuality;

/// Face shape categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceShape {
    /// Balanced proportions, slightly longer than wide
    Oval,
    /// Equal length and width, soft angles
    Round,
    /// Equal length and width, angular jaw
    Square,
    /// Wide forehead, narrow chin
    Heart,
    /// Narrow forehead and chin, wide cheekbones
    Diamond,
    /// Significantly longer than wide
    Oblong,
    /// Unable to classify with confidence
    Unknown,
}

impl FaceShape {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Oval => "Oval",
            Self::Round => "Round",
            Self::Square => "Square",
            Self::Heart => "Heart",
            Self::Diamond => "Diamond",
            Self::Oblong => "Oblong",
            Self::Unknown => "Unknown",
        }
    }
    
    /// Get all variants (for iteration)
    pub fn all() -> &'static [FaceShape] {
        &[
            FaceShape::Oval,
            FaceShape::Round,
            FaceShape::Square,
            FaceShape::Heart,
            FaceShape::Diamond,
            FaceShape::Oblong,
        ]
    }
}

/// Face shape classification result
#[derive(Debug, Clone)]
pub struct FaceShapeResult {
    /// Primary classification
    pub shape: FaceShape,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Top-2 scores for transparency
    pub scores: [(FaceShape, f32); 2],
    /// Explanation of classification
    pub reason: String,
}

impl Default for FaceShapeResult {
    fn default() -> Self {
        Self {
            shape: FaceShape::Unknown,
            confidence: 0.0,
            scores: [(FaceShape::Unknown, 0.0), (FaceShape::Unknown, 0.0)],
            reason: String::new(),
        }
    }
}

/// Configuration for shape classification thresholds
#[derive(Debug, Clone)]
pub struct ShapeThresholds {
    /// Length/width ratio threshold for oblong vs others
    pub oblong_ratio: f32,
    /// Length/width ratio threshold for round vs oval
    pub round_ratio: f32,
    /// Jaw/cheek ratio threshold for square vs round
    pub square_jaw_ratio: f32,
    /// Taper threshold for heart shape
    pub heart_taper: f32,
    /// Cheekbone prominence for diamond
    pub diamond_cheek_prominence: f32,
}

impl Default for ShapeThresholds {
    fn default() -> Self {
        Self {
            oblong_ratio: 1.6,
            round_ratio: 1.2,
            square_jaw_ratio: 0.9,
            heart_taper: 0.25,
            diamond_cheek_prominence: 1.1,
        }
    }
}

/// Rule-based face shape classifier
pub struct ShapeClassifier {
    thresholds: ShapeThresholds,
}

impl ShapeClassifier {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            thresholds: ShapeThresholds::default(),
        }
    }
    
    /// Create with custom thresholds
    pub fn with_thresholds(thresholds: ShapeThresholds) -> Self {
        Self { thresholds }
    }
    
    /// Classify face shape from measurements
    pub fn classify(&self, measurements: &FaceMeasurements, quality: &BeautyQuality) -> FaceShapeResult {
        // Quality penalty
        let quality_factor = quality.overall_confidence;
        
        // Score each shape
        let mut scores: Vec<(FaceShape, f32, String)> = FaceShape::all()
            .iter()
            .map(|&shape| {
                let (score, reason) = self.score_shape(shape, measurements);
                (shape, score * quality_factor, reason)
            })
            .collect();
        
        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Check margin between top-1 and top-2
        let (top_shape, top_score, top_reason) = scores[0].clone();
        let (second_shape, second_score, _) = scores[1].clone();
        
        let margin = top_score - second_score;
        let min_confidence = 0.3;
        
        // Confidence = base score + margin bonus
        let confidence = if top_score > min_confidence && margin > 0.1 {
            (top_score + margin * 0.5).clamp(0.0, 1.0)
        } else if top_score > min_confidence {
            top_score * 0.8 // Penalize low margin
        } else {
            0.0
        };
        
        let shape = if confidence > 0.3 {
            top_shape
        } else {
            FaceShape::Unknown
        };
        
        FaceShapeResult {
            shape,
            confidence,
            scores: [(top_shape, top_score), (second_shape, second_score)],
            reason: top_reason,
        }
    }
    
    /// Score how well measurements match a face shape
    fn score_shape(&self, shape: FaceShape, m: &FaceMeasurements) -> (f32, String) {
        let t = &self.thresholds;
        
        match shape {
            FaceShape::Oval => {
                // Oval: length/cheek ~1.3-1.5, balanced proportions
                let length_score = Self::gaussian_score(m.length_to_cheek_ratio, 1.4, 0.2);
                let jaw_score = Self::gaussian_score(m.jaw_to_cheek_ratio, 0.8, 0.15);
                let forehead_score = Self::gaussian_score(m.forehead_to_cheek_ratio, 0.9, 0.15);
                let round_score = Self::gaussian_score(m.roundness_score, 0.5, 0.2);
                
                let score = (length_score + jaw_score + forehead_score + round_score) / 4.0;
                let reason = format!(
                    "Length ratio {:.2}, jaw/cheek {:.2}, forehead/cheek {:.2}",
                    m.length_to_cheek_ratio, m.jaw_to_cheek_ratio, m.forehead_to_cheek_ratio
                );
                (score, reason)
            }
            
            FaceShape::Round => {
                // Round: length/cheek ~1.0-1.2, high roundness, equal widths
                let length_score = Self::gaussian_score(m.length_to_cheek_ratio, 1.1, 0.15);
                let jaw_score = Self::gaussian_score(m.jaw_to_cheek_ratio, 0.9, 0.1);
                let roundness = Self::gaussian_score(m.roundness_score, 0.8, 0.15);
                let taper_score = 1.0 - m.face_taper.abs(); // Low taper
                
                let score = (length_score + jaw_score + roundness + taper_score) / 4.0;
                let reason = format!(
                    "Length ratio {:.2}, roundness {:.2}",
                    m.length_to_cheek_ratio, m.roundness_score
                );
                (score, reason)
            }
            
            FaceShape::Square => {
                // Square: length/cheek ~1.0-1.2, low roundness, high jaw ratio
                let length_score = Self::gaussian_score(m.length_to_cheek_ratio, 1.1, 0.15);
                let jaw_score = if m.jaw_to_cheek_ratio > t.square_jaw_ratio {
                    1.0
                } else {
                    m.jaw_to_cheek_ratio / t.square_jaw_ratio
                };
                let angular = Self::gaussian_score(m.roundness_score, 0.2, 0.2);
                
                let score = (length_score + jaw_score + angular) / 3.0;
                let reason = format!(
                    "Jaw/cheek {:.2}, roundness {:.2} (angular)",
                    m.jaw_to_cheek_ratio, m.roundness_score
                );
                (score, reason)
            }
            
            FaceShape::Heart => {
                // Heart: wide forehead, narrow chin, high taper
                let taper_score = if m.face_taper > t.heart_taper {
                    1.0
                } else {
                    m.face_taper / t.heart_taper
                };
                let forehead_score = if m.forehead_to_cheek_ratio > 0.95 {
                    1.0
                } else {
                    m.forehead_to_cheek_ratio / 0.95
                };
                let chin_score = Self::gaussian_score(m.chin_to_jaw_ratio, 0.4, 0.15);
                
                let score = (taper_score + forehead_score + chin_score) / 3.0;
                let reason = format!(
                    "Face taper {:.2}, forehead prominent {:.2}",
                    m.face_taper, m.forehead_to_cheek_ratio
                );
                (score, reason)
            }
            
            FaceShape::Diamond => {
                // Diamond: narrow forehead and jaw, wide cheekbones
                let cheek_vs_forehead = m.cheekbone_width / m.forehead_width.max(0.01);
                let cheek_vs_jaw = m.cheekbone_width / m.jaw_width.max(0.01);
                
                let forehead_score = if cheek_vs_forehead > t.diamond_cheek_prominence {
                    1.0
                } else {
                    cheek_vs_forehead / t.diamond_cheek_prominence
                };
                let jaw_score = if cheek_vs_jaw > t.diamond_cheek_prominence {
                    1.0
                } else {
                    cheek_vs_jaw / t.diamond_cheek_prominence
                };
                
                let score = (forehead_score + jaw_score) / 2.0;
                let reason = format!(
                    "Cheekbone prominence: vs forehead {:.2}, vs jaw {:.2}",
                    cheek_vs_forehead, cheek_vs_jaw
                );
                (score, reason)
            }
            
            FaceShape::Oblong => {
                // Oblong: significantly longer than wide
                let length_score = if m.length_to_cheek_ratio > t.oblong_ratio {
                    1.0
                } else {
                    (m.length_to_cheek_ratio - 1.3) / (t.oblong_ratio - 1.3)
                }.max(0.0);
                
                let even_widths = 1.0 - (m.forehead_to_cheek_ratio - m.jaw_to_cheek_ratio).abs();
                
                let score = length_score * 0.7 + even_widths * 0.3;
                let reason = format!(
                    "Length ratio {:.2} (>{:.2} = oblong)",
                    m.length_to_cheek_ratio, t.oblong_ratio
                );
                (score, reason)
            }
            
            FaceShape::Unknown => (0.0, String::new()),
        }
    }
    
    /// Gaussian-like scoring function
    fn gaussian_score(value: f32, target: f32, sigma: f32) -> f32 {
        let diff = (value - target) / sigma;
        (-0.5 * diff * diff).exp()
    }
}

impl Default for ShapeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_quality() -> BeautyQuality {
        BeautyQuality {
            detection_ok: true,
            pose_ok: true,
            overall_confidence: 0.95,
            ..Default::default()
        }
    }
    
    #[test]
    fn test_oval_classification() {
        let m = FaceMeasurements {
            length_to_cheek_ratio: 1.4,
            jaw_to_cheek_ratio: 0.8,
            forehead_to_cheek_ratio: 0.9,
            roundness_score: 0.5,
            ..Default::default()
        };
        
        let classifier = ShapeClassifier::new();
        let result = classifier.classify(&m, &make_quality());
        
        assert_eq!(result.shape, FaceShape::Oval);
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_round_classification() {
        let m = FaceMeasurements {
            length_to_cheek_ratio: 1.1,
            jaw_to_cheek_ratio: 0.9,
            forehead_to_cheek_ratio: 0.9,
            roundness_score: 0.8,
            face_taper: 0.05,
            ..Default::default()
        };
        
        let classifier = ShapeClassifier::new();
        let result = classifier.classify(&m, &make_quality());
        
        assert_eq!(result.shape, FaceShape::Round);
    }
    
    #[test]
    fn test_low_quality_reduces_confidence() {
        let m = FaceMeasurements {
            length_to_cheek_ratio: 1.4,
            jaw_to_cheek_ratio: 0.8,
            ..Default::default()
        };
        
        let low_quality = BeautyQuality {
            overall_confidence: 0.3,
            ..Default::default()
        };
        
        let classifier = ShapeClassifier::new();
        let result = classifier.classify(&m, &low_quality);
        
        // Low quality should reduce confidence
        assert!(result.confidence < 0.5);
    }
    
    #[test]
    fn test_shape_names() {
        assert_eq!(FaceShape::Oval.name(), "Oval");
        assert_eq!(FaceShape::Heart.name(), "Heart");
    }
}
