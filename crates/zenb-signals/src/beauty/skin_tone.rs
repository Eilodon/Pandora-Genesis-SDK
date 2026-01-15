//! Skin Tone Analysis
//!
//! Analyzes skin undertone and depth from ROI colors.

use super::quality::BeautyQuality;
use super::RoiColors;

/// Skin undertone classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Undertone {
    Warm,
    Cool,
    Neutral,
    Unknown,
}

/// Skin depth classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkinDepth {
    Fair,
    Light,
    Medium,
    Tan,
    Deep,
    Unknown,
}

/// Notes about skin condition
#[derive(Debug, Clone)]
pub enum SkinNote {
    Redness,
    Uneven,
    GoodCondition,
}

/// Skin analysis result
#[derive(Debug, Clone)]
pub struct SkinAnalysis {
    /// Undertone classification
    pub undertone: Undertone,
    /// Undertone confidence (0-1)
    pub undertone_confidence: f32,
    /// Depth classification
    pub depth: SkinDepth,
    /// Notes
    pub notes: Vec<SkinNote>,
}

/// Analyze skin tone from ROI colors
pub fn analyze_skin(colors: &RoiColors, quality: &BeautyQuality) -> Option<SkinAnalysis> {
    if !quality.lighting_ok {
        return None;
    }
    
    // Average the cheek colors (most reliable for skin tone)
    let avg_rgb = [
        (colors.left_cheek[0] + colors.right_cheek[0]) / 2.0,
        (colors.left_cheek[1] + colors.right_cheek[1]) / 2.0,
        (colors.left_cheek[2] + colors.right_cheek[2]) / 2.0,
    ];
    
    // Convert to Lab-like values for analysis
    let (l, a, b) = rgb_to_lab_approx(avg_rgb);
    
    // Determine undertone from a* and b*
    let (undertone, ut_conf) = classify_undertone(a, b);
    
    // Determine depth from L*
    let depth = classify_depth(l);
    
    // Check for redness
    let mut notes = Vec::new();
    if a > 15.0 {
        notes.push(SkinNote::Redness);
    }
    
    // Check evenness (variance across regions)
    let variance = compute_color_variance(colors);
    if variance > 20.0 {
        notes.push(SkinNote::Uneven);
    } else {
        notes.push(SkinNote::GoodCondition);
    }
    
    Some(SkinAnalysis {
        undertone,
        undertone_confidence: ut_conf,
        depth,
        notes,
    })
}

/// Approximate RGB to Lab conversion
fn rgb_to_lab_approx(rgb: [f32; 3]) -> (f32, f32, f32) {
    // Normalize to 0-1
    let r = rgb[0] / 255.0;
    let g = rgb[1] / 255.0;
    let b = rgb[2] / 255.0;
    
    // Approximate L* from luminance
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let l = if y > 0.008856 {
        116.0 * y.powf(1.0/3.0) - 16.0
    } else {
        903.3 * y
    };
    
    // Approximate a* (red-green) and b* (yellow-blue)
    let a = (r - g) * 100.0;
    let b_val = (g - b) * 100.0;
    
    (l, a, b_val)
}

/// Classify undertone from Lab a*, b* values
fn classify_undertone(a: f32, b: f32) -> (Undertone, f32) {
    // Warm: high b* (yellow)
    // Cool: low b* (blue), can have high a* (pink)
    // Neutral: balanced
    
    let warm_score = b / 30.0; // Positive b = warm
    let cool_score = -b / 30.0 + a / 20.0; // Negative b or high a = cool
    
    if warm_score > 0.3 && warm_score > cool_score {
        (Undertone::Warm, (warm_score * 2.0).clamp(0.0, 1.0))
    } else if cool_score > 0.3 && cool_score > warm_score {
        (Undertone::Cool, (cool_score * 2.0).clamp(0.0, 1.0))
    } else {
        let conf = 1.0 - (warm_score - cool_score).abs();
        (Undertone::Neutral, conf.clamp(0.3, 0.8))
    }
}

/// Classify depth from L* value
fn classify_depth(l: f32) -> SkinDepth {
    if l > 80.0 {
        SkinDepth::Fair
    } else if l > 65.0 {
        SkinDepth::Light
    } else if l > 50.0 {
        SkinDepth::Medium
    } else if l > 35.0 {
        SkinDepth::Tan
    } else {
        SkinDepth::Deep
    }
}

/// Compute color variance across ROI regions
fn compute_color_variance(colors: &RoiColors) -> f32 {
    let regions = [
        colors.forehead,
        colors.left_cheek,
        colors.right_cheek,
        colors.chin,
    ];
    
    // Compute mean
    let mut mean = [0.0f32; 3];
    for rgb in &regions {
        mean[0] += rgb[0];
        mean[1] += rgb[1];
        mean[2] += rgb[2];
    }
    mean[0] /= 4.0;
    mean[1] /= 4.0;
    mean[2] /= 4.0;
    
    // Compute variance
    let mut var = 0.0f32;
    for rgb in &regions {
        let dr = rgb[0] - mean[0];
        let dg = rgb[1] - mean[1];
        let db = rgb[2] - mean[2];
        var += dr*dr + dg*dg + db*db;
    }
    
    (var / 4.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_depth_classification() {
        assert_eq!(classify_depth(85.0), SkinDepth::Fair);
        assert_eq!(classify_depth(70.0), SkinDepth::Light);
        assert_eq!(classify_depth(55.0), SkinDepth::Medium);
    }
    
    #[test]
    fn test_lighting_gating() {
        let colors = RoiColors::default();
        let bad_quality = BeautyQuality {
            lighting_ok: false,
            ..Default::default()
        };
        
        assert!(analyze_skin(&colors, &bad_quality).is_none());
    }
}
