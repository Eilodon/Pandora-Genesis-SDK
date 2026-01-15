//! Makeup Recommendation Engine
//!
//! Generates contour, highlight, and blush recommendations.

use super::face_shape::FaceShape;
use super::measurements::FaceMeasurements;
use super::landmarks::{CanonicalLandmarks, get_point, indices};

/// Makeup zone type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneType {
    Contour,
    Highlight,
    Blush,
    Bronzer,
    BrowShape,
    EyelinerGuide,
}

/// Shade hint relative to skin tone
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadeHint {
    Darker,
    Lighter,
    Warmer,
    Cooler,
    Neutral,
}

/// Makeup style preference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MakeupStyle {
    #[default]
    Natural,
    Glam,
    Office,
}

/// Single makeup zone
#[derive(Debug, Clone)]
pub struct MakeupZone {
    pub zone_type: ZoneType,
    pub name: String,
    pub polygon: Vec<[f32; 2]>,
    pub intensity: f32,
    pub shade: ShadeHint,
    pub note: String,
}

/// Eyebrow recommendation
#[derive(Debug, Clone, Default)]
pub struct BrowRecommendation {
    pub arch_position: f32,
    pub thickness: f32,
    pub tail_tip: String,
}

/// Eye makeup recommendation
#[derive(Debug, Clone, Default)]
pub struct EyeRecommendation {
    pub liner_angle: f32,
    pub tip: String,
}

/// Lip recommendation
#[derive(Debug, Clone, Default)]
pub struct LipRecommendation {
    pub overline: bool,
    pub tip: String,
}

/// Complete makeup plan
#[derive(Debug, Clone, Default)]
pub struct MakeupPlan {
    pub zones: Vec<MakeupZone>,
    pub brow: BrowRecommendation,
    pub eye: EyeRecommendation,
    pub lip: LipRecommendation,
    pub tips: Vec<String>,
}

/// Generate makeup plan
pub fn generate_makeup_plan(
    shape: FaceShape,
    measurements: &FaceMeasurements,
    landmarks: &CanonicalLandmarks,
    style: MakeupStyle,
) -> MakeupPlan {
    let intensity = match style {
        MakeupStyle::Natural => 0.5,
        MakeupStyle::Office => 0.7,
        MakeupStyle::Glam => 1.0,
    };
    
    let mut plan = MakeupPlan::default();
    
    if landmarks.valid {
        plan.zones = generate_zones(shape, landmarks, intensity);
    }
    
    plan.brow = generate_brow_rec(shape, measurements);
    plan.eye = generate_eye_rec(measurements);
    plan.lip = generate_lip_rec(shape);
    plan.tips = generate_tips(shape);
    
    plan
}

fn generate_zones(shape: FaceShape, lm: &CanonicalLandmarks, intensity: f32) -> Vec<MakeupZone> {
    let mut zones = Vec::new();
    
    // Cheek hollow contour (universal)
    for side in ["left", "right"] {
        let cheekbone = if side == "left" {
            get_point(lm, indices::LEFT_CHEEKBONE)
        } else {
            get_point(lm, indices::RIGHT_CHEEKBONE)
        };
        
        zones.push(MakeupZone {
            zone_type: ZoneType::Contour,
            name: format!("{}_cheek", side),
            polygon: circle(cheekbone, 0.1),
            intensity: intensity * match shape {
                FaceShape::Round => 1.0,
                FaceShape::Square => 0.8,
                _ => 0.7,
            },
            shade: ShadeHint::Darker,
            note: "Cheek hollow".to_string(),
        });
    }
    
    // Blush zones
    for side in ["left", "right"] {
        let cheek = if side == "left" {
            get_point(lm, indices::LEFT_CHEEKBONE)
        } else {
            get_point(lm, indices::RIGHT_CHEEKBONE)
        };
        
        zones.push(MakeupZone {
            zone_type: ZoneType::Blush,
            name: format!("{}_blush", side),
            polygon: circle(cheek, 0.12),
            intensity,
            shade: ShadeHint::Neutral,
            note: "Apple of cheek".to_string(),
        });
    }
    
    // Highlight
    let nose = get_point(lm, indices::NOSE_BRIDGE);
    zones.push(MakeupZone {
        zone_type: ZoneType::Highlight,
        name: "nose_bridge".to_string(),
        polygon: circle(nose, 0.05),
        intensity,
        shade: ShadeHint::Lighter,
        note: "Center of nose".to_string(),
    });
    
    zones
}

fn circle(center: [f32; 2], r: f32) -> Vec<[f32; 2]> {
    (0..8).map(|i| {
        let a = std::f32::consts::PI * 2.0 * i as f32 / 8.0;
        [center[0] + r * a.cos(), center[1] + r * a.sin()]
    }).collect()
}

fn generate_brow_rec(shape: FaceShape, m: &FaceMeasurements) -> BrowRecommendation {
    let (arch, tip) = match shape {
        FaceShape::Round => (0.7, "High arch"),
        FaceShape::Square => (0.67, "Soft arch"),
        FaceShape::Heart => (0.6, "Rounded"),
        _ => (0.67, "Natural"),
    };
    
    BrowRecommendation {
        arch_position: arch,
        thickness: m.brow_arch_height,
        tail_tip: tip.to_string(),
    }
}

fn generate_eye_rec(m: &FaceMeasurements) -> EyeRecommendation {
    let angle = if m.canthal_tilt > 5.0 { 25.0 }
                else if m.canthal_tilt < -5.0 { 45.0 }
                else { 35.0 };
    
    EyeRecommendation {
        liner_angle: angle,
        tip: "Follow eye shape".to_string(),
    }
}

fn generate_lip_rec(shape: FaceShape) -> LipRecommendation {
    LipRecommendation {
        overline: matches!(shape, FaceShape::Heart | FaceShape::Oblong),
        tip: "Natural line".to_string(),
    }
}

fn generate_tips(shape: FaceShape) -> Vec<String> {
    match shape {
        FaceShape::Round => vec!["Add angles".to_string()],
        FaceShape::Square => vec!["Soften jaw".to_string()],
        FaceShape::Heart => vec!["Balance forehead".to_string()],
        _ => vec!["Enhance features".to_string()],
    }
}
