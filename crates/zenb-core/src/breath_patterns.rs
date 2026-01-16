//! Breathing Patterns Registry
//!
//! Defines breathing patterns for guided meditation and breath work.
//! Ported from ZenOne TypeScript implementation.
//!
//! # Patterns Available
//! - **4-7-8**: Tranquility (Sleep & Anxiety)
//! - **box**: Focus (Navy SEALs technique)
//! - **calm**: Balance (Coherence)
//! - **coherence**: Heart Health (HRV optimization)
//! - **deep-relax**: Stress Relief
//! - **7-11**: Deep Calm (Panic attacks)
//! - **awake**: Energize (Wake up)
//! - **triangle**: Yoga (Emotional stability)
//! - **tactical**: Advanced Focus
//! - **buteyko**: Light Air (Health)
//! - **wim-hof**: Tummo Power (Immunity)

use crate::phase_machine::PhaseDurations;
use std::collections::HashMap;

/// Color theme for UI visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ColorTheme {
    Warm,
    Cool,
    Neutral,
}

/// Pattern tier (difficulty/experience level)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatternTier {
    Beginner = 1,
    Intermediate = 2,
    Advanced = 3,
}

/// Breathing pattern definition
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BreathPattern {
    /// Unique pattern identifier
    pub id: String,
    /// Display label
    pub label: String,
    /// Category tag
    pub tag: String,
    /// Description of the pattern
    pub description: String,
    /// Phase timings in seconds
    pub timings: PatternTimings,
    /// Color theme for UI
    pub color_theme: ColorTheme,
    /// Recommended number of cycles
    pub recommended_cycles: u32,
    /// Difficulty tier
    pub tier: PatternTier,
    /// Arousal impact: -1.0 (sedative) to 1.0 (stimulant)
    pub arousal_impact: f32,
}

/// Phase timings in seconds
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PatternTimings {
    pub inhale: f32,
    pub hold_in: f32,
    pub exhale: f32,
    pub hold_out: f32,
}

impl PatternTimings {
    /// Convert to PhaseDurations (microseconds)
    pub fn to_phase_durations(&self) -> PhaseDurations {
        PhaseDurations {
            inhale_us: (self.inhale * 1_000_000.0) as u64,
            hold_in_us: (self.hold_in * 1_000_000.0) as u64,
            exhale_us: (self.exhale * 1_000_000.0) as u64,
            hold_out_us: (self.hold_out * 1_000_000.0) as u64,
        }
    }
    
    /// Total cycle duration in seconds
    pub fn total_seconds(&self) -> f32 {
        self.inhale + self.hold_in + self.exhale + self.hold_out
    }
}

impl BreathPattern {
    /// Get PhaseDurations for this pattern
    pub fn to_phase_durations(&self) -> PhaseDurations {
        self.timings.to_phase_durations()
    }
    
    /// Calculate breaths per minute for this pattern
    pub fn breaths_per_minute(&self) -> f32 {
        60.0 / self.timings.total_seconds()
    }
}

/// Get all built-in breathing patterns
pub fn builtin_patterns() -> HashMap<String, BreathPattern> {
    let mut patterns = HashMap::new();
    
    // 4-7-8: Tranquility (Andrew Weil technique)
    patterns.insert("4-7-8".to_string(), BreathPattern {
        id: "4-7-8".to_string(),
        label: "Tranquility".to_string(),
        tag: "Sleep & Anxiety".to_string(),
        description: "A natural tranquilizer for the nervous system.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 7.0, exhale: 8.0, hold_out: 0.0 },
        color_theme: ColorTheme::Warm,
        recommended_cycles: 4,
        tier: PatternTier::Beginner,
        arousal_impact: -0.8,
    });
    
    // Box Breathing (Navy SEALs)
    patterns.insert("box".to_string(), BreathPattern {
        id: "box".to_string(),
        label: "Focus".to_string(),
        tag: "Concentration".to_string(),
        description: "Used by Navy SEALs to heighten performance.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 4.0, exhale: 4.0, hold_out: 4.0 },
        color_theme: ColorTheme::Neutral,
        recommended_cycles: 6,
        tier: PatternTier::Beginner,
        arousal_impact: 0.0,
    });
    
    // Calm (Simple coherence)
    patterns.insert("calm".to_string(), BreathPattern {
        id: "calm".to_string(),
        label: "Balance".to_string(),
        tag: "Coherence".to_string(),
        description: "Restores balance to your heart rate variability.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 0.0, exhale: 6.0, hold_out: 0.0 },
        color_theme: ColorTheme::Cool,
        recommended_cycles: 8,
        tier: PatternTier::Beginner,
        arousal_impact: -0.3,
    });
    
    // Coherence (HeartMath)
    patterns.insert("coherence".to_string(), BreathPattern {
        id: "coherence".to_string(),
        label: "Coherence".to_string(),
        tag: "Heart Health".to_string(),
        description: "Optimizes Heart Rate Variability (HRV). The 'Golden Ratio' of breathing.".to_string(),
        timings: PatternTimings { inhale: 6.0, hold_in: 0.0, exhale: 6.0, hold_out: 0.0 },
        color_theme: ColorTheme::Cool,
        recommended_cycles: 10,
        tier: PatternTier::Intermediate,
        arousal_impact: -0.5,
    });
    
    // Deep Relax (Extended exhale)
    patterns.insert("deep-relax".to_string(), BreathPattern {
        id: "deep-relax".to_string(),
        label: "Deep Rest".to_string(),
        tag: "Stress Relief".to_string(),
        description: "Doubling the exhalation to trigger the parasympathetic system.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 0.0, exhale: 8.0, hold_out: 0.0 },
        color_theme: ColorTheme::Warm,
        recommended_cycles: 6,
        tier: PatternTier::Beginner,
        arousal_impact: -0.9,
    });
    
    // 7-11 (Panic attack relief)
    patterns.insert("7-11".to_string(), BreathPattern {
        id: "7-11".to_string(),
        label: "7-11".to_string(),
        tag: "Deep Calm".to_string(),
        description: "A powerful technique for panic attacks and deep anxiety.".to_string(),
        timings: PatternTimings { inhale: 7.0, hold_in: 0.0, exhale: 11.0, hold_out: 0.0 },
        color_theme: ColorTheme::Warm,
        recommended_cycles: 4,
        tier: PatternTier::Intermediate,
        arousal_impact: -1.0,
    });
    
    // Awake (Energizing)
    patterns.insert("awake".to_string(), BreathPattern {
        id: "awake".to_string(),
        label: "Energize".to_string(),
        tag: "Wake Up".to_string(),
        description: "Fast-paced rhythm to boost alertness and energy levels.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 0.0, exhale: 2.0, hold_out: 0.0 },
        color_theme: ColorTheme::Cool,
        recommended_cycles: 15,
        tier: PatternTier::Intermediate,
        arousal_impact: 0.8,
    });
    
    // Triangle (Yoga)
    patterns.insert("triangle".to_string(), BreathPattern {
        id: "triangle".to_string(),
        label: "Triangle".to_string(),
        tag: "Yoga".to_string(),
        description: "A geometric pattern for emotional stability and control.".to_string(),
        timings: PatternTimings { inhale: 4.0, hold_in: 4.0, exhale: 4.0, hold_out: 0.0 },
        color_theme: ColorTheme::Neutral,
        recommended_cycles: 8,
        tier: PatternTier::Beginner,
        arousal_impact: 0.2,
    });
    
    // Tactical (Extended box)
    patterns.insert("tactical".to_string(), BreathPattern {
        id: "tactical".to_string(),
        label: "Tactical".to_string(),
        tag: "Advanced Focus".to_string(),
        description: "Extended Box Breathing for high-stress situations.".to_string(),
        timings: PatternTimings { inhale: 5.0, hold_in: 5.0, exhale: 5.0, hold_out: 5.0 },
        color_theme: ColorTheme::Neutral,
        recommended_cycles: 5,
        tier: PatternTier::Intermediate,
        arousal_impact: 0.1,
    });
    
    // Buteyko (Reduced breathing)
    patterns.insert("buteyko".to_string(), BreathPattern {
        id: "buteyko".to_string(),
        label: "Light Air".to_string(),
        tag: "Health".to_string(),
        description: "Reduced breathing to improve oxygen uptake (Buteyko Method).".to_string(),
        timings: PatternTimings { inhale: 3.0, hold_in: 0.0, exhale: 3.0, hold_out: 4.0 },
        color_theme: ColorTheme::Cool,
        recommended_cycles: 12,
        tier: PatternTier::Advanced,
        arousal_impact: -0.2,
    });
    
    // Wim Hof (Tummo)
    patterns.insert("wim-hof".to_string(), BreathPattern {
        id: "wim-hof".to_string(),
        label: "Tummo Power".to_string(),
        tag: "Immunity".to_string(),
        description: "Charge the body. Inhale deeply, let go. Repeat.".to_string(),
        timings: PatternTimings { inhale: 2.0, hold_in: 0.0, exhale: 1.0, hold_out: 15.0 },
        color_theme: ColorTheme::Warm,
        recommended_cycles: 30,
        tier: PatternTier::Advanced,
        arousal_impact: 1.0,
    });
    
    patterns
}

/// Get a pattern by ID
pub fn get_pattern(id: &str) -> Option<BreathPattern> {
    builtin_patterns().remove(id)
}

/// Get patterns by tier
pub fn patterns_by_tier(tier: PatternTier) -> Vec<BreathPattern> {
    builtin_patterns()
        .into_values()
        .filter(|p| p.tier == tier)
        .collect()
}

/// Get patterns by arousal impact range
pub fn patterns_by_arousal(min: f32, max: f32) -> Vec<BreathPattern> {
    builtin_patterns()
        .into_values()
        .filter(|p| p.arousal_impact >= min && p.arousal_impact <= max)
        .collect()
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_builtin_patterns_count() {
        let patterns = builtin_patterns();
        assert_eq!(patterns.len(), 11, "Should have 11 built-in patterns");
    }
    
    #[test]
    fn test_478_pattern() {
        let pattern = get_pattern("4-7-8").expect("4-7-8 should exist");
        assert_eq!(pattern.label, "Tranquility");
        assert_eq!(pattern.timings.inhale, 4.0);
        assert_eq!(pattern.timings.hold_in, 7.0);
        assert_eq!(pattern.timings.exhale, 8.0);
        assert!(pattern.arousal_impact < 0.0, "4-7-8 should be sedative");
    }
    
    #[test]
    fn test_box_pattern() {
        let pattern = get_pattern("box").expect("box should exist");
        assert_eq!(pattern.timings.inhale, 4.0);
        assert_eq!(pattern.timings.hold_in, 4.0);
        assert_eq!(pattern.timings.exhale, 4.0);
        assert_eq!(pattern.timings.hold_out, 4.0);
        assert!((pattern.arousal_impact - 0.0).abs() < 0.01, "Box should be neutral");
    }
    
    #[test]
    fn test_pattern_to_phase_durations() {
        let pattern = get_pattern("box").unwrap();
        let durations = pattern.to_phase_durations();
        
        assert_eq!(durations.inhale_us, 4_000_000);
        assert_eq!(durations.hold_in_us, 4_000_000);
        assert_eq!(durations.exhale_us, 4_000_000);
        assert_eq!(durations.hold_out_us, 4_000_000);
    }
    
    #[test]
    fn test_breaths_per_minute() {
        let pattern = get_pattern("box").unwrap();
        let bpm = pattern.breaths_per_minute();
        // Box: 16 seconds per cycle = 3.75 bpm
        assert!((bpm - 3.75).abs() < 0.01);
    }
    
    #[test]
    fn test_patterns_by_tier() {
        let beginners = patterns_by_tier(PatternTier::Beginner);
        assert!(beginners.len() >= 4, "Should have at least 4 beginner patterns");
        
        let advanced = patterns_by_tier(PatternTier::Advanced);
        assert!(advanced.len() >= 2, "Should have at least 2 advanced patterns");
    }
    
    #[test]
    fn test_patterns_by_arousal() {
        let sedatives = patterns_by_arousal(-1.0, -0.5);
        assert!(!sedatives.is_empty(), "Should have sedative patterns");
        
        let stimulants = patterns_by_arousal(0.5, 1.0);
        assert!(!stimulants.is_empty(), "Should have stimulant patterns");
    }
}
