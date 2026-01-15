//! Fatigue and Stress Fusion Layer
//!
//! Combines multiple signals (HRV, RR, attention) into composite
//! fatigue and stress indicators. Uses per-user baseline for personalization.

use super::attention::AttentionMetrics;

/// Fatigue/Stress fusion configuration
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Weight for HRV contribution to stress (0-1)
    pub hrv_weight: f32,
    /// Weight for RR contribution
    pub rr_weight: f32,
    /// Weight for attention metrics
    pub attention_weight: f32,
    /// Weight for drowsiness (for fatigue)
    pub drowsiness_weight: f32,
    /// Baseline window in samples
    pub baseline_window: usize,
    /// EMA alpha for smoothing
    pub smoothing_alpha: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            hrv_weight: 0.4,
            rr_weight: 0.2,
            attention_weight: 0.3,
            drowsiness_weight: 0.1,
            baseline_window: 100,
            smoothing_alpha: 0.2,
        }
    }
}

/// Per-user baseline for fatigue/stress
#[derive(Debug, Clone)]
pub struct FusionBaseline {
    /// Baseline resting HRV RMSSD
    pub hrv_rmssd_baseline: f32,
    /// Baseline resting HR
    pub hr_baseline: f32,
    /// Baseline RR
    pub rr_baseline: f32,
    /// Number of samples used
    pub sample_count: usize,
}

impl Default for FusionBaseline {
    fn default() -> Self {
        Self {
            hrv_rmssd_baseline: 40.0, // Population average
            hr_baseline: 70.0,
            rr_baseline: 14.0,
            sample_count: 0,
        }
    }
}

/// Fatigue level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatigueLevel {
    Alert,
    Mild,
    Moderate,
    Severe,
}

/// Stress level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressLevel {
    Relaxed,
    Low,
    Moderate,
    High,
    Acute,
}

/// Combined fatigue and stress result
#[derive(Debug, Clone)]
pub struct FatigueStressResult {
    /// Fatigue score (0 = alert, 1 = exhausted)
    pub fatigue_score: f32,
    /// Fatigue level classification
    pub fatigue_level: FatigueLevel,
    /// Stress score (0 = relaxed, 1 = acute stress)
    pub stress_score: f32,
    /// Stress level classification
    pub stress_level: StressLevel,
    /// Recovery readiness (0-1, ability to handle load)
    pub recovery_readiness: f32,
    /// Cognitive load estimate (0-1)
    pub cognitive_load: f32,
    /// Confidence in the assessment
    pub confidence: f32,
    /// Recommendations
    pub recommendations: Vec<Recommendation>,
}

/// Actionable recommendation
#[derive(Debug, Clone)]
pub struct Recommendation {
    pub category: RecommendationCategory,
    pub message: String,
    pub priority: u8, // 1 = highest
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationCategory {
    BreathingExercise,
    TakeBreak,
    Hydration,
    Movement,
    Sleep,
    Relaxation,
}

/// Input signals for fusion
#[derive(Debug, Clone, Default)]
pub struct FusionInput {
    /// Current HRV RMSSD (ms)
    pub hrv_rmssd: Option<f32>,
    /// HRV confidence
    pub hrv_confidence: f32,
    /// Current HR (bpm)
    pub hr_bpm: Option<f32>,
    /// Current RR (breaths/min)
    pub rr_bpm: Option<f32>,
    /// Attention metrics
    pub attention: Option<AttentionMetrics>,
    /// Motion level (0-1)
    pub motion: f32,
}

/// Fatigue/Stress Fusion Engine
#[derive(Debug, Clone)]
pub struct FatigueStressFusion {
    config: FusionConfig,
    baseline: FusionBaseline,
    /// History for baseline updates
    hrv_history: Vec<f32>,
    hr_history: Vec<f32>,
    rr_history: Vec<f32>,
    /// Smoothed scores
    fatigue_ema: f32,
    stress_ema: f32,
}

impl FatigueStressFusion {
    pub fn new() -> Self {
        Self::with_config(FusionConfig::default())
    }
    
    pub fn with_config(config: FusionConfig) -> Self {
        Self {
            config,
            baseline: FusionBaseline::default(),
            hrv_history: Vec::new(),
            hr_history: Vec::new(),
            rr_history: Vec::new(),
            fatigue_ema: 0.3,
            stress_ema: 0.3,
        }
    }
    
    /// Load with pre-existing baseline
    pub fn with_baseline(baseline: FusionBaseline) -> Self {
        let mut fusion = Self::new();
        fusion.baseline = baseline;
        fusion
    }
    
    /// Process input signals and produce fatigue/stress assessment
    pub fn process(&mut self, input: &FusionInput) -> FatigueStressResult {
        // Update baseline if good quality data
        self.update_baseline(input);
        
        // Compute component scores
        let hrv_stress = self.compute_hrv_stress(input);
        let rr_stress = self.compute_rr_stress(input);
        let attention_fatigue = self.compute_attention_fatigue(input);
        let drowsiness = input.attention.as_ref()
            .map(|a| a.drowsiness)
            .unwrap_or(0.0);
        
        // --- Stress Score ---
        let cfg = &self.config;
        let stress_raw = hrv_stress * cfg.hrv_weight
            + rr_stress * cfg.rr_weight
            + (1.0 - input.attention.as_ref().map(|a| a.attention_score).unwrap_or(0.5)) * cfg.attention_weight;
        
        // Motion increases perceived stress
        let motion_boost = input.motion * 0.2;
        let stress_raw = (stress_raw + motion_boost).clamp(0.0, 1.0);
        
        // Smooth
        self.stress_ema = self.stress_ema * (1.0 - cfg.smoothing_alpha) 
            + stress_raw * cfg.smoothing_alpha;
        let stress_score = self.stress_ema;
        
        // --- Fatigue Score ---
        let fatigue_raw = attention_fatigue * cfg.attention_weight
            + drowsiness * cfg.drowsiness_weight
            + hrv_stress * 0.3 // Low HRV also indicates fatigue
            + self.compute_hr_fatigue(input) * 0.2;
        
        let fatigue_raw = fatigue_raw.clamp(0.0, 1.0);
        
        // Smooth
        self.fatigue_ema = self.fatigue_ema * (1.0 - cfg.smoothing_alpha) 
            + fatigue_raw * cfg.smoothing_alpha;
        let fatigue_score = self.fatigue_ema;
        
        // Classifications
        let fatigue_level = self.classify_fatigue(fatigue_score);
        let stress_level = self.classify_stress(stress_score);
        
        // Recovery readiness (inverse of fatigue + stress)
        let recovery_readiness = (1.0 - (fatigue_score + stress_score) / 2.0).clamp(0.0, 1.0);
        
        // Cognitive load (attention effort under stress)
        let cognitive_load = if let Some(att) = &input.attention {
            let effort = (1.0 - att.attention_score) * stress_score;
            effort.clamp(0.0, 1.0)
        } else {
            stress_score * 0.5
        };
        
        // Confidence based on data availability
        let confidence = self.compute_confidence(input);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            fatigue_level, stress_level, input
        );
        
        FatigueStressResult {
            fatigue_score,
            fatigue_level,
            stress_score,
            stress_level,
            recovery_readiness,
            cognitive_load,
            confidence,
            recommendations,
        }
    }
    
    /// Export baseline for persistence
    pub fn export_baseline(&self) -> FusionBaseline {
        self.baseline.clone()
    }
    
    /// Reset to default state
    pub fn reset(&mut self) {
        self.baseline = FusionBaseline::default();
        self.hrv_history.clear();
        self.hr_history.clear();
        self.rr_history.clear();
        self.fatigue_ema = 0.3;
        self.stress_ema = 0.3;
    }
    
    // --- Private ---
    
    fn update_baseline(&mut self, input: &FusionInput) {
        // Only update baseline when motion is low (resting state)
        if input.motion > 0.3 {
            return;
        }
        
        // Update histories
        if let Some(hrv) = input.hrv_rmssd {
            if input.hrv_confidence > 0.5 {
                self.hrv_history.push(hrv);
                if self.hrv_history.len() > self.config.baseline_window {
                    self.hrv_history.remove(0);
                }
            }
        }
        
        if let Some(hr) = input.hr_bpm {
            self.hr_history.push(hr);
            if self.hr_history.len() > self.config.baseline_window {
                self.hr_history.remove(0);
            }
        }
        
        if let Some(rr) = input.rr_bpm {
            self.rr_history.push(rr);
            if self.rr_history.len() > self.config.baseline_window {
                self.rr_history.remove(0);
            }
        }
        
        // Update baseline if enough samples
        if self.hrv_history.len() >= 10 {
            let mean = self.hrv_history.iter().sum::<f32>() / self.hrv_history.len() as f32;
            self.baseline.hrv_rmssd_baseline = self.baseline.hrv_rmssd_baseline * 0.9 + mean * 0.1;
        }
        
        if self.hr_history.len() >= 10 {
            let mean = self.hr_history.iter().sum::<f32>() / self.hr_history.len() as f32;
            self.baseline.hr_baseline = self.baseline.hr_baseline * 0.9 + mean * 0.1;
        }
        
        if self.rr_history.len() >= 10 {
            let mean = self.rr_history.iter().sum::<f32>() / self.rr_history.len() as f32;
            self.baseline.rr_baseline = self.baseline.rr_baseline * 0.9 + mean * 0.1;
        }
        
        self.baseline.sample_count += 1;
    }
    
    fn compute_hrv_stress(&self, input: &FusionInput) -> f32 {
        // Low HRV = high stress
        let Some(hrv) = input.hrv_rmssd else {
            return 0.5; // Unknown
        };
        
        let ratio = hrv / self.baseline.hrv_rmssd_baseline.max(1.0);
        // ratio < 1 = below baseline = stress
        let stress = (1.0 - ratio).clamp(-0.5, 1.0);
        (stress * 0.5 + 0.5).clamp(0.0, 1.0) // Map to 0-1
    }
    
    fn compute_rr_stress(&self, input: &FusionInput) -> f32 {
        // High RR = stress, very low RR might indicate relaxation or drowsiness
        let Some(rr) = input.rr_bpm else {
            return 0.5;
        };
        
        let deviation = (rr - self.baseline.rr_baseline).abs();
        let normalized = (deviation / 6.0).clamp(0.0, 1.0); // 6 breaths deviation = high
        normalized * 0.7 // RR is less reliable indicator
    }
    
    fn compute_hr_fatigue(&self, input: &FusionInput) -> f32 {
        // Elevated resting HR can indicate fatigue
        let Some(hr) = input.hr_bpm else {
            return 0.0;
        };
        
        let elevation = (hr - self.baseline.hr_baseline) / 20.0;
        elevation.clamp(0.0, 1.0)
    }
    
    fn compute_attention_fatigue(&self, input: &FusionInput) -> f32 {
        let Some(att) = &input.attention else {
            return 0.0;
        };
        
        // Combine low attention + high blink rate + drowsiness
        let attention_drop = 1.0 - att.attention_score;
        let blink_factor = if att.blink_state.blink_rate > 20.0 {
            (att.blink_state.blink_rate - 20.0) / 20.0
        } else {
            0.0
        };
        
        (attention_drop * 0.5 + att.drowsiness * 0.3 + blink_factor * 0.2).clamp(0.0, 1.0)
    }
    
    fn classify_fatigue(&self, score: f32) -> FatigueLevel {
        if score < 0.25 {
            FatigueLevel::Alert
        } else if score < 0.5 {
            FatigueLevel::Mild
        } else if score < 0.75 {
            FatigueLevel::Moderate
        } else {
            FatigueLevel::Severe
        }
    }
    
    fn classify_stress(&self, score: f32) -> StressLevel {
        if score < 0.2 {
            StressLevel::Relaxed
        } else if score < 0.4 {
            StressLevel::Low
        } else if score < 0.6 {
            StressLevel::Moderate
        } else if score < 0.8 {
            StressLevel::High
        } else {
            StressLevel::Acute
        }
    }
    
    fn compute_confidence(&self, input: &FusionInput) -> f32 {
        let mut conf = 0.0;
        let mut weight = 0.0;
        
        if input.hrv_rmssd.is_some() {
            conf += input.hrv_confidence * 0.4;
            weight += 0.4;
        }
        if input.hr_bpm.is_some() {
            conf += 0.3;
            weight += 0.3;
        }
        if input.attention.is_some() {
            conf += 0.3;
            weight += 0.3;
        }
        
        if weight > 0.0 {
            conf / weight
        } else {
            0.0
        }
    }
    
    fn generate_recommendations(
        &self,
        fatigue: FatigueLevel,
        stress: StressLevel,
        _input: &FusionInput,
    ) -> Vec<Recommendation> {
        let mut recs = Vec::new();
        
        // Stress recommendations
        if matches!(stress, StressLevel::High | StressLevel::Acute) {
            recs.push(Recommendation {
                category: RecommendationCategory::BreathingExercise,
                message: "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s".into(),
                priority: 1,
            });
        }
        
        if matches!(stress, StressLevel::Moderate | StressLevel::High | StressLevel::Acute) {
            recs.push(Recommendation {
                category: RecommendationCategory::TakeBreak,
                message: "Take a 5-minute break away from screens".into(),
                priority: 2,
            });
        }
        
        // Fatigue recommendations
        if matches!(fatigue, FatigueLevel::Severe) {
            recs.push(Recommendation {
                category: RecommendationCategory::Sleep,
                message: "Consider a 20-minute power nap or rest your eyes".into(),
                priority: 1,
            });
        }
        
        if matches!(fatigue, FatigueLevel::Moderate | FatigueLevel::Severe) {
            recs.push(Recommendation {
                category: RecommendationCategory::Movement,
                message: "Stand up and stretch, or take a short walk".into(),
                priority: 2,
            });
            recs.push(Recommendation {
                category: RecommendationCategory::Hydration,
                message: "Drink a glass of water".into(),
                priority: 3,
            });
        }
        
        recs
    }
}

impl Default for FatigueStressFusion {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_baseline() {
        let baseline = FusionBaseline::default();
        assert!(baseline.hrv_rmssd_baseline > 0.0);
    }
    
    #[test]
    fn test_stress_classification() {
        let fusion = FatigueStressFusion::new();
        
        assert_eq!(fusion.classify_stress(0.1), StressLevel::Relaxed);
        assert_eq!(fusion.classify_stress(0.5), StressLevel::Moderate);
        assert_eq!(fusion.classify_stress(0.9), StressLevel::Acute);
    }
    
    #[test]
    fn test_low_hrv_increases_stress() {
        let mut fusion = FatigueStressFusion::new();
        fusion.baseline.hrv_rmssd_baseline = 40.0;
        
        // Low HRV input
        let input = FusionInput {
            hrv_rmssd: Some(20.0), // Half of baseline
            hrv_confidence: 0.8,
            hr_bpm: Some(70.0),
            ..Default::default()
        };
        
        // Process multiple times for EMA convergence
        for _ in 0..5 {
            fusion.process(&input);
        }
        
        let result = fusion.process(&input);
        assert!(result.stress_score > 0.4); // Should indicate stress
    }

    
    #[test]
    fn test_recommendations_generated() {
        let mut fusion = FatigueStressFusion::new();
        
        let input = FusionInput {
            hrv_rmssd: Some(15.0), // Very low HRV
            hrv_confidence: 0.8,
            hr_bpm: Some(90.0), // Elevated HR
            ..Default::default()
        };
        
        // Process multiple times to increase stress
        for _ in 0..10 {
            fusion.process(&input);
        }
        
        let result = fusion.process(&input);
        assert!(!result.recommendations.is_empty());
    }
}
