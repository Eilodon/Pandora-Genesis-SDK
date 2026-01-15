//! Fraud Detection for Financial Services
//!
//! Passive fraud detection using:
//! - Cardiac fingerprinting (unique pulse patterns)
//! - Stress baseline comparison
//! - Coercion detection
//! - Behavioral biometrics

use ndarray::Array1;
use zenb_signals::physio::HrvEstimator;
use zenb_signals::rppg::EnsembleProcessor;

/// Fraud detection result
#[derive(Debug, Clone)]
pub struct FraudResult {
    /// Risk score (0-1, higher = more suspicious)
    pub risk_score: f32,
    /// Risk level classification
    pub risk_level: RiskLevel,
    /// Identity match confidence (if enrolled)
    pub identity_confidence: Option<f32>,
    /// Stress anomaly detected
    pub stress_anomaly: bool,
    /// Coercion indicators
    pub coercion_indicators: CoercionIndicators,
    /// Recommended action
    pub recommended_action: FraudAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FraudAction {
    Approve,
    AdditionalVerification,
    ManualReview,
    Block,
}

#[derive(Debug, Clone, Default)]
pub struct CoercionIndicators {
    pub elevated_stress: bool,
    pub abnormal_behavior: bool,
    pub rushed_interaction: bool,
    pub unusual_time: bool,
}

/// Fraud Detector
pub struct FraudDetector {
    user_profile: Option<UserProfile>,

    ppg_processor: EnsembleProcessor,
    hrv_estimator: HrvEstimator,

    rgb_buffer: Vec<[f32; 3]>,
    stress_baseline: Option<f32>,
    interaction_start_us: i64,
}

#[derive(Debug, Clone)]
struct UserProfile {
    cardiac_template: Vec<f32>,
    stress_baseline_rmssd: f32,
    typical_hr_range: (f32, f32),
}

impl FraudDetector {
    pub fn new() -> Self {
        Self {
            user_profile: None,
            ppg_processor: EnsembleProcessor::new(),
            hrv_estimator: HrvEstimator::new(),
            rgb_buffer: Vec::with_capacity(300),
            stress_baseline: None,
            interaction_start_us: 0,
        }
    }

    /// Enroll user profile for future matching
    pub fn enroll(
        &mut self,
        cardiac_template: Vec<f32>,
        stress_baseline: f32,
        hr_range: (f32, f32),
    ) {
        self.user_profile = Some(UserProfile {
            cardiac_template,
            stress_baseline_rmssd: stress_baseline,
            typical_hr_range: hr_range,
        });
    }

    /// Process frame for fraud detection
    pub fn process_frame(&mut self, rgb_mean: [f32; 3], timestamp_us: i64) -> FraudResult {
        if self.interaction_start_us == 0 {
            self.interaction_start_us = timestamp_us;
        }

        self.rgb_buffer.push(rgb_mean);
        if self.rgb_buffer.len() > 300 {
            self.rgb_buffer.remove(0);
        }

        if self.rgb_buffer.len() < 150 {
            return FraudResult {
                risk_score: 0.0,
                risk_level: RiskLevel::Low,
                identity_confidence: None,
                stress_anomaly: false,
                coercion_indicators: CoercionIndicators::default(),
                recommended_action: FraudAction::Approve,
            };
        }

        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();

        let r_arr = Array1::from_vec(r);
        let g_arr = Array1::from_vec(g.clone());
        let b_arr = Array1::from_vec(b);

        let ppg_result = self.ppg_processor.process_arrays(&r_arr, &g_arr, &b_arr);

        let identity_confidence = if let (Some(ref profile), Some(_)) = (&self.user_profile, &ppg_result) {
            Some(self.match_cardiac_signature(&g, &profile.cardiac_template))
        } else {
            None
        };

        let (stress_anomaly, current_stress) = self.analyze_stress(&g);

        let coercion = self.detect_coercion(
            current_stress,
            ppg_result.as_ref().map(|r| r.bpm),
            timestamp_us,
        );

        let risk_score = self.calculate_risk(identity_confidence, stress_anomaly, &coercion);
        let risk_level = self.classify_risk(risk_score);
        let recommended_action = self.recommend_action(risk_level, &coercion);

        FraudResult {
            risk_score,
            risk_level,
            identity_confidence,
            stress_anomaly,
            coercion_indicators: coercion,
            recommended_action,
        }
    }

    fn match_cardiac_signature(&self, pulse: &[f32], template: &[f32]) -> f32 {
        if pulse.len() < 90 || template.len() < 90 {
            return 0.0;
        }

        let pulse_slice = &pulse[pulse.len() - 90..];
        let template_slice = &template[..90.min(template.len())];

        let pulse_mean: f32 = pulse_slice.iter().sum::<f32>() / pulse_slice.len() as f32;
        let template_mean: f32 = template_slice.iter().sum::<f32>() / template_slice.len() as f32;

        let mut correlation = 0.0;
        let mut pulse_var = 0.0;
        let mut template_var = 0.0;

        let len = pulse_slice.len().min(template_slice.len());
        for i in 0..len {
            let p = pulse_slice[i] - pulse_mean;
            let t = template_slice[i] - template_mean;
            correlation += p * t;
            pulse_var += p * p;
            template_var += t * t;
        }

        if pulse_var > 0.0 && template_var > 0.0 {
            (correlation / (pulse_var.sqrt() * template_var.sqrt())).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    fn analyze_stress(&mut self, pulse: &[f32]) -> (bool, f32) {
        let pulse_arr = Array1::from_vec(pulse.to_vec());

        let hrv = self.hrv_estimator.estimate(&pulse_arr);
        if let Some(metrics) = hrv.metrics.as_ref() {
            let current_stress = 1.0 - (metrics.rmssd_ms / 50.0).min(1.0);

            let anomaly = if let Some(ref profile) = self.user_profile {
                metrics.rmssd_ms < profile.stress_baseline_rmssd * 0.5
            } else if let Some(baseline) = self.stress_baseline {
                current_stress > baseline + 0.3
            } else {
                self.stress_baseline = Some(current_stress);
                false
            };

            (anomaly, current_stress)
        } else {
            (false, 0.0)
        }
    }

    fn detect_coercion(
        &self,
        stress: f32,
        heart_rate: Option<f32>,
        timestamp_us: i64,
    ) -> CoercionIndicators {
        let elevated_stress = stress > 0.7;

        let abnormal_behavior = if let (Some(ref profile), Some(hr)) = (&self.user_profile, heart_rate)
        {
            hr < profile.typical_hr_range.0 || hr > profile.typical_hr_range.1
        } else {
            false
        };

        let interaction_duration_sec =
            (timestamp_us - self.interaction_start_us) as f32 / 1_000_000.0;
        let rushed_interaction = interaction_duration_sec < 5.0;

        let unusual_time = false;

        CoercionIndicators {
            elevated_stress,
            abnormal_behavior,
            rushed_interaction,
            unusual_time,
        }
    }

    fn calculate_risk(
        &self,
        identity_confidence: Option<f32>,
        stress_anomaly: bool,
        coercion: &CoercionIndicators,
    ) -> f32 {
        let mut risk = 0.0;

        if let Some(conf) = identity_confidence {
            risk += (1.0 - conf) * 0.4;
        }

        if stress_anomaly {
            risk += 0.2;
        }

        if coercion.elevated_stress {
            risk += 0.1;
        }
        if coercion.abnormal_behavior {
            risk += 0.15;
        }
        if coercion.rushed_interaction {
            risk += 0.1;
        }
        if coercion.unusual_time {
            risk += 0.05;
        }

        risk.clamp(0.0, 1.0)
    }

    fn classify_risk(&self, score: f32) -> RiskLevel {
        if score < 0.25 {
            RiskLevel::Low
        } else if score < 0.5 {
            RiskLevel::Medium
        } else if score < 0.75 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    fn recommend_action(&self, level: RiskLevel, coercion: &CoercionIndicators) -> FraudAction {
        if coercion.elevated_stress && coercion.abnormal_behavior {
            return FraudAction::ManualReview;
        }

        match level {
            RiskLevel::Low => FraudAction::Approve,
            RiskLevel::Medium => FraudAction::AdditionalVerification,
            RiskLevel::High => FraudAction::ManualReview,
            RiskLevel::Critical => FraudAction::Block,
        }
    }

    pub fn reset(&mut self) {
        self.rgb_buffer.clear();
        self.ppg_processor.reset();
        self.interaction_start_us = 0;
    }
}

impl Default for FraudDetector {
    fn default() -> Self {
        Self::new()
    }
}
