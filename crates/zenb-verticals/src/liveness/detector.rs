//! Liveness Detection Core
//!
//! Multi-modal liveness detection combining:
//! - rPPG pulse detection (primary)
//! - Texture analysis (3D vs 2D)
//! - Micro-movement analysis
//! - Challenge-response verification

use ndarray::Array1;
use zenb_signals::dsp::{MotionDetector, QualityScorer};
use zenb_signals::rppg::EnsembleProcessor;

use super::challenge_response::{ChallengeGenerator};
use super::temporal_consistency::{ConsistencyResult, TemporalConsistencyChecker};
use super::texture_analyzer::{TextureAnalyzer, TextureResult};

/// Spoofing attack type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpoofingType {
    Photo,
    Video,
    Screen,
    Mask,
    Deepfake,
    Unknown,
}

/// Liveness detection result
#[derive(Debug, Clone)]
pub struct LivenessResult {
    pub is_live: bool,
    pub confidence: f32,

    pub has_pulse: bool,
    pub pulse_confidence: f32,
    pub heart_rate_bpm: Option<f32>,

    pub natural_movements: bool,
    pub movement_score: f32,

    pub texture_3d: bool,
    pub texture_confidence: f32,

    pub challenge_passed: Option<bool>,

    pub temporal_consistency: f32,

    pub spoofing_type: Option<SpoofingType>,

    pub reason: String,
}

impl Default for LivenessResult {
    fn default() -> Self {
        Self {
            is_live: false,
            confidence: 0.0,
            has_pulse: false,
            pulse_confidence: 0.0,
            heart_rate_bpm: None,
            natural_movements: false,
            movement_score: 0.0,
            texture_3d: false,
            texture_confidence: 0.0,
            challenge_passed: None,
            temporal_consistency: 0.0,
            spoofing_type: None,
            reason: String::new(),
        }
    }
}

/// Liveness detection configuration
#[derive(Debug, Clone)]
pub struct LivenessConfig {
    pub min_pulse_confidence: f32,
    pub min_texture_confidence: f32,
    pub min_temporal_consistency: f32,
    pub enable_challenge: bool,
    pub min_frames: usize,
    pub sample_rate: f32,
    pub rppg_buffer_sec: f32,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            min_pulse_confidence: 0.6,
            min_texture_confidence: 0.7,
            min_temporal_consistency: 0.8,
            enable_challenge: false,
            min_frames: 90,
            sample_rate: 30.0,
            rppg_buffer_sec: 5.0,
        }
    }
}

/// Liveness Detector
#[allow(dead_code)] // quality_scorer and motion_detector reserved for future enhancement
pub struct LivenessDetector {
    config: LivenessConfig,

    ppg_processor: EnsembleProcessor,
    quality_scorer: QualityScorer,
    motion_detector: MotionDetector,

    texture_analyzer: TextureAnalyzer,
    challenge_generator: Option<ChallengeGenerator>,
    temporal_checker: TemporalConsistencyChecker,

    frame_count: usize,
    rgb_buffer: Vec<[f32; 3]>,
    landmark_history: Vec<Vec<[f32; 2]>>,
}


impl LivenessDetector {
    pub fn new() -> Self {
        Self::with_config(LivenessConfig::default())
    }

    pub fn with_config(config: LivenessConfig) -> Self {
        let buffer_size = (config.rppg_buffer_sec * config.sample_rate) as usize;

        Self {
            ppg_processor: EnsembleProcessor::new(),
            quality_scorer: QualityScorer::new(),
            motion_detector: MotionDetector::new(),
            texture_analyzer: TextureAnalyzer::new(),
            challenge_generator: if config.enable_challenge {
                Some(ChallengeGenerator::new())
            } else {
                None
            },
            temporal_checker: TemporalConsistencyChecker::new(),
            config,
            frame_count: 0,
            rgb_buffer: Vec::with_capacity(buffer_size),
            landmark_history: Vec::with_capacity(30),
        }
    }

    /// Process a single frame
    pub fn process_frame(
        &mut self,
        rgb_mean: [f32; 3],
        landmarks: &[[f32; 2]],
        _timestamp_us: i64,
    ) -> LivenessResult {
        self.frame_count += 1;

        self.rgb_buffer.push(rgb_mean);
        let max_len = (self.config.rppg_buffer_sec * self.config.sample_rate) as usize;
        if self.rgb_buffer.len() > max_len {
            self.rgb_buffer.remove(0);
        }

        self.landmark_history.push(landmarks.to_vec());
        if self.landmark_history.len() > 30 {
            self.landmark_history.remove(0);
        }

        if self.frame_count < self.config.min_frames {
            return LivenessResult {
                reason: format!(
                    "Collecting data: {}/{} frames",
                    self.frame_count, self.config.min_frames
                ),
                ..Default::default()
            };
        }

        let (has_pulse, pulse_confidence, heart_rate) = self.detect_pulse();
        let (natural_movements, movement_score) = self.analyze_movements();
        let texture_result = self.texture_analyzer.analyze(&self.rgb_buffer);
        let consistency = self.temporal_checker.check(&self.landmark_history);

        let challenge_passed = if let Some(ref mut gen) = self.challenge_generator {
            Some(gen.verify(landmarks))
        } else {
            None
        };

        self.make_decision(
            has_pulse,
            pulse_confidence,
            heart_rate,
            natural_movements,
            movement_score,
            &texture_result,
            &consistency,
            challenge_passed,
        )
    }

    fn detect_pulse(&mut self) -> (bool, f32, Option<f32>) {
        if self.rgb_buffer.len() < 90 {
            return (false, 0.0, None);
        }

        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();

        let r = Array1::from_vec(r);
        let g = Array1::from_vec(g);
        let b = Array1::from_vec(b);

        if let Some(result) = self.ppg_processor.process_arrays(&r, &g, &b) {
            let has_pulse = result.confidence > self.config.min_pulse_confidence;
            (has_pulse, result.confidence, Some(result.bpm))
        } else {
            (false, 0.0, None)
        }
    }

    fn analyze_movements(&self) -> (bool, f32) {
        if self.landmark_history.len() < 10 {
            return (false, 0.0);
        }

        let mut total_variance = 0.0;
        let key_landmarks = [4, 133, 362, 152, 10];

        for &idx in &key_landmarks {
            let positions: Vec<[f32; 2]> = self
                .landmark_history
                .iter()
                .filter_map(|lm| lm.get(idx).copied())
                .collect();

            if positions.len() < 5 {
                continue;
            }

            let mean_x: f32 = positions.iter().map(|p| p[0]).sum::<f32>() / positions.len() as f32;
            let mean_y: f32 = positions.iter().map(|p| p[1]).sum::<f32>() / positions.len() as f32;

            let var: f32 = positions
                .iter()
                .map(|p| (p[0] - mean_x).powi(2) + (p[1] - mean_y).powi(2))
                .sum::<f32>()
                / positions.len() as f32;

            total_variance += var;
        }

        let avg_variance = total_variance / key_landmarks.len() as f32;

        let is_natural = avg_variance > 0.0001 && avg_variance < 0.01;
        let score = if is_natural {
            1.0 - (avg_variance - 0.001).abs() / 0.01
        } else {
            0.0
        };

        (is_natural, score.clamp(0.0, 1.0))
    }

    fn make_decision(
        &self,
        has_pulse: bool,
        pulse_confidence: f32,
        heart_rate: Option<f32>,
        natural_movements: bool,
        movement_score: f32,
        texture: &TextureResult,
        consistency: &ConsistencyResult,
        challenge_passed: Option<bool>,
    ) -> LivenessResult {
        let mut score = 0.0;
        let mut max_score = 0.0;

        max_score += 0.4;
        if has_pulse {
            score += 0.4 * pulse_confidence;
        }

        max_score += 0.25;
        if texture.is_3d {
            score += 0.25 * texture.confidence;
        }

        max_score += 0.2;
        if natural_movements {
            score += 0.2 * movement_score;
        }

        max_score += 0.15;
        score += 0.15 * consistency.score;

        if let Some(passed) = challenge_passed {
            if passed {
                score += 0.1;
            } else {
                score -= 0.2;
            }
        }

        let confidence = (score / max_score).clamp(0.0, 1.0);
        let is_live = confidence > 0.65 && has_pulse;

        let spoofing_type = if !is_live {
            self.detect_spoofing_type(has_pulse, texture, natural_movements)
        } else {
            None
        };

        let reason = self.generate_reason(
            is_live,
            has_pulse,
            texture.is_3d,
            natural_movements,
            consistency.score,
            challenge_passed,
        );

        LivenessResult {
            is_live,
            confidence,
            has_pulse,
            pulse_confidence,
            heart_rate_bpm: heart_rate,
            natural_movements,
            movement_score,
            texture_3d: texture.is_3d,
            texture_confidence: texture.confidence,
            challenge_passed,
            temporal_consistency: consistency.score,
            spoofing_type,
            reason,
        }
    }

    fn detect_spoofing_type(
        &self,
        has_pulse: bool,
        texture: &TextureResult,
        natural_movements: bool,
    ) -> Option<SpoofingType> {
        if !has_pulse && !texture.is_3d && !natural_movements {
            Some(SpoofingType::Photo)
        } else if !has_pulse && !texture.is_3d && natural_movements {
            Some(SpoofingType::Video)
        } else if !has_pulse && texture.is_3d {
            Some(SpoofingType::Mask)
        } else if has_pulse && !texture.is_3d {
            Some(SpoofingType::Screen)
        } else {
            Some(SpoofingType::Unknown)
        }
    }

    fn generate_reason(
        &self,
        is_live: bool,
        has_pulse: bool,
        texture_3d: bool,
        natural_movements: bool,
        consistency: f32,
        challenge: Option<bool>,
    ) -> String {
        if is_live {
            "LIVE: Pulse detected, natural movements, 3D texture confirmed".to_string()
        } else {
            let mut reasons = Vec::new();
            if !has_pulse {
                reasons.push("No pulse detected");
            }
            if !texture_3d {
                reasons.push("2D texture (possible photo/screen)");
            }
            if !natural_movements {
                reasons.push("Unnatural movement pattern");
            }
            if consistency < 0.5 {
                reasons.push("Temporal inconsistency");
            }
            if challenge == Some(false) {
                reasons.push("Challenge failed");
            }
            format!("SPOOF: {}", reasons.join(", "))
        }
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.rgb_buffer.clear();
        self.landmark_history.clear();
        self.ppg_processor = EnsembleProcessor::new();
        self.temporal_checker.reset();
        if let Some(ref mut gen) = self.challenge_generator {
            gen.reset();
        }
    }

    /// Get current config
    pub fn config(&self) -> &LivenessConfig {
        &self.config
    }
}

impl Default for LivenessDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liveness_config_defaults() {
        let config = LivenessConfig::default();
        assert!((config.min_pulse_confidence - 0.6).abs() < 0.01);
        assert!((config.min_texture_confidence - 0.7).abs() < 0.01);
        assert_eq!(config.min_frames, 90);
    }

    #[test]
    fn test_liveness_result_default() {
        let result = LivenessResult::default();
        assert!(!result.is_live);
        assert!(!result.has_pulse);
        assert!(result.spoofing_type.is_none());
    }

    #[test]
    fn test_liveness_detector_creation() {
        let detector = LivenessDetector::new();
        assert_eq!(detector.frame_count, 0);
        assert!(detector.rgb_buffer.is_empty());
    }

    #[test]
    fn test_liveness_detector_initial_frames() {
        let mut detector = LivenessDetector::new();
        let landmarks = vec![[0.5, 0.5]; 468];
        
        // Process less than min_frames should return collecting data
        let result = detector.process_frame([128.0, 128.0, 128.0], &landmarks, 0);
        assert!(!result.is_live);
        assert!(result.reason.contains("Collecting"));
    }

    #[test]
    fn test_liveness_detector_reset() {
        let mut detector = LivenessDetector::new();
        detector.frame_count = 100;
        detector.rgb_buffer.push([1.0, 1.0, 1.0]);
        
        detector.reset();
        
        assert_eq!(detector.frame_count, 0);
        assert!(detector.rgb_buffer.is_empty());
    }

    #[test]
    fn test_spoofing_type_detection() {
        let detector = LivenessDetector::new();
        let texture = TextureResult {
            is_3d: false,
            confidence: 0.5,
            color_gamut: 0.5,
            frequency_score: 0.5,
            specular_score: 0.5,
        };
        
        // No pulse, no 3D, no movements = Photo
        let spoof = detector.detect_spoofing_type(false, &texture, false);
        assert_eq!(spoof, Some(SpoofingType::Photo));
        
        // No pulse, no 3D, has movements = Video
        let spoof = detector.detect_spoofing_type(false, &texture, true);
        assert_eq!(spoof, Some(SpoofingType::Video));
    }
}
