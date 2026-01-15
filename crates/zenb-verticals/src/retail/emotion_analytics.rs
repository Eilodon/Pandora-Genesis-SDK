//! Emotion Analytics for Retail
//!
//! Analyzes customer emotional responses for:
//! - Product engagement
//! - Experience optimization
//! - Purchase intent prediction

use zenb_signals::beauty::CanonicalLandmarks;

use crate::shared::{BasicEmotion, EmotionResult, GazeEstimator, MicroExpressionAnalyzer};

use super::engagement::{EngagementResult, EngagementScorer};
use super::timeline::{EmotionSnapshot, EmotionTimeline};

/// Customer insights result
#[derive(Debug, Clone)]
pub struct CustomerInsights {
    /// Current emotion
    pub current_emotion: BasicEmotion,
    /// Emotion confidence
    pub emotion_confidence: f32,

    /// Valence (-1 to +1)
    pub valence: f32,
    /// Arousal (0 to 1)
    pub arousal: f32,

    /// Engagement score (0-1)
    pub engagement_score: f32,
    /// Engagement classification
    pub engagement_level: EngagementLevel,

    /// Attention score (0-1)
    pub attention_score: f32,
    /// Is looking at product/display
    pub is_attending: bool,

    /// Stress indicators (0-1)
    pub stress_level: f32,

    /// Purchase intent prediction (0-1)
    pub purchase_intent: f32,

    /// Session duration (seconds)
    pub session_duration_sec: f32,

    /// Emotional journey summary
    pub emotion_journey: Vec<EmotionSnapshot>,
}

/// Engagement level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngagementLevel {
    Disengaged,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Emotion Analytics configuration
#[derive(Debug, Clone)]
pub struct EmotionAnalyticsConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Smoothing factor
    pub smoothing_alpha: f32,
    /// Timeline snapshot interval (seconds)
    pub snapshot_interval_sec: f32,
    /// Enable purchase intent prediction
    pub enable_purchase_prediction: bool,
}

impl Default for EmotionAnalyticsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            smoothing_alpha: 0.2,
            snapshot_interval_sec: 5.0,
            enable_purchase_prediction: true,
        }
    }
}

/// Emotion Analytics Analyzer
pub struct EmotionAnalytics {
    config: EmotionAnalyticsConfig,

    // Shared components
    micro_expression: MicroExpressionAnalyzer,
    gaze_estimator: GazeEstimator,

    // Retail-specific
    engagement_scorer: EngagementScorer,
    timeline: EmotionTimeline,

    // State
    session_start_us: i64,
    frame_count: usize,
    valence_history: Vec<f32>,
    arousal_history: Vec<f32>,
}

impl EmotionAnalytics {
    pub fn new() -> Self {
        Self::with_config(EmotionAnalyticsConfig::default())
    }

    pub fn with_config(config: EmotionAnalyticsConfig) -> Self {
        Self {
            micro_expression: MicroExpressionAnalyzer::new(),
            gaze_estimator: GazeEstimator::new(),
            engagement_scorer: EngagementScorer::new(),
            timeline: EmotionTimeline::new(config.snapshot_interval_sec),
            config,
            session_start_us: 0,
            frame_count: 0,
            valence_history: Vec::with_capacity(300),
            arousal_history: Vec::with_capacity(300),
        }
    }

    /// Calibrate baseline (neutral expression)
    pub fn calibrate(&mut self, landmarks: &CanonicalLandmarks) {
        self.micro_expression.calibrate(landmarks);
    }

    /// Process a single frame
    pub fn process_frame(
        &mut self,
        landmarks: &CanonicalLandmarks,
        head_pose: Option<[f32; 3]>,
        timestamp_us: i64,
    ) -> CustomerInsights {
        if self.frame_count == 0 {
            self.session_start_us = timestamp_us;
        }
        self.frame_count += 1;

        let emotion = self.micro_expression.analyze(landmarks);
        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);

        self.valence_history.push(emotion.valence);
        self.arousal_history.push(emotion.arousal);
        if self.valence_history.len() > 300 {
            self.valence_history.remove(0);
            self.arousal_history.remove(0);
        }

        let engagement = self.engagement_scorer.score(&emotion, &gaze);

        self.timeline.update(&emotion, timestamp_us);

        let stress_level = self.calculate_stress(&emotion);

        let purchase_intent = if self.config.enable_purchase_prediction {
            self.predict_purchase_intent(&emotion, &engagement)
        } else {
            0.0
        };

        let session_duration_sec = (timestamp_us - self.session_start_us) as f32 / 1_000_000.0;

        CustomerInsights {
            current_emotion: emotion.primary_emotion,
            emotion_confidence: emotion.confidence,
            valence: emotion.valence,
            arousal: emotion.arousal,
            engagement_score: engagement.score,
            engagement_level: self.classify_engagement(engagement.score),
            attention_score: (1.0 - gaze.deviation).clamp(0.0, 1.0),
            is_attending: gaze.on_screen,
            stress_level,
            purchase_intent,
            session_duration_sec,
            emotion_journey: self.timeline.get_snapshots(),
        }
    }

    fn calculate_stress(&self, emotion: &EmotionResult) -> f32 {
        let stress = if emotion.valence < 0.0 {
            emotion.arousal * (-emotion.valence)
        } else {
            0.0
        };

        stress.clamp(0.0, 1.0)
    }

    fn predict_purchase_intent(&self, emotion: &EmotionResult, engagement: &EngagementResult) -> f32 {
        let valence_factor = (emotion.valence + 1.0) / 2.0;
        let arousal_factor = 1.0 - (emotion.arousal - 0.5).abs() * 2.0;
        let engagement_factor = engagement.score;

        let intent = valence_factor * 0.4 + arousal_factor * 0.2 + engagement_factor * 0.4;

        intent.clamp(0.0, 1.0)
    }

    fn classify_engagement(&self, score: f32) -> EngagementLevel {
        if score < 0.2 {
            EngagementLevel::Disengaged
        } else if score < 0.4 {
            EngagementLevel::Low
        } else if score < 0.6 {
            EngagementLevel::Medium
        } else if score < 0.8 {
            EngagementLevel::High
        } else {
            EngagementLevel::VeryHigh
        }
    }

    /// Get aggregated session statistics
    pub fn get_session_stats(&self) -> SessionStats {
        let avg_valence = if !self.valence_history.is_empty() {
            self.valence_history.iter().sum::<f32>() / self.valence_history.len() as f32
        } else {
            0.0
        };

        let avg_arousal = if !self.arousal_history.is_empty() {
            self.arousal_history.iter().sum::<f32>() / self.arousal_history.len() as f32
        } else {
            0.0
        };

        let peak_valence = if !self.valence_history.is_empty() {
            self.valence_history
                .iter()
                .cloned()
                .fold(f32::MIN, f32::max)
        } else {
            0.0
        };

        let min_valence = if !self.valence_history.is_empty() {
            self.valence_history
                .iter()
                .cloned()
                .fold(f32::MAX, f32::min)
        } else {
            0.0
        };

        SessionStats {
            avg_valence,
            avg_arousal,
            peak_valence,
            min_valence,
            emotion_journey: self.timeline.get_snapshots(),
        }
    }

    pub fn reset(&mut self) {
        self.micro_expression.reset();
        self.gaze_estimator.reset();
        self.engagement_scorer.reset();
        self.timeline.reset();
        self.valence_history.clear();
        self.arousal_history.clear();
        self.frame_count = 0;
        self.session_start_us = 0;
    }
}

/// Session statistics
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub avg_valence: f32,
    pub avg_arousal: f32,
    pub peak_valence: f32,
    pub min_valence: f32,
    pub emotion_journey: Vec<EmotionSnapshot>,
}

impl Default for EmotionAnalytics {
    fn default() -> Self {
        Self::new()
    }
}
