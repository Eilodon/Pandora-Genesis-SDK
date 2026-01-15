//! Engagement Scoring
//!
//! Combines emotion and attention signals into engagement metric.

use crate::shared::{EmotionResult, GazeResult};

/// Engagement result
#[derive(Debug, Clone)]
pub struct EngagementResult {
    /// Overall engagement score (0-1)
    pub score: f32,
    /// Attention contribution
    pub attention_score: f32,
    /// Emotion contribution
    pub emotion_score: f32,
    /// Duration contribution
    pub duration_score: f32,
}

/// Engagement Scorer
pub struct EngagementScorer {
    attention_history: Vec<f32>,
    emotion_history: Vec<f32>,
    window_size: usize,
}

impl EngagementScorer {
    pub fn new() -> Self {
        Self {
            attention_history: Vec::with_capacity(90),
            emotion_history: Vec::with_capacity(90),
            window_size: 90,
        }
    }

    /// Score engagement from emotion and gaze
    pub fn score(&mut self, emotion: &EmotionResult, gaze: &GazeResult) -> EngagementResult {
        let attention = if gaze.on_screen { 1.0 } else { 0.3 };
        self.attention_history.push(attention);
        if self.attention_history.len() > self.window_size {
            self.attention_history.remove(0);
        }

        let emotion_score = (emotion.valence + 1.0) / 2.0 * 0.6 + emotion.arousal * 0.4;
        self.emotion_history.push(emotion_score);
        if self.emotion_history.len() > self.window_size {
            self.emotion_history.remove(0);
        }

        let avg_attention = self.attention_history.iter().sum::<f32>()
            / self.attention_history.len().max(1) as f32;
        let avg_emotion = self.emotion_history.iter().sum::<f32>()
            / self.emotion_history.len().max(1) as f32;

        let duration_score = (self.attention_history.len() as f32 / self.window_size as f32).min(1.0);

        let score = avg_attention * 0.4 + avg_emotion * 0.4 + duration_score * 0.2;

        EngagementResult {
            score: score.clamp(0.0, 1.0),
            attention_score: avg_attention,
            emotion_score: avg_emotion,
            duration_score,
        }
    }

    pub fn reset(&mut self) {
        self.attention_history.clear();
        self.emotion_history.clear();
    }
}

impl Default for EngagementScorer {
    fn default() -> Self {
        Self::new()
    }
}
