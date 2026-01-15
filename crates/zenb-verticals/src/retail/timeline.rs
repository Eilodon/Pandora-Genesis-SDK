//! Emotion Timeline
//!
//! Tracks emotional journey over time for session analysis.

use crate::shared::{BasicEmotion, EmotionResult};

const EMOTION_COUNT: usize = 7;

/// Emotion snapshot at a point in time
#[derive(Debug, Clone)]
pub struct EmotionSnapshot {
    /// Timestamp (seconds from session start)
    pub timestamp_sec: f32,
    /// Primary emotion
    pub emotion: BasicEmotion,
    /// Valence
    pub valence: f32,
    /// Arousal
    pub arousal: f32,
}

/// Emotion Timeline tracker
pub struct EmotionTimeline {
    snapshots: Vec<EmotionSnapshot>,
    interval_sec: f32,
    last_snapshot_us: i64,
    session_start_us: i64,

    valence_acc: f32,
    arousal_acc: f32,
    emotion_counts: [u32; EMOTION_COUNT],
    sample_count: u32,
}

impl EmotionTimeline {
    pub fn new(interval_sec: f32) -> Self {
        Self {
            snapshots: Vec::with_capacity(100),
            interval_sec,
            last_snapshot_us: 0,
            session_start_us: 0,
            valence_acc: 0.0,
            arousal_acc: 0.0,
            emotion_counts: [0; EMOTION_COUNT],
            sample_count: 0,
        }
    }

    /// Update timeline with new emotion
    pub fn update(&mut self, emotion: &EmotionResult, timestamp_us: i64) {
        if self.session_start_us == 0 {
            self.session_start_us = timestamp_us;
            self.last_snapshot_us = timestamp_us;
        }

        self.valence_acc += emotion.valence;
        self.arousal_acc += emotion.arousal;
        let idx = emotion.primary_emotion as usize;
        if idx < EMOTION_COUNT {
            self.emotion_counts[idx] += 1;
        }
        self.sample_count += 1;

        let elapsed_us = timestamp_us - self.last_snapshot_us;
        let interval_us = (self.interval_sec * 1_000_000.0) as i64;

        if elapsed_us >= interval_us && self.sample_count > 0 {
            let avg_valence = self.valence_acc / self.sample_count as f32;
            let avg_arousal = self.arousal_acc / self.sample_count as f32;

            let dominant_idx = self
                .emotion_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let dominant_emotion = match dominant_idx {
                0 => BasicEmotion::Neutral,
                1 => BasicEmotion::Happy,
                2 => BasicEmotion::Sad,
                3 => BasicEmotion::Angry,
                4 => BasicEmotion::Fearful,
                5 => BasicEmotion::Disgusted,
                6 => BasicEmotion::Surprised,
                _ => BasicEmotion::Neutral,
            };

            let timestamp_sec = (timestamp_us - self.session_start_us) as f32 / 1_000_000.0;

            self.snapshots.push(EmotionSnapshot {
                timestamp_sec,
                emotion: dominant_emotion,
                valence: avg_valence,
                arousal: avg_arousal,
            });

            self.valence_acc = 0.0;
            self.arousal_acc = 0.0;
            self.emotion_counts = [0; EMOTION_COUNT];
            self.sample_count = 0;
            self.last_snapshot_us = timestamp_us;
        }
    }

    /// Get all snapshots
    pub fn get_snapshots(&self) -> Vec<EmotionSnapshot> {
        self.snapshots.clone()
    }

    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.last_snapshot_us = 0;
        self.session_start_us = 0;
        self.valence_acc = 0.0;
        self.arousal_acc = 0.0;
        self.emotion_counts = [0; EMOTION_COUNT];
        self.sample_count = 0;
    }
}

impl Default for EmotionTimeline {
    fn default() -> Self {
        Self::new(5.0)
    }
}
