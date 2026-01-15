# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET - PART 5

# 6. PHASE 4: RETAIL ANALYTICS

## Timeline: Day 26-32 (7 days)

### 6.1 Emotion Analytics Core

**File:** `crates/zenb-verticals/src/retail/emotion_analytics.rs`

```rust
//! Emotion Analytics for Retail
//!
//! Analyzes customer emotional responses for:
//! - Product engagement
//! - Experience optimization
//! - Purchase intent prediction

use zenb_signals::beauty::landmarks::CanonicalLandmarks;
use zenb_core::estimators::UkfEstimator;

use crate::shared::{
    MicroExpressionAnalyzer, EmotionResult, BasicEmotion,
    GazeEstimator, GazeResult,
};

use super::engagement::{EngagementScorer, EngagementResult};
use super::timeline::{EmotionTimeline, EmotionSnapshot};

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
        
        // 1. Micro-expression analysis
        let emotion = self.micro_expression.analyze(landmarks);
        
        // 2. Gaze estimation
        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);
        
        // 3. Update histories
        self.valence_history.push(emotion.valence);
        self.arousal_history.push(emotion.arousal);
        if self.valence_history.len() > 300 {
            self.valence_history.remove(0);
            self.arousal_history.remove(0);
        }
        
        // 4. Engagement scoring
        let engagement = self.engagement_scorer.score(&emotion, &gaze);
        
        // 5. Update timeline
        self.timeline.update(&emotion, timestamp_us);
        
        // 6. Calculate stress level
        let stress_level = self.calculate_stress(&emotion);
        
        // 7. Predict purchase intent
        let purchase_intent = if self.config.enable_purchase_prediction {
            self.predict_purchase_intent(&emotion, &engagement)
        } else {
            0.0
        };
        
        // 8. Session duration
        let session_duration_sec = (timestamp_us - self.session_start_us) as f32 / 1_000_000.0;
        
        CustomerInsights {
            current_emotion: emotion.primary_emotion,
            emotion_confidence: emotion.confidence,
            valence: emotion.valence,
            arousal: emotion.arousal,
            engagement_score: engagement.score,
            engagement_level: self.classify_engagement(engagement.score),
            attention_score: 1.0 - gaze.deviation,
            is_attending: gaze.on_screen,
            stress_level,
            purchase_intent,
            session_duration_sec,
            emotion_journey: self.timeline.get_snapshots(),
        }
    }
    
    fn calculate_stress(&self, emotion: &EmotionResult) -> f32 {
        // High arousal + negative valence = stress
        let stress = if emotion.valence < 0.0 {
            emotion.arousal * (-emotion.valence)
        } else {
            0.0
        };
        
        stress.clamp(0.0, 1.0)
    }
    
    fn predict_purchase_intent(&self, emotion: &EmotionResult, engagement: &EngagementResult) -> f32 {
        // Simple heuristic model
        // High engagement + positive valence + moderate arousal = high intent
        
        let valence_factor = (emotion.valence + 1.0) / 2.0; // Normalize to 0-1
        let arousal_factor = 1.0 - (emotion.arousal - 0.5).abs() * 2.0; // Peak at 0.5
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
        
        let peak_valence = self.valence_history.iter().cloned().fold(f32::MIN, f32::max);
        let min_valence = self.valence_history.iter().cloned().fold(f32::MAX, f32::min);
        
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
```

---

### 6.2 Engagement Scorer

**File:** `crates/zenb-verticals/src/retail/engagement.rs`

```rust
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
            window_size: 90, // 3 seconds at 30fps
        }
    }
    
    /// Score engagement from emotion and gaze
    pub fn score(&mut self, emotion: &EmotionResult, gaze: &GazeResult) -> EngagementResult {
        // Attention score from gaze
        let attention = if gaze.on_screen { 1.0 } else { 0.3 };
        self.attention_history.push(attention);
        if self.attention_history.len() > self.window_size {
            self.attention_history.remove(0);
        }
        
        // Emotion score (positive valence + moderate arousal = engaged)
        let emotion_score = (emotion.valence + 1.0) / 2.0 * 0.6 + emotion.arousal * 0.4;
        self.emotion_history.push(emotion_score);
        if self.emotion_history.len() > self.window_size {
            self.emotion_history.remove(0);
        }
        
        // Average scores
        let avg_attention = self.attention_history.iter().sum::<f32>() 
            / self.attention_history.len() as f32;
        let avg_emotion = self.emotion_history.iter().sum::<f32>() 
            / self.emotion_history.len() as f32;
        
        // Duration score (longer attention = more engaged)
        let duration_score = (self.attention_history.len() as f32 / self.window_size as f32).min(1.0);
        
        // Combined score
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
```

---

### 6.3 Emotion Timeline

**File:** `crates/zenb-verticals/src/retail/timeline.rs`

```rust
//! Emotion Timeline
//!
//! Tracks emotional journey over time for session analysis.

use crate::shared::{EmotionResult, BasicEmotion};

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
    
    // Accumulator for averaging
    valence_acc: f32,
    arousal_acc: f32,
    emotion_counts: [u32; 7], // One for each BasicEmotion
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
            emotion_counts: [0; 7],
            sample_count: 0,
        }
    }
    
    /// Update timeline with new emotion
    pub fn update(&mut self, emotion: &EmotionResult, timestamp_us: i64) {
        if self.session_start_us == 0 {
            self.session_start_us = timestamp_us;
            self.last_snapshot_us = timestamp_us;
        }
        
        // Accumulate
        self.valence_acc += emotion.valence;
        self.arousal_acc += emotion.arousal;
        self.emotion_counts[emotion.primary_emotion as usize] += 1;
        self.sample_count += 1;
        
        // Check if interval elapsed
        let elapsed_us = timestamp_us - self.last_snapshot_us;
        let interval_us = (self.interval_sec * 1_000_000.0) as i64;
        
        if elapsed_us >= interval_us && self.sample_count > 0 {
            // Create snapshot
            let avg_valence = self.valence_acc / self.sample_count as f32;
            let avg_arousal = self.arousal_acc / self.sample_count as f32;
            
            // Find dominant emotion
            let dominant_idx = self.emotion_counts
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
            
            // Reset accumulator
            self.valence_acc = 0.0;
            self.arousal_acc = 0.0;
            self.emotion_counts = [0; 7];
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
        self.emotion_counts = [0; 7];
        self.sample_count = 0;
    }
}

impl Default for EmotionTimeline {
    fn default() -> Self {
        Self::new(5.0)
    }
}
```

---

### 6.4 Retail Module Entry Point

**File:** `crates/zenb-verticals/src/retail/mod.rs`

```rust
//! Retail Emotion Analytics Module
//!
//! Customer emotion and engagement analysis for retail optimization.

pub mod emotion_analytics;
pub mod engagement;
pub mod timeline;

pub use emotion_analytics::{
    EmotionAnalytics, EmotionAnalyticsConfig, 
    CustomerInsights, EngagementLevel, SessionStats,
};
pub use engagement::{EngagementScorer, EngagementResult};
pub use timeline::{EmotionTimeline, EmotionSnapshot};
```

---

# 7. IMPLEMENTATION TIMELINE SUMMARY

## 📅 GANTT CHART

```
Week 1 (Day 1-7)
├── Day 1-2: Phase 0 - Infrastructure setup
│   ├── Create zenb-verticals crate
│   ├── Setup Cargo.toml with dependencies
│   └── Update workspace
│
└── Day 3-7: Phase 1 - Shared Components
    ├── eye_metrics.rs (EAR, PERCLOS, blink)
    ├── gaze_estimator.rs (head pose + eye gaze)
    └── micro_expression.rs (AU detection)

Week 2 (Day 8-14)
└── Phase 2 - Liveness Detection
    ├── detector.rs (core liveness logic)
    ├── texture_analyzer.rs (3D vs 2D)
    ├── challenge_response.rs (interactive verification)
    ├── temporal_consistency.rs (anti-replay)
    └── Integration tests

Week 3-4 (Day 15-25)
└── Phase 3 - Driver Monitoring System
    ├── dms.rs (core DMS logic)
    ├── drowsiness.rs (PERCLOS-based)
    ├── distraction.rs (gaze-based)
    ├── cardiac_monitor.rs (emergency detection)
    └── Integration tests

Week 5 (Day 26-32)
└── Phase 4 - Retail Analytics
    ├── emotion_analytics.rs (core analytics)
    ├── engagement.rs (engagement scoring)
    ├── timeline.rs (emotion journey)
    └── Integration tests

Week 6+ (Day 33+)
└── Phase 5 - Fintech & Education (Optional)
    ├── fintech/fraud_detector.rs
    ├── fintech/cardiac_fingerprint.rs
    ├── education/proctoring.rs
    └── education/behavior_scorer.rs
```

---

## 🎯 MILESTONE CHECKLIST

### Phase 0: Infrastructure ✅
- [ ] Create `crates/zenb-verticals/Cargo.toml`
- [ ] Create `crates/zenb-verticals/src/lib.rs`
- [ ] Update workspace `Cargo.toml`
- [ ] Verify `cargo build` passes

### Phase 1: Shared Components ✅
- [ ] `shared/eye_metrics.rs` - EAR, PERCLOS, blink detection
- [ ] `shared/gaze_estimator.rs` - Gaze direction estimation
- [ ] `shared/micro_expression.rs` - AU detection, emotion classification
- [ ] `shared/mod.rs` - Module exports
- [ ] Unit tests for all shared components

### Phase 2: Liveness Detection ✅
- [ ] `liveness/detector.rs` - Core liveness logic
- [ ] `liveness/texture_analyzer.rs` - 3D vs 2D detection
- [ ] `liveness/challenge_response.rs` - Interactive verification
- [ ] `liveness/temporal_consistency.rs` - Anti-replay
- [ ] `liveness/mod.rs` - Module exports
- [ ] Integration tests with mock data
- [ ] Benchmark: <50ms per frame

### Phase 3: Driver Monitoring ✅
- [ ] `automotive/dms.rs` - Core DMS logic
- [ ] `automotive/drowsiness.rs` - PERCLOS-based detection
- [ ] `automotive/distraction.rs` - Gaze-based detection
- [ ] `automotive/cardiac_monitor.rs` - Emergency detection
- [ ] `automotive/mod.rs` - Module exports
- [ ] Integration tests
- [ ] Safety validation: fail-safe defaults

### Phase 4: Retail Analytics ✅
- [ ] `retail/emotion_analytics.rs` - Core analytics
- [ ] `retail/engagement.rs` - Engagement scoring
- [ ] `retail/timeline.rs` - Emotion journey
- [ ] `retail/mod.rs` - Module exports
- [ ] Integration tests
- [ ] Privacy compliance review

---

## 🔧 TESTING STRATEGY

### Unit Tests
```rust
// Example test structure
#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_mock_landmarks() -> CanonicalLandmarks {
        // Create 468 landmarks with realistic positions
        let mut points = vec![[0.5, 0.5]; 468];
        // Set key landmarks...
        CanonicalLandmarks {
            points,
            inter_ocular_px: 100.0,
            origin: [500.0, 400.0],
            valid: true,
        }
    }
    
    #[test]
    fn test_liveness_with_pulse() {
        let mut detector = LivenessDetector::new();
        // Simulate frames with pulse...
        let result = detector.process_frame(...);
        assert!(result.has_pulse);
    }
}
```

### Integration Tests
```rust
// crates/zenb-verticals/tests/liveness_integration.rs
use zenb_verticals::liveness::LivenessDetector;
use zenb_signals::rppg::EnsembleProcessor;

#[test]
fn test_liveness_full_pipeline() {
    // Load test video frames
    // Process through full pipeline
    // Verify liveness detection
}
```

### Benchmark Tests
```rust
// crates/zenb-verticals/benches/liveness_bench.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn liveness_benchmark(c: &mut Criterion) {
    c.bench_function("liveness_frame", |b| {
        let mut detector = LivenessDetector::new();
        b.iter(|| {
            detector.process_frame(rgb, landmarks, timestamp)
        })
    });
}
```

---

## 📦 DEPLOYMENT CHECKLIST

### Pre-release
- [ ] All tests pass
- [ ] Benchmarks meet targets (<50ms/frame)
- [ ] Documentation complete
- [ ] API review complete
- [ ] Security audit complete

### FFI Bindings
- [ ] Create `zenb-verticals-uniffi` crate
- [ ] Define UDL interface
- [ ] Generate Swift/Kotlin bindings
- [ ] Test on iOS/Android

### SDK Release
- [ ] Version bump
- [ ] Changelog update
- [ ] Package for distribution
- [ ] Update examples

---

*Xem PART6 để biết chi tiết về Fintech, Education modules và Safety Framework...*
