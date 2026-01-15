# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET - PART 3

# 4. PHASE 2: LIVENESS DETECTION MODULE

## Timeline: Day 8-14 (7 days)

### 4.1 Liveness Detector Core

**File:** `crates/zenb-verticals/src/liveness/detector.rs`

```rust
//! Liveness Detection Core
//!
//! Multi-modal liveness detection combining:
//! - rPPG pulse detection (primary - unique differentiator)
//! - Texture analysis (3D vs 2D detection)
//! - Micro-movement analysis
//! - Challenge-response verification

use zenb_signals::rppg::{EnsembleProcessor, EnsembleResult};
use zenb_signals::dsp::{MotionDetector, QualityScorer, QualityScore};
use zenb_signals::beauty::landmarks::CanonicalLandmarks;

use super::texture_analyzer::{TextureAnalyzer, TextureResult};
use super::challenge_response::{ChallengeGenerator, ChallengeResult};
use super::temporal_consistency::{TemporalConsistencyChecker, ConsistencyResult};

/// Spoofing attack type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpoofingType {
    /// Printed photo attack
    Photo,
    /// Video replay attack
    Video,
    /// Screen display attack
    Screen,
    /// 3D mask attack
    Mask,
    /// AI-generated deepfake
    Deepfake,
    /// Unknown attack type
    Unknown,
}

/// Liveness detection result
#[derive(Debug, Clone)]
pub struct LivenessResult {
    /// Final liveness decision
    pub is_live: bool,
    /// Overall confidence (0-1)
    pub confidence: f32,
    
    // Individual signals
    /// rPPG pulse detected
    pub has_pulse: bool,
    /// Pulse confidence
    pub pulse_confidence: f32,
    /// Heart rate if detected
    pub heart_rate_bpm: Option<f32>,
    
    /// Natural micro-movements detected
    pub natural_movements: bool,
    /// Movement score
    pub movement_score: f32,
    
    /// 3D texture detected (vs 2D)
    pub texture_3d: bool,
    /// Texture confidence
    pub texture_confidence: f32,
    
    /// Challenge-response passed (if used)
    pub challenge_passed: Option<bool>,
    
    /// Temporal consistency score
    pub temporal_consistency: f32,
    
    /// Detected spoofing type (if not live)
    pub spoofing_type: Option<SpoofingType>,
    
    /// Detailed reason for decision
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
    /// Minimum pulse confidence for liveness
    pub min_pulse_confidence: f32,
    /// Minimum texture confidence
    pub min_texture_confidence: f32,
    /// Minimum temporal consistency
    pub min_temporal_consistency: f32,
    /// Enable challenge-response mode
    pub enable_challenge: bool,
    /// Minimum frames for decision
    pub min_frames: usize,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// rPPG buffer size (seconds)
    pub rppg_buffer_sec: f32,
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            min_pulse_confidence: 0.6,
            min_texture_confidence: 0.7,
            min_temporal_consistency: 0.8,
            enable_challenge: false,
            min_frames: 90, // 3 seconds at 30fps
            sample_rate: 30.0,
            rppg_buffer_sec: 5.0,
        }
    }
}

/// Liveness Detector
pub struct LivenessDetector {
    config: LivenessConfig,
    
    // Core processors (REUSE from zenb-signals)
    ppg_processor: EnsembleProcessor,
    quality_scorer: QualityScorer,
    motion_detector: MotionDetector,
    
    // New components
    texture_analyzer: TextureAnalyzer,
    challenge_generator: Option<ChallengeGenerator>,
    temporal_checker: TemporalConsistencyChecker,
    
    // State
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
    ///
    /// # Arguments
    /// * `rgb_mean` - Mean RGB values from face ROI [R, G, B]
    /// * `landmarks` - 468 MediaPipe landmarks
    /// * `timestamp_us` - Frame timestamp in microseconds
    ///
    /// # Returns
    /// Liveness result (may be incomplete if not enough frames)
    pub fn process_frame(
        &mut self,
        rgb_mean: [f32; 3],
        landmarks: &[[f32; 2]],
        timestamp_us: i64,
    ) -> LivenessResult {
        self.frame_count += 1;
        
        // Add to buffers
        self.rgb_buffer.push(rgb_mean);
        if self.rgb_buffer.len() > (self.config.rppg_buffer_sec * self.config.sample_rate) as usize {
            self.rgb_buffer.remove(0);
        }
        
        self.landmark_history.push(landmarks.to_vec());
        if self.landmark_history.len() > 30 {
            self.landmark_history.remove(0);
        }
        
        // Not enough data yet
        if self.frame_count < self.config.min_frames {
            return LivenessResult {
                reason: format!("Collecting data: {}/{} frames", 
                    self.frame_count, self.config.min_frames),
                ..Default::default()
            };
        }
        
        // 1. rPPG Pulse Detection (PRIMARY SIGNAL)
        let (has_pulse, pulse_confidence, heart_rate) = self.detect_pulse();
        
        // 2. Micro-movement Analysis
        let (natural_movements, movement_score) = self.analyze_movements();
        
        // 3. Texture Analysis (3D vs 2D)
        let texture_result = self.texture_analyzer.analyze(&self.rgb_buffer);
        
        // 4. Temporal Consistency
        let consistency = self.temporal_checker.check(&self.landmark_history);
        
        // 5. Challenge-Response (if enabled)
        let challenge_passed = if let Some(ref mut gen) = self.challenge_generator {
            Some(gen.verify(landmarks))
        } else {
            None
        };
        
        // 6. Fusion Decision
        self.make_decision(
            has_pulse, pulse_confidence, heart_rate,
            natural_movements, movement_score,
            &texture_result,
            &consistency,
            challenge_passed,
        )
    }
    
    fn detect_pulse(&mut self) -> (bool, f32, Option<f32>) {
        if self.rgb_buffer.len() < 90 {
            return (false, 0.0, None);
        }
        
        // Convert buffer to arrays for EnsembleProcessor
        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();
        
        // Process with ensemble
        if let Some(result) = self.ppg_processor.process_arrays(&r, &g, &b) {
            let has_pulse = result.confidence > self.config.min_pulse_confidence;
            (has_pulse, result.confidence, Some(result.heart_rate_bpm))
        } else {
            (false, 0.0, None)
        }
    }
    
    fn analyze_movements(&self) -> (bool, f32) {
        if self.landmark_history.len() < 10 {
            return (false, 0.0);
        }
        
        // Compute micro-movement variance
        let mut total_variance = 0.0;
        let key_landmarks = [4, 133, 362, 152, 10]; // nose, eyes, chin, forehead
        
        for &idx in &key_landmarks {
            let positions: Vec<[f32; 2]> = self.landmark_history
                .iter()
                .filter_map(|lm| lm.get(idx).copied())
                .collect();
            
            if positions.len() < 5 {
                continue;
            }
            
            // Compute variance
            let mean_x: f32 = positions.iter().map(|p| p[0]).sum::<f32>() / positions.len() as f32;
            let mean_y: f32 = positions.iter().map(|p| p[1]).sum::<f32>() / positions.len() as f32;
            
            let var: f32 = positions.iter()
                .map(|p| (p[0] - mean_x).powi(2) + (p[1] - mean_y).powi(2))
                .sum::<f32>() / positions.len() as f32;
            
            total_variance += var;
        }
        
        let avg_variance = total_variance / key_landmarks.len() as f32;
        
        // Natural micro-movements have small but non-zero variance
        // Photos have near-zero, videos may have periodic patterns
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
        // Weighted scoring
        let mut score = 0.0;
        let mut max_score = 0.0;
        
        // Pulse (40% weight - primary signal)
        max_score += 0.4;
        if has_pulse {
            score += 0.4 * pulse_confidence;
        }
        
        // Texture (25% weight)
        max_score += 0.25;
        if texture.is_3d {
            score += 0.25 * texture.confidence;
        }
        
        // Movements (20% weight)
        max_score += 0.2;
        if natural_movements {
            score += 0.2 * movement_score;
        }
        
        // Temporal consistency (15% weight)
        max_score += 0.15;
        score += 0.15 * consistency.score;
        
        // Challenge bonus (if enabled)
        if let Some(passed) = challenge_passed {
            if passed {
                score += 0.1;
            } else {
                score -= 0.2;
            }
        }
        
        let confidence = (score / max_score).clamp(0.0, 1.0);
        let is_live = confidence > 0.65 && has_pulse;
        
        // Determine spoofing type if not live
        let spoofing_type = if !is_live {
            self.detect_spoofing_type(has_pulse, texture, natural_movements)
        } else {
            None
        };
        
        let reason = self.generate_reason(
            is_live, has_pulse, texture.is_3d, natural_movements, 
            consistency.score, challenge_passed
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
```

---

### 4.2 Texture Analyzer

**File:** `crates/zenb-verticals/src/liveness/texture_analyzer.rs`

```rust
//! Texture Analysis for 3D vs 2D Detection
//!
//! Detects photo/screen attacks by analyzing:
//! - Color distribution (screens have limited gamut)
//! - Frequency content (prints have halftone patterns)
//! - Specular highlights (3D faces have natural reflections)

/// Texture analysis result
#[derive(Debug, Clone, Default)]
pub struct TextureResult {
    /// Is 3D texture (real face)
    pub is_3d: bool,
    /// Confidence score
    pub confidence: f32,
    /// Color gamut score (0 = limited, 1 = full)
    pub color_gamut: f32,
    /// Frequency score (0 = halftone, 1 = natural)
    pub frequency_score: f32,
    /// Specular score (0 = matte, 1 = natural specular)
    pub specular_score: f32,
}

/// Texture Analyzer
pub struct TextureAnalyzer {
    min_samples: usize,
}

impl TextureAnalyzer {
    pub fn new() -> Self {
        Self { min_samples: 30 }
    }
    
    /// Analyze texture from RGB history
    pub fn analyze(&self, rgb_history: &[[f32; 3]]) -> TextureResult {
        if rgb_history.len() < self.min_samples {
            return TextureResult::default();
        }
        
        // 1. Color gamut analysis
        let color_gamut = self.analyze_color_gamut(rgb_history);
        
        // 2. Frequency analysis (simplified - check for periodic patterns)
        let frequency_score = self.analyze_frequency(rgb_history);
        
        // 3. Specular analysis (check for natural highlight variations)
        let specular_score = self.analyze_specular(rgb_history);
        
        // Combine scores
        let combined = color_gamut * 0.3 + frequency_score * 0.4 + specular_score * 0.3;
        let is_3d = combined > 0.6;
        
        TextureResult {
            is_3d,
            confidence: combined,
            color_gamut,
            frequency_score,
            specular_score,
        }
    }
    
    fn analyze_color_gamut(&self, rgb_history: &[[f32; 3]]) -> f32 {
        // Real faces have wider color distribution
        // Screens/prints have limited gamut
        
        let mut r_min = 255.0f32;
        let mut r_max = 0.0f32;
        let mut g_min = 255.0f32;
        let mut g_max = 0.0f32;
        let mut b_min = 255.0f32;
        let mut b_max = 0.0f32;
        
        for rgb in rgb_history {
            r_min = r_min.min(rgb[0]);
            r_max = r_max.max(rgb[0]);
            g_min = g_min.min(rgb[1]);
            g_max = g_max.max(rgb[1]);
            b_min = b_min.min(rgb[2]);
            b_max = b_max.max(rgb[2]);
        }
        
        let r_range = r_max - r_min;
        let g_range = g_max - g_min;
        let b_range = b_max - b_min;
        
        // Natural faces have ~10-30 range in each channel due to blood flow
        let expected_range = 20.0;
        let avg_range = (r_range + g_range + b_range) / 3.0;
        
        // Score based on how close to expected range
        let score = 1.0 - ((avg_range - expected_range).abs() / expected_range).min(1.0);
        score.clamp(0.0, 1.0)
    }
    
    fn analyze_frequency(&self, rgb_history: &[[f32; 3]]) -> f32 {
        // Check for periodic patterns (halftone, moire)
        // Real faces have smooth, non-periodic variations
        
        if rgb_history.len() < 10 {
            return 0.5;
        }
        
        // Compute autocorrelation of green channel
        let g: Vec<f32> = rgb_history.iter().map(|rgb| rgb[1]).collect();
        let mean: f32 = g.iter().sum::<f32>() / g.len() as f32;
        let centered: Vec<f32> = g.iter().map(|&v| v - mean).collect();
        
        // Check for periodic peaks in autocorrelation
        let mut max_corr = 0.0f32;
        for lag in 2..g.len().min(20) {
            let mut corr = 0.0;
            for i in 0..g.len() - lag {
                corr += centered[i] * centered[i + lag];
            }
            corr /= (g.len() - lag) as f32;
            max_corr = max_corr.max(corr.abs());
        }
        
        // High autocorrelation = periodic = likely screen/print
        let score = 1.0 - (max_corr / 100.0).min(1.0);
        score.clamp(0.0, 1.0)
    }
    
    fn analyze_specular(&self, rgb_history: &[[f32; 3]]) -> f32 {
        // Real faces have subtle specular variations
        // Matte prints/screens have uniform lighting
        
        // Check for occasional bright spots (specular highlights)
        let brightness: Vec<f32> = rgb_history
            .iter()
            .map(|rgb| (rgb[0] + rgb[1] + rgb[2]) / 3.0)
            .collect();
        
        let mean: f32 = brightness.iter().sum::<f32>() / brightness.len() as f32;
        let std: f32 = (brightness.iter()
            .map(|&b| (b - mean).powi(2))
            .sum::<f32>() / brightness.len() as f32)
            .sqrt();
        
        // Natural faces have std ~5-15
        let expected_std = 10.0;
        let score = 1.0 - ((std - expected_std).abs() / expected_std).min(1.0);
        score.clamp(0.0, 1.0)
    }
}

impl Default for TextureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 4.3 Challenge Response Generator

**File:** `crates/zenb-verticals/src/liveness/challenge_response.rs`

```rust
//! Challenge-Response Verification
//!
//! Generates random challenges (blink, turn head, smile)
//! and verifies user compliance.

use rand::Rng;

/// Challenge types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChallengeType {
    Blink,
    TurnLeft,
    TurnRight,
    Smile,
    OpenMouth,
    RaiseBrows,
}

/// Challenge state
#[derive(Debug, Clone)]
pub struct Challenge {
    pub challenge_type: ChallengeType,
    pub instruction: String,
    pub timeout_ms: u64,
    pub started_at_us: i64,
}

/// Challenge result
#[derive(Debug, Clone)]
pub struct ChallengeResult {
    pub passed: bool,
    pub challenge: ChallengeType,
    pub response_time_ms: f32,
}

/// Challenge Generator
pub struct ChallengeGenerator {
    current_challenge: Option<Challenge>,
    completed_challenges: Vec<ChallengeResult>,
    required_challenges: usize,
}

impl ChallengeGenerator {
    pub fn new() -> Self {
        Self {
            current_challenge: None,
            completed_challenges: Vec::new(),
            required_challenges: 2,
        }
    }
    
    /// Generate a new random challenge
    pub fn generate(&mut self, timestamp_us: i64) -> Challenge {
        let mut rng = rand::thread_rng();
        let challenge_type = match rng.gen_range(0..6) {
            0 => ChallengeType::Blink,
            1 => ChallengeType::TurnLeft,
            2 => ChallengeType::TurnRight,
            3 => ChallengeType::Smile,
            4 => ChallengeType::OpenMouth,
            _ => ChallengeType::RaiseBrows,
        };
        
        let instruction = match challenge_type {
            ChallengeType::Blink => "Please blink your eyes",
            ChallengeType::TurnLeft => "Please turn your head left",
            ChallengeType::TurnRight => "Please turn your head right",
            ChallengeType::Smile => "Please smile",
            ChallengeType::OpenMouth => "Please open your mouth",
            ChallengeType::RaiseBrows => "Please raise your eyebrows",
        }.to_string();
        
        let challenge = Challenge {
            challenge_type,
            instruction,
            timeout_ms: 5000,
            started_at_us: timestamp_us,
        };
        
        self.current_challenge = Some(challenge.clone());
        challenge
    }
    
    /// Verify if current challenge is completed
    pub fn verify(&mut self, landmarks: &[[f32; 2]]) -> bool {
        let challenge = match &self.current_challenge {
            Some(c) => c.clone(),
            None => return false,
        };
        
        let passed = match challenge.challenge_type {
            ChallengeType::Blink => self.detect_blink(landmarks),
            ChallengeType::TurnLeft => self.detect_head_turn(landmarks, -20.0),
            ChallengeType::TurnRight => self.detect_head_turn(landmarks, 20.0),
            ChallengeType::Smile => self.detect_smile(landmarks),
            ChallengeType::OpenMouth => self.detect_open_mouth(landmarks),
            ChallengeType::RaiseBrows => self.detect_raised_brows(landmarks),
        };
        
        if passed {
            self.completed_challenges.push(ChallengeResult {
                passed: true,
                challenge: challenge.challenge_type,
                response_time_ms: 0.0, // Would need timestamp
            });
            self.current_challenge = None;
        }
        
        passed
    }
    
    /// Check if all required challenges are completed
    pub fn is_complete(&self) -> bool {
        self.completed_challenges.len() >= self.required_challenges
    }
    
    fn detect_blink(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }
        
        // Simple EAR check
        let left_top = landmarks[159];
        let left_bottom = landmarks[145];
        let left_inner = landmarks[133];
        let left_outer = landmarks[33];
        
        let v = ((left_top[0] - left_bottom[0]).powi(2) + (left_top[1] - left_bottom[1]).powi(2)).sqrt();
        let h = ((left_inner[0] - left_outer[0]).powi(2) + (left_inner[1] - left_outer[1]).powi(2)).sqrt();
        
        let ear = v / h.max(0.001);
        ear < 0.15 // Eyes closed
    }
    
    fn detect_head_turn(&self, landmarks: &[[f32; 2]], expected_yaw: f32) -> bool {
        if landmarks.len() < 468 {
            return false;
        }
        
        let nose = landmarks[4];
        let left_cheek = landmarks[234];
        let right_cheek = landmarks[454];
        
        let left_dist = ((nose[0] - left_cheek[0]).powi(2) + (nose[1] - left_cheek[1]).powi(2)).sqrt();
        let right_dist = ((nose[0] - right_cheek[0]).powi(2) + (nose[1] - right_cheek[1]).powi(2)).sqrt();
        
        let asymmetry = (right_dist - left_dist) / (left_dist + right_dist).max(0.001);
        let estimated_yaw = asymmetry * 45.0;
        
        if expected_yaw < 0.0 {
            estimated_yaw < -15.0
        } else {
            estimated_yaw > 15.0
        }
    }
    
    fn detect_smile(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }
        
        let left_corner = landmarks[61];
        let right_corner = landmarks[291];
        let upper_lip = landmarks[0];
        let lower_lip = landmarks[17];
        
        let width = ((left_corner[0] - right_corner[0]).powi(2) + (left_corner[1] - right_corner[1]).powi(2)).sqrt();
        let height = ((upper_lip[0] - lower_lip[0]).powi(2) + (upper_lip[1] - lower_lip[1]).powi(2)).sqrt();
        
        // Smile = wide mouth, relatively closed
        width / height.max(0.001) > 3.0
    }
    
    fn detect_open_mouth(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }
        
        let upper_lip = landmarks[0];
        let lower_lip = landmarks[17];
        
        let height = ((upper_lip[0] - lower_lip[0]).powi(2) + (upper_lip[1] - lower_lip[1]).powi(2)).sqrt();
        height > 0.05 // Mouth open
    }
    
    fn detect_raised_brows(&self, landmarks: &[[f32; 2]]) -> bool {
        if landmarks.len() < 468 {
            return false;
        }
        
        let left_brow = landmarks[105];
        let left_eye = landmarks[159];
        let right_brow = landmarks[334];
        let right_eye = landmarks[386];
        
        let left_dist = (left_brow[1] - left_eye[1]).abs();
        let right_dist = (right_brow[1] - right_eye[1]).abs();
        
        (left_dist + right_dist) / 2.0 > 0.08 // Brows raised
    }
    
    pub fn reset(&mut self) {
        self.current_challenge = None;
        self.completed_challenges.clear();
    }
}

impl Default for ChallengeGenerator {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 4.4 Temporal Consistency Checker

**File:** `crates/zenb-verticals/src/liveness/temporal_consistency.rs`

```rust
//! Temporal Consistency Analysis
//!
//! Detects video replay attacks by analyzing:
//! - Landmark motion patterns
//! - Frame-to-frame consistency
//! - Periodic artifacts (video loops)

/// Consistency check result
#[derive(Debug, Clone, Default)]
pub struct ConsistencyResult {
    /// Overall consistency score (0-1)
    pub score: f32,
    /// Motion is natural (not looped)
    pub natural_motion: bool,
    /// No periodic patterns detected
    pub no_loops: bool,
    /// Frame transitions are smooth
    pub smooth_transitions: bool,
}

/// Temporal Consistency Checker
pub struct TemporalConsistencyChecker {
    motion_history: Vec<f32>,
    window_size: usize,
}

impl TemporalConsistencyChecker {
    pub fn new() -> Self {
        Self {
            motion_history: Vec::with_capacity(300),
            window_size: 90,
        }
    }
    
    /// Check temporal consistency from landmark history
    pub fn check(&mut self, landmark_history: &[Vec<[f32; 2]>]) -> ConsistencyResult {
        if landmark_history.len() < 10 {
            return ConsistencyResult::default();
        }
        
        // Compute frame-to-frame motion
        let mut motions = Vec::new();
        for i in 1..landmark_history.len() {
            let motion = self.compute_motion(&landmark_history[i-1], &landmark_history[i]);
            motions.push(motion);
        }
        
        self.motion_history.extend(motions.iter());
        if self.motion_history.len() > 300 {
            self.motion_history.drain(0..self.motion_history.len() - 300);
        }
        
        // 1. Check for natural motion (not too regular)
        let natural_motion = self.check_natural_motion(&motions);
        
        // 2. Check for loops (periodic patterns)
        let no_loops = self.check_no_loops();
        
        // 3. Check smooth transitions (no sudden jumps)
        let smooth_transitions = self.check_smooth_transitions(&motions);
        
        let score = (natural_motion as u8 as f32 * 0.4
            + no_loops as u8 as f32 * 0.3
            + smooth_transitions as u8 as f32 * 0.3)
            .clamp(0.0, 1.0);
        
        ConsistencyResult {
            score,
            natural_motion,
            no_loops,
            smooth_transitions,
        }
    }
    
    fn compute_motion(&self, prev: &[[f32; 2]], curr: &[[f32; 2]]) -> f32 {
        if prev.len() != curr.len() || prev.is_empty() {
            return 0.0;
        }
        
        let key_indices = [4, 133, 362, 152]; // nose, eyes, chin
        let mut total = 0.0;
        let mut count = 0;
        
        for &idx in &key_indices {
            if idx < prev.len() && idx < curr.len() {
                let dx = curr[idx][0] - prev[idx][0];
                let dy = curr[idx][1] - prev[idx][1];
                total += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }
        
        if count > 0 { total / count as f32 } else { 0.0 }
    }
    
    fn check_natural_motion(&self, motions: &[f32]) -> bool {
        if motions.len() < 5 {
            return true;
        }
        
        // Natural motion has some variance
        let mean: f32 = motions.iter().sum::<f32>() / motions.len() as f32;
        let variance: f32 = motions.iter()
            .map(|&m| (m - mean).powi(2))
            .sum::<f32>() / motions.len() as f32;
        
        // Too regular = suspicious
        variance > 0.00001
    }
    
    fn check_no_loops(&self) -> bool {
        if self.motion_history.len() < self.window_size {
            return true;
        }
        
        // Check autocorrelation for periodic patterns
        let recent = &self.motion_history[self.motion_history.len() - self.window_size..];
        let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        
        let mut max_corr = 0.0f32;
        for lag in 10..recent.len() / 2 {
            let mut corr = 0.0;
            for i in 0..recent.len() - lag {
                corr += (recent[i] - mean) * (recent[i + lag] - mean);
            }
            corr /= (recent.len() - lag) as f32;
            max_corr = max_corr.max(corr.abs());
        }
        
        // High correlation = loop detected
        max_corr < 0.5
    }
    
    fn check_smooth_transitions(&self, motions: &[f32]) -> bool {
        if motions.len() < 3 {
            return true;
        }
        
        // Check for sudden jumps
        for i in 1..motions.len() {
            let diff = (motions[i] - motions[i-1]).abs();
            if diff > 0.1 {
                return false; // Sudden jump detected
            }
        }
        
        true
    }
    
    pub fn reset(&mut self) {
        self.motion_history.clear();
    }
}

impl Default for TemporalConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 4.5 Liveness Module Entry Point

**File:** `crates/zenb-verticals/src/liveness/mod.rs`

```rust
//! Liveness Detection Module
//!
//! Multi-modal liveness detection for authentication.

pub mod detector;
pub mod texture_analyzer;
pub mod challenge_response;
pub mod temporal_consistency;

pub use detector::{
    LivenessDetector, LivenessConfig, LivenessResult, SpoofingType,
};
pub use texture_analyzer::{TextureAnalyzer, TextureResult};
pub use challenge_response::{ChallengeGenerator, Challenge, ChallengeType, ChallengeResult};
pub use temporal_consistency::{TemporalConsistencyChecker, ConsistencyResult};
```

---

*Tiếp tục trong PART4 - Driver Monitoring System...*
