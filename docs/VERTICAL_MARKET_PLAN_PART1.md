# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET EXPANSION
## AGOLOS Platform - Super Detailed Execution Guide

> **Tài liệu này được tạo dựa trên deep audit toàn bộ codebase**
> **Ngày tạo:** 2026-01-16
> **Phiên bản:** 1.0

---

# 📋 MỤC LỤC

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Phase 0: Chuẩn bị cơ sở hạ tầng](#2-phase-0-chuẩn-bị-cơ-sở-hạ-tầng)
3. [Phase 1: Shared Components](#3-phase-1-shared-components)
4. [Phase 2: Liveness Detection](#4-phase-2-liveness-detection)
5. [Phase 3: Driver Monitoring System](#5-phase-3-driver-monitoring-system)
6. [Phase 4: Retail Analytics](#6-phase-4-retail-analytics)
7. [Phase 5: Fintech & Education](#7-phase-5-fintech--education)

---

# 1. TỔNG QUAN KIẾN TRÚC

## 1.1 Cấu trúc thư mục mới

```
crates/
├── zenb-signals/           # ✅ EXISTING - Core signal processing
├── zenb-core/              # ✅ EXISTING - Engine & state management
├── zenb-uniffi/            # ✅ EXISTING - FFI bindings
├── zenb-verticals/         # 🆕 NEW - Vertical market modules
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── shared/         # Shared components
│   │   │   ├── mod.rs
│   │   │   ├── eye_metrics.rs
│   │   │   ├── gaze_estimator.rs
│   │   │   ├── micro_expression.rs
│   │   │   ├── attention_tracker.rs
│   │   │   └── stress_profiler.rs
│   │   ├── liveness/       # Liveness Detection
│   │   │   ├── mod.rs
│   │   │   ├── detector.rs
│   │   │   ├── texture_analyzer.rs
│   │   │   ├── challenge_response.rs
│   │   │   └── temporal_consistency.rs
│   │   ├── automotive/     # Driver Monitoring
│   │   │   ├── mod.rs
│   │   │   ├── dms.rs
│   │   │   ├── drowsiness.rs
│   │   │   ├── distraction.rs
│   │   │   └── cardiac_monitor.rs
│   │   ├── retail/         # Retail Analytics
│   │   │   ├── mod.rs
│   │   │   ├── emotion_analytics.rs
│   │   │   ├── engagement.rs
│   │   │   └── timeline.rs
│   │   ├── fintech/        # Fraud Detection
│   │   │   ├── mod.rs
│   │   │   ├── fraud_detector.rs
│   │   │   ├── cardiac_fingerprint.rs
│   │   │   └── coercion_detector.rs
│   │   └── education/      # Exam Proctoring
│   │       ├── mod.rs
│   │       ├── proctoring.rs
│   │       ├── gaze_tracker.rs
│   │       └── behavior_scorer.rs
│   └── tests/
└── zenb-verticals-uniffi/  # 🆕 NEW - FFI for verticals
```

## 1.2 Dependency Graph

```
zenb-verticals
    ├── zenb-signals (rPPG, DSP, Vision, Physio)
    ├── zenb-core (Engine, UKF, PhilosophicalState)
    └── ndarray, nalgebra (math)

zenb-verticals-uniffi
    ├── zenb-verticals
    └── uniffi
```

---

# 2. PHASE 0: CHUẨN BỊ CƠ SỞ HẠ TẦNG

## Timeline: Day 1-2

### Task 0.1: Tạo crate zenb-verticals

**File:** `crates/zenb-verticals/Cargo.toml`

```toml
[package]
name = "zenb-verticals"
version = "0.1.0"
edition = "2021"
description = "Vertical market modules for AGOLOS platform"

[dependencies]
# Internal dependencies
zenb-signals = { path = "../zenb-signals" }
zenb-core = { path = "../zenb-core" }

# Math
ndarray = "0.15"
nalgebra = "0.32"

# Serialization
serde = { version = "1.0", features = ["derive"] }

# Logging
log = "0.4"

[dev-dependencies]
approx = "0.5"

[features]
default = ["liveness", "automotive", "retail"]
liveness = []
automotive = []
retail = []
fintech = []
education = []
```

### Task 0.2: Tạo lib.rs entry point

**File:** `crates/zenb-verticals/src/lib.rs`

```rust
//! Vertical Market Modules for AGOLOS Platform
//!
//! # Modules
//!
//! - `shared`: Common components reused across verticals
//! - `liveness`: Liveness detection for authentication
//! - `automotive`: Driver monitoring system
//! - `retail`: Emotion analytics for retail
//! - `fintech`: Fraud detection
//! - `education`: Exam proctoring

pub mod shared;

#[cfg(feature = "liveness")]
pub mod liveness;

#[cfg(feature = "automotive")]
pub mod automotive;

#[cfg(feature = "retail")]
pub mod retail;

#[cfg(feature = "fintech")]
pub mod fintech;

#[cfg(feature = "education")]
pub mod education;

// Re-exports for convenience
pub use shared::*;
```

### Task 0.3: Update workspace Cargo.toml

**Edit:** `Cargo.toml` (root)

```toml
[workspace]
members = [
    "crates/zenb-core",
    "crates/zenb-signals",
    "crates/zenb-uniffi",
    "crates/zenb-verticals",  # ADD THIS
    # ... other members
]
```

---

# 3. PHASE 1: SHARED COMPONENTS

## Timeline: Day 3-7 (5 days)

### 3.1 Eye Metrics Analyzer

**File:** `crates/zenb-verticals/src/shared/eye_metrics.rs`

**Mục đích:** Tính toán các chỉ số mắt từ 468 landmarks

**Reuse từ existing code:**
- `zenb-signals::beauty::landmarks::indices` - Eye landmark indices
- `zenb-signals::beauty::landmarks::landmark_distance` - Distance calculation

```rust
//! Eye Metrics Analysis
//!
//! Computes eye-related metrics from MediaPipe 468 landmarks:
//! - Eye Aspect Ratio (EAR) for blink detection
//! - PERCLOS for drowsiness
//! - Blink rate and duration

use zenb_signals::beauty::landmarks::{CanonicalLandmarks, landmark_distance};

/// MediaPipe eye landmark indices
pub mod eye_indices {
    // Left eye contour (6 points for EAR)
    pub const LEFT_EYE_TOP_1: usize = 159;
    pub const LEFT_EYE_TOP_2: usize = 158;
    pub const LEFT_EYE_BOTTOM_1: usize = 145;
    pub const LEFT_EYE_BOTTOM_2: usize = 153;
    pub const LEFT_EYE_INNER: usize = 133;
    pub const LEFT_EYE_OUTER: usize = 33;
    
    // Right eye contour
    pub const RIGHT_EYE_TOP_1: usize = 386;
    pub const RIGHT_EYE_TOP_2: usize = 385;
    pub const RIGHT_EYE_BOTTOM_1: usize = 374;
    pub const RIGHT_EYE_BOTTOM_2: usize = 380;
    pub const RIGHT_EYE_INNER: usize = 362;
    pub const RIGHT_EYE_OUTER: usize = 263;
}

/// Eye Aspect Ratio result
#[derive(Debug, Clone, Default)]
pub struct EarResult {
    /// Left eye EAR (0 = closed, ~0.3 = open)
    pub left_ear: f32,
    /// Right eye EAR
    pub right_ear: f32,
    /// Average EAR
    pub avg_ear: f32,
    /// Is eye closed (EAR < threshold)
    pub is_closed: bool,
}

/// Blink detection result
#[derive(Debug, Clone, Default)]
pub struct BlinkResult {
    /// Blink detected in this frame
    pub blink_detected: bool,
    /// Blink duration in ms (if blink just ended)
    pub blink_duration_ms: Option<f32>,
    /// Blinks per minute (rolling average)
    pub blinks_per_minute: f32,
}

/// PERCLOS result (Percentage of Eye Closure)
#[derive(Debug, Clone, Default)]
pub struct PerclosResult {
    /// PERCLOS value (0-1, >0.4 = drowsy)
    pub perclos: f32,
    /// Drowsiness level (0-1)
    pub drowsiness_level: f32,
    /// Is drowsy flag
    pub is_drowsy: bool,
}

/// Eye Metrics Analyzer configuration
#[derive(Debug, Clone)]
pub struct EyeMetricsConfig {
    /// EAR threshold for closed eye
    pub ear_threshold: f32,
    /// PERCLOS window in seconds
    pub perclos_window_sec: f32,
    /// PERCLOS threshold for drowsiness
    pub perclos_threshold: f32,
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Minimum blink duration in ms
    pub min_blink_duration_ms: f32,
    /// Maximum blink duration in ms
    pub max_blink_duration_ms: f32,
}

impl Default for EyeMetricsConfig {
    fn default() -> Self {
        Self {
            ear_threshold: 0.21,
            perclos_window_sec: 60.0,
            perclos_threshold: 0.4,
            sample_rate: 30.0,
            min_blink_duration_ms: 100.0,
            max_blink_duration_ms: 400.0,
        }
    }
}

/// Eye Metrics Analyzer
pub struct EyeMetricsAnalyzer {
    config: EyeMetricsConfig,
    // Blink detection state
    eye_was_closed: bool,
    close_start_frame: usize,
    current_frame: usize,
    blink_history: Vec<i64>, // timestamps of blinks
    // PERCLOS state
    ear_history: Vec<f32>,
}

impl EyeMetricsAnalyzer {
    pub fn new() -> Self {
        Self::with_config(EyeMetricsConfig::default())
    }
    
    pub fn with_config(config: EyeMetricsConfig) -> Self {
        let window_frames = (config.perclos_window_sec * config.sample_rate) as usize;
        Self {
            config,
            eye_was_closed: false,
            close_start_frame: 0,
            current_frame: 0,
            blink_history: Vec::with_capacity(100),
            ear_history: Vec::with_capacity(window_frames),
        }
    }
    
    /// Compute Eye Aspect Ratio from landmarks
    pub fn compute_ear(&self, landmarks: &CanonicalLandmarks) -> EarResult {
        if !landmarks.valid || landmarks.points.len() < 468 {
            return EarResult::default();
        }
        
        let left_ear = self.compute_single_ear(
            landmarks,
            eye_indices::LEFT_EYE_TOP_1,
            eye_indices::LEFT_EYE_TOP_2,
            eye_indices::LEFT_EYE_BOTTOM_1,
            eye_indices::LEFT_EYE_BOTTOM_2,
            eye_indices::LEFT_EYE_INNER,
            eye_indices::LEFT_EYE_OUTER,
        );
        
        let right_ear = self.compute_single_ear(
            landmarks,
            eye_indices::RIGHT_EYE_TOP_1,
            eye_indices::RIGHT_EYE_TOP_2,
            eye_indices::RIGHT_EYE_BOTTOM_1,
            eye_indices::RIGHT_EYE_BOTTOM_2,
            eye_indices::RIGHT_EYE_INNER,
            eye_indices::RIGHT_EYE_OUTER,
        );
        
        let avg_ear = (left_ear + right_ear) / 2.0;
        let is_closed = avg_ear < self.config.ear_threshold;
        
        EarResult {
            left_ear,
            right_ear,
            avg_ear,
            is_closed,
        }
    }
    
    fn compute_single_ear(
        &self,
        landmarks: &CanonicalLandmarks,
        top1: usize,
        top2: usize,
        bottom1: usize,
        bottom2: usize,
        inner: usize,
        outer: usize,
    ) -> f32 {
        // EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        // Where p1=outer, p2=top1, p3=top2, p4=inner, p5=bottom2, p6=bottom1
        
        let v1 = landmark_distance(landmarks, top1, bottom1);
        let v2 = landmark_distance(landmarks, top2, bottom2);
        let h = landmark_distance(landmarks, inner, outer);
        
        if h < 0.001 {
            return 0.0;
        }
        
        (v1 + v2) / (2.0 * h)
    }
    
    /// Update blink detection with new EAR
    pub fn update_blink(&mut self, ear: &EarResult, timestamp_us: i64) -> BlinkResult {
        self.current_frame += 1;
        
        let mut result = BlinkResult::default();
        
        // Detect blink transition
        if ear.is_closed && !self.eye_was_closed {
            // Eye just closed
            self.close_start_frame = self.current_frame;
        } else if !ear.is_closed && self.eye_was_closed {
            // Eye just opened - blink ended
            let duration_frames = self.current_frame - self.close_start_frame;
            let duration_ms = duration_frames as f32 / self.config.sample_rate * 1000.0;
            
            // Validate blink duration
            if duration_ms >= self.config.min_blink_duration_ms
                && duration_ms <= self.config.max_blink_duration_ms
            {
                result.blink_detected = true;
                result.blink_duration_ms = Some(duration_ms);
                self.blink_history.push(timestamp_us);
                
                // Clean old blinks (keep last 60 seconds)
                let cutoff = timestamp_us - 60_000_000;
                self.blink_history.retain(|&t| t > cutoff);
            }
        }
        
        self.eye_was_closed = ear.is_closed;
        
        // Calculate blinks per minute
        if !self.blink_history.is_empty() {
            let window_sec = 60.0f32.min(
                (timestamp_us - self.blink_history[0]) as f32 / 1_000_000.0
            ).max(1.0);
            result.blinks_per_minute = self.blink_history.len() as f32 / window_sec * 60.0;
        }
        
        result
    }
    
    /// Update PERCLOS calculation
    pub fn update_perclos(&mut self, ear: &EarResult) -> PerclosResult {
        let window_frames = (self.config.perclos_window_sec * self.config.sample_rate) as usize;
        
        // Add to history
        self.ear_history.push(ear.avg_ear);
        if self.ear_history.len() > window_frames {
            self.ear_history.remove(0);
        }
        
        if self.ear_history.len() < 30 {
            return PerclosResult::default();
        }
        
        // Calculate PERCLOS (percentage of frames with closed eyes)
        let closed_count = self.ear_history
            .iter()
            .filter(|&&e| e < self.config.ear_threshold)
            .count();
        
        let perclos = closed_count as f32 / self.ear_history.len() as f32;
        
        // Map PERCLOS to drowsiness level (0-1)
        let drowsiness_level = if perclos < 0.15 {
            0.0
        } else if perclos < self.config.perclos_threshold {
            (perclos - 0.15) / (self.config.perclos_threshold - 0.15) * 0.5
        } else {
            0.5 + (perclos - self.config.perclos_threshold) / (1.0 - self.config.perclos_threshold) * 0.5
        };
        
        PerclosResult {
            perclos,
            drowsiness_level: drowsiness_level.clamp(0.0, 1.0),
            is_drowsy: perclos > self.config.perclos_threshold,
        }
    }
    
    /// Full analysis pipeline
    pub fn analyze(
        &mut self,
        landmarks: &CanonicalLandmarks,
        timestamp_us: i64,
    ) -> (EarResult, BlinkResult, PerclosResult) {
        let ear = self.compute_ear(landmarks);
        let blink = self.update_blink(&ear, timestamp_us);
        let perclos = self.update_perclos(&ear);
        (ear, blink, perclos)
    }
    
    /// Reset analyzer state
    pub fn reset(&mut self) {
        self.eye_was_closed = false;
        self.close_start_frame = 0;
        self.current_frame = 0;
        self.blink_history.clear();
        self.ear_history.clear();
    }
}

impl Default for EyeMetricsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ear_threshold() {
        let config = EyeMetricsConfig::default();
        assert!((config.ear_threshold - 0.21).abs() < 0.01);
    }
    
    #[test]
    fn test_perclos_threshold() {
        let config = EyeMetricsConfig::default();
        assert!((config.perclos_threshold - 0.4).abs() < 0.01);
    }
}
```

---

*Tiếp tục trong PART2...*
