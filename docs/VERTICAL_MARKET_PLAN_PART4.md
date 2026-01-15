# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET - PART 4

# 5. PHASE 3: DRIVER MONITORING SYSTEM

## Timeline: Day 15-25 (11 days)

### 5.1 DMS Core

**File:** `crates/zenb-verticals/src/automotive/dms.rs`

```rust
//! Driver Monitoring System Core
//!
//! Comprehensive driver state monitoring combining:
//! - Drowsiness detection (PERCLOS, blink rate)
//! - Distraction detection (gaze tracking)
//! - Stress monitoring (HRV analysis)
//! - Cardiac event detection (emergency)

use zenb_signals::physio::{HrvEstimator, HrvResult, RespirationEstimator, RespirationResult};
use zenb_signals::rppg::EnsembleProcessor;
use zenb_signals::dsp::MotionDetector;
use zenb_signals::beauty::landmarks::CanonicalLandmarks;

use crate::shared::{
    EyeMetricsAnalyzer, EarResult, BlinkResult, PerclosResult,
    GazeEstimator, GazeResult, GazeTarget,
};

use super::drowsiness::{DrowsinessDetector, DrowsinessLevel};
use super::distraction::{DistractionDetector, DistractionLevel};
use super::cardiac_monitor::{CardiacMonitor, CardiacAlert};

/// Recommended driver action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriverAction {
    /// Normal driving, no intervention needed
    Normal,
    /// Suggest taking a break soon
    SuggestBreak,
    /// Warning: take a break now
    TakeBreak,
    /// Critical: pull over immediately
    PullOver,
    /// Emergency: cardiac event detected
    Emergency,
}

impl DriverAction {
    pub fn priority(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::SuggestBreak => 1,
            Self::TakeBreak => 2,
            Self::PullOver => 3,
            Self::Emergency => 4,
        }
    }
    
    pub fn message(&self) -> &'static str {
        match self {
            Self::Normal => "Drive safely",
            Self::SuggestBreak => "Consider taking a break soon",
            Self::TakeBreak => "Please take a break",
            Self::PullOver => "Pull over when safe",
            Self::Emergency => "EMERGENCY: Seek medical attention",
        }
    }
}

/// Driver state snapshot
#[derive(Debug, Clone)]
pub struct DriverState {
    /// Drowsiness level (0-1)
    pub drowsiness_level: f32,
    /// Drowsiness classification
    pub drowsiness: DrowsinessLevel,
    
    /// Distraction level (0-1)
    pub distraction_level: f32,
    /// Distraction classification
    pub distraction: DistractionLevel,
    
    /// Stress level (0-1)
    pub stress_level: f32,
    
    /// Current heart rate (if available)
    pub heart_rate_bpm: Option<f32>,
    /// HRV metrics (if available)
    pub hrv: Option<HrvResult>,
    /// Respiration rate (if available)
    pub respiration_brpm: Option<f32>,
    
    /// Cardiac alert (if any)
    pub cardiac_alert: Option<CardiacAlert>,
    
    /// Gaze target
    pub gaze_target: GazeTarget,
    /// Is looking at road
    pub eyes_on_road: bool,
    
    /// Blink rate (blinks per minute)
    pub blink_rate: f32,
    /// PERCLOS value
    pub perclos: f32,
    
    /// Recommended action
    pub recommended_action: DriverAction,
    
    /// Confidence in assessment
    pub confidence: f32,
    
    /// Timestamp
    pub timestamp_us: i64,
}

impl Default for DriverState {
    fn default() -> Self {
        Self {
            drowsiness_level: 0.0,
            drowsiness: DrowsinessLevel::Alert,
            distraction_level: 0.0,
            distraction: DistractionLevel::Focused,
            stress_level: 0.0,
            heart_rate_bpm: None,
            hrv: None,
            respiration_brpm: None,
            cardiac_alert: None,
            gaze_target: GazeTarget::Screen,
            eyes_on_road: true,
            blink_rate: 15.0,
            perclos: 0.0,
            recommended_action: DriverAction::Normal,
            confidence: 0.0,
            timestamp_us: 0,
        }
    }
}

/// DMS Configuration
#[derive(Debug, Clone)]
pub struct DmsConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// PERCLOS threshold for drowsiness
    pub perclos_threshold: f32,
    /// Gaze off-road time threshold (seconds)
    pub gaze_off_road_threshold_sec: f32,
    /// Stress HRV threshold (RMSSD below this = stressed)
    pub stress_hrv_threshold_ms: f32,
    /// Enable cardiac monitoring
    pub enable_cardiac_monitor: bool,
    /// Cardiac anomaly HR threshold (sudden change)
    pub cardiac_hr_change_threshold: f32,
}

impl Default for DmsConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            perclos_threshold: 0.4,
            gaze_off_road_threshold_sec: 2.0,
            stress_hrv_threshold_ms: 20.0,
            enable_cardiac_monitor: true,
            cardiac_hr_change_threshold: 30.0,
        }
    }
}

/// Driver Monitoring System
pub struct DriverMonitoringSystem {
    config: DmsConfig,
    
    // Shared components
    eye_metrics: EyeMetricsAnalyzer,
    gaze_estimator: GazeEstimator,
    
    // Physio processors (REUSE from zenb-signals)
    ppg_processor: EnsembleProcessor,
    hrv_estimator: HrvEstimator,
    respiration_estimator: RespirationEstimator,
    
    // DMS-specific
    drowsiness_detector: DrowsinessDetector,
    distraction_detector: DistractionDetector,
    cardiac_monitor: Option<CardiacMonitor>,
    
    // State
    rgb_buffer: Vec<[f32; 3]>,
    pulse_buffer: Vec<f32>,
    frame_count: usize,
    session_start_us: i64,
}

impl DriverMonitoringSystem {
    pub fn new() -> Self {
        Self::with_config(DmsConfig::default())
    }
    
    pub fn with_config(config: DmsConfig) -> Self {
        Self {
            eye_metrics: EyeMetricsAnalyzer::new(),
            gaze_estimator: GazeEstimator::new(),
            ppg_processor: EnsembleProcessor::new(),
            hrv_estimator: HrvEstimator::new(),
            respiration_estimator: RespirationEstimator::new(),
            drowsiness_detector: DrowsinessDetector::new(),
            distraction_detector: DistractionDetector::with_threshold(
                config.gaze_off_road_threshold_sec
            ),
            cardiac_monitor: if config.enable_cardiac_monitor {
                Some(CardiacMonitor::new())
            } else {
                None
            },
            config,
            rgb_buffer: Vec::with_capacity(300),
            pulse_buffer: Vec::with_capacity(300),
            frame_count: 0,
            session_start_us: 0,
        }
    }
    
    /// Process a single frame
    pub fn process_frame(
        &mut self,
        rgb_mean: [f32; 3],
        landmarks: &CanonicalLandmarks,
        head_pose: Option<[f32; 3]>,
        timestamp_us: i64,
    ) -> DriverState {
        if self.frame_count == 0 {
            self.session_start_us = timestamp_us;
        }
        self.frame_count += 1;
        
        // Update buffers
        self.rgb_buffer.push(rgb_mean);
        if self.rgb_buffer.len() > 300 {
            self.rgb_buffer.remove(0);
        }
        
        // 1. Eye metrics (EAR, blink, PERCLOS)
        let (ear, blink, perclos) = self.eye_metrics.analyze(landmarks, timestamp_us);
        
        // 2. Gaze estimation
        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);
        
        // 3. Drowsiness detection
        let drowsiness = self.drowsiness_detector.update(&perclos, &blink);
        
        // 4. Distraction detection
        let distraction = self.distraction_detector.update(&gaze, timestamp_us);
        
        // 5. Physiological signals (if enough data)
        let (heart_rate, hrv, respiration, stress_level) = self.process_physio();
        
        // 6. Cardiac monitoring
        let cardiac_alert = if let Some(ref mut monitor) = self.cardiac_monitor {
            monitor.check(heart_rate, &hrv)
        } else {
            None
        };
        
        // 7. Determine recommended action
        let recommended_action = self.determine_action(
            &drowsiness,
            &distraction,
            stress_level,
            &cardiac_alert,
        );
        
        // 8. Calculate confidence
        let confidence = self.calculate_confidence(landmarks, &ear);
        
        DriverState {
            drowsiness_level: drowsiness.level,
            drowsiness: drowsiness.classification,
            distraction_level: distraction.level,
            distraction: distraction.classification,
            stress_level,
            heart_rate_bpm: heart_rate,
            hrv,
            respiration_brpm: respiration,
            cardiac_alert,
            gaze_target: gaze.target,
            eyes_on_road: gaze.on_screen, // Screen = road view
            blink_rate: blink.blinks_per_minute,
            perclos: perclos.perclos,
            recommended_action,
            confidence,
            timestamp_us,
        }
    }
    
    fn process_physio(&mut self) -> (Option<f32>, Option<HrvResult>, Option<f32>, f32) {
        if self.rgb_buffer.len() < 150 {
            return (None, None, None, 0.0);
        }
        
        // Extract pulse signal
        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();
        
        let ppg_result = self.ppg_processor.process_arrays(&r, &g, &b);
        
        let heart_rate = ppg_result.as_ref().map(|r| r.heart_rate_bpm);
        
        // HRV from pulse waveform
        let hrv = if let Some(ref result) = ppg_result {
            // Use green channel as pulse proxy
            let pulse = ndarray::Array1::from_vec(g.clone());
            self.hrv_estimator.estimate(&pulse)
        } else {
            None
        };
        
        // Respiration
        let respiration = if self.rgb_buffer.len() >= 200 {
            let pulse = ndarray::Array1::from_vec(g);
            self.respiration_estimator.estimate(&pulse).map(|r| r.brpm)
        } else {
            None
        };
        
        // Stress level from HRV
        let stress_level = if let Some(ref h) = hrv {
            // Low RMSSD = high stress
            let rmssd = h.rmssd_ms;
            if rmssd < self.config.stress_hrv_threshold_ms {
                1.0 - (rmssd / self.config.stress_hrv_threshold_ms)
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        (heart_rate, hrv, respiration, stress_level.clamp(0.0, 1.0))
    }
    
    fn determine_action(
        &self,
        drowsiness: &super::drowsiness::DrowsinessResult,
        distraction: &super::distraction::DistractionResult,
        stress_level: f32,
        cardiac_alert: &Option<CardiacAlert>,
    ) -> DriverAction {
        // Priority: Emergency > PullOver > TakeBreak > SuggestBreak > Normal
        
        // Check cardiac emergency first
        if let Some(alert) = cardiac_alert {
            if alert.is_emergency {
                return DriverAction::Emergency;
            }
        }
        
        // Check drowsiness
        let drowsy_action = match drowsiness.classification {
            DrowsinessLevel::Alert => DriverAction::Normal,
            DrowsinessLevel::Mild => DriverAction::SuggestBreak,
            DrowsinessLevel::Moderate => DriverAction::TakeBreak,
            DrowsinessLevel::Severe => DriverAction::PullOver,
        };
        
        // Check distraction
        let distract_action = match distraction.classification {
            DistractionLevel::Focused => DriverAction::Normal,
            DistractionLevel::Mild => DriverAction::Normal,
            DistractionLevel::Moderate => DriverAction::SuggestBreak,
            DistractionLevel::Severe => DriverAction::TakeBreak,
        };
        
        // Check stress
        let stress_action = if stress_level > 0.8 {
            DriverAction::SuggestBreak
        } else {
            DriverAction::Normal
        };
        
        // Return highest priority action
        [drowsy_action, distract_action, stress_action]
            .into_iter()
            .max_by_key(|a| a.priority())
            .unwrap_or(DriverAction::Normal)
    }
    
    fn calculate_confidence(&self, landmarks: &CanonicalLandmarks, ear: &EarResult) -> f32 {
        let mut confidence = 1.0;
        
        // Reduce confidence if landmarks invalid
        if !landmarks.valid {
            confidence *= 0.3;
        }
        
        // Reduce confidence if eyes closed (can't track gaze)
        if ear.is_closed {
            confidence *= 0.7;
        }
        
        // Reduce confidence if not enough data
        if self.rgb_buffer.len() < 90 {
            confidence *= self.rgb_buffer.len() as f32 / 90.0;
        }
        
        confidence.clamp(0.0, 1.0)
    }
    
    /// Get driving session duration
    pub fn session_duration_sec(&self, current_us: i64) -> f32 {
        (current_us - self.session_start_us) as f32 / 1_000_000.0
    }
    
    /// Reset DMS state
    pub fn reset(&mut self) {
        self.eye_metrics.reset();
        self.gaze_estimator.reset();
        self.drowsiness_detector.reset();
        self.distraction_detector.reset();
        if let Some(ref mut monitor) = self.cardiac_monitor {
            monitor.reset();
        }
        self.rgb_buffer.clear();
        self.pulse_buffer.clear();
        self.frame_count = 0;
    }
}

impl Default for DriverMonitoringSystem {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.2 Drowsiness Detector

**File:** `crates/zenb-verticals/src/automotive/drowsiness.rs`

```rust
//! Drowsiness Detection
//!
//! Combines multiple indicators:
//! - PERCLOS (primary)
//! - Blink rate changes
//! - Yawn detection (future)

use crate::shared::{PerclosResult, BlinkResult};

/// Drowsiness level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrowsinessLevel {
    /// Fully alert
    Alert,
    /// Slightly drowsy
    Mild,
    /// Moderately drowsy
    Moderate,
    /// Severely drowsy
    Severe,
}

/// Drowsiness detection result
#[derive(Debug, Clone)]
pub struct DrowsinessResult {
    /// Drowsiness level (0-1)
    pub level: f32,
    /// Classification
    pub classification: DrowsinessLevel,
    /// PERCLOS contribution
    pub perclos_score: f32,
    /// Blink rate contribution
    pub blink_score: f32,
}

/// Drowsiness Detector
pub struct DrowsinessDetector {
    // Thresholds
    perclos_mild: f32,
    perclos_moderate: f32,
    perclos_severe: f32,
    
    // Blink rate baseline
    baseline_blink_rate: Option<f32>,
    blink_history: Vec<f32>,
    
    // Smoothing
    smoothed_level: f32,
    alpha: f32,
}

impl DrowsinessDetector {
    pub fn new() -> Self {
        Self {
            perclos_mild: 0.15,
            perclos_moderate: 0.30,
            perclos_severe: 0.50,
            baseline_blink_rate: None,
            blink_history: Vec::with_capacity(60),
            smoothed_level: 0.0,
            alpha: 0.1,
        }
    }
    
    /// Update drowsiness detection
    pub fn update(&mut self, perclos: &PerclosResult, blink: &BlinkResult) -> DrowsinessResult {
        // Update blink baseline
        self.blink_history.push(blink.blinks_per_minute);
        if self.blink_history.len() > 60 {
            self.blink_history.remove(0);
        }
        
        if self.baseline_blink_rate.is_none() && self.blink_history.len() >= 30 {
            self.baseline_blink_rate = Some(
                self.blink_history.iter().sum::<f32>() / self.blink_history.len() as f32
            );
        }
        
        // PERCLOS score (primary indicator)
        let perclos_score = perclos.perclos;
        
        // Blink rate score (deviation from baseline)
        let blink_score = if let Some(baseline) = self.baseline_blink_rate {
            let deviation = (blink.blinks_per_minute - baseline).abs() / baseline.max(1.0);
            // Both too high and too low blink rates indicate drowsiness
            if blink.blinks_per_minute < baseline * 0.5 {
                0.5 // Very low blink rate = drowsy
            } else if blink.blinks_per_minute > baseline * 1.5 {
                0.3 // High blink rate = fighting drowsiness
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Combined level
        let raw_level = perclos_score * 0.7 + blink_score * 0.3;
        
        // Smooth
        self.smoothed_level = self.smoothed_level * (1.0 - self.alpha) + raw_level * self.alpha;
        
        // Classify
        let classification = if self.smoothed_level < self.perclos_mild {
            DrowsinessLevel::Alert
        } else if self.smoothed_level < self.perclos_moderate {
            DrowsinessLevel::Mild
        } else if self.smoothed_level < self.perclos_severe {
            DrowsinessLevel::Moderate
        } else {
            DrowsinessLevel::Severe
        };
        
        DrowsinessResult {
            level: self.smoothed_level,
            classification,
            perclos_score,
            blink_score,
        }
    }
    
    pub fn reset(&mut self) {
        self.baseline_blink_rate = None;
        self.blink_history.clear();
        self.smoothed_level = 0.0;
    }
}

impl Default for DrowsinessDetector {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.3 Distraction Detector

**File:** `crates/zenb-verticals/src/automotive/distraction.rs`

```rust
//! Distraction Detection
//!
//! Monitors driver attention based on gaze direction.

use crate::shared::{GazeResult, GazeTarget};

/// Distraction level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistractionLevel {
    /// Focused on road
    Focused,
    /// Brief glances away
    Mild,
    /// Extended looks away
    Moderate,
    /// Severely distracted
    Severe,
}

/// Distraction detection result
#[derive(Debug, Clone)]
pub struct DistractionResult {
    /// Distraction level (0-1)
    pub level: f32,
    /// Classification
    pub classification: DistractionLevel,
    /// Time looking away (seconds)
    pub off_road_duration_sec: f32,
    /// Current gaze target
    pub gaze_target: GazeTarget,
}

/// Distraction Detector
pub struct DistractionDetector {
    /// Time threshold for distraction warning (seconds)
    off_road_threshold_sec: f32,
    
    /// State
    off_road_start_us: Option<i64>,
    total_off_road_us: i64,
    window_start_us: i64,
    
    /// Smoothing
    smoothed_level: f32,
    alpha: f32,
}

impl DistractionDetector {
    pub fn new() -> Self {
        Self::with_threshold(2.0)
    }
    
    pub fn with_threshold(threshold_sec: f32) -> Self {
        Self {
            off_road_threshold_sec: threshold_sec,
            off_road_start_us: None,
            total_off_road_us: 0,
            window_start_us: 0,
            smoothed_level: 0.0,
            alpha: 0.1,
        }
    }
    
    /// Update distraction detection
    pub fn update(&mut self, gaze: &GazeResult, timestamp_us: i64) -> DistractionResult {
        // Initialize window
        if self.window_start_us == 0 {
            self.window_start_us = timestamp_us;
        }
        
        // Track off-road gaze
        if !gaze.on_screen {
            if self.off_road_start_us.is_none() {
                self.off_road_start_us = Some(timestamp_us);
            }
        } else {
            if let Some(start) = self.off_road_start_us {
                self.total_off_road_us += timestamp_us - start;
            }
            self.off_road_start_us = None;
        }
        
        // Current off-road duration
        let current_off_road_us = if let Some(start) = self.off_road_start_us {
            timestamp_us - start
        } else {
            0
        };
        
        let off_road_duration_sec = current_off_road_us as f32 / 1_000_000.0;
        
        // Calculate distraction level
        let raw_level = (off_road_duration_sec / self.off_road_threshold_sec).min(1.0);
        
        // Smooth
        self.smoothed_level = self.smoothed_level * (1.0 - self.alpha) + raw_level * self.alpha;
        
        // Classify
        let classification = if self.smoothed_level < 0.25 {
            DistractionLevel::Focused
        } else if self.smoothed_level < 0.5 {
            DistractionLevel::Mild
        } else if self.smoothed_level < 0.75 {
            DistractionLevel::Moderate
        } else {
            DistractionLevel::Severe
        };
        
        // Reset window periodically (every 30 seconds)
        if timestamp_us - self.window_start_us > 30_000_000 {
            self.total_off_road_us = 0;
            self.window_start_us = timestamp_us;
        }
        
        DistractionResult {
            level: self.smoothed_level,
            classification,
            off_road_duration_sec,
            gaze_target: gaze.target,
        }
    }
    
    pub fn reset(&mut self) {
        self.off_road_start_us = None;
        self.total_off_road_us = 0;
        self.window_start_us = 0;
        self.smoothed_level = 0.0;
    }
}

impl Default for DistractionDetector {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.4 Cardiac Monitor

**File:** `crates/zenb-verticals/src/automotive/cardiac_monitor.rs`

```rust
//! Cardiac Event Monitor
//!
//! Detects potential cardiac emergencies from HR/HRV patterns.

use zenb_signals::physio::HrvResult;

/// Cardiac alert
#[derive(Debug, Clone)]
pub struct CardiacAlert {
    /// Is this an emergency
    pub is_emergency: bool,
    /// Alert type
    pub alert_type: CardiacAlertType,
    /// Message
    pub message: String,
    /// Confidence
    pub confidence: f32,
}

/// Types of cardiac alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardiacAlertType {
    /// Sudden HR drop
    Bradycardia,
    /// Sudden HR spike
    Tachycardia,
    /// Irregular rhythm
    Arrhythmia,
    /// Very low HRV
    LowHrv,
}

/// Cardiac Monitor
pub struct CardiacMonitor {
    hr_history: Vec<f32>,
    hrv_history: Vec<f32>,
    
    // Thresholds
    bradycardia_threshold: f32,
    tachycardia_threshold: f32,
    hr_change_threshold: f32,
    low_hrv_threshold: f32,
}

impl CardiacMonitor {
    pub fn new() -> Self {
        Self {
            hr_history: Vec::with_capacity(60),
            hrv_history: Vec::with_capacity(60),
            bradycardia_threshold: 50.0,
            tachycardia_threshold: 120.0,
            hr_change_threshold: 30.0,
            low_hrv_threshold: 10.0,
        }
    }
    
    /// Check for cardiac anomalies
    pub fn check(
        &mut self,
        heart_rate: Option<f32>,
        hrv: &Option<HrvResult>,
    ) -> Option<CardiacAlert> {
        let hr = heart_rate?;
        
        // Update history
        self.hr_history.push(hr);
        if self.hr_history.len() > 60 {
            self.hr_history.remove(0);
        }
        
        if let Some(h) = hrv {
            self.hrv_history.push(h.rmssd_ms);
            if self.hrv_history.len() > 60 {
                self.hrv_history.remove(0);
            }
        }
        
        // Check for bradycardia
        if hr < self.bradycardia_threshold {
            return Some(CardiacAlert {
                is_emergency: hr < 40.0,
                alert_type: CardiacAlertType::Bradycardia,
                message: format!("Low heart rate: {:.0} BPM", hr),
                confidence: 0.8,
            });
        }
        
        // Check for tachycardia
        if hr > self.tachycardia_threshold {
            return Some(CardiacAlert {
                is_emergency: hr > 150.0,
                alert_type: CardiacAlertType::Tachycardia,
                message: format!("High heart rate: {:.0} BPM", hr),
                confidence: 0.8,
            });
        }
        
        // Check for sudden HR change
        if self.hr_history.len() >= 10 {
            let recent_avg: f32 = self.hr_history[self.hr_history.len()-5..].iter().sum::<f32>() / 5.0;
            let older_avg: f32 = self.hr_history[self.hr_history.len()-10..self.hr_history.len()-5].iter().sum::<f32>() / 5.0;
            
            let change = (recent_avg - older_avg).abs();
            if change > self.hr_change_threshold {
                return Some(CardiacAlert {
                    is_emergency: change > 50.0,
                    alert_type: CardiacAlertType::Arrhythmia,
                    message: format!("Sudden HR change: {:.0} BPM", change),
                    confidence: 0.7,
                });
            }
        }
        
        // Check for very low HRV
        if let Some(h) = hrv {
            if h.rmssd_ms < self.low_hrv_threshold {
                return Some(CardiacAlert {
                    is_emergency: false,
                    alert_type: CardiacAlertType::LowHrv,
                    message: format!("Very low HRV: {:.1} ms", h.rmssd_ms),
                    confidence: 0.6,
                });
            }
        }
        
        None
    }
    
    pub fn reset(&mut self) {
        self.hr_history.clear();
        self.hrv_history.clear();
    }
}

impl Default for CardiacMonitor {
    fn default() -> Self {
        Self::new()
    }
}
```

---

### 5.5 Automotive Module Entry Point

**File:** `crates/zenb-verticals/src/automotive/mod.rs`

```rust
//! Automotive Driver Monitoring System
//!
//! Comprehensive driver state monitoring for vehicle safety.

pub mod dms;
pub mod drowsiness;
pub mod distraction;
pub mod cardiac_monitor;

pub use dms::{DriverMonitoringSystem, DmsConfig, DriverState, DriverAction};
pub use drowsiness::{DrowsinessDetector, DrowsinessLevel, DrowsinessResult};
pub use distraction::{DistractionDetector, DistractionLevel, DistractionResult};
pub use cardiac_monitor::{CardiacMonitor, CardiacAlert, CardiacAlertType};
```

---

*Tiếp tục trong PART5 - Retail Analytics & Implementation Timeline...*
