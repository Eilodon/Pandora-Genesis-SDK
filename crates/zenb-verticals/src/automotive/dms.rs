//! Driver Monitoring System Core
//!
//! Comprehensive driver state monitoring combining:
//! - Drowsiness detection (PERCLOS, blink rate)
//! - Distraction detection (gaze tracking)
//! - Stress monitoring (HRV analysis)
//! - Cardiac event detection (emergency)

use ndarray::Array1;
use zenb_signals::beauty::CanonicalLandmarks;
use zenb_signals::physio::{HrvConfig, HrvEstimator, HrvResult, RespirationConfig, RespirationEstimator};
use zenb_signals::rppg::{EnsembleConfig, EnsembleProcessor};

use crate::shared::{
    EarResult, EyeMetricsAnalyzer, EyeMetricsConfig, GazeEstimator, GazeTarget,
};

use super::cardiac_monitor::{CardiacAlert, CardiacMonitor};
use super::distraction::{DistractionDetector, DistractionLevel, DistractionResult};
use super::drowsiness::{DrowsinessDetector, DrowsinessLevel, DrowsinessResult};

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
    frame_count: usize,
    session_start_us: i64,
}

impl DriverMonitoringSystem {
    pub fn new() -> Self {
        Self::with_config(DmsConfig::default())
    }

    pub fn with_config(config: DmsConfig) -> Self {
        let sample_rate = config.sample_rate.max(1.0);

        let mut eye_config = EyeMetricsConfig::default();
        eye_config.perclos_threshold = config.perclos_threshold;
        eye_config.sample_rate = sample_rate;

        let mut ensemble_cfg = EnsembleConfig::default();
        ensemble_cfg.sample_rate = sample_rate;
        ensemble_cfg.window_size = (sample_rate * 3.0).round().max(30.0) as usize;

        let mut hrv_cfg = HrvConfig::default();
        hrv_cfg.sample_rate = sample_rate;

        let mut resp_cfg = RespirationConfig::default();
        resp_cfg.sample_rate = sample_rate;

        let mut cardiac_monitor = if config.enable_cardiac_monitor {
            Some(CardiacMonitor::new())
        } else {
            None
        };
        if let Some(ref mut monitor) = cardiac_monitor {
            monitor.set_hr_change_threshold(config.cardiac_hr_change_threshold);
        }

        Self {
            eye_metrics: EyeMetricsAnalyzer::with_config(eye_config),
            gaze_estimator: GazeEstimator::new(),
            ppg_processor: EnsembleProcessor::with_config(ensemble_cfg),
            hrv_estimator: HrvEstimator::with_config(hrv_cfg),
            respiration_estimator: RespirationEstimator::with_config(resp_cfg),
            drowsiness_detector: DrowsinessDetector::new(),
            distraction_detector: DistractionDetector::with_threshold(
                config.gaze_off_road_threshold_sec,
            ),
            cardiac_monitor,
            config,
            rgb_buffer: Vec::with_capacity(300),
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
            eyes_on_road: gaze.on_screen,
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

        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();

        let r_arr = Array1::from_vec(r);
        let g_arr = Array1::from_vec(g);
        let b_arr = Array1::from_vec(b);

        let ppg_result = self.ppg_processor.process_arrays(&r_arr, &g_arr, &b_arr);

        let heart_rate = ppg_result.as_ref().map(|r| r.bpm);

        let hrv = if ppg_result.is_some() {
            let hrv_result = self.hrv_estimator.estimate(&g_arr);
            if hrv_result.metrics.is_some() {
                Some(hrv_result)
            } else {
                None
            }
        } else {
            None
        };

        let respiration = if self.rgb_buffer.len() >= 200 {
            self.respiration_estimator
                .estimate(&g_arr)
                .map(|r| r.brpm)
        } else {
            None
        };

        let stress_level = if let Some(ref h) = hrv {
            if let Some(metrics) = h.metrics.as_ref() {
                let rmssd = metrics.rmssd_ms;
                if rmssd < self.config.stress_hrv_threshold_ms {
                    1.0 - (rmssd / self.config.stress_hrv_threshold_ms)
                } else {
                    0.0
                }
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
        drowsiness: &DrowsinessResult,
        distraction: &DistractionResult,
        stress_level: f32,
        cardiac_alert: &Option<CardiacAlert>,
    ) -> DriverAction {
        if let Some(alert) = cardiac_alert {
            if alert.is_emergency {
                return DriverAction::Emergency;
            }
        }

        let drowsy_action = match drowsiness.classification {
            DrowsinessLevel::Alert => DriverAction::Normal,
            DrowsinessLevel::Mild => DriverAction::SuggestBreak,
            DrowsinessLevel::Moderate => DriverAction::TakeBreak,
            DrowsinessLevel::Severe => DriverAction::PullOver,
        };

        let distract_action = match distraction.classification {
            DistractionLevel::Focused => DriverAction::Normal,
            DistractionLevel::Mild => DriverAction::Normal,
            DistractionLevel::Moderate => DriverAction::SuggestBreak,
            DistractionLevel::Severe => DriverAction::TakeBreak,
        };

        let stress_action = if stress_level > 0.8 {
            DriverAction::SuggestBreak
        } else {
            DriverAction::Normal
        };

        [drowsy_action, distract_action, stress_action]
            .into_iter()
            .max_by_key(|a| a.priority())
            .unwrap_or(DriverAction::Normal)
    }

    fn calculate_confidence(&self, landmarks: &CanonicalLandmarks, ear: &EarResult) -> f32 {
        let mut confidence = 1.0;

        if !landmarks.valid {
            confidence *= 0.3;
        }

        if ear.is_closed {
            confidence *= 0.7;
        }

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
        self.ppg_processor.reset();
        self.rgb_buffer.clear();
        self.frame_count = 0;
        self.session_start_us = 0;
    }
}

impl Default for DriverMonitoringSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dms_config_defaults() {
        let config = DmsConfig::default();
        assert!((config.sample_rate - 30.0).abs() < 0.01);
        assert!((config.perclos_threshold - 0.4).abs() < 0.01);
        assert!((config.gaze_off_road_threshold_sec - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_driver_action_priority() {
        assert!(DriverAction::Emergency.priority() > DriverAction::PullOver.priority());
        assert!(DriverAction::PullOver.priority() > DriverAction::TakeBreak.priority());
        assert!(DriverAction::TakeBreak.priority() > DriverAction::SuggestBreak.priority());
        assert!(DriverAction::SuggestBreak.priority() > DriverAction::Normal.priority());
    }

    #[test]
    fn test_driver_action_messages() {
        assert!(!DriverAction::Normal.message().is_empty());
        assert!(!DriverAction::Emergency.message().is_empty());
        assert!(DriverAction::Emergency.message().contains("EMERGENCY"));
    }

    #[test]
    fn test_driver_state_default() {
        let state = DriverState::default();
        assert_eq!(state.drowsiness, DrowsinessLevel::Alert);
        assert_eq!(state.distraction, DistractionLevel::Focused);
        assert_eq!(state.recommended_action, DriverAction::Normal);
        assert!(state.eyes_on_road);
    }

    #[test]
    fn test_dms_creation() {
        let dms = DriverMonitoringSystem::new();
        assert_eq!(dms.frame_count, 0);
        assert!(dms.rgb_buffer.is_empty());
    }

    #[test]
    fn test_dms_reset() {
        let mut dms = DriverMonitoringSystem::new();
        dms.frame_count = 100;
        dms.rgb_buffer.push([1.0, 1.0, 1.0]);
        
        dms.reset();
        
        assert_eq!(dms.frame_count, 0);
        assert!(dms.rgb_buffer.is_empty());
    }
}
