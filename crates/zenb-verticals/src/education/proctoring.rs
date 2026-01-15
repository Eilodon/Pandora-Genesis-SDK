//! Exam Proctoring System
//!
//! Monitors exam integrity through:
//! - Identity verification (liveness)
//! - Gaze tracking (looking away)
//! - Multi-person detection
//! - Suspicious behavior scoring

use zenb_signals::beauty::CanonicalLandmarks;

use crate::liveness::LivenessDetector;
use crate::shared::{GazeEstimator, GazeResult};

/// Proctoring result
#[derive(Debug, Clone)]
pub struct ProctoringResult {
    /// Identity verified (liveness passed)
    pub identity_verified: bool,
    /// Is looking at screen
    pub eyes_on_screen: bool,
    /// Gaze deviation from center
    pub gaze_deviation: f32,
    /// Multiple people detected
    pub multiple_people: bool,
    /// Suspicious behavior score (0-1)
    pub suspicion_score: f32,
    /// Specific violations detected
    pub violations: Vec<Violation>,
    /// Recommended action
    pub action: ProctoringAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    LookingAway,
    MultipleFaces,
    FaceNotVisible,
    SuspiciousMovement,
    IdentityMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProctoringAction {
    Continue,
    Warning,
    FlagForReview,
    Terminate,
}

/// Proctoring configuration
#[derive(Debug, Clone)]
pub struct ProctoringConfig {
    /// Maximum gaze deviation before warning
    pub max_gaze_deviation: f32,
    /// Maximum time looking away (seconds)
    pub max_look_away_sec: f32,
    /// Enable liveness checks
    pub enable_liveness: bool,
    /// Warning threshold for suspicion score
    pub warning_threshold: f32,
    /// Terminate threshold
    pub terminate_threshold: f32,
}

impl Default for ProctoringConfig {
    fn default() -> Self {
        Self {
            max_gaze_deviation: 0.3,
            max_look_away_sec: 3.0,
            enable_liveness: true,
            warning_threshold: 0.5,
            terminate_threshold: 0.9,
        }
    }
}

/// Exam Proctoring System
pub struct ExamProctoring {
    config: ProctoringConfig,

    gaze_estimator: GazeEstimator,
    liveness_detector: Option<LivenessDetector>,

    look_away_start_us: Option<i64>,
    total_look_away_us: i64,
    violation_history: Vec<(i64, Violation)>,
    frame_count: usize,
    session_start_us: i64,
}

impl ExamProctoring {
    pub fn new() -> Self {
        Self::with_config(ProctoringConfig::default())
    }

    pub fn with_config(config: ProctoringConfig) -> Self {
        Self {
            gaze_estimator: GazeEstimator::new(),
            liveness_detector: if config.enable_liveness {
                Some(LivenessDetector::new())
            } else {
                None
            },
            config,
            look_away_start_us: None,
            total_look_away_us: 0,
            violation_history: Vec::new(),
            frame_count: 0,
            session_start_us: 0,
        }
    }

    /// Process frame for proctoring
    pub fn process_frame(
        &mut self,
        landmarks: &CanonicalLandmarks,
        rgb_mean: Option<[f32; 3]>,
        head_pose: Option<[f32; 3]>,
        face_count: usize,
        timestamp_us: i64,
    ) -> ProctoringResult {
        if self.frame_count == 0 {
            self.session_start_us = timestamp_us;
        }
        self.frame_count += 1;

        let mut violations = Vec::new();

        if !landmarks.valid {
            violations.push(Violation::FaceNotVisible);
        }

        let multiple_people = face_count > 1;
        if multiple_people {
            violations.push(Violation::MultipleFaces);
            self.record_violation(timestamp_us, Violation::MultipleFaces);
        }

        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);
        let eyes_on_screen = gaze.on_screen;

        if !eyes_on_screen {
            if self.look_away_start_us.is_none() {
                self.look_away_start_us = Some(timestamp_us);
            }

            let look_away_duration =
                (timestamp_us - self.look_away_start_us.unwrap()) as f32 / 1_000_000.0;
            if look_away_duration > self.config.max_look_away_sec {
                violations.push(Violation::LookingAway);
                self.record_violation(timestamp_us, Violation::LookingAway);
            }
        } else {
            if let Some(start) = self.look_away_start_us {
                self.total_look_away_us += timestamp_us - start;
            }
            self.look_away_start_us = None;
        }

        let identity_verified = if let (Some(ref mut detector), Some(rgb)) =
            (&mut self.liveness_detector, rgb_mean)
        {
            if self.frame_count % 30 == 0 {
                let liveness = detector.process_frame(rgb, &landmarks.points, timestamp_us);
                if !liveness.is_live && liveness.confidence > 0.5 {
                    violations.push(Violation::IdentityMismatch);
                }
                liveness.is_live
            } else {
                true
            }
        } else {
            true
        };

        let suspicion_score = self.calculate_suspicion(&violations, &gaze);
        let action = self.determine_action(suspicion_score, &violations);

        ProctoringResult {
            identity_verified,
            eyes_on_screen,
            gaze_deviation: gaze.deviation,
            multiple_people,
            suspicion_score,
            violations,
            action,
        }
    }

    fn record_violation(&mut self, timestamp_us: i64, violation: Violation) {
        let recent_cutoff = timestamp_us - 5_000_000;
        if !self
            .violation_history
            .iter()
            .any(|(t, v)| *t > recent_cutoff && *v == violation)
        {
            self.violation_history.push((timestamp_us, violation));
        }
    }

    fn calculate_suspicion(&self, current_violations: &[Violation], gaze: &GazeResult) -> f32 {
        let mut score = 0.0;

        for v in current_violations {
            score += match v {
                Violation::LookingAway => 0.2,
                Violation::MultipleFaces => 0.4,
                Violation::FaceNotVisible => 0.3,
                Violation::SuspiciousMovement => 0.15,
                Violation::IdentityMismatch => 0.5,
            };
        }

        if gaze.deviation > self.config.max_gaze_deviation {
            score += (gaze.deviation - self.config.max_gaze_deviation) * 0.5;
        }

        let recent_violations = self.violation_history.len();
        score += (recent_violations as f32 * 0.05).min(0.3);

        score.clamp(0.0, 1.0)
    }

    fn determine_action(&self, score: f32, violations: &[Violation]) -> ProctoringAction {
        if violations.contains(&Violation::IdentityMismatch) {
            return ProctoringAction::Terminate;
        }

        if score >= self.config.terminate_threshold {
            ProctoringAction::Terminate
        } else if score >= self.config.warning_threshold {
            ProctoringAction::FlagForReview
        } else if !violations.is_empty() {
            ProctoringAction::Warning
        } else {
            ProctoringAction::Continue
        }
    }

    /// Get session statistics
    pub fn get_session_stats(&self, current_us: i64) -> SessionStats {
        let duration_sec = (current_us - self.session_start_us) as f32 / 1_000_000.0;
        let look_away_percent = if duration_sec > 0.0 {
            (self.total_look_away_us as f32 / 1_000_000.0) / duration_sec * 100.0
        } else {
            0.0
        };

        SessionStats {
            duration_sec,
            look_away_percent,
            violation_count: self.violation_history.len(),
        }
    }

    pub fn reset(&mut self) {
        self.gaze_estimator.reset();
        if let Some(ref mut detector) = self.liveness_detector {
            detector.reset();
        }
        self.look_away_start_us = None;
        self.total_look_away_us = 0;
        self.violation_history.clear();
        self.frame_count = 0;
        self.session_start_us = 0;
    }
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub duration_sec: f32,
    pub look_away_percent: f32,
    pub violation_count: usize,
}

impl Default for ExamProctoring {
    fn default() -> Self {
        Self::new()
    }
}
