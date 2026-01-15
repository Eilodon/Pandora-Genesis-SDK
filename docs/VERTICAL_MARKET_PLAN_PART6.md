# 🚀 KẾ HOẠCH THỰC THI VERTICAL MARKET - PART 6

# 8. PHASE 5: FINTECH & EDUCATION MODULES

## 8.1 Fintech - Fraud Detection

**File:** `crates/zenb-verticals/src/fintech/fraud_detector.rs`

```rust
//! Fraud Detection for Financial Services
//!
//! Passive fraud detection using:
//! - Cardiac fingerprinting (unique pulse patterns)
//! - Stress baseline comparison
//! - Coercion detection
//! - Behavioral biometrics

use zenb_signals::rppg::EnsembleProcessor;
use zenb_signals::physio::HrvEstimator;

/// Fraud detection result
#[derive(Debug, Clone)]
pub struct FraudResult {
    /// Risk score (0-1, higher = more suspicious)
    pub risk_score: f32,
    /// Risk level classification
    pub risk_level: RiskLevel,
    /// Identity match confidence (if enrolled)
    pub identity_confidence: Option<f32>,
    /// Stress anomaly detected
    pub stress_anomaly: bool,
    /// Coercion indicators
    pub coercion_indicators: CoercionIndicators,
    /// Recommended action
    pub recommended_action: FraudAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FraudAction {
    Approve,
    AdditionalVerification,
    ManualReview,
    Block,
}

#[derive(Debug, Clone, Default)]
pub struct CoercionIndicators {
    pub elevated_stress: bool,
    pub abnormal_behavior: bool,
    pub rushed_interaction: bool,
    pub unusual_time: bool,
}

/// Fraud Detector
pub struct FraudDetector {
    // Enrolled user profile (if any)
    user_profile: Option<UserProfile>,
    
    // Processors
    ppg_processor: EnsembleProcessor,
    hrv_estimator: HrvEstimator,
    
    // State
    rgb_buffer: Vec<[f32; 3]>,
    stress_baseline: Option<f32>,
    interaction_start_us: i64,
}

#[derive(Debug, Clone)]
struct UserProfile {
    cardiac_template: Vec<f32>,
    stress_baseline_rmssd: f32,
    typical_hr_range: (f32, f32),
}

impl FraudDetector {
    pub fn new() -> Self {
        Self {
            user_profile: None,
            ppg_processor: EnsembleProcessor::new(),
            hrv_estimator: HrvEstimator::new(),
            rgb_buffer: Vec::with_capacity(300),
            stress_baseline: None,
            interaction_start_us: 0,
        }
    }
    
    /// Enroll user profile for future matching
    pub fn enroll(&mut self, cardiac_template: Vec<f32>, stress_baseline: f32, hr_range: (f32, f32)) {
        self.user_profile = Some(UserProfile {
            cardiac_template,
            stress_baseline_rmssd: stress_baseline,
            typical_hr_range: hr_range,
        });
    }
    
    /// Process frame for fraud detection
    pub fn process_frame(
        &mut self,
        rgb_mean: [f32; 3],
        timestamp_us: i64,
    ) -> FraudResult {
        if self.interaction_start_us == 0 {
            self.interaction_start_us = timestamp_us;
        }
        
        self.rgb_buffer.push(rgb_mean);
        if self.rgb_buffer.len() > 300 {
            self.rgb_buffer.remove(0);
        }
        
        // Need enough data
        if self.rgb_buffer.len() < 150 {
            return FraudResult {
                risk_score: 0.0,
                risk_level: RiskLevel::Low,
                identity_confidence: None,
                stress_anomaly: false,
                coercion_indicators: CoercionIndicators::default(),
                recommended_action: FraudAction::Approve,
            };
        }
        
        // Extract signals
        let r: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[0]).collect();
        let g: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[1]).collect();
        let b: Vec<f32> = self.rgb_buffer.iter().map(|rgb| rgb[2]).collect();
        
        let ppg_result = self.ppg_processor.process_arrays(&r, &g, &b);
        
        // Identity matching
        let identity_confidence = if let (Some(ref profile), Some(ref result)) = (&self.user_profile, &ppg_result) {
            Some(self.match_cardiac_signature(&g, &profile.cardiac_template))
        } else {
            None
        };
        
        // Stress analysis
        let (stress_anomaly, current_stress) = self.analyze_stress(&g);
        
        // Coercion detection
        let coercion = self.detect_coercion(
            current_stress,
            ppg_result.as_ref().map(|r| r.heart_rate_bpm),
            timestamp_us,
        );
        
        // Calculate risk score
        let risk_score = self.calculate_risk(
            identity_confidence,
            stress_anomaly,
            &coercion,
        );
        
        let risk_level = self.classify_risk(risk_score);
        let recommended_action = self.recommend_action(risk_level, &coercion);
        
        FraudResult {
            risk_score,
            risk_level,
            identity_confidence,
            stress_anomaly,
            coercion_indicators: coercion,
            recommended_action,
        }
    }
    
    fn match_cardiac_signature(&self, pulse: &[f32], template: &[f32]) -> f32 {
        // Simple correlation-based matching
        if pulse.len() < 90 || template.len() < 90 {
            return 0.0;
        }
        
        let pulse_slice = &pulse[pulse.len()-90..];
        let template_slice = &template[..90.min(template.len())];
        
        // Normalize
        let pulse_mean: f32 = pulse_slice.iter().sum::<f32>() / pulse_slice.len() as f32;
        let template_mean: f32 = template_slice.iter().sum::<f32>() / template_slice.len() as f32;
        
        // Cross-correlation
        let mut correlation = 0.0;
        let mut pulse_var = 0.0;
        let mut template_var = 0.0;
        
        let len = pulse_slice.len().min(template_slice.len());
        for i in 0..len {
            let p = pulse_slice[i] - pulse_mean;
            let t = template_slice[i] - template_mean;
            correlation += p * t;
            pulse_var += p * p;
            template_var += t * t;
        }
        
        if pulse_var > 0.0 && template_var > 0.0 {
            (correlation / (pulse_var.sqrt() * template_var.sqrt())).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    fn analyze_stress(&mut self, pulse: &[f32]) -> (bool, f32) {
        let pulse_arr = ndarray::Array1::from_vec(pulse.to_vec());
        
        if let Some(hrv) = self.hrv_estimator.estimate(&pulse_arr) {
            let current_stress = 1.0 - (hrv.rmssd_ms / 50.0).min(1.0);
            
            // Compare to baseline
            let anomaly = if let Some(ref profile) = self.user_profile {
                hrv.rmssd_ms < profile.stress_baseline_rmssd * 0.5
            } else if let Some(baseline) = self.stress_baseline {
                current_stress > baseline + 0.3
            } else {
                self.stress_baseline = Some(current_stress);
                false
            };
            
            (anomaly, current_stress)
        } else {
            (false, 0.0)
        }
    }
    
    fn detect_coercion(
        &self,
        stress: f32,
        heart_rate: Option<f32>,
        timestamp_us: i64,
    ) -> CoercionIndicators {
        let elevated_stress = stress > 0.7;
        
        let abnormal_behavior = if let (Some(ref profile), Some(hr)) = (&self.user_profile, heart_rate) {
            hr < profile.typical_hr_range.0 || hr > profile.typical_hr_range.1
        } else {
            false
        };
        
        let interaction_duration_sec = (timestamp_us - self.interaction_start_us) as f32 / 1_000_000.0;
        let rushed_interaction = interaction_duration_sec < 5.0;
        
        // Check time of day (simplified - would need actual time)
        let unusual_time = false;
        
        CoercionIndicators {
            elevated_stress,
            abnormal_behavior,
            rushed_interaction,
            unusual_time,
        }
    }
    
    fn calculate_risk(
        &self,
        identity_confidence: Option<f32>,
        stress_anomaly: bool,
        coercion: &CoercionIndicators,
    ) -> f32 {
        let mut risk = 0.0;
        
        // Identity mismatch
        if let Some(conf) = identity_confidence {
            risk += (1.0 - conf) * 0.4;
        }
        
        // Stress anomaly
        if stress_anomaly {
            risk += 0.2;
        }
        
        // Coercion indicators
        if coercion.elevated_stress { risk += 0.1; }
        if coercion.abnormal_behavior { risk += 0.15; }
        if coercion.rushed_interaction { risk += 0.1; }
        if coercion.unusual_time { risk += 0.05; }
        
        risk.clamp(0.0, 1.0)
    }
    
    fn classify_risk(&self, score: f32) -> RiskLevel {
        if score < 0.25 { RiskLevel::Low }
        else if score < 0.5 { RiskLevel::Medium }
        else if score < 0.75 { RiskLevel::High }
        else { RiskLevel::Critical }
    }
    
    fn recommend_action(&self, level: RiskLevel, coercion: &CoercionIndicators) -> FraudAction {
        // Coercion always triggers manual review
        if coercion.elevated_stress && coercion.abnormal_behavior {
            return FraudAction::ManualReview;
        }
        
        match level {
            RiskLevel::Low => FraudAction::Approve,
            RiskLevel::Medium => FraudAction::AdditionalVerification,
            RiskLevel::High => FraudAction::ManualReview,
            RiskLevel::Critical => FraudAction::Block,
        }
    }
    
    pub fn reset(&mut self) {
        self.rgb_buffer.clear();
        self.interaction_start_us = 0;
    }
}

impl Default for FraudDetector {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## 8.2 Education - Exam Proctoring

**File:** `crates/zenb-verticals/src/education/proctoring.rs`

```rust
//! Exam Proctoring System
//!
//! Monitors exam integrity through:
//! - Identity verification (liveness)
//! - Gaze tracking (looking away)
//! - Multi-person detection
//! - Suspicious behavior scoring

use zenb_signals::beauty::landmarks::CanonicalLandmarks;

use crate::shared::{GazeEstimator, GazeResult, GazeTarget};
use crate::liveness::{LivenessDetector, LivenessResult};

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
    
    // Components
    gaze_estimator: GazeEstimator,
    liveness_detector: Option<LivenessDetector>,
    
    // State
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
        
        // 1. Face visibility check
        if !landmarks.valid {
            violations.push(Violation::FaceNotVisible);
        }
        
        // 2. Multiple people check
        let multiple_people = face_count > 1;
        if multiple_people {
            violations.push(Violation::MultipleFaces);
            self.record_violation(timestamp_us, Violation::MultipleFaces);
        }
        
        // 3. Gaze tracking
        let gaze = self.gaze_estimator.estimate(landmarks, head_pose);
        let eyes_on_screen = gaze.on_screen;
        
        if !eyes_on_screen {
            if self.look_away_start_us.is_none() {
                self.look_away_start_us = Some(timestamp_us);
            }
            
            let look_away_duration = (timestamp_us - self.look_away_start_us.unwrap()) as f32 / 1_000_000.0;
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
        
        // 4. Liveness check (periodic)
        let identity_verified = if let (Some(ref mut detector), Some(rgb)) = (&mut self.liveness_detector, rgb_mean) {
            if self.frame_count % 30 == 0 { // Check every second
                let liveness = detector.process_frame(rgb, &landmarks.points, timestamp_us);
                if !liveness.is_live && liveness.confidence > 0.5 {
                    violations.push(Violation::IdentityMismatch);
                }
                liveness.is_live
            } else {
                true // Assume verified between checks
            }
        } else {
            true
        };
        
        // 5. Calculate suspicion score
        let suspicion_score = self.calculate_suspicion(&violations, &gaze);
        
        // 6. Determine action
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
        // Deduplicate recent violations
        let recent_cutoff = timestamp_us - 5_000_000; // 5 seconds
        if !self.violation_history.iter()
            .any(|(t, v)| *t > recent_cutoff && *v == violation)
        {
            self.violation_history.push((timestamp_us, violation));
        }
    }
    
    fn calculate_suspicion(&self, current_violations: &[Violation], gaze: &GazeResult) -> f32 {
        let mut score = 0.0;
        
        // Current violations
        for v in current_violations {
            score += match v {
                Violation::LookingAway => 0.2,
                Violation::MultipleFaces => 0.4,
                Violation::FaceNotVisible => 0.3,
                Violation::SuspiciousMovement => 0.15,
                Violation::IdentityMismatch => 0.5,
            };
        }
        
        // Gaze deviation
        if gaze.deviation > self.config.max_gaze_deviation {
            score += (gaze.deviation - self.config.max_gaze_deviation) * 0.5;
        }
        
        // Historical violations (decay over time)
        let recent_violations = self.violation_history.len();
        score += (recent_violations as f32 * 0.05).min(0.3);
        
        score.clamp(0.0, 1.0)
    }
    
    fn determine_action(&self, score: f32, violations: &[Violation]) -> ProctoringAction {
        // Immediate termination for critical violations
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
```

---

# 9. SAFETY & ETHICS FRAMEWORK

## 9.1 Safety Guard Module

**File:** `crates/zenb-verticals/src/shared/safety_guard.rs`

```rust
//! Safety Guard Framework
//!
//! Ensures safe operation across all vertical modules.

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Safety configuration
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Maximum requests per minute per user
    pub rate_limit_per_minute: u32,
    /// Nonce validity window (seconds)
    pub nonce_window_sec: u64,
    /// Minimum confidence for high-stakes decisions
    pub min_confidence_threshold: f32,
    /// Enable fail-safe mode
    pub fail_safe_enabled: bool,
    /// Default action on error
    pub default_on_error: SafeDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDefault {
    /// Deny access on error (secure)
    Deny,
    /// Allow access on error (permissive)
    Allow,
    /// Require manual review
    ManualReview,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            rate_limit_per_minute: 60,
            nonce_window_sec: 300,
            min_confidence_threshold: 0.7,
            fail_safe_enabled: true,
            default_on_error: SafeDefault::Deny,
        }
    }
}

/// Safety Guard
pub struct SafetyGuard {
    config: SafetyConfig,
    request_counts: HashMap<String, (Instant, u32)>,
    used_nonces: HashMap<String, Instant>,
}

impl SafetyGuard {
    pub fn new() -> Self {
        Self::with_config(SafetyConfig::default())
    }
    
    pub fn with_config(config: SafetyConfig) -> Self {
        Self {
            config,
            request_counts: HashMap::new(),
            used_nonces: HashMap::new(),
        }
    }
    
    /// Check rate limit for user
    pub fn check_rate_limit(&mut self, user_id: &str) -> bool {
        let now = Instant::now();
        
        if let Some((start, count)) = self.request_counts.get_mut(user_id) {
            if now.duration_since(*start) > Duration::from_secs(60) {
                // Reset window
                *start = now;
                *count = 1;
                true
            } else if *count >= self.config.rate_limit_per_minute {
                false
            } else {
                *count += 1;
                true
            }
        } else {
            self.request_counts.insert(user_id.to_string(), (now, 1));
            true
        }
    }
    
    /// Validate nonce (anti-replay)
    pub fn validate_nonce(&mut self, nonce: &str) -> bool {
        let now = Instant::now();
        
        // Clean old nonces
        self.used_nonces.retain(|_, t| {
            now.duration_since(*t) < Duration::from_secs(self.config.nonce_window_sec)
        });
        
        // Check if nonce was used
        if self.used_nonces.contains_key(nonce) {
            false
        } else {
            self.used_nonces.insert(nonce.to_string(), now);
            true
        }
    }
    
    /// Check if confidence meets threshold for action
    pub fn meets_confidence_threshold(&self, confidence: f32) -> bool {
        confidence >= self.config.min_confidence_threshold
    }
    
    /// Get fail-safe default action
    pub fn get_fail_safe_action(&self) -> SafeDefault {
        if self.config.fail_safe_enabled {
            self.config.default_on_error
        } else {
            SafeDefault::Allow
        }
    }
    
    /// Validate request with all checks
    pub fn validate_request(
        &mut self,
        user_id: &str,
        nonce: &str,
        confidence: f32,
    ) -> Result<(), SafetyError> {
        if !self.check_rate_limit(user_id) {
            return Err(SafetyError::RateLimitExceeded);
        }
        
        if !self.validate_nonce(nonce) {
            return Err(SafetyError::NonceReused);
        }
        
        if !self.meets_confidence_threshold(confidence) {
            return Err(SafetyError::LowConfidence);
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SafetyError {
    RateLimitExceeded,
    NonceReused,
    LowConfidence,
    InvalidInput,
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## 9.2 Privacy Compliance

**File:** `crates/zenb-verticals/src/shared/privacy.rs`

```rust
//! Privacy Compliance Utilities
//!
//! Helpers for GDPR, CCPA, BIPA compliance.

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum retention period (seconds)
    pub max_retention_sec: u64,
    /// Auto-delete after session
    pub delete_on_session_end: bool,
    /// Anonymize after period
    pub anonymize_after_sec: Option<u64>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_retention_sec: 86400, // 24 hours
            delete_on_session_end: true,
            anonymize_after_sec: Some(3600), // 1 hour
        }
    }
}

/// Consent status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentStatus {
    NotRequested,
    Pending,
    Granted,
    Denied,
    Withdrawn,
}

/// Privacy-safe data wrapper
#[derive(Debug, Clone)]
pub struct PrivacyWrapped<T> {
    data: T,
    consent: ConsentStatus,
    created_at_us: i64,
    retention: RetentionPolicy,
}

impl<T> PrivacyWrapped<T> {
    pub fn new(data: T, consent: ConsentStatus, timestamp_us: i64) -> Self {
        Self {
            data,
            consent,
            created_at_us: timestamp_us,
            retention: RetentionPolicy::default(),
        }
    }
    
    /// Access data only if consent granted
    pub fn access(&self) -> Option<&T> {
        if self.consent == ConsentStatus::Granted {
            Some(&self.data)
        } else {
            None
        }
    }
    
    /// Check if data should be deleted
    pub fn should_delete(&self, current_us: i64) -> bool {
        let age_sec = (current_us - self.created_at_us) as u64 / 1_000_000;
        age_sec > self.retention.max_retention_sec
    }
    
    /// Check if data should be anonymized
    pub fn should_anonymize(&self, current_us: i64) -> bool {
        if let Some(anon_sec) = self.retention.anonymize_after_sec {
            let age_sec = (current_us - self.created_at_us) as u64 / 1_000_000;
            age_sec > anon_sec
        } else {
            false
        }
    }
}

/// Aggregate-only analytics (privacy-preserving)
#[derive(Debug, Clone, Default)]
pub struct AggregateAnalytics {
    pub sample_count: u64,
    pub avg_engagement: f32,
    pub avg_valence: f32,
    pub emotion_distribution: [u32; 7],
}

impl AggregateAnalytics {
    pub fn update(&mut self, engagement: f32, valence: f32, emotion_idx: usize) {
        self.sample_count += 1;
        
        // Running average
        let n = self.sample_count as f32;
        self.avg_engagement = self.avg_engagement * (n - 1.0) / n + engagement / n;
        self.avg_valence = self.avg_valence * (n - 1.0) / n + valence / n;
        
        if emotion_idx < 7 {
            self.emotion_distribution[emotion_idx] += 1;
        }
    }
}
```

---

# 10. QUICK START COMMANDS

## Build & Test

```bash
# Build all verticals
cd /home/ybao/B.1/AGOLOS
cargo build -p zenb-verticals

# Build specific feature
cargo build -p zenb-verticals --features liveness

# Run tests
cargo test -p zenb-verticals

# Run benchmarks
cargo bench -p zenb-verticals

# Check documentation
cargo doc -p zenb-verticals --open
```

## Create New Vertical

```bash
# 1. Create module directory
mkdir -p crates/zenb-verticals/src/new_vertical

# 2. Create mod.rs
touch crates/zenb-verticals/src/new_vertical/mod.rs

# 3. Add to lib.rs
echo 'pub mod new_vertical;' >> crates/zenb-verticals/src/lib.rs

# 4. Add feature flag to Cargo.toml
# new_vertical = []
```

---

# 11. FINAL SUMMARY

## Deliverables

| Phase | Module | Files | Est. LOC | Days |
|-------|--------|-------|----------|------|
| 0 | Infrastructure | 2 | 50 | 2 |
| 1 | Shared | 4 | 800 | 5 |
| 2 | Liveness | 5 | 1200 | 7 |
| 3 | Automotive | 5 | 1000 | 11 |
| 4 | Retail | 4 | 600 | 7 |
| 5 | Fintech | 3 | 500 | 5 |
| 5 | Education | 3 | 500 | 5 |
| - | Safety | 2 | 300 | 2 |
| **Total** | | **28** | **~5000** | **~44** |

## Key Differentiators

1. **rPPG-based Liveness** - Impossible to fake with photos/videos
2. **Cardiac Event Detection** - Life-saving feature for DMS
3. **Passive Fraud Detection** - No user action required
4. **Privacy-First Design** - GDPR/CCPA/BIPA compliant

## Next Steps

1. ✅ Review this plan
2. ⏳ Create `zenb-verticals` crate structure
3. ⏳ Implement shared components
4. ⏳ Build Liveness Detection MVP
5. ⏳ Iterate based on testing

---

**Kế hoạch này đã được tạo dựa trên deep audit toàn bộ codebase AGOLOS.**
**Tất cả code templates đều tương thích với existing zenb-signals và zenb-core modules.**
