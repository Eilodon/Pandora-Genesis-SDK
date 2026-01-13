use blake3::Hasher;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::config::ZenbConfig;

// ============================================================================
// PR4: STRICT TIME HELPERS — Prevent Wraparound
// ============================================================================

/// PR4: Compute time delta with saturating subtraction to prevent wraparound.
/// If clocks go backwards (now < last), returns 0 instead of wrapping to huge value.
///
/// # Arguments
/// * `now_us` - Current timestamp in microseconds
/// * `last_us` - Previous timestamp in microseconds
///
/// # Returns
/// Elapsed time in microseconds, guaranteed non-negative (0 if clocks went backwards)
#[inline]
pub fn dt_us(now_us: i64, last_us: i64) -> u64 {
    if now_us >= last_us {
        (now_us - last_us) as u64
    } else {
        // Clock went backwards - return 0 instead of wrapping
        0
    }
}

/// PR4: Compute time delta in seconds with saturating subtraction.
/// Convenience wrapper around dt_us for floating-point calculations.
#[inline]
pub fn dt_sec(now_us: i64, last_us: i64) -> f32 {
    (dt_us(now_us, last_us) as f32) / 1_000_000.0
}

// ============================================================================
// INPUT LAYER: Multi-Dimensional Observation Space
// ============================================================================

/// Location type classification for environmental context.
/// Used by Active Inference to model expected sensory patterns and affordances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LocationType {
    /// Home environment: typically low stress, high autonomy
    Home,
    /// Work environment: higher cognitive load, social constraints
    Work,
    /// Transit: unpredictable, limited control, potential stressors
    Transit,
}

/// Digital application category for context-aware interventions.
/// Enables the system to understand current cognitive engagement patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AppCategory {
    /// Social media, messaging apps
    Social,
    /// Productivity tools, work applications
    Productivity,
    /// Entertainment, games, streaming
    Entertainment,
    /// Health, meditation, fitness apps
    Wellness,
    /// Browser, general purpose
    Browser,
    /// Unknown or unclassified
    Other,
}

/// Biological metrics from wearable sensors or physiological monitoring.
/// These form the primary sensory input for homeostatic regulation.
/// All values are optional to handle partial sensor availability gracefully.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioMetrics {
    /// Heart rate in beats per minute (BPM).
    /// Active Inference uses this as a proxy for autonomic arousal state.
    /// Typical range: 40-200 BPM.
    pub hr_bpm: Option<f32>,

    /// Heart Rate Variability (HRV) measured as RMSSD in milliseconds.
    /// Higher HRV generally indicates better parasympathetic tone and stress resilience.
    /// Typical range: 20-100 ms for adults.
    pub hrv_rmssd: Option<f32>,

    /// Respiratory rate in breaths per minute.
    /// Key input for breath-based interventions and arousal assessment.
    /// Typical range: 8-20 breaths/min.
    pub respiratory_rate: Option<f32>,
}

/// Environmental context capturing the user's physical and social surroundings.
/// Provides priors for expected stressors and available intervention affordances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalContext {
    /// Type of location the user is currently in.
    /// Influences intervention appropriateness (e.g., no audio in meetings).
    pub location_type: Option<LocationType>,

    /// Ambient noise level in decibels (dB) or normalized 0-1 scale.
    /// High noise can indicate stressful environment or limit audio interventions.
    pub noise_level: Option<f32>,

    /// Whether the device is currently charging.
    /// Affects power budget for continuous monitoring and interventions.
    pub is_charging: bool,
}

/// Digital context capturing the user's interaction with technology.
/// Critical for understanding cognitive load and digital wellbeing patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalContext {
    /// Category of the currently active application.
    /// Helps infer cognitive state and intervention timing.
    pub active_app_category: Option<AppCategory>,

    /// Interaction intensity normalized to [0, 1].
    /// 0 = passive consumption, 1 = high-frequency interaction (typing, swiping).
    /// Derived from touch events, keyboard activity, or app usage patterns.
    pub interaction_intensity: Option<f32>,

    /// Notification pressure: rate or volume of incoming notifications.
    /// High values indicate potential for interruption-driven stress.
    /// Can be measured as notifications per hour or cumulative attention demand.
    pub notification_pressure: Option<f32>,
}

/// Cognitive context capturing semantic and vocal signals for Cognitive OS capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveContext {
    /// Cognitive load normalized to [0.0, 1.0] (screen complexity / task switching).
    pub cognitive_load: Option<f32>,
    /// Voice valence in [-1.0, 1.0] (negative to positive affect).
    pub voice_valence: Option<f32>,
    /// Voice arousal in [0.0, 1.0] (intensity of voice).
    pub voice_arousal: Option<f32>,
    /// Screen text sentiment in [-1.0, 1.0].
    pub screen_text_sentiment: Option<f32>,
}

/// Root observation structure representing the complete sensory input to the AI kernel.
/// This is the "generative model's sensory layer" in Active Inference terminology.
/// The system uses these observations to update beliefs about hidden states and select actions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Observation {
    /// Timestamp of the observation in microseconds since epoch.
    /// Critical for temporal inference and time-series analysis.
    pub timestamp_us: i64,

    /// Biological sensor data (heart rate, HRV, respiration).
    /// Optional to handle cases where wearables are not connected.
    pub bio_metrics: Option<BioMetrics>,

    /// Environmental context (location, noise, charging state).
    /// Optional but provides important priors for intervention planning.
    pub environmental_context: Option<EnvironmentalContext>,

    /// Digital context (app usage, interaction patterns, notifications).
    /// Optional but essential for digital wellbeing interventions.
    pub digital_context: Option<DigitalContext>,

    /// Cognitive context (semantic load, voice sentiment, screen content).
    /// Optional to preserve backward compatibility.
    pub cognitive_context: Option<CognitiveContext>,
}

// ============================================================================
// INTERNAL MODEL: Factorized Belief State
// ============================================================================

/// Biological state factor: represents autonomic nervous system state.
/// Used for homeostatic regulation and breath guidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BioState {
    /// Parasympathetic dominance: low arousal, relaxed
    Calm,
    /// Sympathetic activation: elevated arousal, stress response
    Aroused,
    /// Low energy state: physical or mental exhaustion
    Fatigue,
}

/// Cognitive state factor: represents attentional and mental processing state.
/// Used for productivity optimization and cognitive load management.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveState {
    /// Sustained attention on task, low distraction
    Focus,
    /// Fragmented attention, high context switching
    Distracted,
    /// Optimal engagement: high focus + low effort (Csikszentmihalyi flow)
    Flow,
}

/// Social state factor: represents social engagement and interaction load.
/// Used for managing social overwhelm and communication boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialState {
    /// Alone or minimal social interaction
    Solitary,
    /// Active social engagement (meetings, conversations)
    Interactive,
    /// Excessive social demands, need for withdrawal
    Overwhelmed,
}

/// CANONICAL DECISION (PR1): This type is RENAMED to CausalBeliefState.
/// The canonical BeliefState is in crate::belief::BeliefState (5-mode collapsed representation).
/// This factorized 3-factor representation is used ONLY by the causal reasoning layer
/// for extracting state values into the causal graph's variable space.
///
/// Factorized belief state representation for causal extraction.
/// Instead of a single mode, this maintains probability distributions over
/// multiple independent (or weakly coupled) state factors. This enables:
/// 1. More nuanced state representation (e.g., "Calm but Distracted")
/// 2. Partial observability handling (update only observed factors)
/// 3. Multi-objective policy optimization (balance bio, cognitive, social goals)
///
/// Each distribution is a probability vector summing to 1.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalBeliefState {
    /// Probability distribution over biological states [Calm, Aroused, Fatigue].
    /// Index 0 = Calm, 1 = Aroused, 2 = Fatigue.
    /// Used for: breath guidance, stress interventions, energy management.
    pub bio_state: [f32; 3],

    /// Probability distribution over cognitive states [Focus, Distracted, Flow].
    /// Index 0 = Focus, 1 = Distracted, 2 = Flow.
    /// Used for: notification management, app suggestions, break timing.
    pub cognitive_state: [f32; 3],

    /// Probability distribution over social states [Solitary, Interactive, Overwhelmed].
    /// Index 0 = Solitary, 1 = Interactive, 2 = Overwhelmed.
    /// Used for: communication boundaries, social recovery interventions.
    pub social_state: [f32; 3],

    /// Overall confidence in the belief state (0-1).
    /// Low confidence triggers more exploratory policies or requests for more data.
    pub confidence: f32,

    /// Timestamp of last belief update in microseconds.
    pub last_update_us: i64,

    /// Optional cognitive context extracted for causal mapping.
    pub cognitive_context: Option<CognitiveContext>,
}

impl Default for CausalBeliefState {
    fn default() -> Self {
        Self {
            // Uniform prior: maximum uncertainty
            bio_state: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            cognitive_state: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            social_state: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            confidence: 0.0,
            last_update_us: 0,
            cognitive_context: None,
        }
    }
}

impl CausalBeliefState {
    /// Get the most likely biological state (MAP estimate).
    pub fn most_likely_bio_state(&self) -> BioState {
        let max_idx = self
            .bio_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => BioState::Calm,
            1 => BioState::Aroused,
            _ => BioState::Fatigue,
        }
    }

    /// Get the most likely cognitive state (MAP estimate).
    pub fn most_likely_cognitive_state(&self) -> CognitiveState {
        let max_idx = self
            .cognitive_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => CognitiveState::Focus,
            1 => CognitiveState::Distracted,
            _ => CognitiveState::Flow,
        }
    }

    /// Get the most likely social state (MAP estimate).
    pub fn most_likely_social_state(&self) -> SocialState {
        let max_idx = self
            .social_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => SocialState::Solitary,
            1 => SocialState::Interactive,
            _ => SocialState::Overwhelmed,
        }
    }
}

// ============================================================================
// EXISTING DOMAIN TYPES (Event Sourcing Infrastructure)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId([u8; 16]);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().into_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    pub session_id: SessionId,
    pub seq: u64,
    pub ts_us: i64,
    pub event: Event,
    pub meta: serde_json::Value,
}

/// PR2: Event priority classification for audit guarantees.
/// Critical events MUST NEVER be dropped silently.
/// HighFreq events can be coalesced/backpressured with visibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventPriority {
    /// MUST NEVER DROP: Decision/Trauma/Error/Session lifecycle/Config/Schema changes
    Critical,
    /// Can be coalesced or backpressured: Sensor data, belief updates
    HighFreq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    SessionStarted {
        mode: String,
    },
    SensorFeaturesIngested {
        features: Vec<f32>,
        downsampled: bool,
    },
    ControlDecisionMade {
        decision: ControlDecision,
    },
    PatternAdjusted {
        pattern_id: String,
    },
    CycleCompleted {
        cycles: u32,
    },
    SessionEnded {},
    Tombstone {},
    BeliefUpdated {
        p: [f32; 5],
        conf: f32,
        mode: u8,
    },
    BeliefUpdatedV2 {
        p: [f32; 5],
        conf: f32,
        mode: u8,
        free_energy_ema: f32,
        lr: f32,
        resonance_score: f32,
    },
    PolicyChosen {
        mode: u8,
        reason_bits: u32,
        conf: f32,
    },
    ControlDecisionDenied {
        reason: String,
        timestamp: i64,
    },
    ConfigUpdated {
        config: ZenbConfig,
    },
}

impl Event {
    /// PR2: Classify event priority for audit guarantees.
    /// Critical events MUST be persisted; HighFreq can be coalesced.
    pub fn priority(&self) -> EventPriority {
        match self {
            // CRITICAL: Session lifecycle (forensic requirement)
            Event::SessionStarted { .. } => EventPriority::Critical,
            Event::SessionEnded { .. } => EventPriority::Critical,

            // CRITICAL: Control decisions and denials (safety audit)
            Event::ControlDecisionMade { .. } => EventPriority::Critical,
            Event::ControlDecisionDenied { .. } => EventPriority::Critical,

            // CRITICAL: Configuration changes (schema evolution)
            Event::ConfigUpdated { .. } => EventPriority::Critical,

            // CRITICAL: Pattern adjustments (intervention tracking)
            Event::PatternAdjusted { .. } => EventPriority::Critical,

            // CRITICAL: Tombstone (trauma/error markers)
            Event::Tombstone { .. } => EventPriority::Critical,

            // HIGH-FREQ: Sensor data (can be downsampled)
            Event::SensorFeaturesIngested { .. } => EventPriority::HighFreq,

            // HIGH-FREQ: Belief updates (1-2Hz, can coalesce)
            Event::BeliefUpdated { .. } => EventPriority::HighFreq,
            Event::BeliefUpdatedV2 { .. } => EventPriority::HighFreq,
            Event::PolicyChosen { .. } => EventPriority::HighFreq,

            // HIGH-FREQ: Cycle completions (low-frequency telemetry)
            Event::CycleCompleted { .. } => EventPriority::HighFreq,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlDecision {
    pub target_rate_bpm: f32,
    pub confidence: f32,
    pub recommended_poll_interval_ms: u64,
}

#[derive(Error, Debug)]
pub enum DomainError {
    #[error("invalid sequence: expected {expected} got {got}")]
    InvalidSequence { expected: u64, got: u64 },
    #[error("replay error: {0}")]
    ReplayError(String),
}

impl Envelope {
    pub fn event_type_code(&self) -> u16 {
        match self.event {
            Event::SessionStarted { .. } => 1,
            Event::SensorFeaturesIngested { .. } => 2,
            Event::ControlDecisionMade { .. } => 3,
            Event::PatternAdjusted { .. } => 4,
            Event::CycleCompleted { .. } => 5,
            Event::SessionEnded { .. } => 6,
            Event::Tombstone { .. } => 7,
            Event::BeliefUpdated { .. } => 8,
            Event::PolicyChosen { .. } => 9,
            Event::ControlDecisionDenied { .. } => 10,
            Event::BeliefUpdatedV2 { .. } => 11,
            Event::ConfigUpdated { .. } => 12,
        }
    }

    /// Serialize meta field to bytes for storage.
    pub fn meta_as_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&self.meta)
    }
}

/// Deterministic breath engine state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BreathState {
    pub session_active: bool,
    pub total_cycles: u64,
    pub last_decision: Option<ControlDecision>,
    pub current_mode: Option<u8>,
    pub belief_conf: Option<f32>,
    pub belief_p: Option<[f32; 5]>,
    pub config_hash: Option<[u8; 32]>,
}

/// Helper: Convert f32 to canonical fixed-point representation for deterministic hashing
/// This ensures identical hashes across platforms regardless of floating-point representation
fn f32_to_canonical(val: f32) -> i64 {
    const SCALE: f32 = 1_000_000.0;

    // Handle edge cases for complete determinism
    if val.is_nan() {
        return i64::MAX;
    }
    if val == f32::INFINITY {
        return i64::MAX - 1;
    }
    if val == f32::NEG_INFINITY {
        return i64::MIN;
    }

    // Clamp to safe range to prevent overflow
    let clamped = val.clamp(-2147.0, 2147.0);
    (clamped * SCALE).round() as i64
}

impl BreathState {
    pub fn apply(&mut self, envelope: &Envelope) {
        match &envelope.event {
            Event::SessionStarted { .. } => {
                self.session_active = true;
            }
            Event::ControlDecisionMade { decision } => {
                self.last_decision = Some(decision.clone());
            }
            Event::CycleCompleted { cycles } => {
                self.total_cycles += *cycles as u64;
            }
            Event::SessionEnded { .. } => {
                self.session_active = false;
            }
            Event::BeliefUpdated { p, conf, mode } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
                self.belief_p = Some(*p);
            }
            Event::BeliefUpdatedV2 {
                p,
                conf,
                mode,
                free_energy_ema: _,
                lr: _,
                resonance_score: _,
            } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
                self.belief_p = Some(*p);
            }
            Event::PolicyChosen {
                mode,
                reason_bits: _,
                conf,
            } => {
                self.current_mode = Some(*mode);
                self.belief_conf = Some(*conf);
            }
            Event::ControlDecisionDenied {
                reason: _,
                timestamp: _,
            } => {
                // Denials do not change deterministic breath state, but are recorded as events for auditing.
            }
            Event::ConfigUpdated { config } => {
                let bytes = serde_json::to_vec(config).expect("serialization should not fail");
                let mut hasher = Hasher::new();
                hasher.update(&bytes);
                let out = hasher.finalize();
                self.config_hash = Some(*out.as_bytes());
            }
            _ => {}
        }
    }

    pub fn hash(&self) -> [u8; 32] {
        // Deterministic manual hashing instead of JSON to ensure cross-platform consistency
        let mut hasher = Hasher::new();

        // Hash session_active
        hasher.update(&[if self.session_active { 1u8 } else { 0u8 }]);

        // Hash total_cycles
        hasher.update(&self.total_cycles.to_le_bytes());

        // Hash last_decision with fixed-point conversion
        match &self.last_decision {
            Some(decision) => {
                hasher.update(&[1u8]);
                let target_fixed = f32_to_canonical(decision.target_rate_bpm);
                let conf_fixed = f32_to_canonical(decision.confidence);
                hasher.update(&target_fixed.to_le_bytes());
                hasher.update(&conf_fixed.to_le_bytes());
                hasher.update(&decision.recommended_poll_interval_ms.to_le_bytes());
            }
            None => {
                hasher.update(&[0u8]);
            }
        }

        // Hash current_mode
        match self.current_mode {
            Some(mode) => {
                hasher.update(&[1u8]);
                hasher.update(&[mode]);
            }
            None => {
                hasher.update(&[0u8]);
            }
        }

        // Hash belief_conf with fixed-point conversion
        match self.belief_conf {
            Some(conf) => {
                hasher.update(&[1u8]);
                let conf_fixed = f32_to_canonical(conf);
                hasher.update(&conf_fixed.to_le_bytes());
            }
            None => {
                hasher.update(&[0u8]);
            }
        }

        // Hash belief_p array with fixed-point conversion
        match self.belief_p {
            Some(p) => {
                hasher.update(&[1u8]);
                for val in p.iter() {
                    let val_fixed = f32_to_canonical(*val);
                    hasher.update(&val_fixed.to_le_bytes());
                }
            }
            None => {
                hasher.update(&[0u8]);
            }
        }

        // Hash config_hash
        match &self.config_hash {
            Some(hash) => {
                hasher.update(&[1u8]);
                hasher.update(hash);
            }
            None => {
                hasher.update(&[0u8]);
            }
        }

        let out = hasher.finalize();
        *out.as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_us_normal_forward() {
        // Normal case: time moves forward
        let now = 1000000;
        let last = 500000;
        assert_eq!(dt_us(now, last), 500000);
    }

    #[test]
    fn test_dt_us_backwards_clock() {
        // PR4: Clock went backwards - should return 0, not wrap
        let now = 500000;
        let last = 1000000;
        assert_eq!(dt_us(now, last), 0);

        // Verify we don't get huge wraparound value
        assert!(dt_us(now, last) < 1000);
    }

    #[test]
    fn test_dt_us_same_timestamp() {
        // Edge case: same timestamp
        let ts = 1000000;
        assert_eq!(dt_us(ts, ts), 0);
    }

    #[test]
    fn test_dt_sec_conversion() {
        // Verify microseconds to seconds conversion
        let now = 2000000; // 2 seconds
        let last = 1000000; // 1 second
        let dt = dt_sec(now, last);
        assert!((dt - 1.0).abs() < 0.0001); // Should be ~1.0 second
    }

    #[test]
    fn test_dt_sec_backwards_clock() {
        // PR4: Backwards clock in dt_sec should also return 0
        let now = 1000000;
        let last = 2000000;
        assert_eq!(dt_sec(now, last), 0.0);
    }
}
