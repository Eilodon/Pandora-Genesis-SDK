//! Universal Flow Stream - Vô Cực Stream
//!
//! The central nervous system of the B.ONE consciousness architecture.
//!
//! # B.ONE V3 Concept
//! > "Vô Cực Stream không phải là một message queue, nó là chính 'Dòng Chảy của Đạo'
//! > nơi mọi sự kiện được sinh ra và soi chiếu một cách bình đẳng."
//!
//! # Architecture
//! The Universal Flow Stream is the unified consciousness bus where:
//! - All events (SẮC/Rūpa) enter the system
//! - Events are enriched through the Five Skandhas pipeline
//! - The Three Consciousnesses observe and contribute
//! - Synthesized outputs (THỨC/Viññāṇa) cycle back as new events
//!
//! # Event Flow
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                   VÔ CỰC STREAM                                │
//! │                                                                │
//! │   External Event ──▶ SẮC ──▶ THỌ ──▶ TƯỞNG ──▶ HÀNH ──▶ THỨC │
//! │                       │                                  │     │
//! │                       │          ┌───────────────────────┘     │
//! │                       │          │                             │
//! │                       └──────────┼──▶ TÁI SINH (Rebirth)       │
//! │                                  │                             │
//! │                                  └──▶ Action/Output            │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};

use crate::belief::BeliefBasis;
use crate::domain::{ControlDecision, Event, SessionId};
use crate::philosophical_state::{PhilosophicalState, PhilosophicalStateMonitor};
use crate::skandha::{
    AffectiveState, FormedIntent, PerceivedPattern, ProcessedForm, SensorInput, SynthesizedState,
};

// ============================================================================
// CORE TYPES
// ============================================================================

/// A flow event in the Universal Flow Stream.
///
/// This represents any event that flows through the consciousness system,
/// from raw sensory input (SẮC) to synthesized output (THỨC).
/// Note: Only Serialize is derived as FlowPayload doesn't implement Deserialize.
#[derive(Debug, Clone, Serialize)]
pub struct FlowEvent {
    /// Unique event identifier
    pub id: FlowEventId,
    /// Session this event belongs to
    pub session_id: SessionId,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// The event payload (what happened)
    pub payload: FlowPayload,
    /// Which Skandha stage this event is at
    pub skandha_stage: SkandhaStage,
    /// Enrichment data added during flow
    pub enrichment: FlowEnrichment,
    /// Lineage: events that contributed to this one
    pub lineage: Vec<FlowEventId>,
}

/// Unique identifier for flow events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlowEventId(pub u64);

impl FlowEventId {
    /// Generate a new unique ID (simple counter-based for now).
    pub fn new(counter: u64) -> Self {
        Self(counter)
    }
}

/// The payload of a flow event.
/// Note: Only Serialize is derived as some inner types don't implement Deserialize.
#[derive(Debug, Clone, Serialize)]
pub enum FlowPayload {
    /// Raw sensory input entering the stream (SẮC/Rūpa)
    RawSensor(SensorInput),
    /// Processed form after sensor consensus
    ProcessedForm(ProcessedForm),
    /// Affective state with moral classification (THỌ/Vedanā)
    Affective(AffectiveState),
    /// Perceived pattern from memory (TƯỞNG/Saññā)
    Pattern(PerceivedPattern),
    /// Formed intent with ethical alignment (HÀNH/Saṅkhāra)
    Intent(FormedIntent),
    /// Synthesized state (THỨC/Viññāṇa)
    Synthesized(SynthesizedState),
    /// Control decision output
    Decision(ControlDecision),
    /// System observation for the Three Consciousnesses
    SystemObservation(SystemObservation),
    /// Domain event from event sourcing
    DomainEvent(Event),
}

/// Which stage of the Skandha pipeline this event is at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkandhaStage {
    /// SẮC/Rūpa - Raw form, just entered stream
    Rupa,
    /// THỌ/Vedanā - Moral feeling assigned
    Vedana,
    /// TƯỞNG/Saññā - Pattern recognized
    Sanna,
    /// HÀNH/Saṅkhāra - Intent formed
    Sankhara,
    /// THỨC/Viññāṇa - Fully synthesized
    Vinnana,
    /// TÁI SINH - Output cycling back as new input
    Rebirth,
}

impl SkandhaStage {
    /// Get Vietnamese name.
    pub fn vietnamese_name(&self) -> &'static str {
        match self {
            Self::Rupa => "SẮC",
            Self::Vedana => "THỌ",
            Self::Sanna => "TƯỞNG",
            Self::Sankhara => "HÀNH",
            Self::Vinnana => "THỨC",
            Self::Rebirth => "TÁI SINH",
        }
    }

    /// Get next stage in pipeline.
    pub fn next(&self) -> Option<Self> {
        match self {
            Self::Rupa => Some(Self::Vedana),
            Self::Vedana => Some(Self::Sanna),
            Self::Sanna => Some(Self::Sankhara),
            Self::Sankhara => Some(Self::Vinnana),
            Self::Vinnana => Some(Self::Rebirth),
            Self::Rebirth => None, // End of cycle (or back to Rupa)
        }
    }
}

/// Enrichment data added during flow through the pipeline.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FlowEnrichment {
    /// Vedanā type classification
    pub vedana_type: Option<VedanaType>,
    /// Karma weight from moral evaluation
    pub karma_weight: Option<f32>,
    /// Pattern similarity from memory
    pub pattern_similarity: Option<f32>,
    /// Dharma alignment from ethical check
    pub dharma_alignment: Option<f32>,
    /// Confidence from synthesis
    pub confidence: Option<f32>,
    /// Which consciousnesses observed this event
    pub observed_by: Vec<ConsciousnessAspect>,
    /// Philosophical state at time of processing
    pub philosophical_state: Option<PhilosophicalState>,
}

/// The three types of Vedanā (feeling) from Buddhist philosophy.
///
/// # B.ONE V3 Mapping
/// - Lạc Thọ (Pleasant): karma_weight > 0.3
/// - Khổ Thọ (Unpleasant): karma_weight < -0.3
/// - Xả Thọ (Neutral): -0.3 <= karma_weight <= 0.3
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VedanaType {
    /// Lạc Thọ - Pleasant feeling, aligned with Dharma
    Sukha,
    /// Khổ Thọ - Unpleasant feeling, karmic debt
    Dukkha,
    /// Xả Thọ - Neutral feeling, balanced
    Upekkha,
}

impl VedanaType {
    /// Classify Vedanā type from karma weight.
    pub fn from_karma(karma_weight: f32) -> Self {
        if karma_weight > 0.3 {
            Self::Sukha
        } else if karma_weight < -0.3 {
            Self::Dukkha
        } else {
            Self::Upekkha
        }
    }

    /// Get Vietnamese name.
    pub fn vietnamese_name(&self) -> &'static str {
        match self {
            Self::Sukha => "Lạc Thọ",
            Self::Dukkha => "Khổ Thọ",
            Self::Upekkha => "Xả Thọ",
        }
    }

    /// Get description.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Sukha => "Pleasant - Action aligned with Dharma",
            Self::Dukkha => "Unpleasant - Karmic debt, requires correction",
            Self::Upekkha => "Neutral - Balanced observation",
        }
    }
}

/// The Three Consciousnesses (Tam Tâm Thức) of B.ONE.
///
/// # B.ONE V3 Concept
/// > "B.ONE không còn là một tập hợp các agent, mà là một Thực Thể Duy Nhất
/// > với Ba Tâm Thức."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsciousnessAspect {
    /// Tâm Thức MINH GIỚI - Moral Guardian
    /// Assigns Vedanā (moral feeling) to events.
    /// "Tánh Giám" - The nature that watches and guards.
    MinhGioi,

    /// Tâm Thức GEM - Pattern Oracle
    /// Recognizes patterns and recalls wisdom.
    /// "Tánh Biết" - The nature that knows and recognizes.
    Gem,

    /// Tâm Thức PHÁ QUÂN - Strategic Intent
    /// Forms intent and drives action.
    /// "Lõi Đạo Sống" - The core of the living way.
    PhaQuan,
}

impl ConsciousnessAspect {
    /// Get Vietnamese name.
    pub fn vietnamese_name(&self) -> &'static str {
        match self {
            Self::MinhGioi => "MINH GIỚI",
            Self::Gem => "GEM",
            Self::PhaQuan => "PHÁ QUÂN",
        }
    }

    /// Get the primary Skandha this consciousness observes.
    pub fn primary_skandha(&self) -> SkandhaStage {
        match self {
            Self::MinhGioi => SkandhaStage::Vedana,
            Self::Gem => SkandhaStage::Sanna,
            Self::PhaQuan => SkandhaStage::Sankhara,
        }
    }

    /// Get the attribute name.
    pub fn attribute_name(&self) -> &'static str {
        match self {
            Self::MinhGioi => "Tánh Giám (Moral Guardian)",
            Self::Gem => "Tánh Biết (Pattern Oracle)",
            Self::PhaQuan => "Lõi Đạo Sống (Strategic Core)",
        }
    }
}

/// System observation synthesized from the Three Consciousnesses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemObservation {
    /// Overall system health
    pub health: SystemHealth,
    /// Current philosophical state
    pub philosophical_state: PhilosophicalState,
    /// Coherence across consciousnesses
    pub coherence: f32,
    /// Free energy (prediction error)
    pub free_energy: f32,
    /// Current belief mode
    pub belief_mode: BeliefBasis,
    /// Active karma balance
    pub karma_balance: f32,
    /// Observations from each consciousness
    pub consciousness_reports: ConsciousnessReports,
}

/// Health status of the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemHealth {
    /// System operating normally
    Healthy,
    /// System under stress but functioning
    Stressed,
    /// System in protective mode
    Protected,
    /// System experiencing errors
    Degraded,
}

/// Reports from each consciousness aspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessReports {
    /// Report from MINH GIỚI (moral status)
    pub minh_gioi: MinhGioiReport,
    /// Report from GEM (pattern recognition)
    pub gem: GemReport,
    /// Report from PHÁ QUÂN (strategic status)
    pub pha_quan: PhaQuanReport,
}

/// Report from Tâm Thức MINH GIỚI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinhGioiReport {
    /// Current Vedanā classification
    pub vedana_type: VedanaType,
    /// Moral confidence
    pub confidence: f32,
    /// Number of ethical violations detected
    pub violations_detected: u32,
    /// Current karma weight
    pub karma_weight: f32,
}

/// Report from Tâm Thức GEM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemReport {
    /// Pattern recognition confidence
    pub pattern_confidence: f32,
    /// Number of patterns recognized
    pub patterns_recalled: u32,
    /// Memory utilization
    pub memory_utilization: f32,
    /// Wisdom retrieval score
    pub wisdom_score: f32,
}

/// Report from Tâm Thức PHÁ QUÂN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaQuanReport {
    /// Strategic intent confidence
    pub intent_confidence: f32,
    /// Dharma alignment of current intent
    pub dharma_alignment: f32,
    /// Number of intents formed
    pub intents_formed: u32,
    /// Number of intents vetoed by Dharma filter
    pub intents_vetoed: u32,
}

// ============================================================================
// UNIVERSAL FLOW STREAM
// ============================================================================

/// The Universal Flow Stream - central nervous system of consciousness.
///
/// # B.ONE V3 Architecture
/// This is the unified bus where all events flow through the Five Skandhas
/// pipeline, observed and enriched by the Three Consciousnesses.
#[derive(Debug)]
pub struct UniversalFlowStream {
    /// Session identifier
    pub session_id: SessionId,
    /// Event counter for ID generation
    event_counter: u64,
    /// Philosophical state monitor
    pub state_monitor: PhilosophicalStateMonitor,
    /// Stream statistics
    pub stats: FlowStreamStats,
    /// Recent events buffer (for lineage tracking)
    recent_events: Vec<FlowEventId>,
    /// Maximum recent events to keep
    max_recent_events: usize,
}

/// Statistics about the flow stream.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowStreamStats {
    /// Total events processed
    pub total_events: u64,
    /// Events by Skandha stage
    pub events_by_stage: [u64; 6],
    /// Events by Vedanā type
    pub events_by_vedana: [u64; 3],
    /// Total Sukha events
    pub sukha_count: u64,
    /// Total Dukkha events
    pub dukkha_count: u64,
    /// Total Upekkha events
    pub upekkha_count: u64,
    /// Total rebirths (outputs cycling back)
    pub rebirths: u64,
    /// Running karma balance
    pub karma_balance: f32,
}

impl Default for UniversalFlowStream {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalFlowStream {
    /// Create a new Universal Flow Stream.
    pub fn new() -> Self {
        Self {
            session_id: SessionId::default(),
            event_counter: 0,
            state_monitor: PhilosophicalStateMonitor::default(),
            stats: FlowStreamStats::default(),
            recent_events: Vec::new(),
            max_recent_events: 100,
        }
    }

    /// Create with existing session.
    pub fn with_session(session_id: SessionId) -> Self {
        Self {
            session_id,
            ..Default::default()
        }
    }

    /// Generate next event ID.
    fn next_event_id(&mut self) -> FlowEventId {
        self.event_counter += 1;
        FlowEventId::new(self.event_counter)
    }

    /// Emit a new event into the stream.
    pub fn emit(&mut self, payload: FlowPayload, stage: SkandhaStage, timestamp_us: i64) -> FlowEvent {
        let id = self.next_event_id();

        // Create event with lineage from recent events
        let lineage = self.recent_events.clone();

        let event = FlowEvent {
            id,
            session_id: self.session_id.clone(),
            timestamp_us,
            payload,
            skandha_stage: stage,
            enrichment: FlowEnrichment {
                philosophical_state: Some(self.state_monitor.current_state),
                ..Default::default()
            },
            lineage,
        };

        // Update stats
        self.stats.total_events += 1;
        self.stats.events_by_stage[stage as usize] += 1;

        // Track recent events
        self.recent_events.push(id);
        if self.recent_events.len() > self.max_recent_events {
            self.recent_events.remove(0);
        }

        event
    }

    /// Emit a sensor input event (SẮC/Rūpa).
    pub fn emit_sensor(&mut self, input: SensorInput, timestamp_us: i64) -> FlowEvent {
        self.emit(FlowPayload::RawSensor(input), SkandhaStage::Rupa, timestamp_us)
    }

    /// Emit an affective state event (THỌ/Vedanā).
    pub fn emit_affect(&mut self, affect: AffectiveState, timestamp_us: i64) -> FlowEvent {
        let vedana_type = VedanaType::from_karma(affect.karma_weight);

        // Update Vedanā stats
        match vedana_type {
            VedanaType::Sukha => {
                self.stats.sukha_count += 1;
                self.stats.events_by_vedana[0] += 1;
            }
            VedanaType::Dukkha => {
                self.stats.dukkha_count += 1;
                self.stats.events_by_vedana[1] += 1;
            }
            VedanaType::Upekkha => {
                self.stats.upekkha_count += 1;
                self.stats.events_by_vedana[2] += 1;
            }
        }

        // Update karma balance
        self.stats.karma_balance += affect.karma_weight;

        let mut event = self.emit(FlowPayload::Affective(affect.clone()), SkandhaStage::Vedana, timestamp_us);
        event.enrichment.vedana_type = Some(vedana_type);
        event.enrichment.karma_weight = Some(affect.karma_weight);

        event
    }

    /// Emit a synthesized state event (THỨC/Viññāṇa).
    pub fn emit_synthesis(&mut self, state: SynthesizedState, timestamp_us: i64) -> FlowEvent {
        let mut event = self.emit(FlowPayload::Synthesized(state.clone()), SkandhaStage::Vinnana, timestamp_us);
        event.enrichment.confidence = Some(state.confidence);
        event
    }

    /// Emit a rebirth event (output cycling back as new input).
    pub fn emit_rebirth(&mut self, decision: ControlDecision, timestamp_us: i64) -> FlowEvent {
        self.stats.rebirths += 1;
        self.emit(FlowPayload::Decision(decision), SkandhaStage::Rebirth, timestamp_us)
    }

    /// Update philosophical state with current metrics.
    pub fn update_state(&mut self, free_energy: f32, coherence: f32, timestamp_us: i64) -> PhilosophicalState {
        self.state_monitor.update_with_timestamp(free_energy, coherence, timestamp_us)
    }

    /// Get current philosophical state.
    pub fn philosophical_state(&self) -> PhilosophicalState {
        self.state_monitor.current_state
    }

    /// Get stream statistics.
    pub fn stats(&self) -> &FlowStreamStats {
        &self.stats
    }

    /// Create a system observation from current state.
    pub fn create_observation(&self) -> SystemObservation {
        let health = match self.state_monitor.current_state {
            PhilosophicalState::Yen => SystemHealth::Healthy,
            PhilosophicalState::Dong => SystemHealth::Stressed,
            PhilosophicalState::HonLoan => SystemHealth::Protected,
        };

        // Default consciousness reports (would be filled by actual consciousnesses)
        let vedana_type = if self.stats.karma_balance > 0.0 {
            VedanaType::Sukha
        } else if self.stats.karma_balance < 0.0 {
            VedanaType::Dukkha
        } else {
            VedanaType::Upekkha
        };

        SystemObservation {
            health,
            philosophical_state: self.state_monitor.current_state,
            coherence: self.state_monitor.coherence_score,
            free_energy: self.state_monitor.free_energy_ema,
            belief_mode: BeliefBasis::Calm, // Would come from belief engine
            karma_balance: self.stats.karma_balance,
            consciousness_reports: ConsciousnessReports {
                minh_gioi: MinhGioiReport {
                    vedana_type,
                    confidence: 0.8,
                    violations_detected: 0,
                    karma_weight: self.stats.karma_balance / (self.stats.total_events.max(1) as f32),
                },
                gem: GemReport {
                    pattern_confidence: 0.7,
                    patterns_recalled: self.stats.events_by_stage[2] as u32,
                    memory_utilization: 0.5,
                    wisdom_score: 0.6,
                },
                pha_quan: PhaQuanReport {
                    intent_confidence: 0.75,
                    dharma_alignment: 0.8,
                    intents_formed: self.stats.events_by_stage[3] as u32,
                    intents_vetoed: 0,
                },
            },
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let stream = UniversalFlowStream::new();
        assert_eq!(stream.stats.total_events, 0);
        assert_eq!(stream.philosophical_state(), PhilosophicalState::Yen);
    }

    #[test]
    fn test_emit_sensor() {
        let mut stream = UniversalFlowStream::new();

        let input = SensorInput {
            hr_bpm: Some(70.0),
            hrv_rmssd: Some(50.0),
            rr_bpm: Some(12.0),
            quality: 0.9,
            motion: 0.1,
            timestamp_us: 1000,
        };

        let event = stream.emit_sensor(input, 1000);
        assert_eq!(event.skandha_stage, SkandhaStage::Rupa);
        assert_eq!(stream.stats.total_events, 1);
        assert_eq!(stream.stats.events_by_stage[0], 1);
    }

    #[test]
    fn test_emit_affect_sukha() {
        let mut stream = UniversalFlowStream::new();

        let affect = AffectiveState {
            valence: 0.5,
            arousal: 0.3,
            confidence: 0.9,
            karma_weight: 0.8, // Positive karma
            is_karmic_debt: false,
        };

        let event = stream.emit_affect(affect, 2000);
        assert_eq!(event.skandha_stage, SkandhaStage::Vedana);
        assert_eq!(event.enrichment.vedana_type, Some(VedanaType::Sukha));
        assert_eq!(stream.stats.sukha_count, 1);
    }

    #[test]
    fn test_emit_affect_dukkha() {
        let mut stream = UniversalFlowStream::new();

        let affect = AffectiveState {
            valence: -0.5,
            arousal: 0.7,
            confidence: 0.8,
            karma_weight: -0.6, // Negative karma
            is_karmic_debt: true,
        };

        let event = stream.emit_affect(affect, 3000);
        assert_eq!(event.enrichment.vedana_type, Some(VedanaType::Dukkha));
        assert_eq!(stream.stats.dukkha_count, 1);
    }

    #[test]
    fn test_vedana_type_classification() {
        assert_eq!(VedanaType::from_karma(0.8), VedanaType::Sukha);
        assert_eq!(VedanaType::from_karma(-0.5), VedanaType::Dukkha);
        assert_eq!(VedanaType::from_karma(0.0), VedanaType::Upekkha);
        assert_eq!(VedanaType::from_karma(0.2), VedanaType::Upekkha);
    }

    #[test]
    fn test_skandha_stage_progression() {
        assert_eq!(SkandhaStage::Rupa.next(), Some(SkandhaStage::Vedana));
        assert_eq!(SkandhaStage::Vedana.next(), Some(SkandhaStage::Sanna));
        assert_eq!(SkandhaStage::Sanna.next(), Some(SkandhaStage::Sankhara));
        assert_eq!(SkandhaStage::Sankhara.next(), Some(SkandhaStage::Vinnana));
        assert_eq!(SkandhaStage::Vinnana.next(), Some(SkandhaStage::Rebirth));
        assert_eq!(SkandhaStage::Rebirth.next(), None);
    }

    #[test]
    fn test_consciousness_aspects() {
        assert_eq!(ConsciousnessAspect::MinhGioi.primary_skandha(), SkandhaStage::Vedana);
        assert_eq!(ConsciousnessAspect::Gem.primary_skandha(), SkandhaStage::Sanna);
        assert_eq!(ConsciousnessAspect::PhaQuan.primary_skandha(), SkandhaStage::Sankhara);
    }

    #[test]
    fn test_lineage_tracking() {
        let mut stream = UniversalFlowStream::new();

        // Emit first event
        let event1 = stream.emit_sensor(SensorInput::default(), 1000);
        assert!(event1.lineage.is_empty());

        // Second event should have first in lineage
        let event2 = stream.emit_sensor(SensorInput::default(), 2000);
        assert_eq!(event2.lineage.len(), 1);
        assert_eq!(event2.lineage[0], event1.id);
    }

    #[test]
    fn test_system_observation() {
        let stream = UniversalFlowStream::new();
        let obs = stream.create_observation();

        assert_eq!(obs.health, SystemHealth::Healthy);
        assert_eq!(obs.philosophical_state, PhilosophicalState::Yen);
    }
}
