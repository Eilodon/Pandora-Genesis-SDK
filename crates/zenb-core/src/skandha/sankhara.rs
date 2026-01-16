//! Sankhara (Hành) - Intent Formation and Decision Module
//! 
//! This module implements the "brain" of the AGOLOS decision-making pipeline.
//! It unifies:
//! - Dharma Filter (ethical alignment)
//! - Safety Guards (trauma, confidence, bounds, rate limit)
//! - EFE Calculator (Expected Free Energy for policy selection)
//! - Intent Tracking (for karmic feedback loop)
//!
//! # B.ONE V3: Di Hồn Đại Pháp
//! This is the result of the Grand Refactoring - moving all decision logic
//! from `Engine::make_control` into a dedicated, composable module.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::domain::ControlDecision;
use crate::universal_flow::FlowEventId;
use crate::safety::DharmaFilter;
use crate::policy::{ActionPolicy, BetaMetaLearner, EFECalculator, PolicyAdapter, PolicyEvaluation, PolicyLibrary};
use crate::safety_swarm::{
    BreathBoundsGuard, Clamp, ComfortGuard, ConfidenceGuard, Guard, PatternPatch,
    RateLimitGuard, ResourceGuard, TraumaGuard, TraumaSource,
};
use crate::belief::{BeliefState, Context as BeliefCtx, PhysioState};
use crate::estimator::Estimate;

// Re-export types used in the public API
pub use super::{AffectiveState, FormedIntent, IntentAction, PerceivedPattern, ProcessedForm};

// ============================================================================
// Intent Identification (Karmic Traceability)
// ============================================================================

/// Global counter for intent IDs (thread-safe)
static INTENT_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a deliberation intent.
/// Used to trace actions back to their decision context for learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentId(pub u64);

impl IntentId {
    /// Generate a new unique IntentId
    pub fn new() -> Self {
        Self(INTENT_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
    
    /// Create from raw value (for deserialization/testing)
    pub fn from_raw(value: u64) -> Self {
        Self(value)
    }
    
    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for IntentId {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Deliberation Result (The Single Decision Output)
// ============================================================================

/// Metadata associated with a deliberation for learning and auditing.
#[derive(Debug, Clone, Serialize)]
pub struct DeliberationMeta {
    /// Expected Free Energy score of the selected policy (lower = better)
    pub efe_score: f32,
    
    /// Confidence in the decision (0-1)
    pub confidence: f32,
    
    /// EFE precision parameter (beta) used for this decision
    pub efe_beta: f32,
    
    /// Epistemic value (exploration bonus)
    pub epistemic_value: f32,
    
    /// Pragmatic value (exploitation bonus)
    pub pragmatic_value: f32,
    
    /// Safety guard bits (which guards were triggered)
    pub guard_bits: u32,
    
    /// Causal graph success probability for this context
    pub causal_success_prob: f32,
}

impl Default for DeliberationMeta {
    fn default() -> Self {
        Self {
            efe_score: 0.0,
            confidence: 0.5,
            efe_beta: 1.0,
            epistemic_value: 0.0,
            pragmatic_value: 0.0,
            guard_bits: 0,
            causal_success_prob: 0.5,
        }
    }
}

/// The result of a complete deliberation cycle.
/// This is the SINGLE output from `ZenbSankhara::deliberate()`.
/// 
/// # Design Philosophy
/// - Struct over Tuple: Named fields provide clarity ("Chánh Ngữ")
/// - Intent Tracking: Every decision has a traceable ID for karma learning
/// - Audit Trail: Original intent preserved for debugging and transparency
#[derive(Debug, Clone, Serialize)]
pub struct DeliberationResult {
    /// The final control decision to send to actuators
    pub decision: ControlDecision,
    
    /// Unique identifier for this intent (for tracing outcomes)
    pub intent_id: IntentId,
    
    /// Link to FlowStream event (for persistence/lineage)
    pub flow_event_id: Option<FlowEventId>,
    
    /// Original intent before safety guards modified it
    pub original_intent: Option<FormedIntent>,
    
    /// Reason if the decision was adjusted or denied
    pub adjustment_reason: Option<String>,
    
    /// Whether this decision should be persisted to DB
    pub should_persist: bool,
    
    /// Metadata for learning (EFE scores, guard bits, etc.)
    pub meta: DeliberationMeta,
}

impl Default for DeliberationResult {
    fn default() -> Self {
        Self {
            decision: ControlDecision {
                target_rate_bpm: 6.0,
                confidence: 0.5,
                recommended_poll_interval_ms: 500,
                intent_id: None,
            },
            intent_id: IntentId::new(),
            flow_event_id: None,
            original_intent: None,
            adjustment_reason: None,
            should_persist: false,
            meta: DeliberationMeta::default(),
        }
    }
}

impl DeliberationResult {
    /// Create a safe fallback decision
    pub fn safe_fallback(reason: String) -> Self {
        Self {
            decision: ControlDecision {
                target_rate_bpm: 6.0, // Baseline gentle breathing
                confidence: 0.3,
                recommended_poll_interval_ms: 1000,
                intent_id: None,
            },
            intent_id: IntentId::new(),
            flow_event_id: None,
            original_intent: None,
            adjustment_reason: Some(reason),
            should_persist: true,
            meta: DeliberationMeta::default(),
        }
    }
    
    /// Check if decision was denied/adjusted
    pub fn was_adjusted(&self) -> bool {
        self.adjustment_reason.is_some()
    }
    
    /// Get denial reason if any
    pub fn deny_reason(&self) -> Option<&str> {
        self.adjustment_reason.as_deref()
    }
}

// ============================================================================
// Context Snapshot (For Karmic Learning)
// ============================================================================

/// Snapshot of the context at decision time.
/// Stored with IntentId for precise reward attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    /// Belief mode at decision time
    pub belief_mode: u8,
    
    /// Confidence at decision time
    pub confidence: f32,
    
    /// Free energy (surprise) at decision time
    pub free_energy: f32,
    
    /// Arousal level
    pub arousal: f32,
    
    /// Valence level
    pub valence: f32,
    
    /// Goal ID
    pub goal_id: i64,
    
    /// Pattern ID
    pub pattern_id: i64,
    
    /// Timestamp when decision was made
    pub timestamp_us: i64,
}

impl Default for ContextSnapshot {
    fn default() -> Self {
        Self {
            belief_mode: 0,
            confidence: 0.5,
            free_energy: 0.0,
            arousal: 0.5,
            valence: 0.0,
            goal_id: 0,
            pattern_id: 0,
            timestamp_us: 0,
        }
    }
}

// ============================================================================
// Intent Tracker (Persistent Karma Registry)
// ============================================================================

/// Tracked intent with full context for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedIntent {
    /// Unique intent ID
    pub id: IntentId,
    
    /// Link to FlowStream event
    pub flow_event_id: Option<FlowEventId>,
    
    /// The action that was taken
    pub action: IntentAction,
    
    /// Context snapshot at decision time
    pub context: ContextSnapshot,
    
    /// Target BPM that was decided
    pub target_bpm: f32,
    
    /// When outcome was received (None if pending)
    pub outcome_ts_us: Option<i64>,
    
    /// Whether action was successful (None if no feedback yet)
    pub success: Option<bool>,
}

/// Intent tracker for karmic learning.
/// Maintains a registry of recent intents for reward attribution.
#[derive(Debug, Default)]
pub struct IntentTracker {
    /// Active intents awaiting outcome
    active: std::collections::HashMap<IntentId, TrackedIntent>,
    
    /// Maximum age before intent expires (5 minutes default)
    max_age_us: i64,
    
    /// Maximum number of tracked intents
    max_capacity: usize,
}

impl IntentTracker {
    /// Create a new intent tracker with default settings
    pub fn new() -> Self {
        Self {
            active: std::collections::HashMap::new(),
            max_age_us: 5 * 60 * 1_000_000, // 5 minutes
            max_capacity: 100,
        }
    }
    
    /// Create a new intent tracker with custom settings
    pub fn with_config(max_age_us: i64, max_capacity: usize) -> Self {
        Self {
            active: std::collections::HashMap::new(),
            max_age_us,
            max_capacity,
        }
    }
    
    /// Track a new intent
    pub fn track(&mut self, intent: TrackedIntent) {
        // Evict old entries if at capacity
        if self.active.len() >= self.max_capacity {
            self.evict_oldest();
        }
        self.active.insert(intent.id, intent);
    }
    
    /// Get a tracked intent by ID
    pub fn get(&self, id: IntentId) -> Option<&TrackedIntent> {
        self.active.get(&id)
    }
    
    /// Get a mutable reference to a tracked intent
    pub fn get_mut(&mut self, id: IntentId) -> Option<&mut TrackedIntent> {
        self.active.get_mut(&id)
    }
    
    /// Record outcome for an intent
    pub fn record_outcome(&mut self, id: IntentId, success: bool, ts_us: i64) -> Option<&TrackedIntent> {
        if let Some(intent) = self.active.get_mut(&id) {
            intent.outcome_ts_us = Some(ts_us);
            intent.success = Some(success);
            Some(intent)
        } else {
            None
        }
    }
    
    /// Remove and return a completed intent
    pub fn complete(&mut self, id: IntentId) -> Option<TrackedIntent> {
        self.active.remove(&id)
    }
    
    /// Clean up expired intents
    pub fn cleanup(&mut self, current_ts_us: i64) {
        self.active.retain(|_, intent| {
            current_ts_us - intent.context.timestamp_us < self.max_age_us
        });
    }
    
    /// Evict the oldest intent
    fn evict_oldest(&mut self) {
        if let Some(oldest_id) = self
            .active
            .iter()
            .min_by_key(|(_, intent)| intent.context.timestamp_us)
            .map(|(id, _)| *id)
        {
            self.active.remove(&oldest_id);
        }
    }
    
    /// Get number of active intents
    pub fn len(&self) -> usize {
        self.active.len()
    }
    
    /// Check if tracker is empty
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }
    
    /// Get all pending intents (no outcome yet)
    pub fn pending(&self) -> impl Iterator<Item = &TrackedIntent> {
        self.active.values().filter(|i| i.success.is_none())
    }
    
    /// Serialize active intents for persistence
    pub fn serialize_for_persistence(&self) -> Vec<TrackedIntent> {
        self.active.values().cloned().collect()
    }
    
    /// Restore intents from persistence
    pub fn restore_from_persistence(&mut self, intents: Vec<TrackedIntent>) {
        for intent in intents {
            self.active.insert(intent.id, intent);
        }
    }
}

// ============================================================================
// KARMA ENGINE (B.ONE V3: Full Karmic Feedback Loop)
// ============================================================================

/// Karmic outcome for learning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KarmicOutcome {
    /// Positive outcome - action aligned with Dharma
    Sukha { reward: f32 },
    /// Negative outcome - karmic debt incurred
    Dukkha { severity: f32 },
    /// Neutral outcome - no significant change
    Upekkha,
}

impl KarmicOutcome {
    /// Calculate karma weight from outcome.
    pub fn karma_weight(&self) -> f32 {
        match self {
            Self::Sukha { reward } => reward.clamp(0.0, 1.0),
            Self::Dukkha { severity } => -severity.clamp(0.0, 1.0),
            Self::Upekkha => 0.0,
        }
    }
}

/// Statistics tracked by the Karma Engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KarmaStats {
    /// Total intents processed
    pub total_intents: u64,
    /// Successful intents
    pub sukha_count: u64,
    /// Failed intents
    pub dukkha_count: u64,
    /// Neutral outcomes
    pub upekkha_count: u64,
    /// Running karma balance
    pub karma_balance: f32,
    /// Average karma weight (EMA)
    pub karma_ema: f32,
    /// Success rate (0-1)
    pub success_rate: f32,
    /// Total karmic debt accumulated
    pub total_debt: f32,
    /// Total karmic merit accumulated
    pub total_merit: f32,
}

impl KarmaStats {
    /// Update statistics with new outcome.
    pub fn record_outcome(&mut self, outcome: &KarmicOutcome) {
        self.total_intents += 1;
        let weight = outcome.karma_weight();

        match outcome {
            KarmicOutcome::Sukha { .. } => {
                self.sukha_count += 1;
                self.total_merit += weight;
            }
            KarmicOutcome::Dukkha { .. } => {
                self.dukkha_count += 1;
                self.total_debt += weight.abs();
            }
            KarmicOutcome::Upekkha => {
                self.upekkha_count += 1;
            }
        }

        // Update karma balance
        self.karma_balance += weight;

        // Update EMA (alpha = 0.1)
        const KARMA_EMA_ALPHA: f32 = 0.1;
        self.karma_ema = KARMA_EMA_ALPHA * weight + (1.0 - KARMA_EMA_ALPHA) * self.karma_ema;

        // Update success rate
        self.success_rate = self.sukha_count as f32 / self.total_intents.max(1) as f32;
    }
}

/// The Karma Engine - Full karmic feedback loop for learning.
///
/// # B.ONE V3: Nhân Quả Báo Ứng
///
/// This engine tracks the karmic consequences of all intents:
/// - Every intent is recorded with full context
/// - Outcomes are mapped to karmic weights (Sukha/Dukkha/Upekkha)
/// - Statistics guide future policy selection
/// - Karmic debt triggers corrective behavior
///
/// # Design Philosophy
/// > "Karma is not punishment. Karma is the accumulated memory of outcomes
/// > that guides future decisions toward alignment with Dharma."
#[derive(Debug, Default)]
pub struct KarmaEngine {
    /// Intent tracker for tracing outcomes back to decisions
    pub intent_tracker: IntentTracker,

    /// Karma statistics
    pub stats: KarmaStats,

    /// Threshold for triggering karmic debt warning
    pub debt_warning_threshold: f32,

    /// Threshold for triggering karmic debt emergency
    pub debt_emergency_threshold: f32,

    /// Learning rate for policy weight updates
    pub learning_rate: f32,

    /// Recent outcomes for pattern analysis (circular buffer)
    recent_outcomes: Vec<(IntentId, KarmicOutcome)>,

    /// Maximum recent outcomes to keep
    max_recent_outcomes: usize,
}

impl KarmaEngine {
    /// Create a new Karma Engine with default settings.
    pub fn new() -> Self {
        Self {
            intent_tracker: IntentTracker::new(),
            stats: KarmaStats::default(),
            debt_warning_threshold: 3.0,
            debt_emergency_threshold: 5.0,
            learning_rate: 0.05,
            recent_outcomes: Vec::new(),
            max_recent_outcomes: 100,
        }
    }

    /// Create with custom thresholds.
    pub fn with_thresholds(debt_warning: f32, debt_emergency: f32) -> Self {
        Self {
            debt_warning_threshold: debt_warning,
            debt_emergency_threshold: debt_emergency,
            ..Default::default()
        }
    }

    /// Track a new intent (delegate to IntentTracker).
    pub fn track_intent(&mut self, intent: TrackedIntent) {
        self.intent_tracker.track(intent);
    }

    /// Record outcome for an intent and update karma.
    ///
    /// # Arguments
    /// * `intent_id` - The intent to record outcome for
    /// * `success` - Whether the action was successful
    /// * `severity` - Severity of failure (0.0-1.0), ignored if success=true
    /// * `ts_us` - Timestamp of outcome
    ///
    /// # Returns
    /// The karmic outcome and whether it triggered any thresholds.
    pub fn record_outcome(
        &mut self,
        intent_id: IntentId,
        success: bool,
        severity: f32,
        ts_us: i64,
    ) -> (KarmicOutcome, KarmaThresholdStatus) {
        // Determine karmic outcome
        let outcome = if success {
            KarmicOutcome::Sukha {
                reward: 1.0 - severity.min(0.5), // Higher severity = lower reward
            }
        } else if severity > 0.5 {
            KarmicOutcome::Dukkha { severity }
        } else if severity > 0.1 {
            KarmicOutcome::Dukkha {
                severity: severity * 0.5,
            }
        } else {
            KarmicOutcome::Upekkha
        };

        // Record in intent tracker
        self.intent_tracker.record_outcome(intent_id, success, ts_us);

        // Update statistics
        self.stats.record_outcome(&outcome);

        // Record in recent outcomes
        self.recent_outcomes.push((intent_id, outcome));
        if self.recent_outcomes.len() > self.max_recent_outcomes {
            self.recent_outcomes.remove(0);
        }

        // Check thresholds
        let threshold_status = self.check_thresholds();

        (outcome, threshold_status)
    }

    /// Check if karma thresholds are exceeded.
    fn check_thresholds(&self) -> KarmaThresholdStatus {
        let debt = self.stats.total_debt - self.stats.total_merit;

        if debt > self.debt_emergency_threshold {
            KarmaThresholdStatus::Emergency
        } else if debt > self.debt_warning_threshold {
            KarmaThresholdStatus::Warning
        } else {
            KarmaThresholdStatus::Normal
        }
    }

    /// Get current karma balance.
    pub fn karma_balance(&self) -> f32 {
        self.stats.karma_balance
    }

    /// Check if system is in karmic debt.
    pub fn is_in_debt(&self) -> bool {
        self.stats.karma_balance < -self.debt_warning_threshold
    }

    /// Check if system should enter corrective mode.
    pub fn should_enter_corrective_mode(&self) -> bool {
        self.check_thresholds() != KarmaThresholdStatus::Normal
    }

    /// Get success rate for a specific belief mode.
    pub fn success_rate_for_mode(&self, belief_mode: u8) -> f32 {
        let mut success_count = 0u32;
        let mut total_count = 0u32;

        for (id, outcome) in &self.recent_outcomes {
            if let Some(intent) = self.intent_tracker.get(*id) {
                if intent.context.belief_mode == belief_mode {
                    total_count += 1;
                    if matches!(outcome, KarmicOutcome::Sukha { .. }) {
                        success_count += 1;
                    }
                }
            }
        }

        if total_count == 0 {
            0.5 // Default neutral
        } else {
            success_count as f32 / total_count as f32
        }
    }

    /// Compute karma weight for a proposed action based on history.
    ///
    /// # Arguments
    /// * `belief_mode` - Current belief mode
    /// * `action` - Proposed action type
    ///
    /// # Returns
    /// Expected karma weight based on historical outcomes.
    pub fn predict_karma_weight(&self, belief_mode: u8, action: &IntentAction) -> f32 {
        let mut total_weight = 0.0f32;
        let mut count = 0u32;

        for (id, outcome) in &self.recent_outcomes {
            if let Some(intent) = self.intent_tracker.get(*id) {
                if intent.context.belief_mode == belief_mode {
                    // Check if action type matches
                    let action_matches = match (&intent.action, action) {
                        (IntentAction::GuideBreath { .. }, IntentAction::GuideBreath { .. }) => true,
                        (IntentAction::Observe, IntentAction::Observe) => true,
                        (IntentAction::SuggestIntervention, IntentAction::SuggestIntervention) => true,
                        _ => false,
                    };

                    if action_matches {
                        total_weight += outcome.karma_weight();
                        count += 1;
                    }
                }
            }
        }

        if count == 0 {
            0.0 // No history, neutral prediction
        } else {
            total_weight / count as f32
        }
    }

    /// Get karma statistics.
    pub fn stats(&self) -> &KarmaStats {
        &self.stats
    }

    /// Reset karma statistics (use with caution).
    pub fn reset_stats(&mut self) {
        self.stats = KarmaStats::default();
        self.recent_outcomes.clear();
    }

    /// Cleanup expired intents.
    pub fn cleanup(&mut self, current_ts_us: i64) {
        self.intent_tracker.cleanup(current_ts_us);
    }

    /// Get a summary string for logging.
    pub fn summary(&self) -> String {
        format!(
            "Karma[balance={:.2}, rate={:.1}%, debt={:.2}, merit={:.2}]",
            self.stats.karma_balance,
            self.stats.success_rate * 100.0,
            self.stats.total_debt,
            self.stats.total_merit
        )
    }
}

/// Status of karma thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KarmaThresholdStatus {
    /// Karma is within normal bounds
    Normal,
    /// Karma debt is above warning threshold
    Warning,
    /// Karma debt is above emergency threshold
    Emergency,
}

// ============================================================================
// Guard Reference Types (For Engine Integration)
// ============================================================================

/// Safety guard configuration passed to deliberation.
/// This allows Engine to provide state-dependent guard parameters.
#[derive(Debug, Clone)]
pub struct GuardConfig {
    /// Trauma severity thresholds
    pub trauma_hard_th: f32,
    pub trauma_soft_th: f32,
    
    /// Minimum confidence for action
    pub min_confidence: f32,
    
    /// Breath rate bounds
    pub rr_min: f32,
    pub rr_max: f32,
    pub max_delta_rr_per_min: f32,
    pub hold_max_sec: f32,
    
    /// Rate limiting (minimum seconds between decisions)
    pub min_interval_sec: f32,
    
    /// Last patch timestamp for rate limiting
    pub last_patch_sec: Option<f32>,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            trauma_hard_th: 3.0,
            trauma_soft_th: 1.0,
            min_confidence: 0.2,
            rr_min: 4.0,
            rr_max: 12.0,
            max_delta_rr_per_min: 6.0,
            hold_max_sec: 60.0,
            min_interval_sec: 10.0,
            last_patch_sec: None,
        }
    }
}

// ============================================================================
// Enhanced ZenbSankhara (The Unified Brain)
// ============================================================================

/// The unified decision-making module.
/// 
/// # Responsibilities
/// - Ethical filtering (DharmaFilter)
/// - Safety guard evaluation
/// - EFE-based policy selection
/// - Intent tracking for learning
/// 
/// # Usage
/// ```ignore
/// let result = sankhara.deliberate(
///     &estimate,
///     &belief_state,
///     &context,
///     &guard_config,
///     ts_us,
///     &trauma_cache,
///     Some(0.8),
/// );
/// ```
#[derive(Debug)]
pub struct UnifiedSankhara {
    /// Dharma filter for ethical alignment
    pub dharma: DharmaFilter,
    
    /// Intent tracker for karmic learning
    pub intent_tracker: IntentTracker,
    
    /// Policy adapter for learning from verify/outcomes
    pub policy_adapter: PolicyAdapter,

    /// Meta-learner for EFE precision
    pub meta_learner: BetaMetaLearner,
    
    /// EFE precision parameter (beta)
    pub efe_precision_beta: f32,
    
    /// Meta-learner enabled flag
    pub meta_learning_enabled: bool,

    // === State Tracking for Learning ===
    pub last_executed_policy: Option<ActionPolicy>,
    pub last_deliberation: Option<DeliberationMeta>,
}

impl Default for UnifiedSankhara {
    fn default() -> Self {
        Self {
            dharma: DharmaFilter::default(),
            intent_tracker: IntentTracker::new(),
            policy_adapter: PolicyAdapter::default(),
            meta_learner: BetaMetaLearner::default(),
            efe_precision_beta: 1.0,
            meta_learning_enabled: true,
            last_executed_policy: None,
            last_deliberation: None,
        }
    }
}

impl UnifiedSankhara {
    /// Create a new UnifiedSankhara with default settings
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create with custom EFE precision
    pub fn with_efe_beta(efe_beta: f32) -> Self {
        Self {
            efe_precision_beta: efe_beta,
            ..Default::default()
        }
    }
    
    /// Update EFE precision (for meta-learning)
    pub fn set_efe_beta(&mut self, beta: f32) {
        self.efe_precision_beta = beta.clamp(0.1, 10.0);
    }

    /// DELIBERATE: The Core Decision Loop
    /// 
    /// Unifies Safety (Guards) and Will (EFE) into a single atomic decision.
    /// This is the "Brain" of the system.
    pub fn deliberate(
        &mut self,
        estimate: &Estimate,
        belief_state: &BeliefState,
        context: &BeliefCtx,
        guard_config: &GuardConfig,
        ts_us: i64,
        trauma_source: &dyn TraumaSource,
        causal_success_prob: Option<f32>,
        pattern_id: i64,
        goal_id: i64,
    ) -> DeliberationResult {
        let mut meta = DeliberationMeta::default();
        meta.efe_beta = self.efe_precision_beta;
        meta.causal_success_prob = causal_success_prob.unwrap_or(0.5);

        // 0. Base Calculation
        let base_bpm = 6.0;
        let proposed_bpm = estimate.rr_bpm.unwrap_or(base_bpm)
            .clamp(guard_config.rr_min, guard_config.rr_max);

        // 1. EFE Policy Selection (The "Will")
        let efe_calc = EFECalculator::new(self.efe_precision_beta);
        
        let policies = vec![
            PolicyLibrary::calming_breath(),
            PolicyLibrary::energizing_breath(),
            PolicyLibrary::focus_mode(),
            PolicyLibrary::observe(),
        ];
        
        // Predict future (Simplified: current belief as proxy)
        // TODO: Integrate CausalGraph prediction properly
        let predicted_state = belief_state.to_5mode_array();
        let predicted_uncertainty = belief_state.conf; // Should be variance

        let mut evaluations: Vec<PolicyEvaluation> = policies
            .iter()
            .map(|policy| {
                efe_calc.compute_efe(
                    policy,
                    &belief_state.to_5mode_array(),
                    crate::belief::uncertainty_from_confidence(belief_state.conf),
                    &predicted_state,
                    crate::belief::uncertainty_from_confidence(predicted_uncertainty),
                )
            })
            .collect();

        // Deterministic sampling
        let rng_value = ((ts_us % 1000) as f32) / 1000.0;
        efe_calc.compute_selection_probabilities(&mut evaluations);
        
        let selected_policy = efe_calc.sample_policy(&evaluations, rng_value).clone();
        
        // Get metadata from selection
        if let Some(eval) = evaluations.iter().find(|e| e.policy.description() == selected_policy.description()) {
            meta.efe_score = eval.expected_free_energy;
            meta.epistemic_value = eval.epistemic_value;
            meta.pragmatic_value = eval.pragmatic_value;
        }

        // 2. Policy Adaptation & Causal Veto (The "Intuition")
        let mut final_policy = selected_policy;
        let mut adjustment_reason = None;

        // A. Causal Veto
        if let Some(prob) = causal_success_prob {
             if prob < 0.3 && meta.confidence > 0.5 {
                 // Historical failure likely, force observation
                 final_policy = PolicyLibrary::observe();
                 adjustment_reason = Some("causal_veto".to_string());
             }
        }

        // B. Policy Adapter Masking
        if self.policy_adapter.is_policy_masked(&final_policy) {
            final_policy = PolicyLibrary::observe();
            adjustment_reason = Some("adapter_masked".to_string());
        }

        // 3. Convert to Control Decision
        let control_decision = final_policy.to_control_decision(proposed_bpm, estimate.confidence);

        // 4. Safety Guards (The "Conscience")
        // Build PatternPatch
        let patch = PatternPatch {
            target_bpm: control_decision.target_rate_bpm,
            hold_sec: 30.0, // Default hold limit
            pattern_id,
            goal: goal_id,
        };
        
        let phys_state = PhysioState {
            hr_bpm: estimate.hr_bpm,
            rr_bpm: estimate.rr_bpm,
            rmssd: estimate.rmssd,
            confidence: estimate.confidence,
        };

        // Instantiate Guards
        let guard_results = self.check_safety_guards(
            &patch, 
            belief_state, 
            &phys_state, 
            context, 
            guard_config, 
            ts_us, 
            trauma_source
        );

        // 5. Finalize Result
        match guard_results {
            Ok((safe_patch, rule_bits)) => {
                meta.guard_bits = rule_bits;
                
                // Create Intent
                let intent_id = IntentId::new();
                let result = DeliberationResult {
                    decision: ControlDecision {
                        target_rate_bpm: safe_patch.target_bpm,
                        confidence: control_decision.confidence,
                        recommended_poll_interval_ms: control_decision.recommended_poll_interval_ms,
                        intent_id: None, // Will be set by Engine
                    },
                    intent_id,
                    flow_event_id: None, // Assigned by FlowStream later
                    original_intent: None, // TODO fill from FormedIntent
                    adjustment_reason, 
                    should_persist: true,
                    meta: meta.clone()
                };

                // Track Intent (Karma)
                let tracked = TrackedIntent {
                    id: intent_id,
                    flow_event_id: None,
                    action: intent_action_from_policy(&final_policy),
                    context: ContextSnapshot {
                        belief_mode: belief_state.mode as u8,
                        confidence: belief_state.conf,
                        free_energy: 0.0, // TODO
                        arousal: belief_state.p[1], // Stress mode
                        valence: 0.0,
                        goal_id: 0,
                        pattern_id: 0,
                        timestamp_us: ts_us,
                    },
                    target_bpm: safe_patch.target_bpm,
                    outcome_ts_us: None,
                    success: None,
                };
                self.intent_tracker.track(tracked);

                // Update internal state for feedback loop
                self.last_executed_policy = Some(final_policy);
                self.last_deliberation = Some(meta.clone());

                result
            },
            Err(deny_reason) => {
                // B.ONE V3: Track denied intents too for complete karma history
                // This enables learning from failures and trauma analysis
                let intent_id = IntentId::new();
                
                // Track the denied intent with full context
                let tracked = TrackedIntent {
                    id: intent_id,
                    flow_event_id: None,
                    action: intent_action_from_policy(&final_policy),
                    context: ContextSnapshot {
                        belief_mode: belief_state.mode as u8,
                        confidence: belief_state.conf,
                        free_energy: 0.0,
                        arousal: belief_state.p[1],
                        valence: 0.0,
                        goal_id: 0,
                        pattern_id: 0,
                        timestamp_us: ts_us,
                    },
                    target_bpm: patch.target_bpm,
                    outcome_ts_us: None,
                    success: None, // Will be marked as failure implicitly (denied)
                };
                self.intent_tracker.track(tracked);
                
                // Create fallback with intent_id attached
                let mut fallback = DeliberationResult::safe_fallback(deny_reason.to_string());
                fallback.intent_id = intent_id; // Attach intent to fallback too
                fallback
            }
        }
    }

    /// Helper to run safety swarm
    fn check_safety_guards(
        &self,
        patch: &PatternPatch,
        belief: &BeliefState,
        phys: &PhysioState,
        ctx: &BeliefCtx,
        cfg: &GuardConfig,
        ts_us: i64,
        trauma_source: &dyn TraumaSource,
    ) -> Result<(PatternPatch, u32), &'static str> {
        let mut guards: Vec<Box<dyn Guard>> = Vec::new();
        
        // 1. Trauma Guard (The most critical)
        guards.push(Box::new(TraumaGuard {
            source: trauma_source,
            hard_th: cfg.trauma_hard_th,
            soft_th: cfg.trauma_soft_th,
        }));

        // 2. Confidence Guard
        guards.push(Box::new(ConfidenceGuard {
            min_conf: cfg.min_confidence,
        }));

        // 3. Breath Bounds
        guards.push(Box::new(BreathBoundsGuard {
            clamp: Clamp {
                rr_min: cfg.rr_min,
                rr_max: cfg.rr_max,
                hold_max_sec: cfg.hold_max_sec,
                max_delta_rr_per_min: cfg.max_delta_rr_per_min,
            },
        }));

        // 4. Rate Limiting
        guards.push(Box::new(RateLimitGuard {
            min_interval_sec: cfg.min_interval_sec,
            last_patch_sec: cfg.last_patch_sec,
        }));

        // 5. Soft Guards
        guards.push(Box::new(ComfortGuard));
        guards.push(Box::new(ResourceGuard));

        crate::safety_swarm::decide(
            &guards,
            patch,
            belief,
            phys,
            ctx,
            ts_us
        )
    }

    /// Apply feedback from outcomes to update internal models (Learning).
    /// 
    /// Updates:
    /// - Policy Adapter (Q-Learning / Masking)
    /// - Meta-Learner (EFE Precision Beta)
    pub fn apply_feedback(&mut self, success: bool, reward: f32, state_hash: &str) {
        // 1. Update Policy Adapter
        if let Some(ref policy) = self.last_executed_policy {
            let boost = self.policy_adapter.update_with_outcome(state_hash, policy, reward, success);
            
            // Apply exploration boost to EFE precision (inverse relationship)
            // Higher exploration = lower precision (more uniform sampling)
            self.efe_precision_beta = (self.efe_precision_beta / boost).clamp(0.1, 10.0);
        }

        // 2. Update Beta Meta-Learner
        if self.meta_learning_enabled {
            if let Some(ref meta) = self.last_deliberation {
                let was_exploratory = meta.epistemic_value > meta.pragmatic_value;
                let old_beta = self.efe_precision_beta;
                
                self.efe_precision_beta = self.meta_learner.update_beta(
                    self.efe_precision_beta,
                    was_exploratory,
                    success
                );

                if (self.efe_precision_beta - old_beta).abs() > 0.001 {
                    log::info!(
                        "EFE Adaptation: Beta {:.3} -> {:.3} (Success: {}, Explored: {})",
                        old_beta,
                        self.efe_precision_beta,
                        success,
                        was_exploratory
                    );
                }
            }
        }
    }
    
    /// Apply feedback using IntentId to retrieve exact decision context.
    /// 
    /// This is the V2 learning API that enables true karmic feedback loop:
    /// - Looks up the original TrackedIntent by ID  
    /// - Reconstructs the exact decision context
    /// - Updates PolicyAdapter with context-specific state hash
    /// - Records outcome in IntentTracker for persistence
    /// 
    /// # Arguments
    /// * `intent_id` - The IntentId returned with the original decision
    /// * `success` - Whether the action led to positive outcome
    /// * `severity` - Severity of negative outcome (0.0-1.0)
    /// * `ts_us` - Timestamp of outcome feedback
    pub fn apply_feedback_v2(&mut self, intent_id: IntentId, success: bool, severity: f32, ts_us: i64) {
        // 1. Lookup intent in tracker
        if let Some(intent) = self.intent_tracker.get(intent_id) {
            // 2. Reconstruct state hash from original context
            let state_hash = format!("mode_{}_goal_{}_pattern_{}", 
                intent.context.belief_mode,
                intent.context.goal_id,
                intent.context.pattern_id
            );
            
            // 3. Convert IntentAction to ActionPolicy for adapter
            let policy = self.policy_from_action(&intent.action);
            
            // 4. Compute reward
            let reward = if success { 1.0 } else { -severity };
            
            // 5. Update Policy Adapter with exact context
            let boost = self.policy_adapter.update_with_outcome(&state_hash, &policy, reward, success);
            
            // Apply exploration boost
            self.efe_precision_beta = (self.efe_precision_beta / boost).clamp(0.1, 10.0);
            
            // 6. Record outcome in tracker
            self.intent_tracker.record_outcome(intent_id, success, ts_us);
            
            log::info!(
                "Karmic Feedback: Intent {} -> {} (reward={:.2}, state={})",
                intent_id.raw(),
                if success { "SUCCESS" } else { "FAILURE" },
                reward,
                state_hash
            );
        } else {
            log::warn!("Intent {} not found in tracker, falling back to legacy learning", intent_id.raw());
        }
    }
    
    /// Convert IntentAction back to ActionPolicy for policy adapter.
    fn policy_from_action(&self, action: &IntentAction) -> ActionPolicy {
        match action {
            IntentAction::GuideBreath { target_bpm } => {
                if *target_bpm <= 6 {
                    PolicyLibrary::calming_breath()
                } else if *target_bpm >= 8 {
                    PolicyLibrary::energizing_breath()
                } else {
                    PolicyLibrary::focus_mode()
                }
            },
            IntentAction::Observe => PolicyLibrary::observe(),
            IntentAction::SuggestIntervention => PolicyLibrary::suggest_rest(),
            IntentAction::Alert => PolicyLibrary::suggest_rest(),
            IntentAction::SafeFallback => PolicyLibrary::observe(),
        }
    }
}

// Helper to map Policy to IntentAction for tracking
fn intent_action_from_policy(policy: &ActionPolicy) -> IntentAction {
    match policy {
        ActionPolicy::NoAction => IntentAction::Observe,
        ActionPolicy::GuidanceBreath(p) => IntentAction::GuideBreath { target_bpm: p.target_bpm as u8 },
        ActionPolicy::DigitalIntervention(d) => match d.action {
            crate::policy::DigitalActionType::SuggestBreak => IntentAction::SuggestIntervention,
            _ => IntentAction::SuggestIntervention, 
        },
    }
}

// ============================================================================
// Pipeline Integration (SankharaSkandha Trait)
// ============================================================================

impl super::SankharaSkandha for UnifiedSankhara {
    fn form_intent(
        &mut self,
        pattern: &PerceivedPattern,
        affect: &AffectiveState,
    ) -> FormedIntent {
        // === VAJRA-VOID: PRIORITY KARMIC DEBT HANDLING ===
        // If karmic debt detected at Vedana stage, prioritize corrective action
        if affect.is_karmic_debt {
            return FormedIntent {
                action: IntentAction::SafeFallback,
                alignment: 0.3,
                is_sanctioned: true,
                reasoning: format!(
                    "Karmic debt (weight={:.2}) -> corrective action",
                    affect.karma_weight
                ),
            };
        }
        
        // === TRAUMA-ASSOCIATED PATTERN -> SAFE FALLBACK ===
        if pattern.is_trauma_associated {
            return FormedIntent {
                action: IntentAction::SafeFallback,
                alignment: 0.5,
                is_sanctioned: true,
                reasoning: "Trauma-associated pattern -> safe fallback".to_string(),
            };
        }
        
        // === HEURISTIC ACTION SELECTION (Fast Thinking) ===
        // Note: The heavy "Slow Thinking" deliberation happens in deliberate()
        let action = if affect.arousal > 0.8 {
            IntentAction::GuideBreath { target_bpm: 6 } // Slow, calming
        } else if affect.arousal < 0.2 && affect.valence < -0.2 {
            IntentAction::GuideBreath { target_bpm: 8 } // Slightly faster
        } else if pattern.similarity > 0.8 && affect.valence < 0.0 {
            IntentAction::GuideBreath { target_bpm: 6 }
        } else {
            IntentAction::Observe
        };

        FormedIntent {
            action,
            alignment: 1.0,
            is_sanctioned: true,
            reasoning: "Heuristic policy".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_intent_id_uniqueness() {
        let id1 = IntentId::new();
        let id2 = IntentId::new();
        assert_ne!(id1, id2);
        assert!(id2.0 > id1.0);
    }
    
    #[test]
    fn test_deliberation_result_default() {
        let result = DeliberationResult::default();
        assert_eq!(result.decision.target_rate_bpm, 6.0);
        assert!(!result.was_adjusted());
    }
    
    #[test]
    fn test_safe_fallback() {
        let result = DeliberationResult::safe_fallback("trauma_detected".to_string());
        assert!(result.was_adjusted());
        assert_eq!(result.deny_reason(), Some("trauma_detected"));
        assert!(result.should_persist);
    }
    
    #[test]
    fn test_intent_tracker_basic() {
        let mut tracker = IntentTracker::new();
        
        let intent = TrackedIntent {
            id: IntentId::new(),
            flow_event_id: None,
            action: IntentAction::GuideBreath { target_bpm: 6 },
            context: ContextSnapshot::default(),
            target_bpm: 6.0,
            outcome_ts_us: None,
            success: None,
        };
        
        let id = intent.id;
        tracker.track(intent);
        
        assert_eq!(tracker.len(), 1);
        assert!(tracker.get(id).is_some());
        
        // Record outcome
        tracker.record_outcome(id, true, 1000);
        let completed = tracker.get(id).unwrap();
        assert_eq!(completed.success, Some(true));
    }
    
    #[test]
    fn test_intent_tracker_capacity() {
        let mut tracker = IntentTracker::with_config(1_000_000, 3);

        for _ in 0..5 {
            let intent = TrackedIntent {
                id: IntentId::new(),
                flow_event_id: None,
                action: IntentAction::Observe,
                context: ContextSnapshot::default(),
                target_bpm: 6.0,
                outcome_ts_us: None,
                success: None,
            };
            tracker.track(intent);
        }

        // Should have evicted to stay at capacity
        assert!(tracker.len() <= 3);
    }

    // =========================================================================
    // KARMA ENGINE TESTS
    // =========================================================================

    #[test]
    fn test_karma_engine_creation() {
        let engine = KarmaEngine::new();
        assert_eq!(engine.karma_balance(), 0.0);
        assert!(!engine.is_in_debt());
        assert!(!engine.should_enter_corrective_mode());
    }

    #[test]
    fn test_karma_outcome_weights() {
        let sukha = KarmicOutcome::Sukha { reward: 0.8 };
        assert!((sukha.karma_weight() - 0.8).abs() < 0.001);

        let dukkha = KarmicOutcome::Dukkha { severity: 0.6 };
        assert!((dukkha.karma_weight() - (-0.6)).abs() < 0.001);

        let upekkha = KarmicOutcome::Upekkha;
        assert_eq!(upekkha.karma_weight(), 0.0);
    }

    #[test]
    fn test_karma_engine_track_outcomes() {
        let mut engine = KarmaEngine::new();

        // Track an intent
        let intent = TrackedIntent {
            id: IntentId::new(),
            flow_event_id: None,
            action: IntentAction::GuideBreath { target_bpm: 6 },
            context: ContextSnapshot::default(),
            target_bpm: 6.0,
            outcome_ts_us: None,
            success: None,
        };
        let intent_id = intent.id;
        engine.track_intent(intent);

        // Record successful outcome
        let (outcome, status) = engine.record_outcome(intent_id, true, 0.0, 1000);

        assert!(matches!(outcome, KarmicOutcome::Sukha { .. }));
        assert_eq!(status, KarmaThresholdStatus::Normal);
        assert!(engine.karma_balance() > 0.0);
    }

    #[test]
    fn test_karma_engine_debt_threshold() {
        let mut engine = KarmaEngine::with_thresholds(1.0, 2.0); // Low thresholds for testing

        // Generate multiple failures to accumulate debt
        for i in 0..5 {
            let intent = TrackedIntent {
                id: IntentId::new(),
                flow_event_id: None,
                action: IntentAction::GuideBreath { target_bpm: 6 },
                context: ContextSnapshot::default(),
                target_bpm: 6.0,
                outcome_ts_us: None,
                success: None,
            };
            let intent_id = intent.id;
            engine.track_intent(intent);

            let (_, status) = engine.record_outcome(intent_id, false, 0.8, i * 1000);

            if i >= 3 {
                assert!(
                    status != KarmaThresholdStatus::Normal,
                    "Expected threshold breach at iteration {}", i
                );
            }
        }
    }

    #[test]
    fn test_karma_stats_tracking() {
        let mut stats = KarmaStats::default();

        stats.record_outcome(&KarmicOutcome::Sukha { reward: 1.0 });
        assert_eq!(stats.sukha_count, 1);
        assert_eq!(stats.total_intents, 1);
        assert!(stats.success_rate > 0.99);

        stats.record_outcome(&KarmicOutcome::Dukkha { severity: 0.5 });
        assert_eq!(stats.dukkha_count, 1);
        assert_eq!(stats.total_intents, 2);
        assert!((stats.success_rate - 0.5).abs() < 0.01);

        stats.record_outcome(&KarmicOutcome::Upekkha);
        assert_eq!(stats.upekkha_count, 1);
        assert_eq!(stats.total_intents, 3);
    }

    #[test]
    fn test_karma_engine_summary() {
        let engine = KarmaEngine::new();
        let summary = engine.summary();
        assert!(summary.contains("Karma"));
        assert!(summary.contains("balance="));
        assert!(summary.contains("rate="));
    }
}
