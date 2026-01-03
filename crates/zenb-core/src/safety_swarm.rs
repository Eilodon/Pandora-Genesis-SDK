use crate::belief::BeliefState;
use crate::belief::Context as BeliefCtx;
use crate::belief::PhysioState;
use blake3::Hasher;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraumaHit {
    pub sev_eff: f32,
    pub count: u32,
    pub inhibit_until_ts_us: i64,
    pub last_ts_us: i64,
}

pub trait TraumaSource {
    fn query_trauma(&self, sig_hash: &[u8], now_ts_us: i64) -> Option<TraumaHit>;
}

/// TraumaRegistry for recording negative feedback from action outcomes.
/// This enables the system to learn from failures and become more conservative.
pub struct TraumaRegistry {
    /// In-memory trauma records (sig_hash -> TraumaHit)
    records: std::collections::HashMap<Vec<u8>, TraumaHit>,
}

impl TraumaRegistry {
    pub fn new() -> Self {
        Self {
            records: std::collections::HashMap::new(),
        }
    }

    /// Record negative feedback for a specific context.
    /// Implements exponential backoff: each failure increases inhibit duration.
    ///
    /// # Arguments
    /// * `context_hash` - Blake3 hash of the context (goal, mode, pattern, environment)
    /// * `action_type` - Type of action that failed (for logging/debugging)
    /// * `now_ts_us` - Current timestamp in microseconds
    /// * `severity` - Severity of the failure (0.0 to 5.0)
    ///
    /// # Exponential Backoff Strategy
    /// - First failure: 1 hour inhibit
    /// - Second failure: 2 hours inhibit
    /// - Third failure: 4 hours inhibit
    /// - Nth failure: 2^(N-1) hours inhibit (capped at 24 hours)
    pub fn record_negative_feedback(
        &mut self,
        context_hash: [u8; 32],
        action_type: String,
        now_ts_us: i64,
        severity: f32,
    ) {
        const BASE_INHIBIT_HOURS: i64 = 1;
        const MAX_INHIBIT_HOURS: i64 = 24;
        const SEVERITY_EMA_BETA: f32 = 0.3;

        let key = context_hash.to_vec();

        let (new_count, new_sev, inhibit_duration_us) =
            if let Some(existing) = self.records.get(&key) {
                // Existing trauma record - update with exponential backoff
                let new_count = existing.count.saturating_add(1);

                // Exponential backoff: 2^(count-1) hours, capped at MAX_INHIBIT_HOURS
                let backoff_hours =
                    (BASE_INHIBIT_HOURS * (1 << (new_count.min(10) - 1))).min(MAX_INHIBIT_HOURS);
                let inhibit_duration = backoff_hours * 3_600_000_000; // hours to microseconds

                // EMA update for severity
                let new_sev =
                    existing.sev_eff * (1.0 - SEVERITY_EMA_BETA) + severity * SEVERITY_EMA_BETA;

                (new_count, new_sev, inhibit_duration)
            } else {
                // New trauma record - first failure
                let inhibit_duration = BASE_INHIBIT_HOURS * 3_600_000_000; // 1 hour in microseconds
                (1, severity, inhibit_duration)
            };

        let hit = TraumaHit {
            sev_eff: new_sev.clamp(0.0, 5.0),
            count: new_count,
            inhibit_until_ts_us: now_ts_us + inhibit_duration_us,
            last_ts_us: now_ts_us,
        };

        self.records.insert(key, hit);

        // Log for debugging
        eprintln!(
            "TRAUMA RECORDED: action={}, count={}, severity={:.2}, inhibit_until=+{}h",
            action_type,
            new_count,
            new_sev,
            inhibit_duration_us / 3_600_000_000
        );
    }

    /// Query trauma for a specific context.
    pub fn query(&self, context_hash: &[u8]) -> Option<TraumaHit> {
        self.records.get(context_hash).copied()
    }

    /// Clear all trauma records (for testing or reset).
    pub fn clear(&mut self) {
        self.records.clear();
    }

    /// Get all trauma records for persistence.
    /// Returns iterator over (context_hash, TraumaHit) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &TraumaHit)> {
        self.records.iter()
    }

    /// Get number of trauma records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl TraumaSource for TraumaRegistry {
    fn query_trauma(&self, sig_hash: &[u8], _now_ts_us: i64) -> Option<TraumaHit> {
        self.query(sig_hash)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Clamp {
    pub rr_min: f32,
    pub rr_max: f32,
    pub hold_max_sec: f32,
    pub max_delta_rr_per_min: f32,
}

impl Default for Clamp {
    fn default() -> Self {
        Self {
            rr_min: 4.0,
            rr_max: 12.0,
            hold_max_sec: 60.0,
            max_delta_rr_per_min: 6.0,
        }
    }
}

pub enum Vote {
    Deny(&'static str),
    Allow(Clamp, f32, u32), // clamp, score, reason_bits
}

pub trait Guard {
    fn name(&self) -> &'static str;
    fn vote(
        &self,
        belief: &BeliefState,
        phys: &PhysioState,
        patch: &PatternPatch,
        ctx: &BeliefCtx,
        now_ts_us: i64,
    ) -> Vote;
}

#[derive(Debug, Clone, Copy)]
pub struct PatternPatch {
    pub target_bpm: f32,
    pub hold_sec: f32,
    pub pattern_id: i64,
    pub goal: i64,
}

impl PatternPatch {
    pub fn apply_clamp(&mut self, c: &Clamp) {
        // rr to bpm mapping: rr_bpm == breaths per minute
        self.target_bpm = self.target_bpm.clamp(c.rr_min, c.rr_max);
        self.hold_sec = self.hold_sec.min(c.hold_max_sec);
    }
}

/// Hard guard: confidence threshold
pub struct ConfidenceGuard {
    pub min_conf: f32,
}
impl Guard for ConfidenceGuard {
    fn name(&self) -> &'static str {
        "ConfidenceGuard"
    }
    fn vote(
        &self,
        belief: &BeliefState,
        _phys: &PhysioState,
        _patch: &PatternPatch,
        _ctx: &BeliefCtx,
        _now_ts_us: i64,
    ) -> Vote {
        if belief.conf < self.min_conf {
            Vote::Deny("low_belief_conf")
        } else {
            Vote::Allow(Clamp::default(), 1.0, 0x1)
        }
    }
}

/// Hard guard: breath bounds
pub struct BreathBoundsGuard {
    pub clamp: Clamp,
}
impl Guard for BreathBoundsGuard {
    fn name(&self) -> &'static str {
        "BreathBoundsGuard"
    }
    fn vote(
        &self,
        _belief: &BeliefState,
        _phys: &PhysioState,
        _patch: &PatternPatch,
        _ctx: &BeliefCtx,
        _now_ts_us: i64,
    ) -> Vote {
        Vote::Allow(self.clamp, 1.0, 0x2)
    }
}

/// Rate limit guard
pub struct RateLimitGuard {
    pub min_interval_sec: f32,
    pub last_patch_sec: Option<f32>,
}
impl Guard for RateLimitGuard {
    fn name(&self) -> &'static str {
        "RateLimitGuard"
    }
    fn vote(
        &self,
        _belief: &BeliefState,
        _phys: &PhysioState,
        _patch: &PatternPatch,
        _ctx: &BeliefCtx,
        _now_ts_us: i64,
    ) -> Vote {
        if let Some(last) = self.last_patch_sec {
            if last < self.min_interval_sec {
                return Vote::Deny("rate_limited");
            }
        }
        Vote::Allow(Clamp::default(), 1.0, 0x4)
    }
}

/// Soft guard: comfort that reduces hold length when Stress
pub struct ComfortGuard;
impl Guard for ComfortGuard {
    fn name(&self) -> &'static str {
        "ComfortGuard"
    }
    fn vote(
        &self,
        belief: &BeliefState,
        _phys: &PhysioState,
        patch: &PatternPatch,
        _ctx: &BeliefCtx,
        _now_ts_us: i64,
    ) -> Vote {
        if belief.mode == crate::belief::BeliefBasis::Stress && patch.hold_sec > 30.0 {
            let mut c = Clamp::default();
            c.hold_max_sec = 30.0;
            Vote::Allow(c, 0.5, 0x10)
        } else {
            Vote::Allow(Clamp::default(), 0.8, 0x10)
        }
    }
}

/// Soft guard: resource (charging) - penalize heavy features when not charging
pub struct ResourceGuard;
impl Guard for ResourceGuard {
    fn name(&self) -> &'static str {
        "ResourceGuard"
    }
    fn vote(
        &self,
        _belief: &BeliefState,
        _phys: &PhysioState,
        _patch: &PatternPatch,
        ctx: &BeliefCtx,
        _now_ts_us: i64,
    ) -> Vote {
        // If not charging, be conservative and reduce hold_max regardless of incoming patch length (downsample heavy features)
        if !ctx.is_charging {
            let mut c = Clamp::default();
            c.hold_max_sec = c.hold_max_sec.min(45.0);
            Vote::Allow(c, 0.5, 0x20)
        } else {
            Vote::Allow(Clamp::default(), 0.9, 0x20)
        }
    }
}

pub struct TraumaGuard<'a> {
    pub source: &'a dyn TraumaSource,
    pub hard_th: f32,
    pub soft_th: f32,
}

impl<'a> TraumaGuard<'a> {
    fn sig_hash(patch: &PatternPatch, belief: &BeliefState, ctx: &BeliefCtx) -> [u8; 32] {
        trauma_sig_hash(patch.goal, belief.mode as u8, patch.pattern_id, ctx)
    }
}

pub fn trauma_sig_hash(goal: i64, mode: u8, pattern_id: i64, ctx: &BeliefCtx) -> [u8; 32] {
    // Use continuous trigonometric encoding for temporal continuity
    // Map 0-23 hour to circle (0 to 2*PI)
    let hour_angle = (ctx.local_hour as f32 / 24.0) * 2.0 * std::f32::consts::PI;
    // Project to unit circle components (scaled by 100 for integer precision)
    let time_sin = (hour_angle.sin() * 100.0) as i32;
    let time_cos = (hour_angle.cos() * 100.0) as i32;
    
    // Pack other context into bucket
    let context_bucket: u32 = (((ctx.is_charging as u32) & 1) << 8)
        | (((ctx.recent_sessions as u32).min(15)) << 16);

    let mut h = Hasher::new();
    h.update(&goal.to_le_bytes());
    h.update(&mode.to_le_bytes());
    h.update(&pattern_id.to_le_bytes());
    h.update(&time_sin.to_le_bytes());
    h.update(&time_cos.to_le_bytes());
    h.update(&context_bucket.to_le_bytes());
    let out = h.finalize();
    *out.as_bytes()
}

impl<'a> Guard for TraumaGuard<'a> {
    fn name(&self) -> &'static str {
        "TraumaGuard"
    }

    fn vote(
        &self,
        belief: &BeliefState,
        _phys: &PhysioState,
        patch: &PatternPatch,
        ctx: &BeliefCtx,
        now_ts_us: i64,
    ) -> Vote {
        let sig = Self::sig_hash(patch, belief, ctx);
        let Some(hit) = self.source.query_trauma(&sig, now_ts_us) else {
            return Vote::Allow(Clamp::default(), 1.0, 0x0);
        };

        if now_ts_us < hit.inhibit_until_ts_us {
            return Vote::Deny("trauma_inhibit");
        }
        if hit.sev_eff > self.hard_th {
            return Vote::Deny("trauma_hard");
        }
        if hit.sev_eff > self.soft_th {
            let mut c = Clamp::default();
            c.hold_max_sec = c.hold_max_sec.min(20.0);
            c.max_delta_rr_per_min = c.max_delta_rr_per_min.min(2.0);
            return Vote::Allow(c, 0.2, 0x1000);
        }
        Vote::Allow(Clamp::default(), 0.8, 0x1000)
    }
}

/// Consensus decide function - deterministic
pub fn decide(
    guards: &[Box<dyn Guard>],
    patch: &PatternPatch,
    belief: &BeliefState,
    phys: &PhysioState,
    ctx: &BeliefCtx,
    now_ts_us: i64,
) -> Result<(PatternPatch, u32), &'static str> {
    let mut denies: Vec<&'static str> = Vec::new();
    let mut clamps: Vec<Clamp> = Vec::new();
    let mut reason_bits: u32 = 0;
    for g in guards.iter() {
        match g.vote(belief, phys, patch, ctx, now_ts_us) {
            Vote::Deny(s) => {
                denies.push(s);
            }
            Vote::Allow(cl, _score, bits) => {
                clamps.push(cl);
                reason_bits |= bits;
            }
        }
    }
    if !denies.is_empty() {
        return Err(denies[0]);
    }

    // intersect clamps: rr_min = max(rr_min), rr_max = min(rr_max), hold_max_sec = min, max_delta = min
    let mut final_clamp = Clamp::default();
    if !clamps.is_empty() {
        final_clamp.rr_min = clamps
            .iter()
            .map(|c| c.rr_min)
            .fold(final_clamp.rr_min, f32::max);
        final_clamp.rr_max = clamps
            .iter()
            .map(|c| c.rr_max)
            .fold(final_clamp.rr_max, f32::min);
        final_clamp.hold_max_sec = clamps
            .iter()
            .map(|c| c.hold_max_sec)
            .fold(final_clamp.hold_max_sec, f32::min);
        final_clamp.max_delta_rr_per_min = clamps
            .iter()
            .map(|c| c.max_delta_rr_per_min)
            .fold(final_clamp.max_delta_rr_per_min, f32::min);
    }

    // P0.6: Guard Conflict Validation - detect unsatisfiable ranges
    if final_clamp.rr_min > final_clamp.rr_max {
        eprintln!(
            "GUARD CONFLICT: Unsatisfiable range - rr_min={:.2} > rr_max={:.2}. \
             This indicates conflicting guard constraints that cannot be satisfied simultaneously.",
            final_clamp.rr_min, final_clamp.rr_max
        );
        return Err("guard_conflict_unsatisfiable_range");
    }

    if final_clamp.hold_max_sec <= 0.0 {
        eprintln!(
            "GUARD CONFLICT: Invalid hold_max_sec={:.2}. \
             Guards have reduced hold time to zero or negative.",
            final_clamp.hold_max_sec
        );
        return Err("guard_conflict_invalid_hold_time");
    }

    if final_clamp.max_delta_rr_per_min <= 0.0 {
        eprintln!(
            "GUARD CONFLICT: Invalid max_delta_rr_per_min={:.2}. \
             Guards have reduced rate change limit to zero or negative.",
            final_clamp.max_delta_rr_per_min
        );
        return Err("guard_conflict_invalid_rate_limit");
    }

    let mut applied = patch.clone();
    applied.apply_clamp(&final_clamp);
    Ok((applied, reason_bits))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::{BeliefBasis, BeliefState, Context, PhysioState};

    struct DummyTrauma {
        hit: Option<TraumaHit>,
    }
    impl TraumaSource for DummyTrauma {
        fn query_trauma(&self, _sig_hash: &[u8], _now_ts_us: i64) -> Option<TraumaHit> {
            self.hit
        }
    }

    #[test]
    fn consensus_denies_on_low_conf() {
        let guards: Vec<Box<dyn Guard>> = vec![Box::new(ConfidenceGuard { min_conf: 0.5 })];
        let patch = PatternPatch {
            target_bpm: 6.0,
            hold_sec: 20.0,
            pattern_id: 1,
            goal: 2,
        };
        let belief = BeliefState {
            p: [0.2; 5],
            conf: 0.2,
            mode: BeliefBasis::Calm,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(30.0),
            confidence: 0.2,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: false,
            recent_sessions: 0,
        };
        let res = decide(&guards, &patch, &belief, &phys, &ctx, 0);
        assert!(res.is_err());
    }

    #[test]
    fn consensus_allows_and_clamps() {
        let guards: Vec<Box<dyn Guard>> = vec![Box::new(BreathBoundsGuard {
            clamp: Clamp {
                rr_min: 5.0,
                rr_max: 9.0,
                hold_max_sec: 30.0,
                max_delta_rr_per_min: 4.0,
            },
        })];
        let patch = PatternPatch {
            target_bpm: 12.0,
            hold_sec: 60.0,
            pattern_id: 1,
            goal: 2,
        };
        let belief = BeliefState {
            p: [0.2; 5],
            conf: 0.9,
            mode: BeliefBasis::Calm,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(30.0),
            confidence: 0.9,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: false,
            recent_sessions: 0,
        };
        let res = decide(&guards, &patch, &belief, &phys, &ctx, 0);
        assert!(res.is_ok());
        let (applied, bits) = res.unwrap();
        assert!(applied.target_bpm <= 9.0);
        assert!(bits & 0x2 != 0);
    }

    #[test]
    fn trauma_guard_denies_on_hard_threshold() {
        let src = DummyTrauma {
            hit: Some(TraumaHit {
                sev_eff: 2.0,
                count: 3,
                inhibit_until_ts_us: 0,
                last_ts_us: 0,
            }),
        };
        let guards: Vec<Box<dyn Guard>> = vec![Box::new(TraumaGuard {
            source: &src,
            hard_th: 1.5,
            soft_th: 0.7,
        })];
        let patch = PatternPatch {
            target_bpm: 6.0,
            hold_sec: 20.0,
            pattern_id: 1,
            goal: 2,
        };
        let belief = BeliefState {
            p: [0.2; 5],
            conf: 0.9,
            mode: BeliefBasis::Calm,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(30.0),
            confidence: 0.9,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: false,
            recent_sessions: 0,
        };
        let res = decide(&guards, &patch, &belief, &phys, &ctx, 1);
        assert!(res.is_err());
    }

    #[test]
    fn trauma_guard_denies_on_inhibit() {
        let src = DummyTrauma {
            hit: Some(TraumaHit {
                sev_eff: 0.1,
                count: 1,
                inhibit_until_ts_us: 100,
                last_ts_us: 0,
            }),
        };
        let guards: Vec<Box<dyn Guard>> = vec![Box::new(TraumaGuard {
            source: &src,
            hard_th: 1.5,
            soft_th: 0.7,
        })];
        let patch = PatternPatch {
            target_bpm: 6.0,
            hold_sec: 20.0,
            pattern_id: 1,
            goal: 2,
        };
        let belief = BeliefState {
            p: [0.2; 5],
            conf: 0.9,
            mode: BeliefBasis::Calm,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(30.0),
            confidence: 0.9,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: false,
            recent_sessions: 0,
        };
        let res = decide(&guards, &patch, &belief, &phys, &ctx, 10);
        assert!(res.is_err());
    }
}
