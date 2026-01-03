use crate::belief::{BeliefBasis, Context};
use crate::estimator::Estimate;

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    pub decision_epsilon_bpm: f32,
    pub min_decision_interval_us: i64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            decision_epsilon_bpm: 0.1,
            min_decision_interval_us: 250_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveController {
    pub cfg: ControllerConfig,
    pub(crate) last_decision_ts_us: Option<i64>,
    pub(crate) last_decision_bpm: Option<f32>,
}

impl AdaptiveController {
    pub fn new(cfg: ControllerConfig) -> Self {
        Self {
            cfg,
            last_decision_ts_us: None,
            last_decision_bpm: None,
        }
    }

    /// Decide a target rate based on estimate and previous decision; returns (rate_bpm, changed)
    pub fn decide(&mut self, est: &Estimate, ts_us: i64) -> (f32, bool) {
        // If no RR estimate, fallback to last decision or default 6.0
        let base = est.rr_bpm.or(self.last_decision_bpm).unwrap_or(6.0);
        let target = base.clamp(4.0, 12.0);
        let changed = match self.last_decision_bpm {
            Some(prev) => (prev - target).abs() > self.cfg.decision_epsilon_bpm,
            None => true,
        } && match self.last_decision_ts_us {
            Some(last_ts) => (ts_us - last_ts) >= self.cfg.min_decision_interval_us,
            None => true,
        };
        if changed {
            self.last_decision_bpm = Some(target);
            self.last_decision_ts_us = Some(ts_us);
        }
        (target, changed)
    }
}

/// Compute adaptive polling interval based on belief state and context.
///
/// # Logic:
/// - Stress/Focus (High Energy) OR Action taken -> 1000ms (1s)
/// - Calm (Stable) -> 5000ms (5s)
/// - Sleepy (Low Energy) -> 30000ms (30s)
/// - Energize -> 1000ms (1s) - high energy state
/// - If charging -> Cap max at 5000ms (can afford more compute)
///
/// # Arguments
/// * `belief_mode` - Current belief basis mode
/// * `action_taken` - Whether a control action was just taken
/// * `ctx` - Runtime context (charging state, etc.)
///
/// # Returns
/// Recommended polling interval in milliseconds
pub fn compute_poll_interval(belief_mode: BeliefBasis, action_taken: bool, ctx: &Context) -> u64 {
    const FAST_POLL_MS: u64 = 1000; // 1s - high energy or action taken
    const NORMAL_POLL_MS: u64 = 5000; // 5s - calm/stable
    const SLOW_POLL_MS: u64 = 30000; // 30s - sleepy/low energy
    const CHARGING_MAX_MS: u64 = 5000; // Cap at 5s when charging

    // If action was just taken, poll fast to monitor response
    if action_taken {
        return FAST_POLL_MS;
    }

    // Determine base interval from belief state
    let base_interval = match belief_mode {
        BeliefBasis::Stress | BeliefBasis::Focus | BeliefBasis::Energize => FAST_POLL_MS,
        BeliefBasis::Calm => NORMAL_POLL_MS,
        BeliefBasis::Sleepy => SLOW_POLL_MS,
    };

    // Cap at faster interval when charging (can afford more compute)
    if ctx.is_charging {
        base_interval.min(CHARGING_MAX_MS)
    } else {
        base_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::Estimator;

    #[test]
    fn controller_basic_change() {
        let mut c = AdaptiveController::new(ControllerConfig::default());
        let mut e = Estimator::default();
        let est = e.ingest(&[60.0, 30.0, 6.0], 0);
        let (r, changed) = c.decide(&est, 0);
        assert!(changed);
        assert!((r - 6.0).abs() < 1e-3);
    }
}
