use crate::belief::Context;
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
    pub poller: StabilizedPoller,
}

impl AdaptiveController {
    pub fn new(cfg: ControllerConfig) -> Self {
        Self {
            cfg,
            last_decision_ts_us: None,
            last_decision_bpm: None,
            poller: StabilizedPoller::default(),
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

/// Stabilized poller with Lyapunov-inspired damping to avoid oscillations.
#[derive(Debug, Clone, Copy)]
pub struct StabilizedPoller {
    pub previous_free_energy: f32,
    pub previous_interval_ms: u64,
    pub base_interval_ms: f32,
    pub sensitivity: f32,
    pub k_d: f32,
}

impl Default for StabilizedPoller {
    fn default() -> Self {
        Self {
            previous_free_energy: 0.0,
            previous_interval_ms: 5_000,
            base_interval_ms: 5_000.0,
            sensitivity: 10.0,
            k_d: 1.5, // overdamped by design
        }
    }
}

impl StabilizedPoller {
    /// Compute adaptive polling interval with damping and rate limits.
    pub fn compute(
        &mut self,
        free_energy_ema: f32,
        belief_confidence: f32,
        action_taken: bool,
        ctx: &Context,
    ) -> u64 {
        // Immediate feedback loop if an action was taken
        if action_taken {
            self.previous_free_energy = free_energy_ema;
            self.previous_interval_ms = 200;
            return 200;
        }

        let fe = free_energy_ema.clamp(0.0, 1.0);
        let delta_fe = (fe - self.previous_free_energy).max(0.0); // rising energy only
        let damped_urgency = fe + self.k_d * delta_fe;

        let damped_urgency = if damped_urgency.is_finite() {
            damped_urgency
        } else {
            0.0
        };

        // Sensitivity modulated by uncertainty (1 - confidence)
        let sensitivity = self.sensitivity * (1.0 - belief_confidence.clamp(0.0, 1.0));
        let elasticity = 1.0 + (damped_urgency * sensitivity);
        let mut target_ms = self.base_interval_ms / elasticity.max(1.0);

        // Context modifier: if charging, we can afford more compute (poll faster)
        if ctx.is_charging {
            target_ms *= 0.8;
        }

        // Hard bounds
        let mut bounded = target_ms.clamp(200.0, 30_000.0);

        // Rate-of-change limiter: max 50% change per cycle
        let prev = self.previous_interval_ms as f32;
        let min_step = prev * 0.5;
        let max_step = prev * 1.5;
        bounded = bounded.clamp(min_step, max_step).clamp(200.0, 30_000.0);

        let rounded = bounded.round().clamp(200.0, 30_000.0) as u64;
        self.previous_free_energy = fe;
        self.previous_interval_ms = rounded;
        rounded
    }
}

/// Backwards-compatible helper to compute polling interval with stabilization.
pub fn compute_poll_interval(
    poller: &mut StabilizedPoller,
    free_energy_ema: f32,
    belief_confidence: f32,
    action_taken: bool,
    ctx: &Context,
) -> u64 {
    poller.compute(free_energy_ema, belief_confidence, action_taken, ctx)
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

    #[test]
    fn test_poller_stability() {
        let mut poller = StabilizedPoller::default();
        let ctx = Context {
            local_hour: 12,
            is_charging: false,
            recent_sessions: 0,
        };

        // Simulate oscillating free energy; ensure intervals stay bounded and rate-limited
        let mut intervals = Vec::new();
        for i in 0..50 {
            let fe = 0.5 + 0.3 * (2.0 * std::f32::consts::PI * (i as f32) / 10.0).sin();
            let interval = poller.compute(fe, 0.6, false, &ctx);
            intervals.push(interval as f32);
        }

        // Check bounds and rate-of-change constraint
        for w in intervals.windows(2) {
            let prev = w[0];
            let curr = w[1];
            assert!(curr >= prev * 0.5 - 1e-3 && curr <= prev * 1.5 + 1e-3);
            assert!(curr >= 200.0 && curr <= 30_000.0);
        }
    }
}
