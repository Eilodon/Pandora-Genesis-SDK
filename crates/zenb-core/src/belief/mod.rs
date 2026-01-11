use serde::{Deserialize, Serialize};

use crate::config::ZenbConfig;
use crate::resonance::ResonanceFeatures;

/// Belief latent basis (collapsed modes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeliefBasis {
    Calm = 0,
    Stress = 1,
    Focus = 2,
    Sleepy = 3,
    Energize = 4,
}

impl Default for BeliefBasis {
    fn default() -> Self {
        BeliefBasis::Calm
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    pub p: [f32; 5],
    pub conf: f32,
    pub mode: BeliefBasis,
}

impl Default for BeliefState {
    fn default() -> Self {
        Self {
            p: [0.0; 5],
            conf: 0.0,
            mode: BeliefBasis::Calm,
        }
    }
}

impl BeliefState {
    pub fn to_5mode_array(&self) -> [f32; 5] {
        self.p
    }

    pub fn uncertainty(&self) -> f32 {
        uncertainty_from_confidence(self.conf)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FepState {
    pub mu: [f32; 5],
    pub sigma: [f32; 5],
    pub free_energy_ema: f32,
    pub lr: f32,
}

impl Default for FepState {
    fn default() -> Self {
        Self {
            mu: [0.2; 5],
            sigma: [0.5; 5],
            free_energy_ema: 0.0,
            lr: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FepUpdateOut {
    pub belief: BeliefState,
    pub fep: FepState,
    pub resonance_score: f32,
}

/// Lightweight context used by pathways
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context {
    pub local_hour: u8, // 0..23
    pub is_charging: bool,
    pub recent_sessions: u16,
}

/// Minimal sensor/physio shapes for the pathways
#[derive(Debug, Clone, Copy)]
pub struct SensorFeatures {
    pub hr_bpm: Option<f32>,
    pub rmssd: Option<f32>,
    pub rr_bpm: Option<f32>,
    pub quality: f32, // 0..1
    pub motion: f32,  // 0..1
}

#[derive(Debug, Clone, Copy)]
pub struct PhysioState {
    pub hr_bpm: Option<f32>,
    pub rr_bpm: Option<f32>,
    pub rmssd: Option<f32>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct AgentVote {
    pub logits: [f32; 5],
    pub confidence: f32,
    pub reasoning: &'static str,
}

/// math helpers
pub fn softmax(mut logits: [f32; 5]) -> [f32; 5] {
    // NaN/Infinity guard: if any input is non-finite, return uniform distribution
    // This prevents NaN propagation through the belief engine
    if logits.iter().any(|x| !x.is_finite()) {
        log::warn!("softmax: non-finite input detected, returning uniform");
        return [0.2; 5];
    }
    
    // stable softmax
    let max = logits
        .iter()
        .cloned()
        .fold(std::f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 || !sum.is_finite() {
        return [0.2; 5];
    }
    for v in logits.iter_mut() {
        *v /= sum;
    }
    logits
}

pub fn argmax(p: &[f32; 5]) -> usize {
    let mut best = 0usize;
    for (i, v) in p.iter().enumerate() {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

pub fn ema_vec(prev: &[f32; 5], next: &[f32; 5], alpha: f32) -> [f32; 5] {
    let mut out = [0.0f32; 5];
    for i in 0..5 {
        out[i] = prev[i] * (1.0 - alpha) + next[i] * alpha;
    }
    out
}

pub fn uncertainty_from_confidence(conf: f32) -> f32 {
    // Uncertainty is inverse of confidence (simple mapping)
    // Or Shannon entropy? For now, 1.0 - conf
    (1.0 - conf).clamp(0.0, 1.0)
}

pub fn hysteresis_collapse(
    prev: BeliefBasis,
    p: &[f32; 5],
    enter_th: f32,
    exit_th: f32,
) -> BeliefBasis {
    let idx = argmax(p);
    // if current mode is chosen and above exit_th, keep; else only enter if above enter_th
    let curr = prev as usize;
    if idx == curr {
        if p[idx] >= exit_th {
            return prev;
        }
        // remain
        else {
            return prev;
        }
    }
    // switching
    if p[idx] >= enter_th {
        match idx {
            0 => BeliefBasis::Calm,
            1 => BeliefBasis::Stress,
            2 => BeliefBasis::Focus,
            3 => BeliefBasis::Sleepy,
            _ => BeliefBasis::Energize,
        }
    } else {
        prev
    }
}

/// Agent configs (data-only, serializable).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GeminiConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MinhGioiConfig;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhaQuanConfig;

/// Data-oriented agent enum for static dispatch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentStrategy {
    Gemini(GeminiConfig),
    MinhGioi(MinhGioiConfig),
    PhaQuan(PhaQuanConfig),
}

fn gemini_eval(_cfg: &GeminiConfig, x: &SensorFeatures, phys: &PhysioState) -> AgentVote {
    let mut logits = [0.0f32; 5];
    let rm = phys.rmssd.unwrap_or(20.0);
    if rm > 50.0 {
        logits[0] += 2.0;
    } else {
        logits[1] += (50.0 - rm) / 50.0;
    }
    if let Some(rr) = phys.rr_bpm {
        if rr > 12.0 {
            logits[3] += 1.5;
        }
    }
    if x.motion > 0.5 {
        logits[4] += 2.0;
    }
    if phys.hr_bpm.unwrap_or(60.0) > 100.0 {
        logits[1] += 1.0;
    }
    AgentVote {
        logits,
        confidence: phys.confidence.clamp(0.0, 1.0),
        reasoning: "HR/RMSSD/respiration heuristics",
    }
}

fn minh_gioi_eval(_cfg: &MinhGioiConfig, ctx: &Context) -> AgentVote {
    let mut logits = [0.0f32; 5];
    if ctx.local_hour >= 22 || ctx.local_hour <= 6 {
        logits[3] += 2.0;
    }
    if ctx.recent_sessions > 3 {
        logits[2] += 1.0;
    }
    AgentVote {
        logits,
        confidence: 0.7,
        reasoning: "Time-of-day and session cadence heuristics",
    }
}

fn pha_quan_eval(_cfg: &PhaQuanConfig, x: &SensorFeatures, phys: &PhysioState) -> AgentVote {
    let mut logits = [0.0f32; 5];
    // quality pushes confidence and calms if quality good
    logits[0] += x.quality * 2.0;
    logits[4] += x.motion * 1.5;
    let confidence = (x.quality * phys.confidence).clamp(0.0, 1.0);
    AgentVote {
        logits,
        confidence,
        reasoning: "Signal quality and motion heuristics",
    }
}

impl AgentStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            AgentStrategy::Gemini(_) => "Gemini",
            AgentStrategy::MinhGioi(_) => "MinhGioi",
            AgentStrategy::PhaQuan(_) => "PhaQuan",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            AgentStrategy::Gemini(_) => "Analyzes raw physiological data logic (HR/RMSSD).",
            AgentStrategy::MinhGioi(_) => "Monitors environmental and temporal constraints.",
            AgentStrategy::PhaQuan(_) => "Senses signal quality and physical motion.",
        }
    }

    pub fn eval(&self, x: &SensorFeatures, phys: &PhysioState, ctx: &Context) -> AgentVote {
        match self {
            AgentStrategy::Gemini(cfg) => gemini_eval(cfg, x, phys),
            AgentStrategy::MinhGioi(cfg) => minh_gioi_eval(cfg, ctx),
            AgentStrategy::PhaQuan(cfg) => pha_quan_eval(cfg, x, phys),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BeliefDebug {
    pub per_pathway: Vec<(String, AgentVote)>,
}

pub struct BeliefEngine {
    pub agents: Vec<AgentStrategy>,
    pub w: Vec<f32>,
    pub prior_logits: [f32; 5],
    pub smooth_tau_sec: f32,
    pub enter_th: f32,
    pub exit_th: f32,
}

impl BeliefEngine {
    pub fn new() -> Self {
        Self::from_config(&crate::config::BeliefConfig::default())
    }

    pub fn from_config(config: &crate::config::BeliefConfig) -> Self {
        let agents: Vec<AgentStrategy> = vec![
            AgentStrategy::Gemini(GeminiConfig::default()),
            AgentStrategy::MinhGioi(MinhGioiConfig::default()),
            AgentStrategy::PhaQuan(PhaQuanConfig::default()),
        ];
        Self {
            agents,
            w: config.agent_weights.clone(),
            prior_logits: config.prior_logits,
            smooth_tau_sec: config.smooth_tau_sec,
            enter_th: config.enter_threshold,
            exit_th: config.exit_threshold,
        }
    }

    fn basis_label(idx: usize) -> &'static str {
        match idx {
            0 => "Calm",
            1 => "Stress",
            2 => "Focus",
            3 => "Sleepy",
            _ => "Energize",
        }
    }

    /// Process feedback from action outcomes to adjust model parameters.
    /// This implements Active Inference learning: the model adjusts its uncertainty
    /// based on prediction accuracy.
    ///
    /// # Arguments
    /// * `fep_state` - Current FEP state to update
    /// * `config` - Current configuration (will be modified)
    /// * `success` - Whether the action was successful
    ///
    /// # Active Inference Logic
    /// - **Success**: Model was correct → Decrease process noise (increase precision)
    /// - **Failure**: Model was wrong → Increase process noise (acknowledge uncertainty)
    ///
    /// This creates a self-regulating system that becomes more confident when correct
    /// and more cautious when wrong.
    pub fn process_feedback(
        fep_state: &mut FepState,
        config: &mut crate::config::FepConfig,
        success: bool,
    ) {
        const NOISE_ADJUSTMENT_FACTOR: f32 = 0.15;
        const MIN_PROCESS_NOISE: f32 = 0.005;
        const MAX_PROCESS_NOISE: f32 = 0.2;
        const PRECISION_BOOST_FACTOR: f32 = 0.9;
        const PRECISION_PENALTY_FACTOR: f32 = 1.2;

        if success {
            // Success: Model prediction was accurate
            // → Increase precision (decrease noise)
            // → Boost learning rate slightly (model is on track)

            config.process_noise =
                (config.process_noise * PRECISION_BOOST_FACTOR).max(MIN_PROCESS_NOISE);

            // Reduce uncertainty in posterior
            for sigma in fep_state.sigma.iter_mut() {
                *sigma = (*sigma * PRECISION_BOOST_FACTOR).max(0.001);
            }

            // Slight learning rate boost (but respect bounds)
            fep_state.lr = (fep_state.lr * 1.05).min(config.lr_max);

            eprintln!(
                "FEEDBACK: SUCCESS → process_noise={:.4} (decreased), lr={:.3} (boosted)",
                config.process_noise, fep_state.lr
            );
        } else {
            // Failure: Model prediction was inaccurate
            // → Decrease precision (increase noise) - acknowledge high entropy
            // → Reduce learning rate (be more cautious)

            config.process_noise =
                (config.process_noise * PRECISION_PENALTY_FACTOR).min(MAX_PROCESS_NOISE);

            // Increase uncertainty in posterior
            for sigma in fep_state.sigma.iter_mut() {
                *sigma = (*sigma * PRECISION_PENALTY_FACTOR).min(10.0);
            }

            // Reduce learning rate (be more conservative)
            fep_state.lr = (fep_state.lr * 0.85).max(config.lr_min);

            // Increase free energy to reflect surprise
            fep_state.free_energy_ema =
                (fep_state.free_energy_ema + NOISE_ADJUSTMENT_FACTOR).min(10.0);

            eprintln!(
                "FEEDBACK: FAILURE → process_noise={:.4} (increased), lr={:.3} (reduced), FE={:.3}",
                config.process_noise, fep_state.lr, fep_state.free_energy_ema
            );
        }
    }

    pub fn update(
        &self,
        prev: &BeliefState,
        x: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f64,
    ) -> (BeliefState, BeliefDebug) {
        // evaluate pathways deterministically in order
        let mut logits_total = self.prior_logits;
        let mut per = Vec::new();
        let mut weight_sum = 0.0f32;
        let mut conf_sum = 0.0f32;
        for (i, agent) in self.agents.iter().enumerate() {
            let out = agent.eval(x, phys, ctx);

            let vote_p = softmax(out.logits);
            let vote_idx = argmax(&vote_p);
            eprintln!(
                "BELIEF_VOTE: {} votes {} ({:.3}) | conf={:.3} | {}",
                agent.name(),
                Self::basis_label(vote_idx),
                vote_p[vote_idx],
                out.confidence,
                agent.description()
            );

            per.push((agent.name().to_string(), out));
            let w = *self.w.get(i).unwrap_or(&1.0);
            for j in 0..5 {
                logits_total[j] += w * out.confidence * out.logits[j];
            }
            conf_sum += w * out.confidence;
            weight_sum += w;
        }
        let p = softmax(logits_total);
        let alpha = (dt_sec / (self.smooth_tau_sec as f64 + dt_sec)) as f32;
        let p_smooth = ema_vec(&prev.p, &p, alpha);
        let conf = if weight_sum > 0.0 {
            (conf_sum / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mode = hysteresis_collapse(prev.mode, &p_smooth, self.enter_th, self.exit_th);
        let st = BeliefState {
            p: p_smooth,
            conf,
            mode,
        };
        (st, BeliefDebug { per_pathway: per })
    }

    pub fn fused_logits_and_conf(
        &self,
        x: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
    ) -> ([f32; 5], f32) {
        let mut logits_total = self.prior_logits;
        let mut weight_sum = 0.0f32;
        let mut conf_sum = 0.0f32;
        for (i, agent) in self.agents.iter().enumerate() {
            let out = agent.eval(x, phys, ctx);
            let w = *self.w.get(i).unwrap_or(&1.0);
            for j in 0..5 {
                logits_total[j] += w * out.confidence * out.logits[j];
            }
            conf_sum += w * out.confidence;
            weight_sum += w;
        }
        let obs_conf = if weight_sum > 0.0 {
            (conf_sum / weight_sum).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (logits_total, obs_conf)
    }

    pub fn update_fep(
        &self,
        prev_mode: BeliefBasis,
        prev_fep: &FepState,
        x: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f32,
        resonance: ResonanceFeatures,
    ) -> FepUpdateOut {
        self.update_fep_with_config(
            prev_mode,
            prev_fep,
            x,
            phys,
            ctx,
            dt_sec,
            resonance,
            &ZenbConfig::default(),
        )
    }

    pub fn update_fep_with_config(
        &self,
        prev_mode: BeliefBasis,
        prev_fep: &FepState,
        x: &SensorFeatures,
        phys: &PhysioState,
        ctx: &Context,
        dt_sec: f32,
        resonance: ResonanceFeatures,
        cfg: &ZenbConfig,
    ) -> FepUpdateOut {
        // Freeze learning if base learning rate is zero
        if cfg.fep.lr_base == 0.0 {
            return FepUpdateOut {
                belief: BeliefState {
                    p: prev_fep.mu, // Not a real probability; keeps structure but unused downstream in this branch
                    conf: prev_fep.lr,
                    mode: prev_mode,
                },
                fep: FepState {
                    mu: prev_fep.mu,
                    sigma: prev_fep.sigma,
                    free_energy_ema: prev_fep.free_energy_ema,
                    lr: 0.0,
                },
                resonance_score: resonance.resonance_score,
            };
        }

        let eps = 1e-6f32;
        let (fused_logits, obs_conf) = self.fused_logits_and_conf(x, phys, ctx);
        let mu_obs = softmax(fused_logits);

        let obs_conf_adj = (obs_conf * (0.5 + 0.5 * resonance.resonance_score)).clamp(0.0, 1.0);

        let base_obs_var = cfg.fep.base_obs_var;
        let sigma_obs_base = (base_obs_var / (eps + obs_conf_adj)).clamp(0.02, 5.0);
        let process_noise = cfg.fep.process_noise;

        let mut mu_post = [0.0f32; 5];
        let mut sigma_post = [0.0f32; 5];
        let mut pe = 0.0f32;

        let mu_prior = prev_fep.mu;
        for i in 0..5 {
            let sigma_prior = (prev_fep.sigma[i] + process_noise * dt_sec).clamp(0.001, 10.0);
            let sigma_obs = sigma_obs_base;
            let k = (sigma_prior / (sigma_prior + sigma_obs + eps)).clamp(0.0, 1.0);
            let delta = mu_obs[i] - mu_prior[i];
            mu_post[i] = mu_prior[i] + prev_fep.lr * k * delta;
            sigma_post[i] = ((1.0 - k) * sigma_prior).clamp(0.001, 10.0);
            pe += (delta * delta) / (sigma_prior + sigma_obs + eps);
        }

        let sum: f32 = mu_post.iter().sum();
        if sum > eps {
            for v in mu_post.iter_mut() {
                *v /= sum;
            }
        } else {
            mu_post = [0.2; 5];
        }

        let c = 0.5f32;
        pe *= 1.0 + c * (1.0 - resonance.resonance_score);

        let tau_fe = 4.0f32;
        let alpha_fe = if dt_sec <= 0.0 {
            0.0
        } else {
            (dt_sec / (tau_fe + dt_sec)).clamp(0.0, 1.0)
        };
        let free_energy_ema =
            (prev_fep.free_energy_ema * (1.0 - alpha_fe) + pe * alpha_fe).max(0.0);

        let lr_base = cfg.fep.lr_base;
        let lr_min = cfg.fep.lr_min;
        let lr_max = cfg.fep.lr_max;
        let k_lr = 1.2f32;
        let lr = if lr_base <= 0.0 {
            0.0
        } else {
            (lr_base * (-k_lr * free_energy_ema).exp()).clamp(lr_min, lr_max)
        };

        let conf = (obs_conf_adj * (1.0 / (1.0 + free_energy_ema))).clamp(0.0, 1.0);
        let mode = hysteresis_collapse(prev_mode, &mu_post, self.enter_th, self.exit_th);

        FepUpdateOut {
            belief: BeliefState {
                p: mu_post,
                conf,
                mode,
            },
            fep: FepState {
                mu: mu_post,
                sigma: sigma_post,
                free_energy_ema,
                lr,
            },
            resonance_score: resonance.resonance_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ZenbConfig;
    use crate::resonance::ResonanceFeatures;

    #[test]
    fn softmax_basic() {
        let l = [1.0, 2.0, 3.0, 2.0, 1.0];
        let p = softmax(l);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn belief_engine_stable() {
        let be = BeliefEngine::new();
        let prev = BeliefState {
            p: [0.2; 5],
            conf: 0.8,
            mode: BeliefBasis::Calm,
        };
        let x = SensorFeatures {
            hr_bpm: Some(60.0),
            rmssd: Some(40.0),
            rr_bpm: Some(6.0),
            quality: 0.9,
            motion: 0.1,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(40.0),
            confidence: 0.9,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: true,
            recent_sessions: 0,
        };
        let (out, dbg) = be.update(&prev, &x, &phys, &ctx, 0.5);
        assert!(out.conf >= 0.0 && out.conf <= 1.0);
        assert_eq!(dbg.per_pathway.len(), 3);
    }

    #[test]
    fn fep_low_obs_conf_small_update() {
        let be = BeliefEngine::new();
        let prev_mode = BeliefBasis::Calm;
        let prev_fep = FepState {
            mu: [0.2; 5],
            sigma: [0.5; 5],
            free_energy_ema: 0.0,
            lr: 0.5,
        };

        let x = SensorFeatures {
            hr_bpm: Some(60.0),
            rmssd: Some(40.0),
            rr_bpm: Some(6.0),
            quality: 0.0,
            motion: 0.0,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(6.0),
            rmssd: Some(40.0),
            confidence: 0.05,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: true,
            recent_sessions: 0,
        };
        let res = ResonanceFeatures {
            phase_diff_norm: 0.0,
            resonance_score: 1.0,
            stability_score: 1.0,
        };

        let out = be.update_fep(prev_mode, &prev_fep, &x, &phys, &ctx, 1.0, res);
        let max_delta = out
            .belief
            .p
            .iter()
            .zip(prev_fep.mu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_delta < 0.25);
    }

    #[test]
    fn fep_high_resonance_higher_lr_than_low_resonance() {
        let be = BeliefEngine::new();
        let prev_mode = BeliefBasis::Calm;
        let prev_fep = FepState {
            mu: [0.2; 5],
            sigma: [0.5; 5],
            free_energy_ema: 0.0,
            lr: 0.6,
        };

        let x = SensorFeatures {
            hr_bpm: Some(60.0),
            rmssd: Some(10.0),
            rr_bpm: Some(14.0),
            quality: 1.0,
            motion: 0.0,
        };
        let phys = PhysioState {
            hr_bpm: Some(60.0),
            rr_bpm: Some(14.0),
            rmssd: Some(10.0),
            confidence: 1.0,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: true,
            recent_sessions: 0,
        };

        let high = ResonanceFeatures {
            phase_diff_norm: 0.0,
            resonance_score: 1.0,
            stability_score: 1.0,
        };
        let low = ResonanceFeatures {
            phase_diff_norm: 1.0,
            resonance_score: 0.0,
            stability_score: 1.0,
        };

        let out_high = be.update_fep(prev_mode, &prev_fep, &x, &phys, &ctx, 1.0, high);
        let out_low = be.update_fep(prev_mode, &prev_fep, &x, &phys, &ctx, 1.0, low);

        assert!(out_high.fep.lr >= out_low.fep.lr);
    }

    #[test]
    fn config_lr_base_zero_freezes_learning() {
        let be = BeliefEngine::new();
        let prev_mode = BeliefBasis::Calm;
        let prev_fep = FepState {
            mu: [0.2; 5],
            sigma: [0.5; 5],
            free_energy_ema: 0.0,
            lr: 0.6,
        };

        let x = SensorFeatures {
            hr_bpm: Some(90.0),
            rmssd: Some(5.0),
            rr_bpm: Some(14.0),
            quality: 1.0,
            motion: 0.0,
        };
        let phys = PhysioState {
            hr_bpm: Some(90.0),
            rr_bpm: Some(14.0),
            rmssd: Some(5.0),
            confidence: 1.0,
        };
        let ctx = Context {
            local_hour: 12,
            is_charging: true,
            recent_sessions: 0,
        };
        let res = ResonanceFeatures {
            phase_diff_norm: 0.0,
            resonance_score: 1.0,
            stability_score: 1.0,
        };

        let mut cfg = ZenbConfig::default();
        cfg.fep.lr_base = 0.0;

        let out = be.update_fep_with_config(prev_mode, &prev_fep, &x, &phys, &ctx, 1.0, res, &cfg);

        // If learning is frozen, posterior should equal prior (up to float noise)
        for i in 0..5 {
            assert!((out.fep.mu[i] - prev_fep.mu[i]).abs() < 1e-6);
        }
        assert!((out.fep.lr - 0.0).abs() < 1e-6);
    }
}
