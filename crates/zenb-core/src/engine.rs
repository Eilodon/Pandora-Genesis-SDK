use crate::controller::AdaptiveController;
use crate::estimator::Estimator;
use crate::safety::SafetyEnvelope;
use crate::safety_swarm::{TraumaSource, TraumaRegistry};
use crate::trauma_cache::TraumaCache;
use crate::resonance::ResonanceTracker;
use crate::breath_engine::BreathEngine;
use crate::domain::ControlDecision;
use crate::config::ZenbConfig;
use crate::causal::{CausalGraph, CausalBuffer, ObservationSnapshot};

/// High-level engine that holds estimator, safety envelope, controller and breath engine.
pub struct Engine {
    pub estimator: Estimator,
    pub safety: SafetyEnvelope,
    pub controller: AdaptiveController,
    pub breath: BreathEngine,
    pub belief_engine: crate::belief::BeliefEngine,
    pub belief_state: crate::belief::BeliefState,
    pub fep_state: crate::belief::FepState,
    pub context: crate::belief::Context,
    pub config: ZenbConfig,
    pub last_ts_us: Option<i64>,
    pub last_control_ts_us: Option<i64>,
    pub last_sf: Option<crate::belief::SensorFeatures>,
    pub last_phys: Option<crate::belief::PhysioState>,
    pub resonance_tracker: ResonanceTracker,
    pub last_resonance_score: f32,
    pub resonance_score_ema: f32,
    pub free_energy_peak: f32,
    pub last_pattern_id: i64,
    pub last_goal: i64,
    pub causal_graph: CausalGraph,
    pub causal_buffer: CausalBuffer,
    pub last_observation: Option<crate::domain::Observation>,
    pub trauma_registry: TraumaRegistry,
    pub trauma_cache: TraumaCache,
}

impl Engine {
    pub fn new(default_bpm: f32) -> Self {
        Self::new_with_config(default_bpm, None)
    }

    pub fn new_with_config(default_bpm: f32, config: Option<ZenbConfig>) -> Self {
        let cfg = config.unwrap_or_default();
        let est = Estimator::default();
        let safety = SafetyEnvelope::new(Default::default());
        let controller = AdaptiveController::new(Default::default());
        let mut breath = BreathEngine::new(crate::breath_engine::BreathMode::Dynamic(default_bpm));
        let target_bpm = if default_bpm > 0.0 { default_bpm } else { cfg.breath.default_target_bpm };
        breath.set_target_bpm(target_bpm);
        Self {
            estimator: est,
            safety,
            controller,
            breath,
            belief_engine: crate::belief::BeliefEngine::from_config(&cfg.belief),
            belief_state: crate::belief::BeliefState::default(),
            fep_state: crate::belief::FepState::default(),
            context: crate::belief::Context { local_hour: 0, is_charging: false, recent_sessions: 0 },
            config: cfg,
            last_ts_us: None,
            last_control_ts_us: None,
            last_sf: None,
            last_phys: None,
            resonance_tracker: ResonanceTracker::default(),
            last_resonance_score: 1.0,
            resonance_score_ema: 1.0,
            free_energy_peak: 0.0,
            last_pattern_id: 0,
            last_goal: 0,
            causal_graph: CausalGraph::with_priors(),
            causal_buffer: CausalBuffer::default_capacity(),
            last_observation: None,
            trauma_registry: TraumaRegistry::new(),
            trauma_cache: TraumaCache::new(),
        }
    }

    pub fn update_config(&mut self, cfg: ZenbConfig) {
        self.config = cfg;
    }

    /// Update runtime context (call from App layer with current local hour / charging / session info)
    pub fn update_context(&mut self, ctx: crate::belief::Context) {
        self.context = ctx;
    }

    /// Convenience: ingest sensor features and update context atomically
    pub fn ingest_sensor_with_context(&mut self, features: &[f32], ts_us: i64, ctx: crate::belief::Context) -> Estimate {
        self.update_context(ctx);
        self.ingest_sensor(features, ts_us)
    }

    /// Ingest sensor features (returns whether a SensorFeaturesIngested event should be persisted according to downsample policy)
    ///
    /// Expected features layout (client must follow this order):
    /// [hr_bpm, rmssd, rr_bpm, quality, motion]
    /// - `quality` and `motion` are optional and will default to 1.0 and 0.0 respectively if omitted.
    pub fn ingest_sensor(&mut self, features: &[f32], ts_us: i64) -> Estimate {
        let est = self.estimator.ingest(features, ts_us);
        // Build SensorFeatures and PhysioState for control-tick belief/FEP update
        debug_assert!(features.len() >= 3, "features layout must be [hr, rmssd, rr, quality, motion] (quality/motion optional)");
        let sf = crate::belief::SensorFeatures { hr_bpm: est.hr_bpm, rmssd: est.rmssd, rr_bpm: est.rr_bpm, quality: features.get(3).cloned().unwrap_or(1.0), motion: features.get(4).cloned().unwrap_or(0.0) };
        let phys = crate::belief::PhysioState { hr_bpm: est.hr_bpm, rr_bpm: est.rr_bpm, rmssd: est.rmssd, confidence: est.confidence };
        self.last_sf = Some(sf);
        self.last_phys = Some(phys);
        est
    }

    /// Advance engine time and compute any cycles (returns cycle count)
    pub fn tick(&mut self, dt_us: u64) -> u64 {
        let (trans, cycles) = self.breath.tick(dt_us);
        
        // Push current observation into causal buffer if available
        // Map canonical belief::BeliefState (5-mode) to CausalBeliefState (3-factor)
        if let Some(ref obs) = self.last_observation {
            let snapshot = ObservationSnapshot {
                timestamp_us: obs.timestamp_us,
                observation: obs.clone(),
                action: None,
                belief_state: Some(crate::domain::CausalBeliefState {
                    // Map 5-mode belief to 3-factor causal representation
                    // p = [Calm, Stress, Focus, Sleepy, Energize]
                    bio_state: [
                        self.belief_state.p[0], // Calm
                        self.belief_state.p[1], // Stress (Aroused)
                        self.belief_state.p[3], // Sleepy (Fatigue)
                    ],
                    cognitive_state: [
                        self.belief_state.p[2], // Focus
                        1.0 - self.belief_state.p[2], // Distracted (inverse of Focus)
                        0.0, // Flow (not directly mapped)
                    ],
                    social_state: [
                        0.33, // Solitary (uniform prior - not tracked in 5-mode)
                        0.33, // Interactive
                        0.33, // Overwhelmed
                    ],
                    confidence: self.belief_state.conf,
                    last_update_us: obs.timestamp_us,
                }),
            };
            self.causal_buffer.push(snapshot);
        }
        
        cycles
    }
    
    /// Ingest a full observation (for causal reasoning layer)
    pub fn ingest_observation(&mut self, observation: crate::domain::Observation) {
        self.last_observation = Some(observation);
    }

    /// Sync trauma hits from persistent storage into the in-memory cache.
    /// This should be called by the Runtime layer on startup to hydrate the cache.
    /// 
    /// # Arguments
    /// * `hits` - Vector of (context_hash, TraumaHit) pairs from persistent storage
    pub fn sync_trauma(&mut self, hits: Vec<([u8; 32], crate::safety_swarm::TraumaHit)>) {
        for (sig, hit) in hits {
            self.trauma_cache.update(sig, hit);
        }
    }

    /// Learn from action outcome feedback.
    /// This method coordinates updates to both the TraumaRegistry and BeliefEngine
    /// based on whether the action was successful.
    /// 
    /// # Arguments
    /// * `success` - Whether the action was successful
    /// * `action_type` - Type of action that was executed (for trauma logging)
    /// * `ts_us` - Timestamp of the outcome
    /// * `severity` - Severity of failure (0.0-5.0), ignored if success=true
    /// 
    /// # Safety Behavior
    /// On failure, the system becomes MORE CONSERVATIVE:
    /// - Trauma registry records the failure with exponential backoff
    /// - Belief engine increases process noise (acknowledges uncertainty)
    /// - Learning rate is reduced (more cautious updates)
    /// 
    /// On success, the system becomes MORE CONFIDENT:
    /// - Process noise decreases (model is accurate)
    /// - Learning rate slightly increases (model is on track)
    pub fn learn_from_outcome(
        &mut self,
        success: bool,
        action_type: String,
        ts_us: i64,
        severity: f32,
    ) {
        // Update belief engine (Active Inference learning)
        crate::belief::BeliefEngine::process_feedback(
            &mut self.fep_state,
            &mut self.config.fep,
            success,
        );

        // Update causal graph weights based on outcome
        let context_state = self.causal_graph.extract_state_values(&self.belief_state);
        let action = crate::causal::ActionPolicy {
            action_type: crate::causal::ActionType::BreathGuidance,
            intensity: 0.8,
        };
        const CAUSAL_LEARNING_RATE: f32 = 0.05;
        self.causal_graph.update_weights(&context_state, &action, success, CAUSAL_LEARNING_RATE);

        // If failure, record trauma to prevent repeating the same mistake
        if !success {
            // Compute context hash for trauma registry
            let context_hash = crate::safety_swarm::trauma_sig_hash(
                self.last_goal,
                self.belief_state.mode as u8,
                self.last_pattern_id,
                &self.context,
            );

            // Record negative feedback with exponential backoff
            self.trauma_registry.record_negative_feedback(
                context_hash,
                action_type,
                ts_us,
                severity,
            );

            // Write-through to trauma cache for immediate availability
            if let Some(hit) = self.trauma_registry.query(&context_hash) {
                self.trauma_cache.update(context_hash, hit);
            }
        }
    }

    /// PR3: Make a control decision from estimate with INTRINSIC SAFETY.
    /// Safety cannot be bypassed - Engine always uses internal trauma_cache.
    /// If cache is empty/uninitialized, returns SafeFallback decision.
    /// 
    /// Returns: (ControlDecision, should_persist, policy_info, deny_reason)
    pub fn make_control(&mut self, est: &Estimate, ts_us: i64) -> (ControlDecision, bool, Option<(u8, u32, f32)>, Option<String>) {
        // Run control-tick belief update (1-2Hz) using cached latest sensor features
        if let (Some(sf), Some(phys)) = (self.last_sf, self.last_phys) {
            // PR4: Use dt_sec helper to prevent wraparound if clocks go backwards
            let dt_sec = match self.last_control_ts_us {
                Some(last) => crate::domain::dt_sec(ts_us, last),
                None => 0.0,
            };
            self.last_control_ts_us = Some(ts_us);

            let guide_phase = self.breath.guide_phase_norm();
            let guide_bpm = {
                let total_us = self.breath.pm.durations.total_us();
                if total_us == 0 { 0.0 } else { 60_000_000f32 / (total_us as f32) }
            };
            let res = self.resonance_tracker.update(ts_us, guide_phase, guide_bpm, phys.rr_bpm, &self.config);
            self.last_resonance_score = res.resonance_score;

            let tau_res = 6.0f32;
            let alpha = if dt_sec <= 0.0 { 0.0 } else { (dt_sec / (tau_res + dt_sec)).clamp(0.0, 1.0) };
            self.resonance_score_ema = (self.resonance_score_ema * (1.0 - alpha) + res.resonance_score * alpha).clamp(0.0, 1.0);

            let out = self.belief_engine.update_fep_with_config(
                self.belief_state.mode,
                &self.fep_state,
                &sf,
                &phys,
                &self.context,
                dt_sec,
                res,
                &self.config,
            );
            self.belief_state = out.belief;
            self.fep_state = out.fep;
            if self.fep_state.free_energy_ema > self.free_energy_peak {
                self.free_energy_peak = self.fep_state.free_energy_ema;
            }
        }

        // propose base target from belief mode
        let base = match self.belief_state.mode {
            crate::belief::BeliefBasis::Calm => 6.0,
            crate::belief::BeliefBasis::Stress => 8.0,
            crate::belief::BeliefBasis::Focus => 5.0,
            crate::belief::BeliefBasis::Sleepy => 6.0,
            crate::belief::BeliefBasis::Energize => 7.0,
        };
        // fallback to estimator rr if present
        let proposed = est.rr_bpm.unwrap_or(base).clamp(4.0, 12.0);
        
        let mut patch = crate::safety_swarm::PatternPatch { target_bpm: proposed, hold_sec: 30.0, pattern_id: self.last_pattern_id, goal: self.last_goal };

        // build guards
        let mut guards: Vec<Box<dyn crate::safety_swarm::Guard>> = Vec::new();
        guards.push(Box::new(crate::safety_swarm::TraumaGuard { source: &self.trauma_cache, hard_th: self.config.safety.trauma_hard_th, soft_th: self.config.safety.trauma_soft_th }));
        guards.extend(vec![
            Box::new(crate::safety_swarm::ConfidenceGuard { min_conf: 0.2 }),
            Box::new(crate::safety_swarm::BreathBoundsGuard { clamp: crate::safety_swarm::Clamp { rr_min: 4.0, rr_max: 12.0, hold_max_sec: 60.0, max_delta_rr_per_min: 6.0 } }),
            Box::new(crate::safety_swarm::RateLimitGuard { min_interval_sec: 10.0, last_patch_sec: None }),
            Box::new(crate::safety_swarm::ComfortGuard),
            Box::new(crate::safety_swarm::ResourceGuard),
        ]);

        let decide = crate::safety_swarm::decide(
            &guards,
            &patch,
            &self.belief_state,
            &crate::belief::PhysioState { hr_bpm: est.hr_bpm, rr_bpm: est.rr_bpm, rmssd: est.rmssd, confidence: est.confidence },
            &self.context,
            ts_us,
        );
        
        // CRITICAL: Causal Veto Check (happens AFTER safety guards but BEFORE final decision)
        // This prevents actions that historically fail in this context
        const MIN_SUCCESS_PROB: f32 = 0.3;
        const HIGH_SUCCESS_PROB: f32 = 0.8;
        
        let context_state = self.causal_graph.extract_state_values(&self.belief_state);
        let breath_action = crate::causal::ActionPolicy {
            action_type: crate::causal::ActionType::BreathGuidance,
            intensity: 0.8,
        };
        let success_prob = self.causal_graph.predict_success_probability(&context_state, &breath_action);
        
        match decide {
            Err(s) => {
                // Denied by safety guards -> freeze and surface reason
                let reason = s.to_string();
                let poll_interval = crate::controller::compute_poll_interval(self.belief_state.mode, false, &self.context);
                (ControlDecision { target_rate_bpm: self.controller.last_decision_bpm.unwrap_or(proposed), confidence: est.confidence, recommended_poll_interval_ms: poll_interval }, false, Some((self.belief_state.mode as u8, 0, self.belief_state.conf)), Some(reason))
            }
            Ok((applied, bits)) => {
                // Safety guards passed - now apply causal veto
                if success_prob < MIN_SUCCESS_PROB {
                    // CAUSAL VETO: Action historically fails in this context
                    eprintln!("DEBUG: Action Vetoed by Causal Graph (Prob: {:.3})", success_prob);
                    
                    // Fallback to safe default: maintain last decision or use gentle baseline
                    let fallback_bpm = self.controller.last_decision_bpm.unwrap_or(6.0);
                    let poll_interval = crate::controller::compute_poll_interval(self.belief_state.mode, false, &self.context);
                    
                    return (ControlDecision { 
                        target_rate_bpm: fallback_bpm, 
                        confidence: est.confidence * 0.5, // Reduced confidence for vetoed action
                        recommended_poll_interval_ms: poll_interval 
                    }, false, Some((self.belief_state.mode as u8, 0, self.belief_state.conf)), Some(format!("causal_veto_low_prob_{:.2}", success_prob)));
                }
                
                // Action approved by both safety guards and causal graph
                let final_bpm = applied.target_bpm;
                let changed = match self.controller.last_decision_bpm { Some(prev) => (prev - final_bpm).abs() > self.controller.cfg.decision_epsilon_bpm, None => true } && match self.controller.last_decision_ts_us { Some(last_ts) => (ts_us - last_ts) >= self.controller.cfg.min_decision_interval_us, None => true };
                if changed {
                    self.controller.last_decision_bpm = Some(final_bpm);
                    self.controller.last_decision_ts_us = Some(ts_us);
                    self.safety.record_patch(ts_us, final_bpm);
                }
                
                // Causal boost: high-success actions get confidence boost
                let confidence_boost = if success_prob > HIGH_SUCCESS_PROB {
                    1.2 // 20% confidence boost for historically successful actions
                } else {
                    1.0
                };
                
                let poll_interval = crate::controller::compute_poll_interval(self.belief_state.mode, changed, &self.context);
                let boosted_confidence = (est.confidence * confidence_boost).min(1.0);
                (ControlDecision { target_rate_bpm: final_bpm, confidence: boosted_confidence, recommended_poll_interval_ms: poll_interval }, changed, Some((self.belief_state.mode as u8, bits, self.belief_state.conf)), None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimator::Estimator;

    #[test]
    fn engine_decision_flow() {
        let mut eng = Engine::new(6.0);
        eng.update_context(crate::belief::Context { local_hour: 23, is_charging: true, recent_sessions: 1 });
        let est = eng.ingest_sensor(&[60.0, 40.0, 6.0], 0);
        // PR3: make_control no longer takes Option<TraumaSource> - intrinsic safety
        let (dec, persist, policy, deny) = eng.make_control(&est, 0);
        assert!(dec.confidence >= 0.0);
        assert!(policy.is_some());
        assert!(deny.is_none());
    }
}
