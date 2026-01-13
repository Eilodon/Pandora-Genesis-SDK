use crate::breath_engine::BreathEngine;
use crate::causal::{CausalBuffer, CausalGraph};
use crate::config::ZenbConfig;
use crate::controller::AdaptiveController;
use crate::domain::ControlDecision;
use crate::estimator::Estimate;
use crate::resonance::ResonanceTracker;
use crate::safety::RuntimeState;

// Phase 2 Decomposition: Extracted subsystems

use crate::safety_subsystem::SafetySubsystem;

// PANDORA PORT: Resilience and adaptive features
use crate::adaptive::{AdaptiveThreshold, AnomalyDetector, ConfidenceTracker};
use crate::circuit_breaker::{CircuitBreakerConfig, CircuitBreakerManager};

/// High-level engine that holds estimator, safety envelope, controller and breath engine.
///
/// # VAJRA-001 Upgrade
/// This engine now includes three new cognitive capabilities:
/// - `HolographicMemory`: FFT-based associative memory (Tưởng Uẩn)
/// - `SheafPerception`: Laplacian-based sensor consensus (Sắc Uẩn)
/// - `DharmaFilter`: Phase-based ethical action filtering (Hành Uẩn)
pub struct Engine {
    pub controller: AdaptiveController,
    pub breath: BreathEngine,
    pub belief_engine: crate::belief::BeliefEngine,
    pub belief_state: crate::belief::BeliefState,
    pub fep_state: crate::belief::FepState,
    pub context: crate::belief::Context,
    pub config: ZenbConfig,
    pub last_sf: Option<crate::belief::SensorFeatures>,
    pub last_phys: Option<crate::belief::PhysioState>,
    pub resonance_tracker: ResonanceTracker,
    pub last_resonance_score: f32,
    pub resonance_score_ema: f32,
    pub free_energy_peak: f32,
    pub last_pattern_id: i64,
    pub last_goal: i64,
    pub causal_graph: CausalGraph,
    pub observation_buffer: CausalBuffer,
    pub last_observation: Option<crate::domain::Observation>,

    // --- EFE / META-LEARNING ---
    /// Current EFE precision (beta) for policy selection
    pub efe_precision_beta: f32,

    /// Meta-learner for adapting beta
    pub efe_meta_learner: crate::policy::BetaMetaLearner,

    /// Last chosen policy info (for learning)
    pub last_selected_policy: Option<crate::policy::PolicyEvaluation>,

    // === SOTA Features ===
    // === SKANDHA CORE (The Brain) ===
    /// Unified Skandha Pipeline (Sắc-Thọ-Tưởng-Hành-Thức)
    pub skandha_pipeline: crate::skandha::zenb::ZenbPipeline,

    /// Last synthesized state from Skandha pipeline
    pub skandha_state: Option<crate::skandha::SynthesizedState>,

    // === Timestamp Tracking ===
    pub timestamp: crate::timestamp::TimestampLog,

    /// Last sheaf energy (for diagnostics)
    pub last_sheaf_energy: f32,

    // === Phase 2: Extracted Subsystems ===


    /// Belief subsystem - encapsulates belief engine, state, FEP, and hysteresis
    /// Replaces belief_engine, belief_state, fep_state, belief_enter_threshold fields

    /// Safety subsystem - encapsulates safety monitor, trauma registry, and dharma filter
    pub safety: SafetySubsystem,

    // === PANDORA PORT: Resilience ===
    /// Circuit breaker for estimator failures (prevents cascade on repeated errors)
    pub circuit_breaker: CircuitBreakerManager,

    // === PANDORA PORT: Adaptive Thresholds ===
    /// Adaptive threshold for belief state transitions
    pub belief_enter_threshold: AdaptiveThreshold,

    /// Adaptive lower bound for respiration rate (bpm)
    /// Auto-calibrates to individual physiology and sensor quality
    pub rr_min_threshold: AdaptiveThreshold,

    /// Adaptive upper bound for respiration rate (bpm)
    /// Auto-calibrates to individual physiology and sensor quality
    pub rr_max_threshold: AdaptiveThreshold,

    /// Anomaly detector for sensor readings
    pub sensor_anomaly_detector: AnomalyDetector,

    /// Confidence tracker for decision outcomes
    pub decision_confidence: ConfidenceTracker,

    // === Enhanced Observation Buffer ===
    /// Minimum samples before triggering PC algorithm
    pub observation_buffer_min_samples: usize,

    // === WEEK 2: Automatic Scientist Integration ===
    /// Automatic Scientist for causal hypothesis discovery
    pub scientist: crate::scientist::AutomaticScientist,

    // === POLICY ADAPTER: Learning from Outcomes ===
    /// Policy adapter for catastrophe detection and stagnation escape
    pub policy_adapter: crate::policy::PolicyAdapter,

    /// Last executed policy (for outcome learning)
    pub last_executed_policy: Option<crate::policy::ActionPolicy>,

    // === TIER 3: Thermodynamic Logic (GENERIC Framework) ===
    /// Thermodynamic engine for GENERIC dynamics
    pub thermo_engine: crate::thermo_logic::ThermodynamicEngine,
}

impl Engine {
    pub fn new(default_bpm: f32) -> Self {
        Self::new_with_config(default_bpm, None)
    }

    /// Test-only constructor: allows zero/negative timestamps for guards that enforce time sanity.
    pub fn new_for_test(default_bpm: f32) -> Self {
        let mut cfg = ZenbConfig::default();

        Self::new_with_config(default_bpm, Some(cfg))
    }

    pub fn new_with_config(starting_arousal: f32, config: Option<ZenbConfig>) -> Self {
        let mut cfg = config.unwrap_or_default();

        let controller = AdaptiveController::new(Default::default());

        let mut breath =
            BreathEngine::new(crate::breath_engine::BreathMode::Dynamic(starting_arousal));
        let target_bpm = if starting_arousal > 0.0 {
            starting_arousal
        } else {
            cfg.breath.default_target_bpm
        };
        breath.set_target_bpm(target_bpm);
        Self {
            controller,
            breath,
            belief_engine: crate::belief::BeliefEngine::from_config(&cfg.belief),
            belief_state: crate::belief::BeliefState::default(),
            fep_state: crate::belief::FepState::default(),
            context: crate::belief::Context {
                local_hour: 0,
                is_charging: false,
                recent_sessions: 0,
            },
            config: cfg.clone(),
            last_sf: None,
            last_phys: None,
            resonance_tracker: ResonanceTracker::default(),
            last_resonance_score: 1.0,
            resonance_score_ema: 1.0,
            free_energy_peak: 0.0,
            last_pattern_id: 0,
            last_goal: 0,
            causal_graph: CausalGraph::with_priors(),
            observation_buffer: CausalBuffer::default_capacity(),
            last_observation: None,
            // EFE Initialization
            efe_precision_beta: cfg.features.efe_precision_beta.unwrap_or(4.0),
            efe_meta_learner: crate::policy::BetaMetaLearner::default(),

            last_selected_policy: None,

            // SKANDHA CORE
            skandha_pipeline: crate::skandha::zenb::zenb_pipeline(&cfg),
            skandha_state: None,
            timestamp: crate::timestamp::TimestampLog::new(),

            last_sheaf_energy: 0.0,

            // Phase 2: Extracted Subsystems

            safety: SafetySubsystem::with_config(&cfg),

            // PANDORA PORT: Resilience
            circuit_breaker: CircuitBreakerManager::new(CircuitBreakerConfig {
                failure_threshold: 3,
                open_cooldown_ms: 5_000,
                half_open_trial: 1,
                max_circuits: 100,
                state_ttl_secs: 3600,
            }),

            // PANDORA PORT: Adaptive Thresholds
            belief_enter_threshold: AdaptiveThreshold::new(
                cfg.belief.enter_threshold,
                0.2,  // min
                0.8,  // max
                0.05, // learning rate
            ),
            rr_min_threshold: AdaptiveThreshold::new(
                4.0, // base: 4.0 bpm (typical minimum)
                3.0, // min: allow down to 3.0 for deep meditation/athletes
                6.0, // max: don't go above 6.0 as minimum
                0.1, // learning rate: adapt faster for physiological bounds
            ),
            rr_max_threshold: AdaptiveThreshold::new(
                12.0, // base: 12.0 bpm (typical maximum)
                8.0,  // min: don't allow max to drop below 8.0
                15.0, // max: allow up to 15.0 for anxiety/exercise recovery
                0.1,  // learning rate: adapt faster for physiological bounds
            ),
            sensor_anomaly_detector: AnomalyDetector::new(50, 2.5),
            decision_confidence: ConfidenceTracker::new(100),

            // Enhanced Observation Buffer
            observation_buffer_min_samples: 30, // Minimum samples before PC algorithm runs

            // WEEK 2: Automatic Scientist
            scientist: crate::scientist::AutomaticScientist::new(),

            // POLICY ADAPTER: Learning from Outcomes
            policy_adapter: crate::policy::PolicyAdapter::new(1.0),

            last_executed_policy: None,

            // TIER 3: Thermodynamic Logic (GENERIC Framework)
            thermo_engine: {
                let mut engine = crate::thermo_logic::ThermodynamicEngine::default();
                if let Some(temp) = cfg.features.thermo_temperature {
                    engine.set_temperature(temp);
                }
                engine
            },
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
    pub fn ingest_sensor_with_context(
        &mut self,
        features: &[f32],
        ts_us: i64,
        ctx: crate::belief::Context,
    ) -> Estimate {
        self.update_context(ctx);
        self.ingest_sensor(features, ts_us)
    }

    /// Ingest sensor features (returns whether a SensorFeaturesIngested event should be persisted according to downsample policy)
    ///
    /// Expected features layout (client must follow this order):
    /// [hr_bpm, rmssd, rr_bpm, quality, motion]
    /// - `quality` and `motion` are optional and will default to 1.0 and 0.0 respectively if omitted.
    ///
    /// # VAJRA-001 Enhancement
    /// When `use_vajra_architecture` is enabled, sensor data passes through
    /// PerceptionSubsystem for consensus filtering before further processing.
    pub fn ingest_sensor(&mut self, features: &[f32], ts_us: i64) -> Estimate {
        // Construct input for Skandha pipeline
        let input = crate::skandha::SensorInput {
            hr_bpm: features.get(0).copied(),
            hrv_rmssd: features.get(1).copied(),
            rr_bpm: features.get(2).copied(),
            quality: features.get(3).copied().unwrap_or(1.0),
            motion: features.get(4).copied().unwrap_or(0.0),
            timestamp_us: ts_us,
        };

        // SAFETY: Circuit Breaker Check
        if self.circuit_breaker.is_open("skandha_pipeline") {
            log::warn!("Skandha pipeline circuit open - using raw fallback");
            return Estimate {
                ts_us,
                hr_bpm: input.hr_bpm,
                rmssd: input.hrv_rmssd,
                rr_bpm: input.rr_bpm,
                confidence: 0.1, // Fallback low confidence
            };
        }

        // CORE PIPELINE EXECUTION (The Skandha Loop)
        // Wrapped in catch_unwind for resilience
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.skandha_pipeline.process(&input)
        }));

        match result {
            Ok(synthesized) => {
                self.circuit_breaker.record_success("skandha_pipeline");

                // Store state for decision loop
                self.skandha_state = Some(synthesized.clone());

                // Extract values from trusted Rupa form
                let form_values = &synthesized.form.values;
                let estimate = Estimate {
                    ts_us,
                    hr_bpm: Some(form_values[0] * 200.0), // Denormalize
                    rmssd: Some(form_values[1] * 100.0),
                    rr_bpm: Some(form_values[2] * 20.0),
                    confidence: synthesized.confidence,
                };

                // Sync legacy fields for compatibility
                self.last_sf = Some(crate::belief::SensorFeatures {
                    hr_bpm: estimate.hr_bpm,
                    rmssd: estimate.rmssd,
                    rr_bpm: estimate.rr_bpm,
                    quality: input.quality,
                    motion: input.motion,
                });

                self.last_phys = Some(crate::belief::PhysioState {
                    hr_bpm: estimate.hr_bpm,
                    rr_bpm: estimate.rr_bpm,
                    rmssd: estimate.rmssd,
                    confidence: estimate.confidence,
                });

                // Store energy for diagnostics
                self.last_sheaf_energy = synthesized.form.energy;

                // Update timestamp log for dt calculation
                let _ = self.timestamp.update_ingest(ts_us);

                estimate
            },
            Err(_) => {
                self.circuit_breaker.record_failure("skandha_pipeline");
                log::error!("Skandha pipeline panic recovered - using raw fallback");
                
                Estimate {
                    ts_us,
                    hr_bpm: input.hr_bpm,
                    rmssd: input.hrv_rmssd,
                    rr_bpm: input.rr_bpm,
                    confidence: 0.0, // Zero confidence on panic
                }
            }
        }
    }

    /// Advance engine time and compute any cycles (returns cycle count)
    ///
    /// # VAJRA-001 Enhancement
    /// When `use_vajra_architecture` is enabled, observations are encoded into
    /// HolographicMemory for FFT-based associative recall.
    pub fn tick(&mut self, dt_us: u64) -> u64 {
        let (_trans, cycles) = self.breath.tick(dt_us);

        // Push current observation into causal buffer if available
        // Map canonical belief::BeliefState (5-mode) to CausalBeliefState (3-factor)
        // Process Causal Memory & Holographic Encoding
        self.process_causal_memory();

        // Process Automatic Scientist
        self.process_scientist();

        // Process Thermodynamic Evolution
        self.integrate_thermodynamics();

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
            self.safety.trauma_cache_mut().update(sig, hit);
        }
    }

    /// Learn from action outcome feedback.
    /// This method coordinates updates to both the TraumaRegistry, BeliefEngine,
    /// and PolicyAdapter based on whether the action was successful.
    ///
    /// # Arguments
    /// * `success` - Whether the action was successful
    /// * `action_type` - String identifier for the action taken
    /// * `ts_us` - Timestamp of the outcome
    /// * `severity` - Severity of negative outcome (0.0 = mild, 1.0 = severe)
    ///
    /// # Behavior
    /// - If unsuccessful:
    ///   - TraumaRegistry is updated with a trauma hit for this context
    ///   - PolicyAdapter detects catastrophe and masks harmful policy
    ///   - Learning rate temporarily decreases (model was wrong)
    ///   - Arousal target temporarily increases (heightened vigilance)
    /// - If successful:
    ///   - PolicyAdapter updates Q-values positively
    ///   - Learning rate slightly increases (model is on track)
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
        if let Err(e) =
            self.causal_graph
                .update_weights(&context_state, &action, success, CAUSAL_LEARNING_RATE)
        {
            log::warn!("Causal Update Failed: {}", e);
        }

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
            self.safety.trauma_registry_mut().record_negative_feedback(
                context_hash,
                action_type,
                ts_us,
                severity,
            );

            // Write-through to trauma cache for immediate availability
            if let Some(hit) = self.safety.trauma_registry().query(&context_hash) {
                self.safety.trauma_cache_mut().update(context_hash, hit);
            }
        }

        // SOTA: Meta-learning for EFE precision
        // Adapt Beta based on exploration/exploitation outcome
        if let Some(last_policy) = &self.last_selected_policy {
            // If epistemic value dominates, it was an exploratory action
            let was_exploratory = last_policy.epistemic_value > last_policy.pragmatic_value;

            let old_beta = self.efe_precision_beta;
            self.efe_precision_beta = self.efe_meta_learner.update_beta(
                self.efe_precision_beta,
                was_exploratory,
                success,
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

        // PANDORA PORT: Update confidence tracker based on outcome
        self.decision_confidence.update(success);

        // PANDORA PORT: Adapt belief threshold based on performance
        // Compute performance delta: positive if success, negative if failure
        let performance_delta = if success { 0.1 } else { -0.1 };
        self.belief_enter_threshold.adapt(performance_delta);

        // === POLICY ADAPTER: Learning from Outcomes ===
        // Update PolicyAdapter if enabled and we have a last executed policy
        if self.config.features.policy_adapter_enabled.unwrap_or(false) {
            if let Some(ref policy) = self.last_executed_policy {
                // Convert success to reward: success=1.0, failure=-severity
                let reward = if success { 1.0 } else { -severity };

                // Generate state hash from belief mode
                let state_hash = format!("{:?}", self.belief_state.mode);

                // Update adapter and get exploration boost
                let exploration_boost =
                    self.policy_adapter
                        .update_with_outcome(&state_hash, policy, reward, success);

                // Apply exploration boost to EFE precision (inverse relationship)
                // Higher exploration = lower precision (more uniform sampling)
                self.efe_precision_beta = (self.efe_precision_beta / exploration_boost)
                    .max(0.1)
                    .min(10.0);

                log::debug!(
                    "Policy outcome: success={}, reward={:.2}, exploration_boost={:.1}x, new_beta={:.2}",
                    success, reward, exploration_boost, self.efe_precision_beta
                );
            }
        }
    }

    /// Check if observation buffer has enough samples for PC algorithm.
    ///
    /// Returns true if buffer length >= observation_buffer_min_samples.
    #[inline]
    pub fn is_ready_for_discovery(&self) -> bool {
        self.observation_buffer.len() >= self.observation_buffer_min_samples
    }

    /// Get current observation buffer size.
    #[inline]
    pub fn observation_buffer_len(&self) -> usize {
        self.observation_buffer.len()
    }

    /// Get circuit breaker statistics for monitoring.
    pub fn circuit_breaker_stats(&self) -> crate::circuit_breaker::CircuitStats {
        self.circuit_breaker.stats()
    }

    /// Get current adaptive threshold values for diagnostics.
    pub fn adaptive_thresholds_info(&self) -> (f32, f32, f32) {
        (
            self.belief_enter_threshold.get(),
            self.belief_enter_threshold.base(),
            self.decision_confidence.success_rate(),
        )
    }

    /// Get current physiological context from Sheaf perception.
    ///
    /// Returns the auto-detected or manually set context that determines
    /// the sheaf's anomaly threshold and diffusion rate.
    pub fn sheaf_context(&self) -> crate::perception::PhysiologicalContext {
        self.skandha_pipeline.rupa.context()
    }

    /// Manually override physiological context.
    ///
    /// Use this to set a specific context instead of auto-detection.
    /// Note: Context will be overwritten on next `ingest_sensor` call
    /// unless auto-detection is disabled.
    ///
    /// # Example
    /// ```ignore
    /// engine.set_sheaf_context(PhysiologicalContext::ModerateExercise);
    /// ```
    pub fn set_sheaf_context(&mut self, context: crate::perception::PhysiologicalContext) {
        self.skandha_pipeline.rupa.set_context(context);
    }

    /// Get sheaf diagnostics: (energy, context, is_adaptive_alpha_enabled)
    pub fn sheaf_diagnostics(&self) -> (f32, crate::perception::PhysiologicalContext, bool) {
        (
            self.last_sheaf_energy,
            self.skandha_pipeline.rupa.context(),
            self.skandha_pipeline.rupa.sheaf.is_adaptive_alpha_enabled(),
        )
    }

    /// Perform thermodynamic step using GENERIC framework.
    ///
    /// Integrates belief state using the GENERIC equation:
    /// `dz/dt = L·∇H + M·∇S`
    ///
    /// Where L is the Poisson bracket (reversible) and M is friction (irreversible).
    ///
    /// # Arguments
    /// * `target` - Target belief state (drives energy minimization)
    /// * `steps` - Number of integration steps
    ///
    /// # Returns
    /// Updated belief state probabilities
    pub fn thermo_step(&mut self, target: &[f32; 5], steps: usize) -> [f32; 5] {
        if !self.config.features.thermo_enabled.unwrap_or(false) {
            return self.belief_state.p;
        }

        // Convert belief state to DVector
        let state = nalgebra::DVector::from_vec(self.belief_state.p.to_vec());
        let target_vec = nalgebra::DVector::from_vec(target.to_vec());

        // Integrate using GENERIC dynamics
        let new_state = self.thermo_engine.integrate(&state, &target_vec, steps);

        // Update belief state
        let mut p = [0.0f32; 5];
        for i in 0..5 {
            p[i] = new_state[i];
        }
        self.belief_state.p = p;

        // Normalize (ensure sum = 1 for probability interpretation)
        let sum: f32 = p.iter().sum();
        if sum > 0.0 {
            for i in 0..5 {
                self.belief_state.p[i] /= sum;
            }
        }

        self.belief_state.p
    }

    /// Get thermodynamic diagnostics.
    ///
    /// # Returns
    /// (free_energy, entropy, temperature, enabled)
    pub fn thermo_info(&self) -> (f32, f32, f32, bool) {
        let state = nalgebra::DVector::from_vec(self.belief_state.p.to_vec());
        let target = nalgebra::DVector::from_vec([0.5f32; 5].to_vec()); // Neutral target for diagnostics

        let free_energy = self.thermo_engine.free_energy(&state, &target);
        let entropy = self.thermo_engine.entropy(&state);
        let temperature = self.thermo_engine.config().temperature;

        (
            free_energy,
            entropy,
            temperature,
            self.config.features.thermo_enabled.unwrap_or(false),
        )
    }

    /// PR3: Make a control decision from estimate with INTRINSIC SAFETY.
    /// Safety cannot be bypassed - Engine always uses internal trauma_cache.
    /// If cache is empty/uninitialized, returns SafeFallback decision.
    ///
    /// Returns: (ControlDecision, should_persist, policy_info, deny_reason)
    pub fn make_control(
        &mut self,
        est: &Estimate,
        ts_us: i64,
    ) -> (
        ControlDecision,
        bool,
        Option<(u8, u32, f32)>,
        Option<String>,
    ) {
        // Run control-tick belief update (1-2Hz) using cached latest sensor features
        if let (Some(sf), Some(phys)) = (self.last_sf, self.last_phys) {
            // PR4: Use dt_sec helper to prevent wraparound if clocks go backwards
            // Update control timestamp and get dt
            let dt_sec = match self.timestamp.update_control(ts_us) {
                Ok(dt) => dt,
                Err(e) => {
                    log::error!("Control loop timestamp error: {}", e);
                    // Fallback to minimal +ve step to avoid division by zero
                    0.001
                }
            };

            let guide_phase = self.breath.guide_phase_norm();
            let guide_bpm = {
                let total_us = self.breath.pm.durations.total_us();
                if total_us == 0 {
                    0.0
                } else {
                    60_000_000f32 / (total_us as f32)
                }
            };
            let res = self.resonance_tracker.update(
                ts_us,
                guide_phase,
                guide_bpm,
                phys.rr_bpm,
                &self.config,
            );
            self.last_resonance_score = res.resonance_score;

            let tau_res = 6.0f32;
            let alpha = if dt_sec <= 0.0 {
                0.0
            } else {
                (dt_sec / (tau_res + dt_sec)).clamp(0.0, 1.0)
            };
            self.resonance_score_ema = (self.resonance_score_ema * (1.0 - alpha)
                + res.resonance_score * alpha)
                .clamp(0.0, 1.0);

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
        // ADAPTIVE BOUNDS: Use platform-aware, auto-calibrating thresholds
        let rr_min = self.rr_min_threshold.get();
        let rr_max = self.rr_max_threshold.get();
        let mut proposed = est.rr_bpm.unwrap_or(base).clamp(rr_min, rr_max);

        // --- EFE POLICY SELECTION ---
        if self.config.features.use_efe_selection {
            use crate::policy::{ActionPolicy, EFECalculator, PolicyLibrary};

            // 1. Instantiate Calculator with current Beta
            let efe_calc = EFECalculator::new(self.efe_precision_beta);

            // 2. Define Candidate Policies
            let policies = vec![
                PolicyLibrary::calming_breath(),
                PolicyLibrary::energizing_breath(),
                PolicyLibrary::focus_mode(),
                PolicyLibrary::observe(),
            ];

            // 3. Predict future state (from Causal Graph)
            // For now, we use current belief as proxy for immediate prediction
            // In future, CausalGraph::predict_state() would be called here
            let predicted_state = self.belief_state.to_5mode_array();
            let predicted_uncertainty = self.belief_state.conf;

            // 4. Compute EFE for each policy
            let mut evaluations: Vec<crate::policy::PolicyEvaluation> = policies
                .into_iter()
                .map(|policy| {
                    efe_calc.compute_efe(
                        &policy,
                        &self.belief_state.to_5mode_array(),
                        self.belief_state.uncertainty(),
                        &predicted_state,
                        crate::belief::uncertainty_from_confidence(predicted_uncertainty),
                    )
                })
                .collect();

            // 5. Select Policy (Softmax)
            // Use deterministic RNG seed from timestamp for reproducibility
            let rng_value = ((ts_us % 1000) as f32) / 1000.0;
            efe_calc.compute_selection_probabilities(&mut evaluations);
            let selected_policy_ref = efe_calc.sample_policy(&evaluations, rng_value);

            // 6. Apply Selection
            // Store for learning loop
            // We need to clone it to store it (sample_policy returns ref)
            let selected_clone = evaluations
                .iter()
                .find(|e| e.policy.description() == selected_policy_ref.description())
                .cloned();
            self.last_selected_policy = selected_clone;

            // Store the ActionPolicy for outcome learning
            self.last_executed_policy = Some(selected_policy_ref.clone());

            // Check if policy is masked by PolicyAdapter
            if self.config.features.policy_adapter_enabled.unwrap_or(false)
                && self.policy_adapter.is_policy_masked(selected_policy_ref)
            {
                log::warn!(
                    "PolicyAdapter MASKED policy: {}, falling back to NoAction",
                    selected_policy_ref.description()
                );
                // Override with safe fallback
                let fallback = crate::policy::PolicyLibrary::observe();
                let decision = fallback.to_control_decision(proposed, est.confidence);
                proposed = decision.target_rate_bpm;
            } else {
                // Convert to BPM target
                // Use current proposed as fallback
                let decision = selected_policy_ref.to_control_decision(proposed, est.confidence);
                proposed = decision.target_rate_bpm;
            }

            // Handle Side Effects (e.g. Digital Interventions)
            if let ActionPolicy::DigitalIntervention(di) = selected_policy_ref {
                // In a full implementation, this would emit a separate event or side-effect
                // For now, we just log it as the primary decision driver
                // (Engine currently focused on Breath, but this lays groundwork for multi-modal)
                log::info!("EFE Selected Digital Intervention: {:?}", di);
            }
        }

        // VAJRA-001: Use Skandha decision if available
        // This overrides EFE logic with the unified pipeline's decision (which may include its own EFE)
        if self.config.features.vajra_enabled {
            if let Some(ref s) = self.skandha_state {
                if let Some(ref d) = s.decision {
                    proposed = d.target_bpm;
                    // Also use confidence from Skandha
                    log::debug!(
                        "Vajra-001: Using Skandha decision {:.2} (conf={:.2})",
                        proposed,
                        d.confidence
                    );
                }
            }
        }

        // VAJRA-001: Dharma Filter - phase-based ethical action filtering
        if self.config.features.vajra_enabled {
            use crate::safety::ComplexDecision;

            // Convert proposed BPM to complex decision
            let baseline_bpm = 6.0;
            let action = ComplexDecision::from_bpm_target(proposed, baseline_bpm);

            // Apply Dharma filter
            match action.filter_with(&self.skandha_pipeline.sankhara.dharma) {
                Some(sanctioned) => {
                    let new_proposed = sanctioned.to_bpm(baseline_bpm);
                    if (new_proposed - proposed).abs() > 0.1 {
                        log::info!(
                            "DharmaFilter: Scaled action from {:.2} to {:.2} BPM (alignment={:.3})",
                            proposed,
                            new_proposed,
                            self.skandha_pipeline
                                .sankhara
                                .dharma
                                .check_alignment(action.vector)
                        );
                    }
                    // ADAPTIVE BOUNDS: Reuse calibrated thresholds after dharma scaling
                    proposed = new_proposed.clamp(rr_min, rr_max);
                }
                None => {
                    // Dharma veto - fall back to baseline
                    log::warn!(
                        "DharmaFilter: Action VETOED (proposed={:.2} BPM, phase={:.3} rad)",
                        proposed,
                        action.phase()
                    );
                    proposed = baseline_bpm;
                }
            }
        }

        let mut patch = crate::safety_swarm::PatternPatch {
            target_bpm: proposed,
            hold_sec: 30.0,
            pattern_id: self.last_pattern_id,
            goal: self.last_goal,
        };

        // build guards
        let decide = {
            let mut guards: Vec<Box<dyn crate::safety_swarm::Guard + '_>> = Vec::new();
            guards.push(Box::new(crate::safety_swarm::TraumaGuard {
                source: self.safety.trauma_cache(),
                hard_th: self.config.safety.trauma_hard_th,
                soft_th: self.config.safety.trauma_soft_th,
            }));
            // Fix: Use controller state for rate limiting (stateless guard needs history)
            let last_patch_sec = self
                .controller
                .last_decision_ts_us
                .map(|last| crate::domain::dt_sec(ts_us, last));

            guards.push(Box::new(crate::safety_swarm::ConfidenceGuard {
                min_conf: 0.2,
            }));
            guards.push(Box::new(crate::safety_swarm::BreathBoundsGuard {
                clamp: crate::safety_swarm::Clamp {
                    rr_min: 4.0,
                    rr_max: 12.0,
                    hold_max_sec: 60.0,
                    max_delta_rr_per_min: 6.0,
                },
            }));
            guards.push(Box::new(crate::safety_swarm::RateLimitGuard {
                min_interval_sec: 10.0,
                last_patch_sec,
            }));
            guards.push(Box::new(crate::safety_swarm::ComfortGuard));
            guards.push(Box::new(crate::safety_swarm::ResourceGuard));

            crate::safety_swarm::decide(
                guards.as_slice(),
                &patch,
                &self.belief_state,
                &crate::belief::PhysioState {
                    hr_bpm: est.hr_bpm,
                    rr_bpm: est.rr_bpm,
                    rmssd: est.rmssd,
                    confidence: est.confidence,
                },
                &self.context,
                ts_us,
            )
        };

        // CRITICAL: Causal Veto Check (happens AFTER safety guards but BEFORE final decision)
        // This prevents actions that historically fail in this context
        const MIN_SUCCESS_PROB: f32 = 0.3;
        const HIGH_SUCCESS_PROB: f32 = 0.8;

        // LTL Safety Monitor Check
        // Verify tempo bounds, panic halt, and safety lock invariants
        let session_duration = self.timestamp.session_duration(ts_us);

        let runtime_state = RuntimeState {
            tempo_scale: proposed / 6.0, // Normalize to baseline 6 BPM
            status: "RUNNING".to_string(),
            session_duration,
            prediction_error: self.fep_state.free_energy_ema,
            last_update_timestamp: ts_us as u64,
        };

        if let Err(violations) = self.safety.monitor_mut().check(&runtime_state) {
            let reason = format!("ltl_violation:{}", violations[0].property_name);
            eprintln!("ENGINE_DENY: LTL safety violation: {:?}", violations);

            let poll_interval = crate::controller::compute_poll_interval(
                &mut self.controller.poller,
                self.fep_state.free_energy_ema,
                self.belief_state.conf,
                false,
                &self.context,
            );

            // Shield tempo to safe bounds
            let safe_tempo = self.safety.monitor().shield_tempo(proposed / 6.0) * 6.0;

            return (
                ControlDecision {
                    target_rate_bpm: safe_tempo,
                    confidence: est.confidence * 0.3, // Heavily reduced confidence
                    recommended_poll_interval_ms: poll_interval,
                },
                false,
                Some((self.belief_state.mode as u8, 0, self.belief_state.conf)),
                Some(reason),
            );
        }

        let context_state = self.causal_graph.extract_state_values(&self.belief_state);
        let breath_action = crate::causal::ActionPolicy {
            action_type: crate::causal::ActionType::BreathGuidance,
            intensity: 0.8,
        };
        let success_prob = self
            .causal_graph
            .predict_success_probability(&context_state, &breath_action);

        match decide {
            Err(s) => {
                // Denied by safety guards -> freeze and surface reason
                let reason = s.to_string();
                eprintln!("ENGINE_DENY: safety_guard reason={}", reason);
                let poll_interval = crate::controller::compute_poll_interval(
                    &mut self.controller.poller,
                    self.fep_state.free_energy_ema,
                    self.belief_state.conf,
                    false,
                    &self.context,
                );
                (
                    ControlDecision {
                        target_rate_bpm: self.controller.last_decision_bpm.unwrap_or(proposed),
                        confidence: est.confidence,
                        recommended_poll_interval_ms: poll_interval,
                    },
                    false,
                    Some((self.belief_state.mode as u8, 0, self.belief_state.conf)),
                    Some(reason),
                )
            }
            Ok((applied, bits)) => {
                // Safety guards passed - now apply causal veto
                if success_prob.value < MIN_SUCCESS_PROB {
                    // CAUSAL VETO: Action historically fails in this context
                    eprintln!(
                        "DEBUG: Action Vetoed by Causal Graph (Prob: {:.3}, Conf: {:.3})",
                        success_prob.value, success_prob.confidence
                    );
                    eprintln!(
                        "ENGINE_DENY: causal_veto_low_prob prob={:.3} conf={:.3} mode={:?}",
                        success_prob.value, success_prob.confidence, self.belief_state.mode
                    );

                    // Fallback to safe default: maintain last decision or use gentle baseline
                    let fallback_bpm = self.controller.last_decision_bpm.unwrap_or(6.0);
                    let poll_interval = crate::controller::compute_poll_interval(
                        &mut self.controller.poller,
                        self.fep_state.free_energy_ema,
                        self.belief_state.conf,
                        false,
                        &self.context,
                    );

                    return (
                        ControlDecision {
                            target_rate_bpm: fallback_bpm,
                            confidence: est.confidence * 0.5, // Reduced confidence for vetoed action
                            recommended_poll_interval_ms: poll_interval,
                        },
                        false,
                        Some((self.belief_state.mode as u8, 0, self.belief_state.conf)),
                        Some(format!("causal_veto_low_prob_{:.2}", success_prob.value)),
                    );
                }

                // Action approved by both safety guards and causal graph
                let final_bpm = applied.target_bpm;
                let changed = match self.controller.last_decision_bpm {
                    Some(prev) => {
                        (prev - final_bpm).abs() > self.controller.cfg.decision_epsilon_bpm
                    }
                    None => true,
                } && match self.controller.last_decision_ts_us {
                    Some(last_ts) => {
                        (ts_us - last_ts) >= self.controller.cfg.min_decision_interval_us
                    }
                    None => true,
                };
                if changed {
                    self.controller.last_decision_bpm = Some(final_bpm);
                    self.controller.last_decision_ts_us = Some(ts_us);
                }

                // Causal boost: high-success actions get confidence boost
                let confidence_boost = if success_prob.value > HIGH_SUCCESS_PROB {
                    1.2 // 20% confidence boost for historically successful actions
                } else {
                    1.0
                };

                let poll_interval = crate::controller::compute_poll_interval(
                    &mut self.controller.poller,
                    self.fep_state.free_energy_ema,
                    self.belief_state.conf,
                    changed,
                    &self.context,
                );
                let boosted_confidence = (est.confidence * confidence_boost).min(1.0);
                (
                    ControlDecision {
                        target_rate_bpm: final_bpm,
                        confidence: boosted_confidence,
                        recommended_poll_interval_ms: poll_interval,
                    },
                    changed,
                    Some((self.belief_state.mode as u8, bits, self.belief_state.conf)),
                    None,
                )
            }
        }
    }
}

// ============================================================================
// TIER 4a: Skandha Pipeline Integration
// ============================================================================
impl Engine {
    /// Execute the full Skandha pipeline (Tier 4a).
    ///
    /// This method acts as the "Vinnana" (Consciousness) orchestrator, processing
    /// raw inputs through the five aggregates to produce a synthesized state.
    ///
    /// # Arguments
    /// * `obs` - Snapshot of current observation (sensor data, context)
    ///
    /// # Returns
    /// Synthesized state including belief, affect, pattern, and decision.
    pub fn process_skandha_pipeline(
        &mut self,
        obs: &crate::domain::Observation,
    ) -> crate::skandha::SynthesizedState {
        // Map Observation to SensorInput
        let bio = obs.bio_metrics.as_ref();
        let timestamp = obs.timestamp_us;

        // Rupa stage input
        let input = crate::skandha::SensorInput {
            hr_bpm: bio.and_then(|b| b.hr_bpm),
            hrv_rmssd: bio.and_then(|b| b.hrv_rmssd),
            rr_bpm: bio.and_then(|b| b.respiratory_rate),
            quality: 1.0, // Default quality if not in bio metrics? assume good if present
            motion: 0.0,  // Motion not currently in bio metrics
            timestamp_us: timestamp,
        };

        // Execute unified pipeline
        let result = self.skandha_pipeline.process(&input);

        // Log synthesis result
        log::debug!(
            "Skandha Synthesis: Conf={:.2} Mode={} FE={:.2}",
            result.confidence,
            result.mode,
            result.free_energy
        );

        result
    }
    // Helper: Process Causal Memory & Holographic Encoding
    fn process_causal_memory(&mut self) {
        if let Some(ref obs) = self.last_observation {
            let snapshot = crate::causal::ObservationSnapshot {
                timestamp_us: obs.timestamp_us,
                observation: obs.clone(),
                action: None,
                belief_state: Some(crate::domain::CausalBeliefState {
                    // Map 5-mode belief to 3-factor causal representation
                    bio_state: [
                        self.belief_state.p[0], // Calm
                        self.belief_state.p[1], // Stress (Aroused)
                        self.belief_state.p[3], // Sleepy (Fatigue)
                    ],
                    cognitive_state: [
                        self.belief_state.p[2],       // Focus
                        1.0 - self.belief_state.p[2], // Distracted (inverse of Focus)
                        0.0,                          // Flow (not directly mapped)
                    ],
                    social_state: [
                        0.33, // Solitary (uniform prior - not tracked in 5-mode)
                        0.33, // Interactive
                        0.33, // Overwhelmed
                    ],
                    confidence: self.belief_state.conf,
                    last_update_us: obs.timestamp_us,
                    cognitive_context: obs.cognitive_context.clone(),
                }),
            };
            self.observation_buffer.push(snapshot);

            // VAJRA-001: Encode into Holographic Memory
            if self.config.features.vajra_enabled {
                // Create context key from belief state
                let key = crate::memory::hologram::encode_context_key(
                    &self.belief_state.p[..],
                    self.skandha_pipeline.sanna.memory.dim(),
                );

                // Create value from observation features
                let bio = obs.bio_metrics.as_ref();
                let obs_features = vec![
                    bio.and_then(|b| b.hr_bpm).unwrap_or(0.0) / 200.0, // Normalize
                    bio.and_then(|b| b.hrv_rmssd).unwrap_or(0.0) / 100.0,
                    bio.and_then(|b| b.respiratory_rate).unwrap_or(0.0) / 20.0,
                    self.belief_state.conf,
                    self.last_resonance_score,
                ];
                let value = crate::memory::hologram::encode_state_value(
                    &obs_features,
                    self.skandha_pipeline.sanna.memory.dim(),
                );

                // Entangle (store) the association
                self.skandha_pipeline.sanna.memory.entangle(&key, &value);

                // Apply decay (forgetting) - 0.999 = very slow decay
                self.skandha_pipeline.sanna.memory.decay(0.999);
            }
        }
    }

    // Helper: Process Automatic Scientist Logic
    fn process_scientist(&mut self) {
        if self.config.features.scientist_enabled.unwrap_or(false) {
            // Feed current belief state as observation [hr, hrv, rr, conf, resonance]
            if let Some(ref obs) = self.last_observation {
                let bio = obs.bio_metrics.as_ref();
                let scientist_obs = [
                    bio.and_then(|b| b.hr_bpm).unwrap_or(60.0) / 200.0, // Normalize to [0,1]
                    bio.and_then(|b| b.hrv_rmssd).unwrap_or(50.0) / 100.0,
                    bio.and_then(|b| b.respiratory_rate).unwrap_or(6.0) / 20.0,
                    self.belief_state.conf,
                    self.last_resonance_score,
                ];
                self.scientist.observe(scientist_obs);

                // Tick scientist state machine
                if self.scientist.tick() {
                    log::debug!(
                        "Scientist state transition: {}",
                        self.scientist.state_name()
                    );
                }

                // Check for crystallized discoveries (using pending queue)
                let discoveries = self.scientist.drain_pending_discoveries();
                for hypothesis in discoveries {
                    log::info!(
                        "Scientist discovered causal edge: {} -> {} (strength={:.2}, confidence={:.2})",
                        hypothesis.from_variable,
                        hypothesis.to_variable,
                        hypothesis.strength,
                        hypothesis.confidence
                    );

                    // WIRE DISCOVERY to CausalGraph
                    let map_var = |idx: u8| -> Option<crate::causal::Variable> {
                        match idx {
                            0 => Some(crate::causal::Variable::HeartRate),
                            1 => Some(crate::causal::Variable::HeartRateVariability),
                            2 => Some(crate::causal::Variable::RespiratoryRate),
                            // 3 (Confidence) and 4 (Resonance) are internal metrics not yet in CausalGraph
                            _ => None,
                        }
                    };

                    if let (Some(cause), Some(effect)) = (
                        map_var(hypothesis.from_variable),
                        map_var(hypothesis.to_variable),
                    ) {
                        let edge = crate::causal::CausalEdge {
                            successes: (hypothesis.confidence * 100.0) as u32,
                            failures: ((1.0 - hypothesis.confidence) * 100.0) as u32,
                            source: crate::causal::CausalSource::Learned {
                                observation_count: 50, // Minimum for confidence
                                confidence_score: hypothesis.confidence,
                            },
                        };
                        self.causal_graph.set_link(cause, effect, edge);
                        log::info!("Scientist Wired Link: {:?} -> {:?}", cause, effect);
                    } else {
                        log::debug!(
                            "Skipping wiring for internal variables {}->{}",
                            hypothesis.from_variable,
                            hypothesis.to_variable
                        );
                    }
                }
            }
        }
    }

    // Helper: Integrate Thermodynamics
    fn integrate_thermodynamics(&mut self) {
        // VAJRA-001: Thermodynamic Evolution (GENERIC)
        // Apply thermodynamic laws for smooth belief state evolution
        if self.config.features.vajra_enabled
            && self.config.features.thermo_enabled.unwrap_or(false)
        {
            // Use Skandha pipeline belief output as the TARGET
            // This makes the system evolve toward the inferred belief state
            // rather than arbitrarily toward max entropy
            let target = if let Some(ref skandha_state) = self.skandha_state {
                // Use Skandha's synthesized belief as thermodynamic attractor
                skandha_state.belief
            } else {
                // Fallback: use current belief state (no drift)
                self.belief_state.p
            };
            
            // Apply thermodynamic step - system smoothly evolves toward target
            // This provides temporal smoothing and prevents abrupt state transitions
            self.thermo_step(&target, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_decision_flow() {
        let mut eng = Engine::new_for_test(6.0);
        eng.update_context(crate::belief::Context {
            local_hour: 23,
            is_charging: true,
            recent_sessions: 1,
        });
        let est = eng.ingest_sensor(&[60.0, 40.0, 6.0], 0);
        // PR3: make_control no longer takes Option<TraumaSource> - intrinsic safety
        let (dec, _persist, policy, deny) = eng.make_control(&est, 0);
        assert!(dec.confidence >= 0.0);
        assert!(policy.is_some());
        assert!(deny.is_none());
    }
}
