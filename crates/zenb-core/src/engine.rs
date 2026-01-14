use crate::breath_engine::BreathEngine;
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

    // === Causal Subsystem (Phase 6) ===
    pub causal: crate::causal_subsystem::CausalSubsystem,

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
    /// Now uses BeliefSubsystem as Vedana stage (Single Source of Truth)
    pub skandha_pipeline: crate::skandha::zenb::ZenbPipelineUnified,

    /// Last synthesized state from Skandha pipeline
    pub skandha_state: Option<crate::skandha::SynthesizedState>,

    // === Timestamp Tracking ===
    pub timestamp: crate::timestamp::TimestampLog,

    /// Last sheaf energy (for diagnostics)
    pub last_sheaf_energy: f32,

    // === Phase 2: Extracted Subsystems ===
    /// Safety subsystem - encapsulates safety monitor, trauma registry, and dharma filter
    pub safety: SafetySubsystem,

    // === PANDORA PORT: Resilience ===
    /// Circuit breaker for estimator failures (prevents cascade on repeated errors)
    pub circuit_breaker: CircuitBreakerManager,

    // === PANDORA PORT: Adaptive Thresholds ===
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

    // === WEEK 2: Automatic Scientist Integration ===
    /// Automatic Scientist for causal hypothesis discovery

    // === POLICY ADAPTER: Learning from Outcomes ===
    /// Policy adapter for catastrophe detection and stagnation escape
    pub policy_adapter: crate::policy::PolicyAdapter,

    /// Last executed policy (for outcome learning)
    pub last_executed_policy: Option<crate::policy::ActionPolicy>,

    // === TIER 3: Thermodynamic Logic (GENERIC Framework) ===
    /// Thermodynamic engine for GENERIC dynamics
    pub thermo_engine: crate::thermo_logic::ThermodynamicEngine,

    // === NEURAL WIRING: Saccade Memory Linker ===
    /// Saccade linker for memory coordinate prediction
    /// Uses LTC network to predict where to look in memory
    pub saccade: crate::memory::SaccadeLinker,
    /// Dedicated holographic memory for saccade predictions
    /// Separate from HDC-based Sanna memory to enable FFT-based recall
    pub saccade_memory: crate::memory::HolographicMemory,
    
    // === VAJRA-VOID: FEP Prediction Loop ===
    /// Last predicted context values (from previous tick)
    /// Used for computing prediction error (surprise)
    pub last_predicted_context: Option<Vec<f32>>,
    /// Exponential moving average of prediction error
    /// High values indicate persistent surprise requiring model update
    pub prediction_error_ema: f32,
}

impl Engine {
    pub fn new(default_bpm: f32) -> Self {
        Self::new_with_config(default_bpm, None)
    }

    /// Test-only constructor: allows zero/negative timestamps for guards that enforce time sanity.
    pub fn new_for_test(default_bpm: f32) -> Self {
        let cfg = ZenbConfig::default();

        Self::new_with_config(default_bpm, Some(cfg))
    }

    pub fn new_with_config(starting_arousal: f32, config: Option<ZenbConfig>) -> Self {
        let cfg = config.unwrap_or_default();

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

            // Causal Subsystem
            causal: crate::causal_subsystem::CausalSubsystem::new(&cfg),

            // EFE Initialization
            efe_precision_beta: cfg.features.efe_precision_beta.unwrap_or(4.0),
            efe_meta_learner: crate::policy::BetaMetaLearner::default(),

            last_selected_policy: None,

            // SKANDHA CORE (Unified)
            skandha_pipeline: crate::skandha::zenb::zenb_pipeline_unified(&cfg),
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

            // WEEK 2: Automatic Scientist

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

            // NEURAL WIRING: Saccade Memory Linker
            saccade: crate::memory::SaccadeLinker::default_for_zenb(),
            saccade_memory: crate::memory::HolographicMemory::default_for_zenb(),
            
            // VAJRA-VOID: FEP Prediction Loop
            last_predicted_context: None,
            prediction_error_ema: 0.0,
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
            hr_bpm: features.first().copied(),
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

                // Feed Causal Scientist
                let causal_feats = [
                    form_values[0],
                    form_values[1],
                    form_values[2],
                    synthesized.confidence,
                    self.last_resonance_score,
                ];
                self.causal.observe(causal_feats, None);

                // Update timestamp log for dt calculation
                let _ = self.timestamp.update_ingest(ts_us);

                estimate
            }
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
    ///
    /// # Neural Wiring Enhancement
    /// Saccade memory linker is called to predict where to look in memory
    /// based on current perception context.
    /// 
    /// # VAJRA-VOID: FEP Prediction Loop
    /// Compares predicted context with actual state to compute prediction error.
    /// High surprise triggers rapid learning and confidence reduction.
    pub fn tick(&mut self, dt_us: u64) -> u64 {
        let (_trans, cycles) = self.breath.tick(dt_us);
        
        // === VAJRA-VOID: FEP COMPARISON (Before updating state) ===
        self.integrate_fep_prediction_loop();

        // Update Causal Subsystem
        if self.causal.tick() {
            // Log any new discoveries
            let discoveries = self.causal.drain_discoveries();
            for hypo in discoveries {
                log::info!(
                    "Agolos Discovery: {} -> {} (strength={:.2})",
                    hypo.from_variable,
                    hypo.to_variable,
                    hypo.strength
                );
            }
        }

        // Process Thermodynamic Evolution
        self.integrate_thermodynamics();

        // NEURAL WIRING: Saccade Memory Recall
        self.integrate_saccade_recall();
        
        // === VAJRA-VOID: DATA REINCARNATION (At end of tick) ===
        self.integrate_data_reincarnation(dt_us as i64);

        cycles
    }

    /// Ingest a full observation (for causal reasoning layer)
    pub fn ingest_observation(&mut self, observation: crate::domain::Observation) {
        // Extract features for Causal Scientist
        let bio = observation.bio_metrics.as_ref();
        let features = [
            bio.and_then(|b| b.hr_bpm).unwrap_or(0.0) / 200.0,
            bio.and_then(|b| b.hrv_rmssd).unwrap_or(0.0) / 100.0,
            bio.and_then(|b| b.respiratory_rate).unwrap_or(0.0) / 20.0,
            self.skandha_pipeline.vedana.confidence(),
            self.last_resonance_score,
        ];

        // Feed to Causal Subsystem
        self.causal.observe(features, Some(observation));
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
        // Update belief engine (Active Inference learning)
        self.skandha_pipeline
            .vedana
            .process_feedback(success, &mut self.config.fep);

        // Update causal graph weights based on outcome
        let context_state = self.causal.graph.extract_state_values(
            self.skandha_pipeline.vedana.state(),
            self.causal.last_observation.as_ref(),
            Some(&self.context),
        );
        let action = crate::causal::ActionPolicy {
            action_type: crate::causal::ActionType::BreathGuidance,
            intensity: 0.8,
        };
        const CAUSAL_LEARNING_RATE: f32 = 0.05;
        if let Err(e) =
            self.causal
                .graph
                .update_weights(&context_state, &action, success, CAUSAL_LEARNING_RATE)
        {
            log::warn!("Causal Update Failed: {}", e);
        }

        // If failure, record trauma to prevent repeating the same mistake
        if !success {
            // Compute context hash for trauma registry
            let context_hash = crate::safety_swarm::trauma_sig_hash(
                self.last_goal,
                self.skandha_pipeline.vedana.mode() as u8,
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
        self.skandha_pipeline
            .vedana
            .adapt_threshold(performance_delta);

        // === POLICY ADAPTER: Learning from Outcomes ===
        // Update PolicyAdapter if enabled and we have a last executed policy
        if self.config.features.policy_adapter_enabled.unwrap_or(false) {
            if let Some(ref policy) = self.last_executed_policy {
                // Convert success to reward: success=1.0, failure=-severity
                let reward = if success { 1.0 } else { -severity };

                // Generate state hash from belief mode
                let state_hash = format!("{:?}", self.skandha_pipeline.vedana.mode());

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

    /// Get circuit breaker statistics for monitoring.
    pub fn circuit_breaker_stats(&self) -> crate::circuit_breaker::CircuitStats {
        self.circuit_breaker.stats()
    }

    /// Get current adaptive threshold values for diagnostics.
    pub fn adaptive_thresholds_info(&self) -> (f32, f32, f32) {
        (
            self.skandha_pipeline.vedana.enter_threshold(),
            self.skandha_pipeline.vedana.enter_threshold_base(),
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
            return *self.skandha_pipeline.vedana.probabilities();
        }

        // Convert belief state to DVector
        let state =
            nalgebra::DVector::from_vec(self.skandha_pipeline.vedana.probabilities().to_vec());
        let target_vec = nalgebra::DVector::from_vec(target.to_vec());

        // Integrate using GENERIC dynamics
        let new_state = self.thermo_engine.integrate(&state, &target_vec, steps);

        // Update belief state
        let mut p = [0.0f32; 5];
        for i in 0..5 {
            p[i] = new_state[i];
        }
        // Normalize (ensure sum = 1 for probability interpretation)
        let sum: f32 = p.iter().sum();
        if sum > 0.0 {
            for i in 0..5 {
                p[i] /= sum;
            }
        }
        self.skandha_pipeline.vedana.set_probabilities(p);

        *self.skandha_pipeline.vedana.probabilities()
    }

    /// Get thermodynamic diagnostics.
    ///
    /// # Returns
    /// (free_energy, entropy, temperature, enabled)
    pub fn thermo_info(&self) -> (f32, f32, f32, bool) {
        let state =
            nalgebra::DVector::from_vec(self.skandha_pipeline.vedana.probabilities().to_vec());
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

            let _out = self.skandha_pipeline.vedana.update_fep(
                &sf,
                &phys,
                &self.context,
                dt_sec,
                res,
                &self.config,
            );
            // State update is handled internally

            let current_fep = self.skandha_pipeline.vedana.free_energy_ema();
            if current_fep > self.free_energy_peak {
                self.free_energy_peak = current_fep;
            }
        }

        // propose base target from belief mode
        let base = match self.skandha_pipeline.vedana.mode() {
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
            let predicted_state = self.skandha_pipeline.vedana.to_5mode_array();
            let predicted_uncertainty = self.skandha_pipeline.vedana.confidence();

            // 4. Compute EFE for each policy
            let mut evaluations: Vec<crate::policy::PolicyEvaluation> = policies
                .into_iter()
                .map(|policy| {
                    efe_calc.compute_efe(
                        &policy,
                        &self.skandha_pipeline.vedana.to_5mode_array(),
                        self.skandha_pipeline.vedana.uncertainty(),
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

        let patch = crate::safety_swarm::PatternPatch {
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
                &self.skandha_pipeline.vedana.state(),
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
            prediction_error: self.skandha_pipeline.vedana.free_energy_ema(),
            last_update_timestamp: ts_us as u64,
        };

        if let Err(violations) = self.safety.monitor_mut().check(&runtime_state) {
            let reason = format!("ltl_violation:{}", violations[0].property_name);
            eprintln!("ENGINE_DENY: LTL safety violation: {:?}", violations);

            let poll_interval = crate::controller::compute_poll_interval(
                &mut self.controller.poller,
                self.skandha_pipeline.vedana.free_energy_ema(),
                self.skandha_pipeline.vedana.confidence(),
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
                Some((
                    self.skandha_pipeline.vedana.mode() as u8,
                    0,
                    self.skandha_pipeline.vedana.confidence(),
                )),
                Some(reason),
            );
        }

        let context_state = self.causal.graph.extract_state_values(
            self.skandha_pipeline.vedana.state(),
            self.causal.last_observation.as_ref(), // Use cached latest observation
            Some(&self.context),
        );
        let breath_action = crate::causal::ActionPolicy {
            action_type: crate::causal::ActionType::BreathGuidance,
            intensity: 0.8,
        };
        let success_prob = self
            .causal
            .graph
            .predict_success_probability(&context_state, &breath_action);

        match decide {
            Err(s) => {
                // Denied by safety guards -> freeze and surface reason
                let reason = s.to_string();
                eprintln!("ENGINE_DENY: safety_guard reason={}", reason);
                let poll_interval = crate::controller::compute_poll_interval(
                    &mut self.controller.poller,
                    self.skandha_pipeline.vedana.free_energy_ema(),
                    self.skandha_pipeline.vedana.confidence(),
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
                    Some((
                        self.skandha_pipeline.vedana.mode() as u8,
                        0,
                        self.skandha_pipeline.vedana.confidence(),
                    )),
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
                        success_prob.value,
                        success_prob.confidence,
                        self.skandha_pipeline.vedana.mode()
                    );

                    // Fallback to safe default: maintain last decision or use gentle baseline
                    let fallback_bpm = self.controller.last_decision_bpm.unwrap_or(6.0);
                    let poll_interval = crate::controller::compute_poll_interval(
                        &mut self.controller.poller,
                        self.skandha_pipeline.vedana.free_energy_ema(),
                        self.skandha_pipeline.vedana.confidence(),
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
                        Some((
                            self.skandha_pipeline.vedana.mode() as u8,
                            0,
                            self.skandha_pipeline.vedana.confidence(),
                        )),
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
                    self.skandha_pipeline.vedana.free_energy_ema(),
                    self.skandha_pipeline.vedana.confidence(),
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
                    Some((
                        self.skandha_pipeline.vedana.mode() as u8,
                        bits,
                        self.skandha_pipeline.vedana.confidence(),
                    )),
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

    // Helper: Integrate Thermodynamics with Entropy-Based Mode Switching
    //
    // VAJRA V5: Life Systems Integration
    // - Dissipative mode (high entropy): Exploration, memory decay
    // - Conservative mode (low entropy): Deep learning, stability
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
                *self.skandha_pipeline.vedana.probabilities()
            };

            // Apply thermodynamic step - system smoothly evolves toward target
            // This provides temporal smoothing and prevents abrupt state transitions
            self.thermo_step(&target, 1);

            // === VAJRA V5: Entropy-Based Mode Switching ===
            let state =
                nalgebra::DVector::from_vec(self.skandha_pipeline.vedana.probabilities().to_vec());
            let entropy = self.thermo_engine.entropy(&state);
            
            // Get entropy thresholds from config or use defaults
            let entropy_high = self.config.features.entropy_high_threshold.unwrap_or(1.5);
            let entropy_low = self.config.features.entropy_low_threshold.unwrap_or(0.5);

            // High entropy (>1.5): Dissipative mode - trigger memory decay for forgetting
            if entropy > entropy_high {
                // Apply memory decay to prevent memory overload
                // Decay rate scales with entropy excess
                let decay_rate = 0.99 - (entropy - entropy_high) * 0.01;
                let decay_rate = decay_rate.clamp(0.95, 0.999);
                self.skandha_pipeline.sanna.memory.decay(decay_rate);
                
                // Increase temperature for more exploration
                let current_temp = self.thermo_engine.config().temperature;
                if current_temp < 2.0 {
                    self.thermo_engine.set_temperature((current_temp * 1.1).min(2.0));
                }
                
                log::debug!(
                    "Thermo: DISSIPATIVE mode (entropy={:.3}), decay_rate={:.4}",
                    entropy, decay_rate
                );
            }
            // Low entropy (<0.5): Conservative mode - deep learning
            else if entropy < entropy_low {
                // Decrease temperature for more exploitation
                let current_temp = self.thermo_engine.config().temperature;
                if current_temp > 0.1 {
                    self.thermo_engine.set_temperature((current_temp * 0.9).max(0.1));
                }
                
                // Boost learning rate for belief updates (done via EFE precision)
                // Higher precision = more exploitation of known good states
                self.efe_precision_beta = (self.efe_precision_beta * 1.05).min(10.0);
                
                log::debug!(
                    "Thermo: CONSERVATIVE mode (entropy={:.3}), beta={:.2}",
                    entropy, self.efe_precision_beta
                );
            }
            // Normal entropy: Balanced mode
            else {
                // Gradually restore temperature to default
                let default_temp = self.config.features.thermo_temperature.unwrap_or(1.0);
                let current_temp = self.thermo_engine.config().temperature;
                let new_temp = current_temp + (default_temp - current_temp) * 0.1;
                self.thermo_engine.set_temperature(new_temp);
            }
        }
    }

    // =========================================================================
    // NEURAL WIRING: Saccade Memory Integration
    // =========================================================================
    
    /// Integrate saccade memory linker for memory-driven context modulation.
    ///
    /// Uses the current perception context (from Sheaf/Skandha) to predict
    /// where to look in holographic memory. High-confidence predictions
    /// can modulate belief updates for temporal consistency.
    ///
    /// # Algorithm
    /// 1. Extract context from skandha_state.form.values
    /// 2. Call saccade.recall_fast() to predict memory coordinates
    /// 3. If prediction succeeds, log for diagnostics
    /// 4. Optionally use prediction to modulate belief (future enhancement)
    fn integrate_saccade_recall(&mut self) {
        // Only proceed if we have a valid skandha state
        let Some(ref skandha_state) = self.skandha_state else {
            return;
        };

        // Convert form values to saccade context (5-dim)
        let context: Vec<f32> = skandha_state.form.values.to_vec();
        
        // Get dt from last tick (default to 0.016 = 60fps if not available)
        let dt = 0.016f32;
        
        // Recall prediction from saccade linker using dedicated holographic memory
        let recalled = self.saccade.recall_fast(
            &context,
            &self.saccade_memory,
            dt,
        );
        
        // If memory recall succeeded
        if let Some(recalled_pattern) = recalled {
            // Compute energy of recalled pattern as confidence measure
            let energy: f32 = recalled_pattern.iter()
                .take(10)
                .map(|c| c.norm_sqr())
                .sum::<f32>()
                .sqrt();
            
            if energy > 0.1 {
                log::debug!(
                    "Saccade: recall energy={:.3}, pattern_len={}",
                    energy,
                    recalled_pattern.len()
                );
            }
        }
        
        // Learn from current observation for future predictions
        // Use belief probabilities as the "actual coordinates" to learn
        let actual_coords = self.skandha_pipeline.vedana.probabilities();
        self.saccade.learn_correction(&context, actual_coords.as_slice());
    }
    
    // =========================================================================
    // VAJRA-VOID: FEP Prediction Loop
    // =========================================================================
    
    /// Integrate FEP prediction loop: compare predicted vs actual context.
    /// 
    /// # Algorithm
    /// 1. Compare last prediction with current skandha_state
    /// 2. Compute prediction error (Euclidean distance)
    /// 3. Update prediction_error_ema
    /// 4. If error > surprise_threshold: trigger rapid learning
    /// 5. Store current state as prediction for next tick
    fn integrate_fep_prediction_loop(&mut self) {
        // Only proceed if we have a valid skandha state
        let Some(ref skandha_state) = self.skandha_state else {
            return;
        };
        
        let actual_context = &skandha_state.form.values;
        
        // Compare with last prediction if available
        if let Some(ref predicted) = self.last_predicted_context {
            let prediction_error = Self::compute_prediction_error(predicted, actual_context);
            
            // Update EMA of prediction error (α = 0.1 for smooth tracking)
            const PREDICTION_EMA_ALPHA: f32 = 0.1;
            self.prediction_error_ema = (1.0 - PREDICTION_EMA_ALPHA) * self.prediction_error_ema
                + PREDICTION_EMA_ALPHA * prediction_error;
            
            // Get surprise threshold from config or use default
            let surprise_threshold = self.config.fep.surprise_threshold.unwrap_or(0.3);
            
            // High surprise triggers FEP actions
            if prediction_error > surprise_threshold {
                log::info!(
                    "FEP SURPRISE: error={:.3} > threshold={:.3}, triggering rapid learning",
                    prediction_error, surprise_threshold
                );
                
                // 1. Force causal graph update (high plasticity mode)
                self.causal.force_discovery();
                
                // 2. Reduce decision confidence (model is uncertain)
                self.decision_confidence.decay(0.8);
                
                // 3. Increase EFE precision beta (more exploitation of known-good states)
                self.efe_precision_beta = (self.efe_precision_beta * 0.9).max(0.5);
            }
        }
        
        // Store current state as prediction for next tick
        // Simple prediction: assume smooth continuation (persistence model)
        // Future: use causal graph or LTC network for more sophisticated prediction
        self.last_predicted_context = Some(actual_context.to_vec());
    }
    
    /// Compute prediction error as Euclidean distance between predicted and actual.
    fn compute_prediction_error(predicted: &[f32], actual: &[f32; 5]) -> f32 {
        predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Predict next context (simple persistence model).
    /// 
    /// Future enhancement: Use causal graph for intervention prediction.
    #[allow(dead_code)]
    fn predict_next_context(&self) -> Vec<f32> {
        if let Some(ref state) = self.skandha_state {
            state.form.values.to_vec()
        } else {
            vec![0.5; 5] // Neutral prediction
        }
    }
    
    // =========================================================================
    // VAJRA-VOID: Data Reincarnation
    // =========================================================================
    
    /// Integrate data reincarnation: feed Vinnana output back as internal sensation.
    /// 
    /// # V3-1 Spec
    /// "Nối output Vinnana (Thức) quay lại làm input Rupa (Sắc) cho vòng lặp sau"
    /// 
    /// This creates a feedback loop where the synthesized belief state
    /// influences the next perception cycle through the causal system.
    fn integrate_data_reincarnation(&mut self, ts_us: i64) {
        // Only proceed if we have a valid synthesized state
        let Some(ref final_state) = self.skandha_state else {
            return;
        };
        
        // Don't reincarnate if confidence is too low (unreliable synthesis)
        if final_state.confidence < 0.3 {
            return;
        }
        
        // Convert "Thức" (consciousness) to "Sắc" (form) - internal sensation
        // Map belief states to virtual physiological signals
        let stress_belief = final_state.belief[1]; // Stress mode probability
        let calm_belief = final_state.belief[0];   // Calm mode probability
        
        // Virtual HR: higher stress -> higher HR
        let virtual_hr = 60.0 + stress_belief * 40.0 - calm_belief * 10.0;
        // Virtual HRV: inverse of stress
        let virtual_hrv = (1.0 - stress_belief) * 60.0 + 20.0;
        
        let internal_features = [
            virtual_hr / 200.0,      // Normalized HR
            virtual_hrv / 100.0,     // Normalized HRV
            final_state.form.values[2], // Keep actual RR
            final_state.confidence,  // Use confidence as quality
            0.0,                     // Zero motion (internal state)
        ];
        
        // Feed to causal subsystem as internal observation
        // This allows the causal graph to learn from self-generated predictions
        self.causal.observe_internal(internal_features, ts_us);
    }
    
    /// Get current prediction error EMA (for diagnostics).
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error_ema
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
