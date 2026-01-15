use crate::breath_engine::BreathEngine;
use crate::config::ZenbConfig;
use crate::controller::AdaptiveController;

use crate::estimator::Estimate;
use crate::resonance::ResonanceTracker;


// Phase 2 Decomposition: Extracted subsystems

use crate::universal_flow::FlowEventId;

use crate::safety_subsystem::SafetySubsystem;
use crate::vinnana_controller::VinnanaController;

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
    // PHASE 3: Removed last_sf and last_phys - redundant with skandha_state.form
    pub resonance_tracker: ResonanceTracker,
    pub last_resonance_score: f32,
    pub resonance_score_ema: f32,
    pub free_energy_peak: f32,
    pub last_pattern_id: i64,
    pub last_goal: i64,

    // === Causal Subsystem (Phase 6) ===
    pub causal: crate::causal_subsystem::CausalSubsystem,

    // === VINNANA CONTROLLER (Thức Uẩn) ===
    /// Supreme Consciousness Controller - orchestrates all Skandha stages
    /// Replaces: skandha_pipeline, skandha_state, thermo_engine, saccade,
    /// saccade_memory, last_predicted_context, prediction_error_ema, philosophical_state
    pub vinnana: VinnanaController,

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

    // === Flow Integration: Event Lineage ===
    /// Current FlowEventId being processed (for traceability)
    /// Set by Runtime before calling ingest methods
    pub current_flow_id: Option<FlowEventId>,
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
            // PHASE 3: Removed last_sf and last_phys initialization
            resonance_tracker: ResonanceTracker::default(),
            last_resonance_score: 1.0,
            resonance_score_ema: 1.0,

            free_energy_peak: 0.0,
            last_pattern_id: 0,
            last_goal: 0,

            // Causal Subsystem
            causal: crate::causal_subsystem::CausalSubsystem::new(&cfg),

            // VINNANA CONTROLLER (Thức Uẩn)
            // Encapsulates: skandha_pipeline, skandha_state, thermo_engine,
            // saccade, saccade_memory, prediction_error_ema, philosophical_state
            vinnana: VinnanaController::new(
                crate::skandha::zenb::zenb_pipeline_unified(&cfg)
            ),

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

            // Flow Integration: Event Lineage
            current_flow_id: None,
        }
    }

    pub fn update_config(&mut self, cfg: ZenbConfig) {
        self.config = cfg;
    }

    /// Update runtime context (call from App layer with current local hour / charging / session info)
    pub fn update_context(&mut self, ctx: crate::belief::Context) {
        self.context = ctx;
    }
    
    /// Set the current FlowEventId for event lineage tracing.
    /// Called by Runtime before ingesting sensor data.
    pub fn set_current_flow_id(&mut self, id: Option<FlowEventId>) {
        self.current_flow_id = id;
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
            self.vinnana.pipeline.process(&input)
        }));

        match result {
            Ok(synthesized) => {
                self.circuit_breaker.record_success("skandha_pipeline");

                // Store state for decision loop (in VinnanaController)
                self.vinnana.last_state = Some(synthesized.clone());

                // Extract values from trusted Rupa form
                let form_values = &synthesized.form.values;
                let estimate = Estimate {
                    ts_us,
                    hr_bpm: Some(form_values[0] * 200.0), // Denormalize
                    rmssd: Some(form_values[1] * 100.0),
                    rr_bpm: Some(form_values[2] * 20.0),
                    confidence: synthesized.confidence,
                };

                // PHASE 3: Removed last_sf and last_phys assignments
                // Data available in skandha_state.form and estimate

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

        // === B.ONE V3: UPDATE PHILOSOPHICAL STATE (YÊNA/ĐỘNG/HỖN LOẠN) ===
        // Compute free energy from prediction error and coherence from Skandha confidence
        let free_energy = self.vinnana.prediction_error_ema;
        let coherence = self.vinnana.last_state.as_ref()
            .map(|s| s.confidence)
            .unwrap_or(0.8); // Default: moderate coherence
        
        let phil_state = self.vinnana.philosophical_state.update_with_timestamp(
            free_energy, 
            coherence, 
            dt_us as i64
        );
        
        // Apply processing config based on philosophical state
        let config = self.vinnana.philosophical_state.get_processing_config();
        self.vinnana.pipeline.config.refinement_enabled = config.refinement_enabled;
        
        // Log state transitions
        if self.vinnana.philosophical_state.just_transitioned() {
            log::info!(
                "B.ONE V3: Consciousness state -> {} (FE={:.3}, C={:.3})",
                phil_state.vietnamese_name(),
                free_energy,
                coherence
            );
        }

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

        // VAJRA-VOID Task 2.2: Feature-gated Krylov stability/energy monitor
        // Detects Hologram memory explosion or NaN corruption (Lanczos instability)
        #[cfg(feature = "vajra_debug")]
        {
            let energy = self.holographic_memory.energy();
            if energy.is_nan() || energy > 1e6 {
                log::error!(
                    "VAJRA PANIC: Hologram energy explosion detected! Energy={}, resetting memory.",
                    energy
                );
                self.holographic_memory.clear();
            }
        }

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
            self.vinnana.pipeline.vedana.confidence(),
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
        self.vinnana.pipeline
            .vedana
            .process_feedback(success, &mut self.config.fep);

        // Update causal graph weights based on outcome
        let context_state = self.causal.graph.extract_state_values(
            self.vinnana.pipeline.vedana.state(),
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
                self.vinnana.pipeline.vedana.mode() as u8,
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
        
        // PANDORA PORT: Update confidence tracker based on outcome
        self.decision_confidence.update(success);

        // PANDORA PORT: Adapt belief threshold based on performance
        let performance_delta = if success { 0.1 } else { -0.1 };
        self.vinnana.pipeline
            .vedana
            .adapt_threshold(performance_delta);

        // === UNIFIED SANKHARA LEARNING ===
        // Delegate EFE meta-learning and policy adaptation to Sankhara
        // (Legacy path - uses mode hash instead of IntentId)
        let reward = if success { 1.0 } else { -severity };
        let state_hash = format!("{:?}", self.vinnana.pipeline.vedana.mode());
        
        self.vinnana.pipeline.sankhara.apply_feedback(success, reward, &state_hash);
    }
    
    /// Learn from action outcome with IntentId traceability (V2 API).
    /// 
    /// This is the preferred learning API for the V3 Karmic Feedback Loop.
    /// It uses IntentId to retrieve the exact decision context from IntentTracker.
    /// 
    /// # Arguments
    /// * `intent_id` - The intent ID returned with the original ControlDecision
    /// * `success` - Whether the action led to positive outcome
    /// * `action_type` - String identifier for the action taken (for trauma)
    /// * `ts_us` - Timestamp of the outcome
    /// * `severity` - Severity of negative outcome (0.0-1.0)
    pub fn learn_from_outcome_v2(
        &mut self,
        intent_id: u64,
        success: bool,
        action_type: String,
        ts_us: i64,
        severity: f32,
    ) {
        // Update belief engine (Active Inference learning)
        self.vinnana.pipeline
            .vedana
            .process_feedback(success, &mut self.config.fep);

        // Update causal graph weights based on outcome
        let context_state = self.causal.graph.extract_state_values(
            self.vinnana.pipeline.vedana.state(),
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
            let context_hash = crate::safety_swarm::trauma_sig_hash(
                self.last_goal,
                self.vinnana.pipeline.vedana.mode() as u8,
                self.last_pattern_id,
                &self.context,
            );
            self.safety.trauma_registry_mut().record_negative_feedback(
                context_hash,
                action_type,
                ts_us,
                severity,
            );
            if let Some(hit) = self.safety.trauma_registry().query(&context_hash) {
                self.safety.trauma_cache_mut().update(context_hash, hit);
            }
        }
        
        // PANDORA PORT: Update confidence tracker
        self.decision_confidence.update(success);
        let performance_delta = if success { 0.1 } else { -0.1 };
        self.vinnana.pipeline.vedana.adapt_threshold(performance_delta);

        // === V2: KARMIC FEEDBACK WITH INTENT TRACING ===
        let intent_id = crate::skandha::sankhara::IntentId::from_raw(intent_id);
        self.vinnana.pipeline.sankhara.apply_feedback_v2(intent_id, success, severity, ts_us);
    }

    /// Get circuit breaker statistics for monitoring.
    pub fn circuit_breaker_stats(&self) -> crate::circuit_breaker::CircuitStats {
        self.circuit_breaker.stats()
    }

    /// Get current adaptive threshold values for diagnostics.
    pub fn adaptive_thresholds_info(&self) -> (f32, f32, f32) {
        (
            self.vinnana.pipeline.vedana.enter_threshold(),
            self.vinnana.pipeline.vedana.enter_threshold_base(),
            self.decision_confidence.success_rate(),
        )
    }

    /// Get current physiological context from Sheaf perception.
    ///
    /// Returns the auto-detected or manually set context that determines
    /// the sheaf's anomaly threshold and diffusion rate.
    pub fn sheaf_context(&self) -> crate::perception::PhysiologicalContext {
        self.vinnana.pipeline.rupa.context()
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
        self.vinnana.pipeline.rupa.set_context(context);
    }

    /// Get sheaf diagnostics: (energy, context, is_adaptive_alpha_enabled)
    pub fn sheaf_diagnostics(&self) -> (f32, crate::perception::PhysiologicalContext, bool) {
        (
            self.last_sheaf_energy,
            self.vinnana.pipeline.rupa.context(),
            self.vinnana.pipeline.rupa.sheaf.is_adaptive_alpha_enabled(),
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
            return *self.vinnana.pipeline.vedana.probabilities();
        }

        // Convert belief state to DVector
        let state =
            nalgebra::DVector::from_vec(self.vinnana.pipeline.vedana.probabilities().to_vec());
        let target_vec = nalgebra::DVector::from_vec(target.to_vec());

        // Integrate using GENERIC dynamics
        let new_state = self.vinnana.thermo.integrate(&state, &target_vec, steps);

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
        self.vinnana.pipeline.vedana.set_probabilities(p);

        *self.vinnana.pipeline.vedana.probabilities()
    }

    /// Get thermodynamic diagnostics.
    ///
    /// # Returns
    /// (free_energy, entropy, temperature, enabled)
    pub fn thermo_info(&self) -> (f32, f32, f32, bool) {
        let state =
            nalgebra::DVector::from_vec(self.vinnana.pipeline.vedana.probabilities().to_vec());
        let target = nalgebra::DVector::from_vec([0.5f32; 5].to_vec()); // Neutral target for diagnostics

        let free_energy = self.vinnana.thermo.free_energy(&state, &target);
        let entropy = self.vinnana.thermo.entropy(&state);
        let temperature = self.vinnana.thermo.config().temperature;

        (
            free_energy,
            entropy,
            temperature,
            self.config.features.thermo_enabled.unwrap_or(false),
        )
    }



    pub fn make_control(
        &mut self,
        est: &crate::estimator::Estimate,
        ts_us: i64,
    ) -> (
        crate::domain::ControlDecision,
        bool,
        Option<(u8, u32, f32)>,
        Option<String>,
    ) {
        // Calculate Causal Probability
        let context_state = self.causal.graph.extract_state_values(
            self.vinnana.pipeline.vedana.state(),
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

        // Build GuardConfig
        let last_patch_sec = self
            .controller
            .last_decision_ts_us
            .map(|last| crate::domain::dt_sec(ts_us, last));

        let guard_config = crate::skandha::sankhara::GuardConfig {
            trauma_hard_th: self.config.safety.trauma_hard_th,
            trauma_soft_th: self.config.safety.trauma_soft_th,
            min_confidence: 0.2,
            rr_min: self.rr_min_threshold.get(),
            rr_max: self.rr_max_threshold.get(),
            hold_max_sec: 60.0,
            max_delta_rr_per_min: 6.0,
            min_interval_sec: 10.0,
            last_patch_sec,
        };

        // DELIBERATE (Unified Sankhara)
        let deliberation = self.vinnana.pipeline.sankhara.deliberate(
            est,
            self.vinnana.pipeline.vedana.state(),
            &self.context,
            &guard_config,
            ts_us,
            self.safety.trauma_cache(),
            Some(success_prob.value),
            self.last_pattern_id as i64,
            self.last_goal as i64,
        );

        // LTL Monitor Verification
        // Verify tempo bounds, panic halt, and safety lock invariants
        let proposed = deliberation.decision.target_rate_bpm;
        let session_duration = self.timestamp.session_duration(ts_us);

        // Required import if not present, but using implicit if available
        use crate::safety::RuntimeState;

        let runtime_state = RuntimeState {
            tempo_scale: proposed / 6.0,
            status: "RUNNING".to_string(),
            session_duration,
            prediction_error: self.vinnana.pipeline.vedana.free_energy_ema(),
            last_update_timestamp: ts_us as u64,
        };

        if let Err(violations) = self.safety.monitor_mut().check(&runtime_state) {
            let reason = format!("ltl_violation:{}", violations[0].property_name);
            eprintln!("ENGINE_DENY: LTL safety violation: {:?}", violations);

            // Calculate safe poll interval from Controller logic (Engine owns poll logic configuration)
            let poll_interval = crate::controller::compute_poll_interval(
                &mut self.controller.poller,
                self.vinnana.pipeline.vedana.free_energy_ema(),
                self.vinnana.pipeline.vedana.confidence(),
                false,
                &self.context,
            );

            // Shield tempo to safe bounds
            let safe_tempo = self.safety.monitor().shield_tempo(proposed / 6.0) * 6.0;

            return (
                crate::domain::ControlDecision {
                    target_rate_bpm: safe_tempo,
                    confidence: est.confidence * 0.3, 
                    recommended_poll_interval_ms: poll_interval,
                    intent_id: None, // LTL fallback has no intent
                },
                false, 
                Some((
                    self.vinnana.pipeline.vedana.mode() as u8,
                    deliberation.meta.guard_bits,
                    self.vinnana.pipeline.vedana.confidence(),
                )),
                Some(reason),
            );
        }

        // Apply Decision
        let final_bpm = deliberation.decision.target_rate_bpm;
        let changed = match self.controller.last_decision_bpm {
            Some(prev) => (prev - final_bpm).abs() > self.controller.cfg.decision_epsilon_bpm,
            None => true,
        } && match self.controller.last_decision_ts_us {
            Some(last_ts) => (ts_us - last_ts) >= self.controller.cfg.min_decision_interval_us,
            None => true,
        };
        
        if changed {
            self.controller.last_decision_bpm = Some(final_bpm);
            self.controller.last_decision_ts_us = Some(ts_us);
        }

        // B.ONE V3: Attach intent_id to decision for karmic traceability
        let mut final_decision = deliberation.decision.clone();
        final_decision.intent_id = Some(deliberation.intent_id.raw());

        (
            final_decision,
            changed,
            Some((
                self.vinnana.pipeline.vedana.mode() as u8,
                deliberation.meta.guard_bits,
                self.vinnana.pipeline.vedana.confidence(),
            )),
            deliberation.adjustment_reason,
        )
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
        let result = self.vinnana.pipeline.process(&input);

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
            let target = if let Some(ref skandha_state) = self.vinnana.last_state {
                // Use Skandha's synthesized belief as thermodynamic attractor
                skandha_state.belief
            } else {
                // Fallback: use current belief state (no drift)
                *self.vinnana.pipeline.vedana.probabilities()
            };

            // Apply thermodynamic step - system smoothly evolves toward target
            // This provides temporal smoothing and prevents abrupt state transitions
            self.thermo_step(&target, 1);

            // === VAJRA V5: Entropy-Based Mode Switching ===
            let state =
                nalgebra::DVector::from_vec(self.vinnana.pipeline.vedana.probabilities().to_vec());
            let entropy = self.vinnana.thermo.entropy(&state);
            
            // Get entropy thresholds from config (FIXED: raised defaults to 2.0/0.3)
            let entropy_high = self.config.features.entropy_high_threshold.unwrap_or(2.0);
            let entropy_low = self.config.features.entropy_low_threshold.unwrap_or(0.3);
            let memory_decay_base = self.config.features.memory_decay_base.unwrap_or(0.995);

            // High entropy (>2.0): Dissipative mode - trigger memory decay for forgetting
            // FIXED: Reduced aggressiveness - decay scales more gently
            if entropy > entropy_high {
                // Apply memory decay to prevent memory overload
                // FIXED: Use config base (0.995) and gentler scaling (0.005)
                let decay_rate = memory_decay_base - (entropy - entropy_high) * 0.005;
                let decay_rate = decay_rate.clamp(0.98, 0.9999);
                self.vinnana.pipeline.sanna.memory.decay(decay_rate);
                
                // Increase temperature for more exploration (gentler: 1.05x instead of 1.1x)
                let current_temp = self.vinnana.thermo.config().temperature;
                if current_temp < 2.0 {
                    self.vinnana.thermo.set_temperature((current_temp * 1.05).min(2.0));
                }
                
                log::debug!(
                    "Thermo: DISSIPATIVE mode (entropy={:.3}), decay_rate={:.4}",
                    entropy, decay_rate
                );
            }
            // Low entropy (<0.3): Conservative mode - deep learning
            else if entropy < entropy_low {
                // Decrease temperature for more exploitation
                let current_temp = self.vinnana.thermo.config().temperature;
                if current_temp > 0.1 {
                    self.vinnana.thermo.set_temperature((current_temp * 0.95).max(0.1));
                }
                
                // Boost learning rate for belief updates (done via EFE precision)
                // Higher precision = more exploitation of known good states
                self.vinnana.pipeline.sankhara.efe_precision_beta = (self.vinnana.pipeline.sankhara.efe_precision_beta * 1.02).min(10.0);
                
                log::debug!(
                    "Thermo: CONSERVATIVE mode (entropy={:.3}), beta={:.2}",
                    entropy, self.vinnana.pipeline.sankhara.efe_precision_beta
                );
            }
            // Normal entropy: Balanced mode
            else {
                // Gradually restore temperature to default
                let default_temp = self.config.features.thermo_temperature.unwrap_or(1.0);
                let current_temp = self.vinnana.thermo.config().temperature;
                let new_temp = current_temp + (default_temp - current_temp) * 0.1;
                self.vinnana.thermo.set_temperature(new_temp);
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
        let Some(ref skandha_state) = self.vinnana.last_state else {
            return;
        };

        // Convert form values to saccade context (5-dim)
        let context: Vec<f32> = skandha_state.form.values.to_vec();
        
        // Get dt from last tick (default to 0.016 = 60fps if not available)
        let dt = 0.016f32;
        
        // Recall prediction from saccade linker using dedicated holographic memory
        let recalled = self.vinnana.saccade.recall_fast(
            &context,
            &self.vinnana.saccade_memory,
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
        let actual_coords = self.vinnana.pipeline.vedana.probabilities();
        self.vinnana.saccade.learn_correction(&context, actual_coords.as_slice());
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
        let Some(ref skandha_state) = self.vinnana.last_state else {
            return;
        };
        
        let actual_context = &skandha_state.form.values;
        
        // Compare with last prediction if available
        if let Some(ref predicted) = self.vinnana.last_predicted_context {
            let prediction_error = Self::compute_prediction_error(predicted, actual_context);
            
            // Update EMA of prediction error (α = 0.1 for smooth tracking)
            const PREDICTION_EMA_ALPHA: f32 = 0.1;
            self.vinnana.prediction_error_ema = (1.0 - PREDICTION_EMA_ALPHA) * self.vinnana.prediction_error_ema
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
                self.vinnana.pipeline.sankhara.efe_precision_beta = (self.vinnana.pipeline.sankhara.efe_precision_beta * 0.9).max(0.5);
            }
        }
        
        // Store current state as prediction for next tick
        // Simple prediction: assume smooth continuation (persistence model)
        // Future: use causal graph or LTC network for more sophisticated prediction
        self.vinnana.last_predicted_context = Some(actual_context.to_vec());
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
        if let Some(ref state) = self.vinnana.last_state {
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
        let Some(ref final_state) = self.vinnana.last_state else {
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
        
        // VAJRA-VOID: Reduce weight of internal predictions to prevent self-reinforcement
        // (Đạo đức vật lý: AI không được tin suy nghĩ của mình hơn thực tế từ sensor)
        const REINCARNATION_WEIGHT: f32 = 0.1;
        
        let internal_features = [
            virtual_hr / 200.0 * REINCARNATION_WEIGHT,      // Normalized HR (weighted)
            virtual_hrv / 100.0 * REINCARNATION_WEIGHT,     // Normalized HRV (weighted)
            final_state.form.values[2] * REINCARNATION_WEIGHT, // RR (weighted)
            final_state.confidence * REINCARNATION_WEIGHT,  // Quality (weighted)
            0.0,                     // Zero motion (internal state)
        ];
        
        // Feed to causal subsystem as internal observation
        // This allows the causal graph to learn from self-generated predictions
        self.causal.observe_internal(internal_features, ts_us);
    }
    
    /// Get current prediction error EMA (for diagnostics).
    pub fn prediction_error(&self) -> f32 {
        self.vinnana.prediction_error_ema
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
        assert!(deny.is_none(), "Denied with reason: {:?}", deny);
    }
}
