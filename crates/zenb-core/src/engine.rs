use crate::breath_engine::BreathEngine;
use crate::causal::{CausalBuffer, CausalGraph, ObservationSnapshot};
use crate::config::ZenbConfig;
use crate::controller::AdaptiveController;
use crate::domain::ControlDecision;
use crate::estimator::Estimator;
use crate::estimator::Estimate;
use crate::estimators::{UkfStateEstimator, Observation as UkfObservation};
use crate::resonance::ResonanceTracker;
use crate::safety::{SafetyMonitor, RuntimeState, DharmaFilter};
use crate::safety_swarm::TraumaRegistry;
use crate::trauma_cache::TraumaCache;

// VAJRA-001: New cognitive architecture imports
use crate::memory::HolographicMemory;
use crate::perception::SheafPerception;
use nalgebra::DVector;

// PANDORA PORT: Resilience and adaptive features
use crate::circuit_breaker::{CircuitBreakerManager, CircuitBreakerConfig};
use crate::adaptive::{AdaptiveThreshold, AnomalyDetector, ConfidenceTracker};

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
    /// @deprecated Use `holographic_memory` instead. Kept for backward compatibility.
    pub causal_buffer: CausalBuffer,
    pub last_observation: Option<crate::domain::Observation>,
    pub trauma_registry: TraumaRegistry,
    pub trauma_cache: TraumaCache,
    pub safety_monitor: SafetyMonitor,
    /// Session start time for tracking duration
    pub session_start_ts_us: Option<i64>,
    
    // --- EFE / META-LEARNING ---
    
    /// Current EFE precision (beta) for policy selection
    pub efe_precision_beta: f32,
    
    /// Meta-learner for adapting beta
    pub efe_meta_learner: crate::policy::BetaMetaLearner,
    
    /// Feature flag to enable EFE-based policy selection
    pub use_efe_selection: bool,
    
    /// Last chosen policy info (for learning)
    pub last_selected_policy: Option<crate::policy::PolicyEvaluation>,
    
    // === SOTA Features ===
    pub legacy_estimator: crate::estimator::Estimator,
    pub ukf_estimator: Option<UkfStateEstimator>,
    pub last_aukf_telemetry: Option<serde_json::Value>,
    pub last_ingest_ts_us: Option<i64>,
    
    // === PC Algorithm ===
    pub observation_buffer: Vec<ObservationSnapshot>,
    pub pc_change_detector: crate::causal::GraphChangeDetector,
    pub pc_learning_enabled: bool,
    pub last_pc_run_ts: i64,
    
    // === VAJRA-001: Holographic Cognitive Architecture ===
    
    /// Holographic Memory (Tưởng Uẩn) - FFT-based associative memory
    /// Replaces causal_buffer for new code paths
    pub holographic_memory: HolographicMemory,
    
    /// Sheaf Perception (Sắc Uẩn) - Laplacian sensor consensus
    pub sheaf_perception: SheafPerception,
    
    /// Dharma Filter (Hành Uẩn) - Phase-based ethical action filtering
    pub dharma_filter: DharmaFilter,
    
    /// Feature flag to enable Vajra-001 architecture
    pub use_vajra_architecture: bool,
    
    /// Last sheaf energy (for diagnostics)
    pub last_sheaf_energy: f32,
    
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
}

impl Engine {
    pub fn new(default_bpm: f32) -> Self {
        Self::new_with_config(default_bpm, None)
    }

    /// Test-only constructor: allows zero/negative timestamps for guards that enforce time sanity.
    pub fn new_for_test(default_bpm: f32) -> Self {
        let mut cfg = ZenbConfig::default();
        cfg.safety.allow_test_time = true;
        Self::new_with_config(default_bpm, Some(cfg))
    }

    pub fn new_with_config(starting_arousal: f32, config: Option<ZenbConfig>) -> Self {
        let mut cfg = config.unwrap_or_default();
        #[cfg(test)]
        { 
            cfg.safety.allow_test_time = true; 
        }
        let controller = AdaptiveController::new(Default::default());
        
        // SOTA: Initialize UKF if enabled
        let ukf_estimator = if cfg.sota.use_ukf {
            Some(UkfStateEstimator::new_adaptive(Some(cfg.sota.ukf_config.clone())))
        } else {
            None
        };
        
        let mut breath = BreathEngine::new(crate::breath_engine::BreathMode::Dynamic(starting_arousal)); // Assuming starting_arousal can be used as initial BPM
        let target_bpm = if starting_arousal > 0.0 {
            starting_arousal
        } else {
            cfg.breath.default_target_bpm
        };
        breath.set_target_bpm(target_bpm);
        Self {
            legacy_estimator: Estimator::default(),
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
            trauma_cache: TraumaCache::default(),
            safety_monitor: SafetyMonitor::new(),
            session_start_ts_us: None,
            // EFE Initialization
            efe_precision_beta: cfg.sota.efe_precision_beta.unwrap_or(4.0),
            efe_meta_learner: crate::policy::BetaMetaLearner::default(),
            use_efe_selection: cfg.sota.use_efe_selection,
            last_selected_policy: None,
            // UKF Initialization
            ukf_estimator,
            last_aukf_telemetry: None,
            last_ingest_ts_us: None,
            
            // PC Init
            observation_buffer: Vec::with_capacity(100),
            pc_change_detector: crate::causal::GraphChangeDetector::default(),
            pc_learning_enabled: cfg.sota.pc_learning_enabled,
            last_pc_run_ts: 0,
            
            // VAJRA-001: Holographic Cognitive Architecture
            holographic_memory: HolographicMemory::default_for_zenb(),
            sheaf_perception: SheafPerception::default_for_zenb(),
            dharma_filter: DharmaFilter::default_for_zenb(),
            use_vajra_architecture: true, // ENABLED: All Vajra components verified and tested
            last_sheaf_energy: 0.0,
            
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
                4.0,  // base: 4.0 bpm (typical minimum)
                3.0,  // min: allow down to 3.0 for deep meditation/athletes
                6.0,  // max: don't go above 6.0 as minimum
                0.1,  // learning rate: adapt faster for physiological bounds
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
    /// SheafPerception for consensus filtering before further processing.
    pub fn ingest_sensor(&mut self, features: &[f32], ts_us: i64) -> Estimate {
        // VAJRA-001: Sheaf Perception - filter inconsistent sensors
        let processed_features: Vec<f32> = if self.use_vajra_architecture && features.len() >= 3 {
            // Pad to 5 sensors if needed
            let mut padded = vec![0.0f32; 5];
            for (i, &f) in features.iter().take(5).enumerate() {
                padded[i] = f;
            }
            // Default quality and motion if not provided
            if features.len() < 4 { padded[3] = 1.0; } // quality
            if features.len() < 5 { padded[4] = 0.0; } // motion

            let sensor_vec = DVector::from_vec(padded.clone());

            // RESILIENCE: Circuit breaker protection for sheaf perception
            if !self.circuit_breaker.is_open("sheaf_perception") {
                // Try sheaf processing with panic catch (Laplacian can fail on degenerate graphs)
                let sheaf_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.sheaf_perception.process(&sensor_vec)
                }));

                match sheaf_result {
                    Ok((diffused, is_anomalous, energy)) => {
                        // Success: record and use sheaf result
                        self.circuit_breaker.record_success("sheaf_perception");

                        // Store energy for diagnostics
                        self.last_sheaf_energy = energy;

                        if is_anomalous {
                            log::warn!(
                                "SheafPerception: Anomalous sensor data detected (energy={:.3})",
                                energy
                            );
                        }

                        // Use diffused values
                        diffused.iter().cloned().collect()
                    }
                    Err(e) => {
                        // Sheaf panicked (Laplacian singularity, numerical issues, etc.)
                        self.circuit_breaker.record_failure("sheaf_perception");
                        log::error!("SheafPerception panicked: {:?}, using raw sensor fallback", e);
                        // Fallback: use raw padded sensor values
                        padded
                    }
                }
            } else {
                // Circuit is open: skip sheaf, use raw sensors
                log::warn!("SheafPerception circuit open, using raw sensor fallback");
                padded
            }
        } else {
            features.to_vec()
        };
        
        let features = &processed_features;
        
        // Legacy ingest for backward compatibility / fallback
        // We always run this to maintain estimator state even if UKF is primary
        let est_legacy = self.legacy_estimator.ingest(features, ts_us);
        
        let mut final_est = est_legacy.clone();
        
        // HYBRID UKF LOGIC
        if let Some(ukf) = &mut self.ukf_estimator {
            // Compute dt (ensure non-negative, handle first sample)
            let dt = if let Some(last_ts) = self.last_ingest_ts_us {
                crate::domain::dt_sec(ts_us, last_ts)
            } else {
                0.0
            };
            
            // Avoid dt=0 updates (bursts)
            if dt > 0.001 { // 1ms minimum
                 let obs = UkfObservation {
                    heart_rate: features.get(0).copied(),
                    hr_confidence: Some(features.get(3).copied().unwrap_or(1.0)),
                    stress_index: features.get(1).map(|v| 100.0 - v), // Inverse HRV as stress proxy? Approx
                    respiration_rate: features.get(2).copied(),
                    facial_valence: None, // Need separate input for this
                };

                // RESILIENCE: Circuit breaker protection for UKF numerical instability
                if !self.circuit_breaker.is_open("ukf_update") {
                    // Try update with panic catch (UKF can panic on matrix singularity)
                    let ukf_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        ukf.update(&obs, dt)
                    }));

                    match ukf_result {
                        Ok(belief) => {
                            // Success: record and use UKF result
                            self.circuit_breaker.record_success("ukf_update");

                            // Convert to Estimate
                            final_est = Estimate {
                                ts_us, // Use current timestamp
                                hr_bpm: Some(belief.arousal * 100.0 + 40.0), // Denormalize approx
                                rr_bpm: Some(belief.rhythm_alignment * 10.0 + 4.0),
                                rmssd: Some((1.0 - belief.arousal) * 100.0),
                                confidence: belief.confidence,
                            };
                        }
                        Err(e) => {
                            // UKF panicked (matrix singularity, numerical instability, etc.)
                            self.circuit_breaker.record_failure("ukf_update");
                            log::error!("UKF update panicked: {:?}, falling back to legacy estimator", e);
                            // final_est already set to est_legacy, no change needed
                        }
                    }
                } else {
                    // Circuit is open: skip UKF, use legacy fallback
                    log::warn!("UKF circuit open, using legacy estimator fallback");
                    // final_est already set to est_legacy
                }
                
                // TELEMETRY: Capture adaptive Q if needed
                if ukf.sample_count() % 10 == 0 {
                    let q_diag = ukf.get_q_diagonal();
                    self.last_aukf_telemetry = Some(serde_json::json!({
                        "aukf_q_diagonal": q_diag,
                        "aukf_sample_count": ukf.sample_count(),
                        "aukf_forgetting_factor": self.config.sota.ukf_config.forgetting_factor,
                    }));
                } else {
                    self.last_aukf_telemetry = None;
                }
            }
        }
        self.last_ingest_ts_us = Some(ts_us);

        // Build SensorFeatures and PhysioState...
        debug_assert!(
            features.len() >= 3,
            "features layout must be [hr, rmssd, rr, quality, motion] (quality/motion optional)"
        );
        let sf = crate::belief::SensorFeatures {
            hr_bpm: final_est.hr_bpm,
            rmssd: final_est.rmssd,
            rr_bpm: final_est.rr_bpm,
            quality: features.get(3).cloned().unwrap_or(1.0),
            motion: features.get(4).cloned().unwrap_or(0.0),
        };
        let phys = crate::belief::PhysioState {
            hr_bpm: final_est.hr_bpm,
            rr_bpm: final_est.rr_bpm,
            rmssd: final_est.rmssd,
            confidence: final_est.confidence,
        };
        self.last_sf = Some(sf);
        self.last_phys = Some(phys);
        final_est
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
            self.causal_buffer.push(snapshot);
            
            // VAJRA-001: Encode into Holographic Memory
            if self.use_vajra_architecture {
                // Create context key from belief state
                let key = crate::memory::hologram::encode_context_key(
                    &self.belief_state.p[..],
                    self.holographic_memory.dim(),
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
                    self.holographic_memory.dim(),
                );
                
                // Entangle (store) the association
                self.holographic_memory.entangle(&key, &value);
                
                // Apply decay (forgetting) - 0.999 = very slow decay
                self.holographic_memory.decay(0.999);
            }
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
        if let Err(e) = self.causal_graph
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
            let dt_sec = match self.last_control_ts_us {
                Some(last) => crate::domain::dt_sec(ts_us, last),
                None => 0.0,
            };
            self.last_control_ts_us = Some(ts_us);

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
        if self.use_efe_selection {
            use crate::policy::{EFECalculator, ActionPolicy, PolicyLibrary};
            
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
            let mut evaluations: Vec<crate::policy::PolicyEvaluation> = policies.into_iter().map(|policy| {
                efe_calc.compute_efe(
                    &policy,
                    &self.belief_state.to_5mode_array(),
                    self.belief_state.uncertainty(),
                    &predicted_state,
                    crate::belief::uncertainty_from_confidence(predicted_uncertainty),
                )
            }).collect();
            
            // 5. Select Policy (Softmax)
            // Use deterministic RNG seed from timestamp for reproducibility
            let rng_value = ((ts_us % 1000) as f32) / 1000.0;
            efe_calc.compute_selection_probabilities(&mut evaluations);
            let selected_policy_ref = efe_calc.sample_policy(&evaluations, rng_value);
            
            // 6. Apply Selection
            // Store for learning loop
            // We need to clone it to store it (sample_policy returns ref)
            let selected_clone = evaluations.iter().find(|e| e.policy.description() == selected_policy_ref.description()).cloned();
            self.last_selected_policy = selected_clone;
            
            // Convert to BPM target
            // Use current proposed as fallback
            let decision = selected_policy_ref.to_control_decision(proposed, est.confidence);
            proposed = decision.target_rate_bpm;
            
            // Handle Side Effects (e.g. Digital Interventions)
            if let ActionPolicy::DigitalIntervention(di) = selected_policy_ref {
                 // In a full implementation, this would emit a separate event or side-effect
                 // For now, we just log it as the primary decision driver
                 // (Engine currently focused on Breath, but this lays groundwork for multi-modal)
                 log::info!("EFE Selected Digital Intervention: {:?}", di);
            }
        }


        // VAJRA-001: Dharma Filter - phase-based ethical action filtering
        if self.use_vajra_architecture {
            use crate::safety::ComplexDecision;
            
            // Convert proposed BPM to complex decision
            let baseline_bpm = 6.0;
            let action = ComplexDecision::from_bpm_target(proposed, baseline_bpm);
            
            // Apply Dharma filter
            match action.filter_with(&self.dharma_filter) {
                Some(sanctioned) => {
                    let new_proposed = sanctioned.to_bpm(baseline_bpm);
                    if (new_proposed - proposed).abs() > 0.1 {
                        log::info!(
                            "DharmaFilter: Scaled action from {:.2} to {:.2} BPM (alignment={:.3})",
                            proposed,
                            new_proposed,
                            self.dharma_filter.check_alignment(action.vector)
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
        let mut guards: Vec<Box<dyn crate::safety_swarm::Guard + '_>> = Vec::new();
        guards.push(Box::new(crate::safety_swarm::TraumaGuard {
            source: &self.trauma_cache,
            hard_th: self.config.safety.trauma_hard_th,
            soft_th: self.config.safety.trauma_soft_th,
            allow_test_time: self.config.safety.allow_test_time,
        }));
        // Fix: Use controller state for rate limiting (stateless guard needs history)
        let last_patch_sec = self
            .controller
            .last_decision_ts_us
            .map(|last| crate::domain::dt_sec(ts_us, last));

        guards.push(Box::new(crate::safety_swarm::ConfidenceGuard { min_conf: 0.2 }));
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

        let decide = crate::safety_swarm::decide(
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
        );

        // CRITICAL: Causal Veto Check (happens AFTER safety guards but BEFORE final decision)
        // This prevents actions that historically fail in this context
        const MIN_SUCCESS_PROB: f32 = 0.3;
        const HIGH_SUCCESS_PROB: f32 = 0.8;

        // LTL Safety Monitor Check
        // Verify tempo bounds, panic halt, and safety lock invariants
        let session_duration = match self.session_start_ts_us {
            Some(start) => crate::domain::dt_sec(ts_us, start),
            None => {
                // Initialize session start time on first control decision
                self.session_start_ts_us = Some(ts_us);
                0.0
            }
        };
        
        let runtime_state = RuntimeState {
            tempo_scale: proposed / 6.0, // Normalize to baseline 6 BPM
            status: "RUNNING".to_string(),
            session_duration,
            prediction_error: self.fep_state.free_energy_ema,
            last_update_timestamp: ts_us as u64,
        };
        
        if let Err(violations) = self.safety_monitor.check(&runtime_state) {
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
            let safe_tempo = self.safety_monitor.shield_tempo(proposed / 6.0) * 6.0;
            
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

