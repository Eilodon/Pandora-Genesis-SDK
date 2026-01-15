//! Vinnana Controller - Thức Uẩn (Consciousness/Synthesis)
//!
//! Supreme Control layer that orchestrates the Ngũ Uẩn pipeline,
//! synthesizing all stages into unified consciousness state.
//!
//! # B.ONE V3 Concept
//! > "Thức Uẩn là Tổng Hợp Tri Thức và Tái Sinh - nhận thức cuối cùng
//! > của hệ thống được phát ngược trở lại làm Sắc mới cho vòng lặp sau."

use crate::domain::Observation;
use crate::memory::{HolographicMemory, SaccadeLinker};
use crate::philosophical_state::PhilosophicalStateMonitor;
use crate::skandha::{zenb::ZenbPipelineUnified, SensorInput, SynthesizedState};
use crate::thermo_logic::ThermodynamicEngine;
use nalgebra::DVector;

/// VinnanaController - Supreme Control of the Consciousness System
///
/// # Responsibilities
/// - Orchestrate Skandha Pipeline (Rupa → Vedana → Sanna → Sankhara → Vinnana)
/// - Integrate thermodynamic dynamics (GENERIC framework)
/// - Manage saccade memory predictions
/// - Execute FEP prediction loop (surprise detection)
/// - Enable data reincarnation (Vinnana → Rupa feedback)
#[derive(Debug)]
pub struct VinnanaController {
    /// Unified Skandha Pipeline (Sắc-Thọ-Tưởng-Hành-Thức)
    pub pipeline: ZenbPipelineUnified,

    /// Last synthesized state
    pub last_state: Option<SynthesizedState>,

    /// Philosophical State Monitor (YÊN/ĐỘNG/HỖN LOẠN)
    pub philosophical_state: PhilosophicalStateMonitor,

    /// Thermodynamic engine for GENERIC dynamics
    pub thermo: ThermodynamicEngine,

    /// Saccade linker for memory coordinate prediction
    pub saccade: SaccadeLinker,

    /// Holographic memory for saccade predictions
    pub saccade_memory: HolographicMemory,

    /// Last predicted context (for FEP surprise detection)
    pub last_predicted_context: Option<Vec<f32>>,

    /// Prediction error EMA
    pub prediction_error_ema: f32,
}

impl VinnanaController {
    /// Create with default configuration.
    pub fn new(pipeline: ZenbPipelineUnified) -> Self {
        Self {
            pipeline,
            last_state: None,
            philosophical_state: PhilosophicalStateMonitor::default(),
            thermo: ThermodynamicEngine::default(),
            saccade: SaccadeLinker::default_for_zenb(),
            saccade_memory: HolographicMemory::default(),
            last_predicted_context: None,
            prediction_error_ema: 0.0,
        }
    }

    /// Execute the full Skandha pipeline.
    ///
    /// This is the "Vinnana" orchestrator, processing inputs through
    /// all five aggregates to produce a synthesized consciousness state.
    pub fn synthesize(&mut self, obs: &Observation) -> SynthesizedState {
        // Map Observation to SensorInput
        let bio = obs.bio_metrics.as_ref();

        let input = SensorInput {
            hr_bpm: bio.and_then(|b| b.hr_bpm),
            hrv_rmssd: bio.and_then(|b| b.hrv_rmssd),
            rr_bpm: bio.and_then(|b| b.respiratory_rate),
            quality: 1.0,
            motion: 0.0,
            timestamp_us: obs.timestamp_us,
        };

        // Execute pipeline
        let result = self.pipeline.process(&input);

        // Update philosophical state based on free energy and confidence
        let coherence = result.confidence;
        self.philosophical_state.update(result.free_energy, coherence);

        // Store for next cycle
        self.last_state = Some(result.clone());

        log::debug!(
            "Vinnana Synthesis: Conf={:.2} Mode={} FE={:.2} PhiloState={:?}",
            result.confidence,
            result.mode,
            result.free_energy,
            self.philosophical_state.current_state
        );

        result
    }

    /// Integrate FEP prediction loop: surprise detection and rapid learning.
    ///
    /// Compares predicted context with actual state to compute prediction error.
    pub fn integrate_fep_loop(&mut self) {
        let Some(current) = &self.last_state else {
            return;
        };

        // Compare with last prediction
        if let Some(predicted) = &self.last_predicted_context {
            let actual = current.belief;
            let error = Self::compute_prediction_error(predicted, &actual);

            // Update EMA
            let alpha = 0.1;
            self.prediction_error_ema =
                alpha * error + (1.0 - alpha) * self.prediction_error_ema;

            // High surprise threshold
            if error > 0.5 {
                log::info!(
                    "FEP Surprise: error={:.3} (ema={:.3}) - triggering rapid learning",
                    error,
                    self.prediction_error_ema
                );
            }
        }

        // Store current as prediction for next tick
        self.last_predicted_context = Some(current.belief.to_vec());
    }

    /// Compute prediction error as Euclidean distance.
    fn compute_prediction_error(predicted: &[f32], actual: &[f32; 5]) -> f32 {
        let mut sum_sq = 0.0f32;
        for (i, &a) in actual.iter().enumerate() {
            let p = predicted.get(i).copied().unwrap_or(0.0);
            sum_sq += (a - p).powi(2);
        }
        sum_sq.sqrt()
    }

    /// Integrate thermodynamic dynamics using GENERIC framework.
    ///
    /// Updates belief state using: dz/dt = L·∇H + M·∇S
    pub fn thermo_step(&mut self, target: &[f32; 5], steps: usize) -> [f32; 5] {
        let Some(state) = &mut self.last_state else {
            return *target;
        };

        // Use thermodynamic engine with DVector
        let current = DVector::from_vec(state.belief.to_vec());
        let target_vec = DVector::from_vec(target.to_vec());
        
        // Integrate multiple steps
        let result = self.thermo.integrate(&current, &target_vec, steps);
        
        // Convert back to array
        let mut new_belief = [0.0f32; 5];
        for (i, v) in result.iter().enumerate().take(5) {
            new_belief[i] = *v;
        }
        state.belief = new_belief;

        new_belief
    }

    /// Integrate saccade memory prediction.
    ///
    /// Uses LTC network to predict where to look in holographic memory.
    pub fn integrate_saccade_recall(&mut self, dt: f32) {
        let Some(state) = &self.last_state else {
            return;
        };

        // Extend belief to match expected context dimension (7 for ZenB)
        let mut context: Vec<f32> = state.belief.to_vec();
        context.extend([state.confidence, 0.0]); // Add confidence and padding

        if let Some(recalled) = self.saccade.recall_fast(&context, &self.saccade_memory, dt) {
            log::debug!("Saccade recall: {} values retrieved", recalled.len());
        }
    }

    /// Integrate data reincarnation: Vinnana → Rupa feedback loop.
    ///
    /// In future: feeds synthesized state back into causal system for next cycle.
    /// Currently: logs the reincarnation event for tracing.
    pub fn integrate_reincarnation(&mut self, _ts_us: i64) {
        let Some(state) = &self.last_state else {
            return;
        };

        // Log the reincarnation event
        // In production: this would create a new observation and feed it
        // back into the causal system for the next cycle
        log::debug!(
            "Data Reincarnation: mode={} conf={:.2} belief={:?}",
            state.mode,
            state.confidence,
            state.belief
        );
    }

    /// Get current philosophical state.
    pub fn philosophical_state(&self) -> crate::philosophical_state::PhilosophicalState {
        self.philosophical_state.current_state
    }

    /// Get prediction error EMA for diagnostics.
    pub fn prediction_error(&self) -> f32 {
        self.prediction_error_ema
    }

    /// Get thermodynamic diagnostics.
    /// Returns (temperature, energy_scale, entropy_scale, enabled)
    pub fn thermo_info(&self) -> (f32, f32, f32, bool) {
        let cfg = self.thermo.config();
        (cfg.temperature, cfg.energy_scale, cfg.entropy_scale, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ZenbConfig;
    use crate::skandha::zenb::zenb_pipeline_unified;

    #[test]
    fn test_vinnana_controller_creation() {
        let cfg = ZenbConfig::default();
        let pipeline = zenb_pipeline_unified(&cfg);
        let controller = VinnanaController::new(pipeline);
        assert!(controller.last_state.is_none());
        assert_eq!(controller.prediction_error_ema, 0.0);
    }
}
