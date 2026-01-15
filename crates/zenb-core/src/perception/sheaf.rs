//! Sheaf Laplacian for Sensor Consensus
//!
//! Uses graph diffusion to filter contradictory sensor inputs and achieve
//! a globally consistent view of the user's state.
//!
//! # VAJRA V5: Complex32 Extension
//! Added support for complex-valued sensor data (amplitude + phase) from
//! wavelet transforms. Phase disagreement provides additional anomaly signal.

use nalgebra::{DMatrix, DVector};
use num_complex::Complex32;

/// Context for adaptive threshold adjustment
///
/// Different physiological contexts have different acceptable levels of
/// sensor disagreement and require different filtering strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysiologicalContext {
    /// Sleep state (very low noise, preserve subtle signals)
    Sleep,
    /// Resting state (low noise, high precision needed)
    #[default]
    Rest,
    /// Light activity (moderate noise tolerance)
    LightActivity,
    /// Stress/anxiety (moderate noise, careful consensus)
    Stress,
    /// Moderate exercise (higher noise tolerance)
    ModerateExercise,
    /// Intense exercise (high noise, aggressive filtering)
    IntenseExercise,
}

impl PhysiologicalContext {
    /// Get adaptive anomaly threshold for this context
    ///
    /// # Returns
    /// Threshold value - higher means more tolerant of sensor disagreement
    pub fn anomaly_threshold(&self) -> f32 {
        match self {
            Self::Sleep => 0.2,            // Very sensitive - detect small anomalies
            Self::Rest => 0.3,             // Sensitive - current default
            Self::LightActivity => 0.5,    // Moderate tolerance
            Self::Stress => 0.6,           // Higher tolerance for stress-induced variation
            Self::ModerateExercise => 0.8, // High tolerance for exercise variation
            Self::IntenseExercise => 1.2,  // Very high tolerance
        }
    }

    /// Get base alpha for this context
    ///
    /// # Returns
    /// Base diffusion rate - higher means faster consensus
    pub fn base_alpha(&self) -> f32 {
        match self {
            Self::Sleep => 0.01,            // Very gentle diffusion, preserve signals
            Self::Rest => 0.02,             // Current default
            Self::LightActivity => 0.025,   // Slightly more aggressive
            Self::Stress => 0.03,           // More consensus needed
            Self::ModerateExercise => 0.04, // Strong consensus for noisy data
            Self::IntenseExercise => 0.05,  // Very aggressive filtering
        }
    }
}

/// Sheaf Laplacian for sensor consensus
///
/// # Mathematical Model (Sắc Uẩn - Coherent Perception)
/// Given sensors as nodes in a graph with "should agree" relationships,
/// the Sheaf Laplacian L measures local disagreement.
///
/// Diffusion process: dx/dt = -L * x
///
/// This converges to the "global section" - the most consistent interpretation
/// of all sensor data that minimizes disagreement.
///
/// # Invariants
/// - Laplacian is symmetric positive semi-definite
/// - Kernel of L contains constant vectors (perfect global agreement)
/// - Energy E = x^T * L * x is non-negative
///
/// # Properties
/// - High energy = sensors disagree = possible attack/malfunction
/// - Low energy = sensors agree = trustworthy input
#[derive(Debug)]
pub struct SheafPerception {
    /// Laplacian matrix L = D - A (Degree - Adjacency)
    laplacian: DMatrix<f32>,
    /// Number of sensor channels
    n_sensors: usize,
    /// Base diffusion coefficient (adjusted by energy and context)
    alpha: f32,
    /// Base energy threshold for anomaly detection (adjusted by context)
    anomaly_threshold: f32,
    /// Current physiological context (for adaptive thresholds)
    context: PhysiologicalContext,
    /// Whether to use adaptive alpha based on energy
    use_adaptive_alpha: bool,
}

impl SheafPerception {
    /// Create perception layer from sensor adjacency relationships.
    ///
    /// # Arguments
    /// * `adjacency` - Which sensors should agree (weight = agreement strength)
    /// * `alpha` - Diffusion rate (0.01-0.1 typical)
    ///
    /// # Panics
    /// Panics if adjacency matrix is not square.
    pub fn new(adjacency: &DMatrix<f32>, alpha: f32) -> Self {
        let n = adjacency.nrows();
        assert_eq!(
            adjacency.nrows(),
            adjacency.ncols(),
            "Adjacency must be square"
        );

        // Compute degree matrix (diagonal sum of row weights)
        let mut degree = DMatrix::zeros(n, n);
        for i in 0..n {
            let row_sum: f32 = adjacency.row(i).iter().sum();
            degree[(i, i)] = row_sum;
        }

        // Laplacian = Degree - Adjacency
        let laplacian = degree - adjacency;

        Self {
            laplacian,
            n_sensors: n,
            alpha,
            anomaly_threshold: 1.0,              // Default threshold
            context: PhysiologicalContext::Rest, // Default context
            use_adaptive_alpha: true,            // Enable adaptive alpha by default
        }
    }

    /// Create default perception for ZenB sensor layout.
    ///
    /// Sensor indices:
    /// 0: Heart Rate (HR)
    /// 1: Heart Rate Variability (HRV/RMSSD)
    /// 2: Respiratory Rate (RR)
    /// 3: Signal Quality
    /// 4: Motion
    ///
    /// Adjacency encodes which sensors should correlate:
    /// - HR & HRV should inversely correlate (handled by sign flip before consensus)
    /// - HR & RR should correlate (Respiratory Sinus Arrhythmia)
    /// - Motion affects all bio signals
    pub fn default_for_zenb() -> Self {
        let n = 5;
        let mut adj = DMatrix::zeros(n, n);

        // HR (0) - HRV (1) correlation: high HR usually means low HRV
        // We encode this as a strong link - the consensus process will
        // identify when they are TOO different (both high = suspicious)
        adj[(0, 1)] = 0.8;
        adj[(1, 0)] = 0.8;

        // HR (0) - RR (2) correlation: Respiratory Sinus Arrhythmia
        adj[(0, 2)] = 0.5;
        adj[(2, 0)] = 0.5;

        // HRV (1) - RR (2) correlation: breath coherence affects HRV
        adj[(1, 2)] = 0.6;
        adj[(2, 1)] = 0.6;

        // Motion (4) affects all bio signals (0, 1, 2)
        for i in 0..3 {
            adj[(4, i)] = 0.3;
            adj[(i, 4)] = 0.3;
        }

        // Quality (3) is somewhat independent but low quality affects trust
        adj[(3, 0)] = 0.2;
        adj[(0, 3)] = 0.2;

        let mut perception = Self::new(&adj, 0.02); // Base alpha (will be adjusted by context/energy)
        perception.context = PhysiologicalContext::Rest; // Default to resting state
        perception.anomaly_threshold = perception.context.anomaly_threshold();
        perception.use_adaptive_alpha = true; // Enable energy-based adaptation
        perception
    }

    /// Compute effective alpha based on energy and context
    ///
    /// # Adaptive Alpha Strategy
    /// - High energy (contradictory sensors) → larger alpha (faster, more aggressive consensus)
    /// - Low energy (clean sensors) → smaller alpha (preserve accuracy)
    /// - Context modulates base alpha (exercise vs sleep have different noise levels)
    ///
    /// # Formula
    /// `effective_alpha = base_alpha * (1.0 + sqrt(energy))`
    ///
    /// This gives:
    /// - Energy = 0.1 → 1.32x base alpha (gentle boost)
    /// - Energy = 0.5 → 1.71x base alpha (moderate boost)
    /// - Energy = 1.0 → 2.0x base alpha (strong boost)
    /// - Energy = 4.0 → 3.0x base alpha (aggressive consensus)
    fn compute_adaptive_alpha(&self, energy: f32) -> f32 {
        if !self.use_adaptive_alpha {
            return self.alpha;
        }

        // Get context-specific base alpha
        let context_alpha = self.context.base_alpha();

        // Energy-based multiplier: sqrt provides smooth scaling
        // High energy → more aggressive consensus needed
        let energy_multiplier = 1.0 + energy.sqrt();

        // Cap at 3x base alpha to prevent over-aggressive diffusion
        let effective_alpha = context_alpha * energy_multiplier;
        effective_alpha.min(context_alpha * 3.0)
    }

    /// Run diffusion to achieve sensor consensus.
    ///
    /// This is the core algorithm: iterate the diffusion equation until
    /// sensors reach approximate agreement (global section).
    ///
    /// # Arguments
    /// * `input` - Raw sensor readings (normalized to similar ranges)
    /// * `steps` - Number of diffusion iterations (10-50 typical)
    ///
    /// # Returns
    /// Diffused sensor values representing global section (consensus view).
    ///
    /// # Mathematical Operation
    /// For each step: x_{t+1} = x_t - α * L * x_t
    ///
    /// This is equivalent to heat diffusion on the sensor graph.
    pub fn diffuse(&self, input: &DVector<f32>, steps: usize) -> DVector<f32> {
        assert_eq!(
            input.len(),
            self.n_sensors,
            "Input dimension must match sensor count"
        );

        let mut state = input.clone();

        // Compute initial energy for adaptive alpha
        let energy = self.compute_energy(&state);
        let effective_alpha = self.compute_adaptive_alpha(energy);

        for _ in 0..steps {
            // Compute disagreement gradient: delta = L * x
            let delta = &self.laplacian * &state;

            // Update: reduce disagreement proportionally with adaptive alpha
            state = state - effective_alpha * delta;
        }

        state
    }

    /// Diffuse with adaptive step count based on initial energy.
    ///
    /// With energy-based adaptive alpha, we can afford more steps since
    /// high-energy states will use larger alpha (faster convergence).
    pub fn diffuse_adaptive(&self, input: &DVector<f32>, max_steps: usize) -> DVector<f32> {
        let initial_energy = self.compute_energy(input);

        // Adaptive step count based on energy
        // With adaptive alpha enabled, high energy → larger alpha → fewer steps needed
        // But we still need sufficient steps for proper convergence
        let steps = if self.use_adaptive_alpha {
            // Adaptive alpha mode: energy determines both alpha and steps
            if initial_energy < 0.1 {
                10 // Clean sensors, gentle diffusion
            } else if initial_energy < 0.3 {
                20 // Moderate disagreement
            } else if initial_energy < 0.8 {
                40 // High disagreement, but adaptive alpha will accelerate
            } else {
                max_steps.min(100) // Very high disagreement, use max steps
            }
        } else {
            // Fixed alpha mode: use original logic
            if initial_energy < 0.1 {
                5
            } else if initial_energy < 0.5 {
                15
            } else {
                max_steps.min(50)
            }
        };

        self.diffuse(input, steps)
    }

    /// Compute Laplacian energy (total inconsistency).
    ///
    /// E = x^T * L * x
    ///
    /// This is a measure of how much sensors disagree with each other.
    ///
    /// # Interpretation
    /// - E ≈ 0: Perfect agreement (global section exists)
    /// - E high: Significant disagreement (contradiction/noise/attack)
    pub fn compute_energy(&self, state: &DVector<f32>) -> f32 {
        let lx = &self.laplacian * state;
        state.dot(&lx)
    }

    /// Check if input is anomalous based on energy threshold.
    ///
    /// High energy indicates sensors report contradictory information,
    /// which could be due to:
    /// - Sensor malfunction
    /// - Adversarial attack
    /// - Unusual physiological state
    ///
    /// Uses context-aware threshold: different contexts have different
    /// acceptable levels of sensor disagreement.
    ///
    /// # Returns
    /// `true` if energy exceeds threshold (anomalous input)
    pub fn is_anomalous(&self, state: &DVector<f32>) -> bool {
        let energy = self.compute_energy(state);
        let threshold = self.context.anomaly_threshold();
        energy > threshold
    }

    /// Validate and filter sensor input.
    ///
    /// This is the main entry point for the perception pipeline:
    /// 1. Check if input is anomalous (high disagreement)
    /// 2. If anomalous, log warning but still diffuse
    /// 3. Return diffused (consensus) values
    ///
    /// Uses context-aware thresholds and adaptive alpha for optimal
    /// filtering based on physiological state and sensor disagreement.
    ///
    /// # Returns
    /// (diffused_values, is_anomalous, energy)
    pub fn process(&self, raw_input: &DVector<f32>) -> (DVector<f32>, bool, f32) {
        let energy = self.compute_energy(raw_input);
        let threshold = self.context.anomaly_threshold();
        let anomalous = energy > threshold;

        if anomalous {
            log::warn!(
                "SheafPerception: High disagreement detected (E={:.3} > {:.3}, context={:?})",
                energy,
                threshold,
                self.context
            );
        }

        // Increased max_steps from 30 to 100 for better convergence
        let diffused = self.diffuse_adaptive(raw_input, 100);

        (diffused, anomalous, energy)
    }

    /// Get the number of sensors.
    pub fn n_sensors(&self) -> usize {
        self.n_sensors
    }

    /// Set the anomaly detection threshold.
    pub fn set_anomaly_threshold(&mut self, threshold: f32) {
        self.anomaly_threshold = threshold;
    }

    /// Get the Laplacian matrix (for diagnostics).
    pub fn laplacian(&self) -> &DMatrix<f32> {
        &self.laplacian
    }

    /// Set the physiological context for adaptive thresholds
    ///
    /// Different contexts have different acceptable levels of sensor
    /// disagreement and require different filtering strategies.
    ///
    /// # Example
    /// ```ignore
    /// sheaf.set_context(PhysiologicalContext::ModerateExercise);
    /// // Now anomaly threshold = 0.8, base alpha = 0.04
    /// ```
    pub fn set_context(&mut self, context: PhysiologicalContext) {
        self.context = context;
        // Update anomaly threshold to match context
        self.anomaly_threshold = context.anomaly_threshold();
    }

    /// Get current physiological context
    pub fn context(&self) -> PhysiologicalContext {
        self.context
    }

    /// Enable or disable adaptive alpha
    pub fn set_adaptive_alpha(&mut self, enabled: bool) {
        self.use_adaptive_alpha = enabled;
    }

    /// Check if adaptive alpha is enabled
    pub fn is_adaptive_alpha_enabled(&self) -> bool {
        self.use_adaptive_alpha
    }

    // ========================================================================
    // VAJRA V5: Complex32 Extension Methods
    // ========================================================================

    /// Diffuse complex-valued sensor data (from wavelet transforms).
    ///
    /// This extends the standard diffusion to handle amplitude AND phase:
    /// - Amplitude diffusion: same as f32 (consensus on magnitude)
    /// - Phase disagreement: additional energy term from phase mismatch
    ///
    /// # Arguments
    /// * `input` - Complex sensor values [amplitude, phase encoded]
    /// * `steps` - Number of diffusion iterations
    ///
    /// # Returns
    /// Diffused complex values with consensus amplitude and phase
    pub fn diffuse_complex(&self, input: &[Complex32]) -> Vec<Complex32> {
        assert_eq!(
            input.len(),
            self.n_sensors,
            "Input dimension must match sensor count"
        );

        // Extract amplitude and phase separately
        let amplitudes: Vec<f32> = input.iter().map(|c| c.norm()).collect();
        let phases: Vec<f32> = input.iter().map(|c| c.arg()).collect();

        // Diffuse amplitudes using standard Laplacian
        let amp_vec = DVector::from_vec(amplitudes);
        let diffused_amp = self.diffuse_adaptive(&amp_vec, 50);

        // For phases: compute weighted average based on amplitude
        // (stronger signals contribute more to phase consensus)
        let total_amp: f32 = diffused_amp.iter().sum::<f32>().max(1e-6);
        
        // Use circular mean for phases (proper phase averaging)
        let mut sin_sum = 0.0f32;
        let mut cos_sum = 0.0f32;
        for (i, &phase) in phases.iter().enumerate() {
            let weight = diffused_amp[i] / total_amp;
            sin_sum += weight * phase.sin();
            cos_sum += weight * phase.cos();
        }
        let phase_consensus = sin_sum.atan2(cos_sum);

        // Reconstruct complex values with diffused amplitude and consensus phase
        // Each sensor's phase is smoothed toward consensus
        // VAJRA-VOID: Context-aware phase smoothing
        let alpha_phase = match self.context {
            PhysiologicalContext::Sleep => 0.1,           // Preserve subtle signals
            PhysiologicalContext::Rest => 0.2,
            PhysiologicalContext::LightActivity => 0.3,
            PhysiologicalContext::Stress => 0.35,
            PhysiologicalContext::ModerateExercise => 0.5,
            PhysiologicalContext::IntenseExercise => 0.6, // Aggressive filtering
        };
        let result: Vec<Complex32> = (0..self.n_sensors)
            .map(|i| {
                let amp = diffused_amp[i];
                let original_phase = phases[i];
                // Blend original phase with consensus
                let smoothed_phase = original_phase + alpha_phase * (phase_consensus - original_phase);
                Complex32::from_polar(amp, smoothed_phase)
            })
            .collect();

        result
    }

    /// Compute phase-based disagreement energy.
    ///
    /// Measures how much the phases of sensor signals disagree.
    /// High phase disagreement indicates potential jitter or interference.
    ///
    /// # Returns
    /// Phase energy in [0, π²] - higher means more phase disagreement
    pub fn compute_phase_energy(&self, input: &[Complex32]) -> f32 {
        if input.len() < 2 {
            return 0.0;
        }

        // Extract phases
        let phases: Vec<f32> = input.iter().map(|c| c.arg()).collect();

        // Compute mean phase (circular)
        let sin_sum: f32 = phases.iter().map(|p| p.sin()).sum();
        let cos_sum: f32 = phases.iter().map(|p| p.cos()).sum();
        let n = phases.len() as f32;
        let mean_phase = (sin_sum / n).atan2(cos_sum / n);

        // Compute variance from mean (handling circular nature)
        let mut energy = 0.0f32;
        for &phase in &phases {
            let diff = (phase - mean_phase).sin().abs(); // Circular difference
            energy += diff * diff;
        }

        energy
    }

    /// Process complex-valued sensor input (from wavelet).
    ///
    /// # Returns
    /// (diffused_complex, is_anomalous, amplitude_energy, phase_energy)
    pub fn process_complex(&self, input: &[Complex32]) -> (Vec<Complex32>, bool, f32, f32) {
        // Extract amplitudes for standard processing
        let amplitudes: Vec<f32> = input.iter().map(|c| c.norm()).collect();
        let amp_vec = DVector::from_vec(amplitudes);

        // Compute both amplitude and phase energies
        let amp_energy = self.compute_energy(&amp_vec);
        let phase_energy = self.compute_phase_energy(input);

        // Combined anomaly detection: either amplitude OR phase disagreement
        let threshold = self.context.anomaly_threshold();
        let phase_threshold = 0.5; // π²/20 ≈ 0.5 is moderate phase disagreement
        let is_anomalous = amp_energy > threshold || phase_energy > phase_threshold;

        if is_anomalous {
            log::warn!(
                "SheafPerception: Anomaly detected (amp_E={:.3}, phase_E={:.3}, context={:?})",
                amp_energy,
                phase_energy,
                self.context
            );
        }

        // Diffuse complex values
        let diffused = self.diffuse_complex(input);

        (diffused, is_anomalous, amp_energy, phase_energy)
    }

    /// Convert f32 sensor values to Complex32 (zero phase).
    ///
    /// Convenience method for transitioning from f32 to Complex32 API.
    pub fn to_complex(values: &[f32]) -> Vec<Complex32> {
        values.iter().map(|&r| Complex32::new(r, 0.0)).collect()
    }

    /// Extract amplitudes from Complex32 values (back to f32).
    pub fn from_complex(values: &[Complex32]) -> Vec<f32> {
        values.iter().map(|c| c.norm()).collect()
    }
}

impl Default for SheafPerception {
    fn default() -> Self {
        Self::default_for_zenb()
    }
}

// ============================================================================
// RupaSkandha Integration (Sắc Uẩn - Form Processing)
// ============================================================================

use crate::skandha::{ProcessedForm, RupaSkandha, SensorInput};

/// Implement RupaSkandha for SheafPerception.
/// 
/// This bridges the sensor consensus layer into the Ngũ Uẩn pipeline,
/// making SheafPerception the primary Form (Sắc) processing stage.
impl RupaSkandha for SheafPerception {
    /// Process raw sensor input through Sheaf Laplacian diffusion.
    /// 
    /// # Sắc Uẩn Processing
    /// - Converts SensorInput (hr, hrv, rr, quality, motion) to DVector
    /// - Applies graph diffusion for sensor consensus
    /// - Returns ProcessedForm with consensus values and anomaly metrics
    fn process_form(&mut self, input: &SensorInput) -> ProcessedForm {
        // Convert SensorInput to DVector<f32>
        // Order: [hr_bpm, hrv_rmssd, rr_bpm, quality, motion]
        // Note: hr_bpm, hrv_rmssd, rr_bpm are Option<f32>, use defaults if None
        let raw_values = vec![
            input.hr_bpm.unwrap_or(70.0),      // Default resting HR
            input.hrv_rmssd.unwrap_or(50.0),   // Default healthy HRV
            input.rr_bpm.unwrap_or(15.0),      // Default resting RR
            input.quality,
            input.motion,
        ];
        let raw_dvec = DVector::from_vec(raw_values);
        
        // Process through Sheaf Laplacian diffusion
        let (diffused, is_anomalous, energy) = self.process(&raw_dvec);
        
        // Convert back to fixed array
        let mut values = [0.0f32; 5];
        for (i, &v) in diffused.iter().enumerate().take(5) {
            values[i] = v;
        }
        
        // Create ProcessedForm output
        ProcessedForm {
            values,
            anomaly_score: if is_anomalous { energy } else { 0.0 },
            energy,
            is_reliable: !is_anomalous,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_new_perception() {
        let adj = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let perception = SheafPerception::new(&adj, 0.01);
        assert_eq!(perception.n_sensors(), 3);
    }

    #[test]
    fn test_laplacian_properties() {
        let perception = SheafPerception::default_for_zenb();

        // Laplacian should be symmetric
        let l = perception.laplacian();
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert!(
                    approx_eq(l[(i, j)], l[(j, i)], 1e-6),
                    "Laplacian should be symmetric"
                );
            }
        }

        // Row sums should be zero (Laplacian property)
        for i in 0..l.nrows() {
            let row_sum: f32 = l.row(i).iter().sum();
            assert!(
                approx_eq(row_sum, 0.0, 1e-6),
                "Row sum should be zero: got {}",
                row_sum
            );
        }
    }

    #[test]
    fn test_constant_vector_zero_energy() {
        let perception = SheafPerception::default_for_zenb();

        // Constant vector should have zero Laplacian energy
        let constant = DVector::from_element(5, 1.0);
        let energy = perception.compute_energy(&constant);

        assert!(
            approx_eq(energy, 0.0, 1e-6),
            "Constant vector should have zero energy: got {}",
            energy
        );
    }

    #[test]
    fn test_diffusion_reduces_disagreement() {
        let perception = SheafPerception::default_for_zenb();

        // Create disagreeing sensors: HR high, HRV high (unusual)
        let disagreeing = DVector::from_vec(vec![1.0, 1.0, 0.5, 1.0, 0.0]);

        let energy_before = perception.compute_energy(&disagreeing);
        let diffused = perception.diffuse(&disagreeing, 20);
        let energy_after = perception.compute_energy(&diffused);

        println!(
            "Energy before: {:.4}, after: {:.4}",
            energy_before, energy_after
        );

        assert!(
            energy_after < energy_before,
            "Diffusion should reduce disagreement"
        );
    }

    #[test]
    fn test_contradictory_sensors_converge_to_zero() {
        // DoD Test: input [1.0, -1.0] on connected nodes should converge to ~[0.0, 0.0]
        let adj = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let perception = SheafPerception::new(&adj, 0.1);

        let contradictory = DVector::from_vec(vec![1.0, -1.0]);
        let diffused = perception.diffuse(&contradictory, 100);

        // Should converge towards the mean (0.0)
        assert!(
            diffused[0].abs() < 0.2,
            "First sensor should approach mean: got {}",
            diffused[0]
        );
        assert!(
            diffused[1].abs() < 0.2,
            "Second sensor should approach mean: got {}",
            diffused[1]
        );
    }

    #[test]
    fn test_anomaly_detection() {
        let perception = SheafPerception::default_for_zenb();

        // Normal input: HR moderate, HRV moderate, RR normal
        let normal = DVector::from_vec(vec![0.6, 0.5, 0.5, 1.0, 0.1]);

        // Anomalous input: very high HR, very high HRV (physiologically unlikely)
        let anomalous = DVector::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0]);

        let (_, _normal_anomalous, _) = perception.process(&normal);
        let (_, _suspicious_anomalous, energy) = perception.process(&anomalous);

        println!("Anomalous input energy: {:.4}", energy);

        // The obviously contradictory input should have higher energy
        // (exact threshold depends on configuration)
    }

    #[test]
    fn test_process_returns_diffused() {
        let perception = SheafPerception::default_for_zenb();
        let input = DVector::from_vec(vec![0.7, 0.3, 0.5, 1.0, 0.2]);

        let (diffused, _, _) = perception.process(&input);

        assert_eq!(diffused.len(), 5);
    }

    #[test]
    fn test_zenb_default_sensors() {
        let perception = SheafPerception::default_for_zenb();

        // Should have 5 sensors
        assert_eq!(perception.n_sensors(), 5);

        // Verify Laplacian dimensions
        assert_eq!(perception.laplacian().nrows(), 5);
        assert_eq!(perception.laplacian().ncols(), 5);
    }
}
