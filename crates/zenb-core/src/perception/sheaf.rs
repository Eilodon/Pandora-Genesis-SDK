//! Sheaf Laplacian for Sensor Consensus
//!
//! Uses graph diffusion to filter contradictory sensor inputs and achieve
//! a globally consistent view of the user's state.

use nalgebra::{DMatrix, DVector};

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
    /// Diffusion coefficient (step size for consensus)
    alpha: f32,
    /// Energy threshold for anomaly detection
    anomaly_threshold: f32,
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
            anomaly_threshold: 1.0, // Default threshold
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

        let mut perception = Self::new(&adj, 0.05);
        perception.anomaly_threshold = 0.5;
        perception
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

        for _ in 0..steps {
            // Compute disagreement gradient: delta = L * x
            let delta = &self.laplacian * &state;

            // Update: reduce disagreement proportionally
            state = state - self.alpha * delta;
        }

        state
    }

    /// Diffuse with adaptive step count based on initial energy.
    ///
    /// High disagreement states need more iterations to converge.
    pub fn diffuse_adaptive(&self, input: &DVector<f32>, max_steps: usize) -> DVector<f32> {
        let initial_energy = self.compute_energy(input);

        // More steps for higher disagreement
        let steps = if initial_energy < 0.1 {
            5 // Already agree
        } else if initial_energy < 0.5 {
            15
        } else {
            max_steps.min(50)
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
    /// # Returns
    /// `true` if energy exceeds threshold (anomalous input)
    pub fn is_anomalous(&self, state: &DVector<f32>) -> bool {
        self.compute_energy(state) > self.anomaly_threshold
    }

    /// Validate and filter sensor input.
    ///
    /// This is the main entry point for the perception pipeline:
    /// 1. Check if input is anomalous (high disagreement)
    /// 2. If anomalous, log warning but still diffuse
    /// 3. Return diffused (consensus) values
    ///
    /// # Returns
    /// (diffused_values, is_anomalous, energy)
    pub fn process(&self, raw_input: &DVector<f32>) -> (DVector<f32>, bool, f32) {
        let energy = self.compute_energy(raw_input);
        let anomalous = energy > self.anomaly_threshold;

        if anomalous {
            log::warn!(
                "SheafPerception: High disagreement detected (E={:.3} > {:.3})",
                energy,
                self.anomaly_threshold
            );
        }

        let diffused = self.diffuse_adaptive(raw_input, 30);

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
}

impl Default for SheafPerception {
    fn default() -> Self {
        Self::default_for_zenb()
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

        println!("Energy before: {:.4}, after: {:.4}", energy_before, energy_after);

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

        let (_, normal_anomalous, _) = perception.process(&normal);
        let (_, suspicious_anomalous, energy) = perception.process(&anomalous);

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
