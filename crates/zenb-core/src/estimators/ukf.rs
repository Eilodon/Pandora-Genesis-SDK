//! Unscented Kalman Filter (UKF) for non-linear state estimation
//!
//! Implements a 5-dimensional state estimator for physiological signals:
//! - Arousal
//! - dArousal/dt (momentum)
//! - Valence  
//! - Attention
//! - Rhythm alignment
//!
//! Reference: UKFStateEstimator.ts (~500 lines)

use nalgebra::{SMatrix, SVector};
use serde::{Deserialize, Serialize};

// State dimension
const N: usize = 5;

type StateVector = SVector<f32, N>;
type CovarianceMatrix = SMatrix<f32, N, N>;

/// UKF Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UkfConfig {
    /// Process noise scale
    pub q_scale: f32,

    /// Measurement noise variances
    pub r_hr: f32,
    pub r_hrv: f32,
    pub r_resp: f32,
    pub r_valence: f32,

    /// UKF parameters
    pub alpha: f32, // Spread of sigma points (default: 0.001)
    pub beta: f32,  // Prior knowledge (default: 2.0 for Gaussian)
    pub kappa: f32, // Secondary scaling (default: 0.0)
}

impl Default for UkfConfig {
    fn default() -> Self {
        Self {
            q_scale: 0.01,
            r_hr: 0.15,
            r_hrv: 0.25,
            r_resp: 0.20,
            r_valence: 0.30,
            alpha: 0.001,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

/// Sage-Husa Adaptive UKF Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AukfConfig {
    /// Base UKF configuration
    pub ukf: UkfConfig,

    /// Forgetting factor for noise estimation (0.95-0.99 typical)
    /// Smaller = faster adaptation, larger = more stable
    pub forgetting_factor: f32,

    /// Minimum window for noise estimation (prevents premature adaptation)
    pub min_samples: usize,

    /// Enable adaptive Q estimation
    pub adapt_q: bool,

    /// Enable adaptive R estimation  
    pub adapt_r: bool,

    /// Maximum Q scale (prevents divergence)
    pub max_q_scale: f32,

    /// Minimum R scale (prevents over-trusting measurements)
    pub min_r_scale: f32,
}

impl Default for AukfConfig {
    fn default() -> Self {
        Self {
            ukf: UkfConfig::default(),
            forgetting_factor: 0.97,
            min_samples: 10,
            adapt_q: true,
            adapt_r: true,
            max_q_scale: 0.5,
            min_r_scale: 0.05,
        }
    }
}

/// Sensor observation
#[derive(Debug, Clone, Default)]
pub struct Observation {
    pub heart_rate: Option<f32>,
    pub hr_confidence: Option<f32>,
    pub stress_index: Option<f32>,
    pub respiration_rate: Option<f32>,
    pub facial_valence: Option<f32>,
}

/// Target state from breathing protocol
#[derive(Debug, Clone, Copy)]
struct TargetState {
    arousal: f32,
    rhythm: f32,
    valence: f32,
}

impl Default for TargetState {
    fn default() -> Self {
        Self {
            arousal: 0.5,
            rhythm: 0.7,
            valence: 0.5,
        }
    }
}

/// UKF Belief state output (compatible with existing zenb-core)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UkfBeliefState {
    pub arousal: f32,
    pub attention: f32,
    pub rhythm_alignment: f32,
    pub valence: f32,

    pub arousal_variance: f32,
    pub attention_variance: f32,
    pub rhythm_variance: f32,

    pub prediction_error: f32,
    pub innovation: f32,
    pub confidence: f32,
}

/// Unscented Kalman Filter State Estimator
pub struct UkfStateEstimator {
    x: StateVector,      // State [arousal, dA/dt, valence, attention, rhythm]
    p: CovarianceMatrix, // Covariance
    config: UkfConfig,
    target: TargetState,

    // UKF weights (precomputed)
    weights_m: Vec<f32>,
    weights_c: Vec<f32>,
    lambda: f32,

    // Time constants (physiological dynamics)
    tau_arousal: f32,
    tau_arousal_vel: f32,
    tau_attention: f32,
    tau_rhythm: f32,
    tau_valence: f32,

    // === SAGE-HUSA AUKF STATE ===
    /// Sample count for minimum window check
    sample_count: usize,

    /// Adaptive Q estimate (process noise covariance)
    q_adaptive: CovarianceMatrix,



    /// Residual statistics for Sage-Husa
    residual_mean: StateVector,
    residual_cov: CovarianceMatrix,

    /// AUKF configuration
    aukf_config: AukfConfig,
}

impl UkfStateEstimator {
    /// Create new UKF estimator
    pub fn new(config: Option<UkfConfig>) -> Self {
        Self::new_adaptive(config.map(|c| AukfConfig {
            ukf: c,
            ..Default::default()
        }))
    }

    /// Create new Adaptive UKF (AUKF) estimator with Sage-Husa noise estimation
    pub fn new_adaptive(config: Option<AukfConfig>) -> Self {
        let aukf_cfg = config.unwrap_or_default();
        let cfg = aukf_cfg.ukf.clone();
        let n = N as f32;
        let lambda = cfg.alpha.powi(2) * (n + cfg.kappa) - n;

        // Precompute weights
        let w0_m = lambda / (n + lambda);
        let w0_c = w0_m + (1.0 - cfg.alpha.powi(2) + cfg.beta);
        let wi = 1.0 / (2.0 * (n + lambda));

        let mut weights_m = vec![wi; 2 * N + 1];
        let mut weights_c = vec![wi; 2 * N + 1];
        weights_m[0] = w0_m;
        weights_c[0] = w0_c;

        Self {
            x: StateVector::from_vec(vec![0.5, 0.0, 0.0, 0.5, 0.0]),
            p: CovarianceMatrix::identity() * 0.2,
            config: cfg.clone(),
            target: TargetState::default(),
            weights_m,
            weights_c,
            lambda,
            tau_arousal: 15.0,
            tau_arousal_vel: 5.0,
            tau_attention: 5.0,
            tau_rhythm: 10.0,
            tau_valence: 8.0,
            // AUKF initialization
            sample_count: 0,
            q_adaptive: CovarianceMatrix::identity() * cfg.q_scale,

            residual_mean: StateVector::zeros(),
            residual_cov: CovarianceMatrix::identity() * 0.2, // Initialize to match P to avoid cold-start drop
            aukf_config: aukf_cfg,
        }
    }

    /// Set target state based on arousal impact
    pub fn set_protocol(&mut self, arousal_impact: f32) {
        self.target = if arousal_impact < -0.5 {
            // Parasympathetic
            TargetState {
                arousal: 0.2,
                rhythm: 0.8,
                valence: 0.6,
            }
        } else if arousal_impact > 0.5 {
            // Sympathetic
            TargetState {
                arousal: 0.7,
                rhythm: 0.6,
                valence: 0.7,
            }
        } else {
            // Balanced
            TargetState {
                arousal: 0.4,
                rhythm: 0.9,
                valence: 0.5,
            }
        };
    }

    /// Main update step
    pub fn update(&mut self, obs: &Observation, dt: f32) -> UkfBeliefState {
        // 1. Prediction
        self.predict(dt);

        // Store predicted state for residual calculation
        let x_pred = self.x;

        // 2. Correction
        self.correct(obs);

        // 3. Adaptive Q Update (Sage-Husa)
        // Residual = x_post - x_pred (state correction magnitude)
        let residual = self.x - x_pred;
        let p_curr = self.p; // Copy p to avoid borrow conflict
        self.update_adaptive_q(&residual, &p_curr);

        // 4. Convert to UkfBeliefState
        self.to_belief_state()
    }

    /// Get current sample count
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Get diagonal elements of adaptive Q matrix (for telemetry)
    pub fn get_q_diagonal(&self) -> [f32; N] {
        let mut diag = [0.0; N];
        for i in 0..N {
            diag[i] = self.q_adaptive[(i, i)];
        }
        diag
    }

    // --- PREDICTION ---

    fn predict(&mut self, dt: f32) {
        // Generate sigma points
        let sigmas = self.generate_sigma_points();

        // Propagate through dynamics
        let sigmas_pred: Vec<StateVector> =
            sigmas.iter().map(|s| self.state_dynamics(s, dt)).collect();

        // Predicted mean
        self.x = self.weighted_mean(&sigmas_pred);

        // Predicted covariance
        self.p = self.weighted_covariance(&sigmas_pred, &self.x);

        // Add process noise (use adaptive Q if enabled)
        if self.aukf_config.adapt_q && self.sample_count >= self.aukf_config.min_samples {
            self.p += self.q_adaptive * dt;
        } else {
            self.p += CovarianceMatrix::identity() * (self.config.q_scale * dt);
        }

        // Update sample count
        self.sample_count += 1;

        // Ensure positive-definiteness after prediction
        self.ensure_psd();
    }

    /// Sage-Husa Q adaptation: estimate process noise from state prediction residuals
    fn update_adaptive_q(&mut self, state_residual: &StateVector, p_post: &CovarianceMatrix) {
        if !self.aukf_config.adapt_q {
            return;
        }

        let b = self.aukf_config.forgetting_factor;
        let d_k = 1.0 - b; // (1 - b) weight for new observation

        // println!("UKF DEBUG: update_adaptive_q count={} min={}", self.sample_count, self.aukf_config.min_samples);

        // Update residual mean: x̄ₖ = b * x̄ₖ₋₁ + (1-b) * dₖ
        self.residual_mean = self.residual_mean * b + state_residual * d_k;

        // Compute residual outer product: dₖ * dₖᵀ
        let residual_outer = state_residual * state_residual.transpose();

        // Update residual covariance: Cₖ = b * Cₖ₋₁ + (1-b) * dₖdₖᵀ
        self.residual_cov = self.residual_cov * b + residual_outer * d_k;

        // Only update Q after minimum samples
        if self.sample_count < self.aukf_config.min_samples {
            return;
        }

        // Sage-Husa Q estimate: Q̂ = Cₖ - Pₖ₊₁|ₖ₊₁
        let q_estimate = self.residual_cov - *p_post;

        // Enforce positive semi-definiteness and bounds
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    // Diagonal: clamp to [0, max_q_scale]
                    self.q_adaptive[(i, j)] = q_estimate[(i, j)]
                        .max(0.0)
                        .min(self.aukf_config.max_q_scale);
                } else {
                    // Off-diagonal: preserve correlation structure but dampen
                    self.q_adaptive[(i, j)] = q_estimate[(i, j)] * 0.5;
                }
            }
        }
    }

    fn state_dynamics(&self, x: &StateVector, dt: f32) -> StateVector {
        let a = x[0]; // Arousal
        let da = x[1]; // dA/dt
        let v = x[2]; // Valence
        let att = x[3]; // Attention
        let r = x[4]; // Rhythm

        // 1. Arousal (logistic growth with momentum)
        let k = 0.1;
        let dda = -k * a * (1.0 - a) - da / self.tau_arousal_vel
            + (self.target.arousal - a) / self.tau_arousal;
        let a_new = a + da * dt;
        let da_new = da + dda * dt;

        // 2. Valence (inverted-U / Yerkes-Dodson)
        let v_optimal = 0.4;
        let v_target = self.target.valence - (a - v_optimal).abs() * 0.5;
        let v_new = v + (v_target - v) / self.tau_valence * dt;

        // 3. Attention (decay + rhythm boost)
        let att_decay = (-dt / self.tau_attention).exp();
        let att_boost = r * 0.1 * dt;
        let att_new = att * att_decay + att_boost;

        // 4. Rhythm (PLL)
        let r_new = r + (self.target.rhythm - r) / self.tau_rhythm * dt;

        StateVector::from_vec(vec![
            a_new.clamp(0.0, 1.0),
            da_new.clamp(-0.5, 0.5),
            v_new.clamp(-1.0, 1.0),
            att_new.clamp(0.0, 1.0),
            r_new.clamp(0.0, 1.0),
        ])
    }

    // --- CORRECTION ---

    fn correct(&mut self, obs: &Observation) {
        // Heart Rate
        if let (Some(hr), Some(conf)) = (obs.heart_rate, obs.hr_confidence) {
            if conf > 0.3 {
                let z = (hr - 50.0) / 70.0;
                let r = self.config.r_hr * (1.0 + (1.0 - conf));
                self.correct_single(z, r, |x| x[0]);
            }
        }

        // Stress Index (HRV)
        if let Some(si) = obs.stress_index {
            let z = (si / 300.0).min(1.0);
            let r = self.config.r_hrv;
            self.correct_single(z, r, |x| x[0] * (1.0 - x[4]));
        }

        // Facial Valence
        if let Some(val) = obs.facial_valence {
            let r = self.config.r_valence;
            self.correct_single(val, r, |x| x[2]);
        }
    }

    fn correct_single<F>(&mut self, z: f32, r: f32, h: F)
    where
        F: Fn(&StateVector) -> f32,
    {
        let sigmas = self.generate_sigma_points();
        let z_sigmas: Vec<f32> = sigmas.iter().map(|s| h(s)).collect();
        let z_pred = self.weighted_mean_1d(&z_sigmas);

        // Innovation covariance S
        let mut s = r;
        for i in 0..z_sigmas.len() {
            let diff = z_sigmas[i] - z_pred;
            s += self.weights_c[i] * diff * diff;
        }

        // Cross-covariance Pxz
        let mut pxz = StateVector::zeros();
        for i in 0..sigmas.len() {
            let x_diff = sigmas[i] - self.x;
            let z_diff = z_sigmas[i] - z_pred;
            pxz += x_diff * (self.weights_c[i] * z_diff);
        }

        // Kalman gain
        let k = pxz / s;

        // Innovation
        let innovation = z - z_pred;

        // Outlier rejection (Mahalanobis)
        let mahalanobis = innovation.abs() / s.sqrt();
        if mahalanobis > 3.0 {
            return; // Reject outlier
        }

        // Update state
        self.x += k * innovation;

        // Update covariance (Joseph form for numerical stability)
        // P = P - K * S * K^T
        self.p -= k * s * k.transpose();

        // Ensure positive-definiteness after measurement update
        self.ensure_psd();
    }

    /// Ensure covariance matrix remains positive semi-definite.
    /// This is critical for numerical stability of the UKF.
    fn ensure_psd(&mut self) {
        const EPSILON: f32 = 1e-6;
        const MAX_VARIANCE: f32 = 10.0;

        // 1. Force symmetry: P = (P + P^T) / 2
        self.p = (self.p + self.p.transpose()) * 0.5;

        // 2. Clamp diagonal elements (variances must be positive)
        for i in 0..N {
            if self.p[(i, i)] < EPSILON {
                self.p[(i, i)] = EPSILON;
            } else if self.p[(i, i)] > MAX_VARIANCE {
                self.p[(i, i)] = MAX_VARIANCE;
            }

            // 3. Check for NaN and replace with default
            if self.p[(i, i)].is_nan() {
                log::warn!("NaN detected in covariance diagonal, resetting");
                self.p[(i, i)] = 0.2;
            }
        }

        // 4. Clamp off-diagonal elements (correlations bounded by variances)
        for i in 0..N {
            for j in 0..N {
                if i != j {
                    let max_cov = (self.p[(i, i)] * self.p[(j, j)]).sqrt();
                    self.p[(i, j)] = self.p[(i, j)].clamp(-max_cov, max_cov);

                    if self.p[(i, j)].is_nan() {
                        self.p[(i, j)] = 0.0;
                    }
                }
            }
        }
    }

    // --- SIGMA POINTS ---

    fn generate_sigma_points(&mut self) -> Vec<StateVector> {
        let n = N as f32;
        let scale = (n + self.lambda).sqrt();

        // Cholesky decomposition with regularization fallback
        let l = match self.p.cholesky() {
            Some(chol) => chol.l(),
            None => {
                // Covariance is not PSD - regularize and retry
                log::warn!("Covariance not PSD, applying regularization");
                self.ensure_psd();

                // Add small diagonal jitter for numerical stability
                let mut p_jittered = self.p;
                for i in 0..N {
                    p_jittered[(i, i)] += 1e-4;
                }

                // Retry Cholesky with jittered matrix
                match p_jittered.cholesky() {
                    Some(chol) => chol.l(),
                    None => {
                        // Last resort: reset to identity (this should be rare)
                        log::error!("Covariance recovery failed, resetting to identity");
                        self.p = CovarianceMatrix::identity() * 0.2;
                        CovarianceMatrix::identity() * 0.447 // sqrt(0.2)
                    }
                }
            }
        };

        let mut sigmas = Vec::with_capacity(2 * N + 1);
        sigmas.push(self.x);

        for i in 0..N {
            let col = l.column(i) * scale;
            sigmas.push(self.x + col);
            sigmas.push(self.x - col);
        }

        sigmas
    }

    // --- HELPERS ---

    fn weighted_mean(&self, vectors: &[StateVector]) -> StateVector {
        let mut mean = StateVector::zeros();
        for (i, v) in vectors.iter().enumerate() {
            mean += v * self.weights_m[i];
        }
        mean
    }

    fn weighted_mean_1d(&self, values: &[f32]) -> f32 {
        values
            .iter()
            .zip(self.weights_m.iter())
            .map(|(v, w)| v * w)
            .sum()
    }

    fn weighted_covariance(&self, vectors: &[StateVector], mean: &StateVector) -> CovarianceMatrix {
        let mut cov = CovarianceMatrix::zeros();
        for (i, v) in vectors.iter().enumerate() {
            let diff = v - mean;
            cov += diff * diff.transpose() * self.weights_c[i];
        }
        cov
    }

    fn to_belief_state(&self) -> UkfBeliefState {
        let prediction_error = ((self.x[0] - self.target.arousal).powi(2)
            + (self.x[4] - self.target.rhythm).powi(2))
        .sqrt()
            / 2.0_f32.sqrt();

        let trace = self.p.trace();
        let confidence = (1.0 - trace / 5.0).clamp(0.0, 1.0);

        UkfBeliefState {
            arousal: self.x[0],
            attention: self.x[3],
            rhythm_alignment: self.x[4],
            valence: self.x[2],
            arousal_variance: self.p[(0, 0)],
            attention_variance: self.p[(3, 3)],
            rhythm_variance: self.p[(4, 4)],
            prediction_error,
            innovation: self.x[1].abs(),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ukf_initialization() {
        let ukf = UkfStateEstimator::new(None);
        assert_eq!(ukf.x.len(), 5);
        assert_eq!(ukf.weights_m.len(), 11); // 2*N + 1
    }

    #[test]
    fn test_ukf_update() {
        let mut ukf = UkfStateEstimator::new(None);

        let obs = Observation {
            heart_rate: Some(70.0),
            hr_confidence: Some(0.8),
            ..Default::default()
        };

        let belief = ukf.update(&obs, 0.1);

        assert!(belief.arousal >= 0.0 && belief.arousal <= 1.0);
        assert!(belief.confidence >= 0.0 && belief.confidence <= 1.0);
    }

    #[test]
    fn test_sigma_points_generation() {
        let mut ukf = UkfStateEstimator::new(None);
        let sigmas = ukf.generate_sigma_points();

        assert_eq!(sigmas.len(), 11); // 2*N + 1

        // First sigma point should be mean
        assert_eq!(sigmas[0], ukf.x);
    }
}
