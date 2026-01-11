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
#[derive(Debug, Clone)]
pub struct UkfConfig {
    /// Process noise scale
    pub q_scale: f32,
    
    /// Measurement noise variances
    pub r_hr: f32,
    pub r_hrv: f32,
    pub r_resp: f32,
    pub r_valence: f32,
    
    /// UKF parameters
    pub alpha: f32,  // Spread of sigma points (default: 0.001)
    pub beta: f32,   // Prior knowledge (default: 2.0 for Gaussian)
    pub kappa: f32,  // Secondary scaling (default: 0.0)
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
    attention: f32,
    rhythm: f32,
    valence: f32,
}

impl Default for TargetState {
    fn default() -> Self {
        Self {
            arousal: 0.5,
            attention: 0.6,
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
    x: StateVector,         // State [arousal, dA/dt, valence, attention, rhythm]
    p: CovarianceMatrix,    // Covariance
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
}

impl UkfStateEstimator {
    /// Create new UKF estimator
    pub fn new(config: Option<UkfConfig>) -> Self {
        let cfg = config.unwrap_or_default();
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
            config: cfg,
            target: TargetState::default(),
            weights_m,
            weights_c,
            lambda,
            tau_arousal: 15.0,
            tau_arousal_vel: 5.0,
            tau_attention: 5.0,
            tau_rhythm: 10.0,
            tau_valence: 8.0,
        }
    }
    
    /// Set target state based on arousal impact
    pub fn set_protocol(&mut self, arousal_impact: f32) {
        self.target = if arousal_impact < -0.5 {
            // Parasympathetic
            TargetState { arousal: 0.2, attention: 0.5, rhythm: 0.8, valence: 0.6 }
        } else if arousal_impact > 0.5 {
            // Sympathetic
            TargetState { arousal: 0.7, attention: 0.8, rhythm: 0.6, valence: 0.7 }
        } else {
            // Balanced
            TargetState { arousal: 0.4, attention: 0.7, rhythm: 0.9, valence: 0.5 }
        };
    }
    
    /// Main update step
    pub fn update(&mut self, obs: &Observation, dt: f32) -> UkfBeliefState {
        // 1. Prediction
        self.predict(dt);
        
        // 2. Correction
        self.correct(obs);
        
        // 3. Convert to UkfBeliefState
        self.to_belief_state()
    }
    
    // --- PREDICTION ---
    
    fn predict(&mut self, dt: f32) {
        // Generate sigma points
        let sigmas = self.generate_sigma_points();
        
        // Propagate through dynamics
        let sigmas_pred: Vec<StateVector> = sigmas
            .iter()
            .map(|s| self.state_dynamics(s, dt))
            .collect();
        
        // Predicted mean
        self.x = self.weighted_mean(&sigmas_pred);
        
        // Predicted covariance
        self.p = self.weighted_covariance(&sigmas_pred, &self.x);
        
        // Add process noise
        self.p += CovarianceMatrix::identity() * (self.config.q_scale * dt);
    }
    
    fn state_dynamics(&self, x: &StateVector, dt: f32) -> StateVector {
        let a = x[0];    // Arousal
        let da = x[1];   // dA/dt
        let v = x[2];    // Valence
        let att = x[3];  // Attention
        let r = x[4];    // Rhythm
        
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
        
        // Update covariance
        self.p -= k * s * k.transpose();
    }
    
    // --- SIGMA POINTS ---
    
    fn generate_sigma_points(&self) -> Vec<StateVector> {
        let n = N as f32;
        let scale = (n + self.lambda).sqrt();
        
        // Cholesky decomposition
        let l = match self.p.cholesky() {
            Some(chol) => chol.l(),
            None => {
                log::warn!("Covariance not PSD, using identity");
                CovarianceMatrix::identity() * 0.2
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
        values.iter().zip(self.weights_m.iter()).map(|(v, w)| v * w).sum()
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
        let ukf = UkfStateEstimator::new(None);
        let sigmas = ukf.generate_sigma_points();
        
        assert_eq!(sigmas.len(), 11); // 2*N + 1
        
        // First sigma point should be mean
        assert_eq!(sigmas[0], ukf.x);
    }
}
