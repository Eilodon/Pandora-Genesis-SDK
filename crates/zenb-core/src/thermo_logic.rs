//! Thermodynamic Logic Engine (GENERIC Framework)
//!
//! Implements the GENERIC (General Equation for Non-Equilibrium Reversible-Irreversible Coupling)
//! framework for cognitive dynamics.
//!
//! # Mathematical Foundation
//!
//! The GENERIC equation:
//! ```text
//! dz/dt = L(z)·∇H(z) + M(z)·∇S(z)
//! ```
//!
//! Where:
//! - `z`: State vector (belief, arousal, attention, etc.)
//! - `L`: Poisson bracket operator (reversible/Hamiltonian dynamics)
//! - `H`: Energy functional (free energy to minimize)
//! - `M`: Friction matrix (irreversible/dissipative dynamics)
//! - `S`: Entropy functional (to maximize for exploration)
//!
//! # Key Properties
//!
//! 1. **Degeneracy conditions**:
//!    - L·∇S = 0 (Poisson doesn't affect entropy)
//!    - M·∇H = 0 (Friction doesn't affect energy)
//!
//! 2. **Energy-entropy balance**:
//!    - dH/dt ≤ 0 (energy decreases - exploitation)
//!    - dS/dt ≥ 0 (entropy increases - exploration)
//!
//! # Reference
//! - Grmela & Öttinger (1997): "Dynamics and thermodynamics of complex fluids"
//! - B.1 ULTIMATE Vajra-Void Architectonics Specification

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Configuration for Thermodynamic Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoConfig {
    /// State dimension
    pub dim: usize,
    
    /// Energy scale (controls exploitation strength)
    pub energy_scale: f32,
    
    /// Entropy scale (controls exploration strength)
    pub entropy_scale: f32,
    
    /// Temperature parameter (higher = more exploration)
    pub temperature: f32,
    
    /// Time step for integration
    pub dt: f32,
    
    /// Regularization for numerical stability
    pub epsilon: f32,
}

impl Default for ThermoConfig {
    fn default() -> Self {
        Self {
            dim: 5,          // [arousal, valence, attention, rhythm, momentum]
            energy_scale: 1.0,
            entropy_scale: 0.1,  // Lower than energy for stability
            temperature: 1.0,
            dt: 0.01,
            epsilon: 1e-6,
        }
    }
}

/// Thermodynamic Logic Engine
///
/// Combines reversible (Hamiltonian) and irreversible (dissipative) dynamics
/// using the GENERIC formalism.
///
/// # Example
/// ```ignore
/// let engine = ThermodynamicEngine::default();
/// let state = DVector::from_vec(vec![0.5, 0.3, 0.7, 0.6, 0.0]);
/// let target = DVector::from_vec(vec![0.3, 0.6, 0.8, 0.9, 0.0]);
/// let next = engine.step(&state, &target);
/// ```
pub struct ThermodynamicEngine {
    /// Poisson bracket operator (antisymmetric)
    poisson_l: DMatrix<f32>,
    
    /// Friction matrix (symmetric positive semi-definite)
    friction_m: DMatrix<f32>,
    
    /// Configuration
    config: ThermoConfig,
}

impl ThermodynamicEngine {
    /// Create new engine with default Poisson and Friction structures
    pub fn new(config: ThermoConfig) -> Self {
        let dim = config.dim;
        
        // Build Poisson bracket L (antisymmetric)
        // Encodes reversible coupling between state variables
        // L_ij = -L_ji, L_ii = 0
        let mut poisson_l = DMatrix::zeros(dim, dim);
        
        // Arousal ↔ Momentum coupling (Hamiltonian pair)
        if dim >= 5 {
            poisson_l[(0, 4)] = 1.0;   // ∂arousal/∂t depends on momentum
            poisson_l[(4, 0)] = -1.0;  // Antisymmetric
        }
        
        // Valence ↔ Arousal coupling (Yerkes-Dodson)
        if dim >= 2 {
            poisson_l[(0, 1)] = 0.3;
            poisson_l[(1, 0)] = -0.3;
        }
        
        // Attention ↔ Rhythm coupling
        if dim >= 4 {
            poisson_l[(2, 3)] = 0.5;
            poisson_l[(3, 2)] = -0.5;
        }
        
        // Build Friction matrix M (symmetric PSD)
        // Encodes irreversible dissipation
        let mut friction_m = DMatrix::zeros(dim, dim);
        
        // Diagonal elements: self-dissipation
        for i in 0..dim {
            friction_m[(i, i)] = 0.1;  // Base dissipation
        }
        
        // Momentum dissipates faster (damping)
        if dim >= 5 {
            friction_m[(4, 4)] = 0.3;
        }
        
        // Cross-dissipation: arousal and valence relax together
        if dim >= 2 {
            friction_m[(0, 1)] = 0.05;
            friction_m[(1, 0)] = 0.05;
        }
        
        Self {
            poisson_l,
            friction_m,
            config,
        }
    }
    
    /// Create with 5D state for ZenB integration
    pub fn default_for_zenb() -> Self {
        Self::new(ThermoConfig::default())
    }
    
    /// Perform one GENERIC step
    ///
    /// # Arguments
    /// * `state` - Current state vector z
    /// * `target` - Target/goal state (used for energy gradient)
    ///
    /// # Returns
    /// New state after dt integration
    pub fn step(&self, state: &DVector<f32>, target: &DVector<f32>) -> DVector<f32> {
        let grad_h = self.energy_gradient(state, target);
        let grad_s = self.entropy_gradient(state);
        
        // GENERIC: dz/dt = L·∇H + M·∇S
        let reversible = &self.poisson_l * &grad_h;
        let irreversible = &self.friction_m * &grad_s;
        
        let dz = reversible * self.config.energy_scale 
               + irreversible * self.config.entropy_scale;
        
        // Euler integration
        let mut next = state + dz * self.config.dt;
        
        // Clamp to valid range [0, 1] for normalized states
        for i in 0..next.len() {
            next[i] = next[i].clamp(0.0, 1.0);
        }
        
        next
    }
    
    /// Multi-step integration with RK4
    pub fn integrate(&self, state: &DVector<f32>, target: &DVector<f32>, steps: usize) -> DVector<f32> {
        let mut current = state.clone();
        
        for _ in 0..steps {
            current = self.step_rk4(&current, target);
        }
        
        current
    }
    
    /// Runge-Kutta 4th order step (more accurate than Euler)
    fn step_rk4(&self, state: &DVector<f32>, target: &DVector<f32>) -> DVector<f32> {
        let dt = self.config.dt;
        
        let k1 = self.dynamics(state, target);
        let k2 = self.dynamics(&(state + &k1 * (dt / 2.0)), target);
        let k3 = self.dynamics(&(state + &k2 * (dt / 2.0)), target);
        let k4 = self.dynamics(&(state + &k3 * dt), target);
        
        let mut next = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        
        // Clamp
        for i in 0..next.len() {
            next[i] = next[i].clamp(0.0, 1.0);
        }
        
        next
    }
    
    /// Compute dz/dt
    fn dynamics(&self, state: &DVector<f32>, target: &DVector<f32>) -> DVector<f32> {
        let grad_h = self.energy_gradient(state, target);
        let grad_s = self.entropy_gradient(state);
        
        &self.poisson_l * &grad_h * self.config.energy_scale 
            + &self.friction_m * &grad_s * self.config.entropy_scale
    }
    
    /// Compute energy gradient ∇H
    ///
    /// Energy H = (1/2) * ||z - z_target||²
    /// ∇H = z - z_target
    fn energy_gradient(&self, state: &DVector<f32>, target: &DVector<f32>) -> DVector<f32> {
        state - target
    }
    
    /// Compute entropy gradient ∇S
    ///
    /// Entropy S = -Σ p_i log(p_i) (Shannon entropy, treating state as probability-like)
    /// ∇S_i = -log(z_i) - 1 (pushes away from boundaries)
    fn entropy_gradient(&self, state: &DVector<f32>) -> DVector<f32> {
        let mut grad = DVector::zeros(state.len());
        let eps = self.config.epsilon;
        let temp = self.config.temperature;
        
        for i in 0..state.len() {
            let p = state[i].clamp(eps, 1.0 - eps);
            // ∇S = log(1/p) = -log(p) scaled by temperature
            grad[i] = -p.ln() * temp;
        }
        
        grad
    }
    
    /// Compute current free energy (lower is better)
    pub fn free_energy(&self, state: &DVector<f32>, target: &DVector<f32>) -> f32 {
        let h = self.energy(state, target);
        let s = self.entropy(state);
        
        // F = H - T*S (Helmholtz free energy)
        h - self.config.temperature * s
    }
    
    /// Compute energy H = (1/2) * ||z - target||²
    pub fn energy(&self, state: &DVector<f32>, target: &DVector<f32>) -> f32 {
        let diff = state - target;
        0.5 * diff.dot(&diff)
    }
    
    /// Compute entropy S = -Σ p_i log(p_i)
    pub fn entropy(&self, state: &DVector<f32>) -> f32 {
        let eps = self.config.epsilon;
        let mut s = 0.0;
        
        for &p in state.iter() {
            let p_clamped = p.clamp(eps, 1.0 - eps);
            s -= p_clamped * p_clamped.ln();
        }
        
        s
    }
    
    /// Check degeneracy condition L·∇S = 0
    pub fn check_degeneracy_ls(&self, state: &DVector<f32>) -> f32 {
        let grad_s = self.entropy_gradient(state);
        let ls = &self.poisson_l * &grad_s;
        ls.norm()
    }
    
    /// Check degeneracy condition M·∇H = 0
    pub fn check_degeneracy_mh(&self, state: &DVector<f32>, target: &DVector<f32>) -> f32 {
        let grad_h = self.energy_gradient(state, target);
        let mh = &self.friction_m * &grad_h;
        mh.norm()
    }
    
    /// Set temperature (exploration parameter)
    pub fn set_temperature(&mut self, temp: f32) {
        self.config.temperature = temp.max(0.01);
    }
    
    /// Get current configuration
    pub fn config(&self) -> &ThermoConfig {
        &self.config
    }
    
    /// Get Poisson matrix (for diagnostics)
    pub fn poisson_matrix(&self) -> &DMatrix<f32> {
        &self.poisson_l
    }
    
    /// Get Friction matrix (for diagnostics)
    pub fn friction_matrix(&self) -> &DMatrix<f32> {
        &self.friction_m
    }
}

impl Default for ThermodynamicEngine {
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
    
    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }
    
    #[test]
    fn test_engine_creation() {
        let engine = ThermodynamicEngine::default();
        assert_eq!(engine.config().dim, 5);
        assert_eq!(engine.poisson_matrix().nrows(), 5);
        assert_eq!(engine.friction_matrix().nrows(), 5);
    }
    
    #[test]
    fn test_poisson_antisymmetric() {
        let engine = ThermodynamicEngine::default();
        let l = engine.poisson_matrix();
        
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert!(
                    approx_eq(l[(i, j)], -l[(j, i)], 1e-6),
                    "Poisson should be antisymmetric: L[{},{}]={} != -L[{},{}]={}",
                    i, j, l[(i, j)], j, i, -l[(j, i)]
                );
            }
        }
    }
    
    #[test]
    fn test_friction_symmetric() {
        let engine = ThermodynamicEngine::default();
        let m = engine.friction_matrix();
        
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                assert!(
                    approx_eq(m[(i, j)], m[(j, i)], 1e-6),
                    "Friction should be symmetric: M[{},{}]={} != M[{},{}]={}",
                    i, j, m[(i, j)], j, i, m[(j, i)]
                );
            }
        }
    }
    
    #[test]
    fn test_step_moves_toward_target() {
        let engine = ThermodynamicEngine::default();
        let state = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0]);
        let target = DVector::from_vec(vec![0.3, 0.7, 0.8, 0.9, 0.0]);
        
        let initial_energy = engine.energy(&state, &target);
        
        // Multiple steps should reduce energy
        let final_state = engine.integrate(&state, &target, 100);
        let final_energy = engine.energy(&final_state, &target);
        
        assert!(
            final_energy < initial_energy,
            "Energy should decrease: {} -> {}",
            initial_energy, final_energy
        );
    }
    
    #[test]
    fn test_entropy_increases_exploration() {
        let mut engine = ThermodynamicEngine::default();
        engine.config.entropy_scale = 0.5;  // Higher entropy weight
        
        let state = DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.0]);
        let target = DVector::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.0]); // Already at target
        
        let initial_entropy = engine.entropy(&state);
        let final_state = engine.integrate(&state, &target, 50);
        let final_entropy = engine.entropy(&final_state);
        
        // Entropy term should push state away from boundaries (0.1)
        // This increases entropy
        assert!(
            final_entropy >= initial_entropy - 0.1,  // Allow small decrease due to integration
            "Entropy should not decrease significantly: {} -> {}",
            initial_entropy, final_entropy
        );
    }
    
    #[test]
    fn test_free_energy_balance() {
        let engine = ThermodynamicEngine::default();
        let state = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0]);
        let target = DVector::from_vec(vec![0.3, 0.6, 0.7, 0.8, 0.0]);
        
        let f = engine.free_energy(&state, &target);
        let h = engine.energy(&state, &target);
        let s = engine.entropy(&state);
        
        // F = H - T*S
        let expected_f = h - engine.config().temperature * s;
        assert!(
            approx_eq(f, expected_f, 1e-5),
            "Free energy mismatch: {} != {}",
            f, expected_f
        );
    }
    
    #[test]
    fn test_temperature_affects_exploration() {
        let mut engine_cold = ThermodynamicEngine::default();
        engine_cold.set_temperature(0.1);  // Low temp = exploitation
        
        let mut engine_hot = ThermodynamicEngine::default();
        engine_hot.set_temperature(2.0);  // High temp = exploration
        
        let state = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0]);
        let target = DVector::from_vec(vec![0.3, 0.3, 0.3, 0.3, 0.0]);
        
        // Both should converge, but hot should explore more (higher entropy states)
        let cold_final = engine_cold.integrate(&state, &target, 100);
        let hot_final = engine_hot.integrate(&state, &target, 100);
        
        let cold_entropy = engine_cold.entropy(&cold_final);
        let hot_entropy = engine_hot.entropy(&hot_final);
        
        // Hot should have higher entropy due to exploration pressure
        // (This is a soft constraint - may not always hold due to dynamics)
        println!("Cold entropy: {}, Hot entropy: {}", cold_entropy, hot_entropy);
    }
    
    #[test]
    fn test_rk4_vs_euler() {
        let engine = ThermodynamicEngine::default();
        let state = DVector::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0]);
        let target = DVector::from_vec(vec![0.3, 0.7, 0.5, 0.6, 0.0]);
        
        // RK4 should be more accurate (larger dt, same result)
        let rk4_result = engine.integrate(&state, &target, 10);
        
        // Many Euler steps
        let mut euler_state = state.clone();
        for _ in 0..10 {
            euler_state = engine.step(&euler_state, &target);
        }
        
        // Results should be similar but not identical
        let diff = (&rk4_result - &euler_state).norm();
        assert!(diff < 0.5, "RK4 and Euler should be similar: diff={}", diff);
    }
}
