//! Liquid Time-Constant (LTC) Neural Networks for Temporal Reasoning
//!
//! Implementation of "Liquid Time-Constant Networks" (Hasani et al., AAAI 2021)
//! and "Closed-form Continuous-time Models" (Nature Machine Intelligence 2022).
//!
//! # VAJRA V5: Generalized LTC Architecture
//! This module now supports multiple domains beyond breath prediction:
//! - Breath rate prediction (original use case)
//! - Memory coordinate prediction (SaccadeLinker)
//! - Arousal/valence prediction (emotion)
//! - Custom domains via GeneralLtcConfig
//!
//! # Key Innovation
//! Time constants τ adapt based on input, allowing the network to:
//! - Speed up for rapid changes (user begins exercise)
//! - Slow down for gradual transitions (meditation settling)
//! - Handle irregular sampling (common on mobile)
//!
//! # Mathematical Foundation
//! ```text
//! ODE: dx/dt = (-x + f(x, I, t)) / τ(x, I)
//!
//! Where:
//! - x: hidden state
//! - I: input signal
//! - τ(x, I): learnable time constant function
//! - f: nonlinearity (tanh in our case)
//! ```
//!
//! # Closed-Form Continuous (CfC) Extension
//! For edge efficiency, we use the closed-form solution:
//! ```text
//! x(t+Δt) = σ_1 · I + (1 - σ_1) · (σ_2 · f + (1 - σ_2) · x(t))
//! ```
//! where σ_i are sigmoid functions of learnable parameters.
//!
//! # Performance
//! - **10-100x fewer parameters** than RNNs/LSTMs
//! - **Interpretable dynamics**: τ has physical meaning
//! - **Robust to irregular sampling**: no fixed dt assumption

use serde::{Deserialize, Serialize};

/// Configuration for LTC predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcConfig {
    /// Base time constant in seconds
    pub tau_base: f32,
    /// Minimum time constant (fastest adaptation)
    pub tau_min: f32,
    /// Maximum time constant (slowest adaptation)
    pub tau_max: f32,
    /// Learning rate for online adaptation
    pub learning_rate: f32,
    /// Number of hidden neurons
    pub hidden_size: usize,
    /// Input dimension
    pub input_size: usize,
}

impl Default for LtcConfig {
    fn default() -> Self {
        Self {
            tau_base: 3.0, // 3 second base time constant
            tau_min: 0.5,  // 500ms minimum (fast adaptation)
            tau_max: 10.0, // 10 second maximum (slow adaptation)
            learning_rate: 0.01,
            hidden_size: 4, // Small for edge deployment
            input_size: 3,  // [hr_norm, hrv_norm, motion]
        }
    }
}

impl LtcConfig {
    /// Configuration for breath rate prediction
    pub fn for_breath_prediction() -> Self {
        Self {
            tau_base: 2.0,
            tau_min: 0.3,
            tau_max: 8.0,
            learning_rate: 0.02,
            hidden_size: 3,
            input_size: 3,
        }
    }
}

// =============================================================================
// VAJRA V5: Generalized LTC Configuration
// =============================================================================

/// Domain-specific preset for LTC configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LtcDomain {
    /// Breath rate prediction (4-15 BPM range)
    Breath,
    /// Memory coordinate prediction (for SaccadeLinker)
    MemoryCoord,
    /// Arousal prediction (0-1 range)
    Arousal,
    /// Valence prediction (-1 to 1 range)
    Valence,
    /// Generic normalized output (0-1 range)
    Normalized,
}

/// Generalized LTC configuration for any domain
///
/// # VAJRA V5 Feature
/// This struct removes domain-specific hardcoding and allows LTC networks
/// to be configured for any prediction task with custom output ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralLtcConfig {
    /// Base LTC parameters
    pub base: LtcConfig,
    /// Domain identifier
    pub domain: LtcDomain,
    /// Output minimum value
    pub output_min: f32,
    /// Output maximum value
    pub output_max: f32,
    /// Output default value (initial prediction)
    pub output_default: f32,
    /// Label for this predictor (for diagnostics)
    pub label: String,
}

impl Default for GeneralLtcConfig {
    fn default() -> Self {
        Self::for_domain(LtcDomain::Normalized)
    }
}

impl GeneralLtcConfig {
    /// Create configuration for a specific domain
    pub fn for_domain(domain: LtcDomain) -> Self {
        match domain {
            LtcDomain::Breath => Self {
                base: LtcConfig::for_breath_prediction(),
                domain,
                output_min: 4.0,
                output_max: 15.0,
                output_default: 6.0,
                label: "BreathPredictor".to_string(),
            },
            LtcDomain::MemoryCoord => Self {
                base: LtcConfig {
                    tau_base: 1.0,
                    tau_min: 0.1,
                    tau_max: 5.0,
                    learning_rate: 0.05,
                    hidden_size: 4,
                    input_size: 5,
                },
                domain,
                output_min: 0.0,
                output_max: 1.0,
                output_default: 0.5,
                label: "MemoryCoordPredictor".to_string(),
            },
            LtcDomain::Arousal => Self {
                base: LtcConfig {
                    tau_base: 2.0,
                    tau_min: 0.5,
                    tau_max: 10.0,
                    learning_rate: 0.02,
                    hidden_size: 3,
                    input_size: 4,
                },
                domain,
                output_min: 0.0,
                output_max: 1.0,
                output_default: 0.5,
                label: "ArousalPredictor".to_string(),
            },
            LtcDomain::Valence => Self {
                base: LtcConfig {
                    tau_base: 3.0,
                    tau_min: 0.5,
                    tau_max: 15.0,
                    learning_rate: 0.01,
                    hidden_size: 3,
                    input_size: 4,
                },
                domain,
                output_min: -1.0,
                output_max: 1.0,
                output_default: 0.0,
                label: "ValencePredictor".to_string(),
            },
            LtcDomain::Normalized => Self {
                base: LtcConfig::default(),
                domain,
                output_min: 0.0,
                output_max: 1.0,
                output_default: 0.5,
                label: "GenericPredictor".to_string(),
            },
        }
    }

    /// Create custom configuration
    pub fn custom(
        base: LtcConfig,
        output_min: f32,
        output_max: f32,
        output_default: f32,
        label: impl Into<String>,
    ) -> Self {
        Self {
            base,
            domain: LtcDomain::Normalized,
            output_min,
            output_max,
            output_default,
            label: label.into(),
        }
    }
}

/// A single LTC neuron with learnable time constant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcNeuron {
    /// Hidden state
    state: f32,
    /// Base time constant
    tau_base: f32,
    /// Time constant modulation weights (input-dependent τ)
    tau_weights: Vec<f32>,
    /// Time constant bias
    tau_bias: f32,
    /// Input weights
    input_weights: Vec<f32>,
    /// Recurrent weight (self-connection)
    recurrent_weight: f32,
    /// Bias
    bias: f32,
}

impl LtcNeuron {
    /// Create a new LTC neuron
    pub fn new(input_size: usize, tau_base: f32) -> Self {
        // Initialize weights with small random values
        let tau_weights = (0..input_size)
            .map(|i| 0.1 * ((i as f32 * 0.7).sin()))
            .collect();
        let input_weights = (0..input_size)
            .map(|i| 0.1 * ((i as f32 * 1.3).cos()))
            .collect();

        Self {
            state: 0.0,
            tau_base,
            tau_weights,
            tau_bias: 0.0,
            input_weights,
            recurrent_weight: 0.2,
            bias: 0.0,
        }
    }

    /// Compute input-dependent time constant
    fn compute_tau(&self, inputs: &[f32], tau_min: f32, tau_max: f32) -> f32 {
        // τ(x, I) = τ_base * sigmoid(w_τ · I + b_τ)
        let linear: f32 = self
            .tau_weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum::<f32>()
            + self.tau_bias;

        let sigmoid = 1.0 / (1.0 + (-linear).exp());

        // Scale to [tau_min, tau_max]
        tau_min + sigmoid * (tau_max - tau_min)
    }

    /// Compute pre-activation
    fn pre_activation(&self, inputs: &[f32]) -> f32 {
        let input_sum: f32 = self
            .input_weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum();

        input_sum + self.recurrent_weight * self.state + self.bias
    }

    /// Advance state using Euler integration
    ///
    /// ODE: dx/dt = (-x + tanh(w_in · I + w_rec · x + b)) / τ(I)
    pub fn step(&mut self, inputs: &[f32], dt: f32, tau_min: f32, tau_max: f32) -> f32 {
        let tau = self.compute_tau(inputs, tau_min, tau_max);
        let pre_act = self.pre_activation(inputs);
        let target = pre_act.tanh();

        // Euler integration: x(t+dt) = x(t) + dt * dx/dt
        let dx = (-self.state + target) / tau;
        self.state += dx * dt;

        // Clamp to prevent numerical issues
        self.state = self.state.clamp(-5.0, 5.0);

        self.state
    }

    /// Closed-form continuous-time step (more stable)
    ///
    /// Uses the exponential solution for better numerical stability:
    /// x(t+dt) = target + (x(t) - target) * exp(-dt/τ)
    pub fn step_cfc(&mut self, inputs: &[f32], dt: f32, tau_min: f32, tau_max: f32) -> f32 {
        let tau = self.compute_tau(inputs, tau_min, tau_max);
        let pre_act = self.pre_activation(inputs);
        let target = pre_act.tanh();

        // Closed-form solution (more stable than Euler)
        let decay = (-dt / tau).exp();
        self.state = target + (self.state - target) * decay;

        self.state
    }

    /// Get current time constant (for diagnostics)
    pub fn current_tau(&self, inputs: &[f32], tau_min: f32, tau_max: f32) -> f32 {
        self.compute_tau(inputs, tau_min, tau_max)
    }

    /// Get current state
    pub fn state(&self) -> f32 {
        self.state
    }

    /// Reset state to zero
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

/// LTC-based breath rate predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcBreathPredictor {
    config: LtcConfig,
    /// Hidden layer neurons
    hidden: Vec<LtcNeuron>,
    /// Output weights
    output_weights: Vec<f32>,
    /// Output bias
    output_bias: f32,
    /// Last prediction (for smoothing)
    last_prediction: f32,
    /// Prediction error accumulator (for learning)
    error_ema: f32,
    /// Total steps
    step_count: u64,
}

impl LtcBreathPredictor {
    /// Create a new LTC breath predictor
    pub fn new(config: LtcConfig) -> Self {
        let hidden: Vec<LtcNeuron> = (0..config.hidden_size)
            .map(|i| {
                let tau = config.tau_base * (1.0 + 0.2 * (i as f32));
                LtcNeuron::new(config.input_size, tau)
            })
            .collect();

        let output_weights: Vec<f32> = (0..config.hidden_size)
            .map(|i| 0.3 / (config.hidden_size as f32) * (1.0 + 0.1 * (i as f32)))
            .collect();

        Self {
            config,
            hidden,
            output_weights,
            output_bias: 6.0, // Default to 6 BPM (typical breathing rate)
            last_prediction: 6.0,
            error_ema: 0.0,
            step_count: 0,
        }
    }

    /// Create with default breath prediction config
    pub fn default_for_breath() -> Self {
        Self::new(LtcConfig::for_breath_prediction())
    }

    /// Predict breath rate given sensor inputs
    ///
    /// # Arguments
    /// * `inputs` - [hr_norm (0-1), hrv_norm (0-1), motion (0-1)]
    /// * `dt` - Time delta in seconds since last call
    ///
    /// # Returns
    /// Predicted breath rate in BPM
    pub fn predict(&mut self, inputs: &[f32], dt: f32) -> f32 {
        // Advance all hidden neurons
        for neuron in &mut self.hidden {
            neuron.step_cfc(inputs, dt, self.config.tau_min, self.config.tau_max);
        }

        // Compute output (weighted sum of hidden states)
        let hidden_sum: f32 = self
            .hidden
            .iter()
            .zip(self.output_weights.iter())
            .map(|(n, w)| n.state() * w)
            .sum();

        // Output transformation: 4-12 BPM range
        let raw_output = hidden_sum + self.output_bias;
        let prediction = raw_output.clamp(4.0, 15.0);

        // Smooth output
        let alpha = (dt / (dt + 0.5)).clamp(0.0, 1.0);
        self.last_prediction = self.last_prediction * (1.0 - alpha) + prediction * alpha;

        self.step_count += 1;
        self.last_prediction
    }

    /// Update predictor based on actual measured breath rate
    ///
    /// # Arguments
    /// * `actual_rr` - Actual measured respiration rate in BPM
    /// * `inputs` - Input features at time of measurement
    pub fn learn_from_measurement(&mut self, actual_rr: f32, _inputs: &[f32]) {
        if actual_rr <= 0.0 || actual_rr > 30.0 {
            return; // Invalid measurement
        }

        let error = actual_rr - self.last_prediction;

        // Update error EMA
        self.error_ema = 0.9 * self.error_ema + 0.1 * error.abs();

        // Simple online learning: adjust output bias and weights
        let lr = self.config.learning_rate;

        // Update output bias
        self.output_bias += lr * error;

        // Update output weights proportionally to hidden states
        for (i, neuron) in self.hidden.iter().enumerate() {
            let state = neuron.state();
            self.output_weights[i] += lr * error * state * 0.1;
            self.output_weights[i] = self.output_weights[i].clamp(-2.0, 2.0);
        }

        // Adapt time constants based on prediction quality
        // If errors are large, reduce tau (faster adaptation)
        // If errors are small, increase tau (more stability)
        if self.error_ema > 1.5 {
            // Large errors: need faster adaptation
            for neuron in &mut self.hidden {
                neuron.tau_bias -= lr * 0.5;
            }
        } else if self.error_ema < 0.3 && self.step_count > 100 {
            // Good predictions: can afford slower, smoother dynamics
            for neuron in &mut self.hidden {
                neuron.tau_bias += lr * 0.2;
            }
        }
    }

    /// Get diagnostic information
    ///
    /// # Returns
    /// (current_prediction, average_tau, error_ema, step_count)
    pub fn diagnostics(&self, inputs: &[f32]) -> (f32, f32, f32, u64) {
        let avg_tau: f32 = self
            .hidden
            .iter()
            .map(|n| n.current_tau(inputs, self.config.tau_min, self.config.tau_max))
            .sum::<f32>()
            / self.hidden.len() as f32;

        (
            self.last_prediction,
            avg_tau,
            self.error_ema,
            self.step_count,
        )
    }

    /// Reset predictor state
    pub fn reset(&mut self) {
        for neuron in &mut self.hidden {
            neuron.reset();
        }
        self.last_prediction = 6.0;
        self.error_ema = 0.0;
    }

    /// Get last prediction
    pub fn last_prediction(&self) -> f32 {
        self.last_prediction
    }
}

// =============================================================================
// VAJRA V5: Generalized LTC Predictor
// =============================================================================

/// Domain-agnostic LTC predictor using GeneralLtcConfig
///
/// # VAJRA V5 Feature
/// This struct provides a unified predictor interface for any domain,
/// with configurable output ranges and automatic scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralLtcPredictor {
    config: GeneralLtcConfig,
    /// Hidden layer neurons
    hidden: Vec<LtcNeuron>,
    /// Output weights
    output_weights: Vec<f32>,
    /// Output bias (in normalized space)
    output_bias: f32,
    /// Last prediction
    last_prediction: f32,
    /// Prediction error accumulator
    error_ema: f32,
    /// Total steps
    step_count: u64,
}

impl GeneralLtcPredictor {
    /// Create predictor for a specific domain
    pub fn new(config: GeneralLtcConfig) -> Self {
        let hidden: Vec<LtcNeuron> = (0..config.base.hidden_size)
            .map(|i| {
                let tau = config.base.tau_base * (1.0 + 0.2 * (i as f32));
                LtcNeuron::new(config.base.input_size, tau)
            })
            .collect();

        let output_weights: Vec<f32> = (0..config.base.hidden_size)
            .map(|i| 0.3 / (config.base.hidden_size as f32) * (1.0 + 0.1 * (i as f32)))
            .collect();

        let last_pred = config.output_default;
        
        Self {
            config,
            hidden,
            output_weights,
            output_bias: 0.0, // Normalized space
            last_prediction: last_pred,
            error_ema: 0.0,
            step_count: 0,
        }
    }

    /// Create predictor for a domain preset
    pub fn for_domain(domain: LtcDomain) -> Self {
        Self::new(GeneralLtcConfig::for_domain(domain))
    }

    /// Predict output value given inputs
    ///
    /// # Arguments
    /// * `inputs` - Input features (must match config.base.input_size)
    /// * `dt` - Time delta in seconds since last call
    ///
    /// # Returns
    /// Prediction in the configured output range
    pub fn predict(&mut self, inputs: &[f32], dt: f32) -> f32 {
        // Advance all hidden neurons
        for neuron in &mut self.hidden {
            neuron.step_cfc(inputs, dt, self.config.base.tau_min, self.config.base.tau_max);
        }

        // Compute output (weighted sum of hidden states)
        let hidden_sum: f32 = self.hidden
            .iter()
            .zip(self.output_weights.iter())
            .map(|(n, w)| n.state() * w)
            .sum();

        // Transform to output range: [min, max]
        let normalized = (hidden_sum + self.output_bias).tanh() * 0.5 + 0.5; // [0, 1]
        let range = self.config.output_max - self.config.output_min;
        let raw_output = self.config.output_min + normalized * range;
        let prediction = raw_output.clamp(self.config.output_min, self.config.output_max);

        // Smooth output
        let alpha = (dt / (dt + 0.5)).clamp(0.0, 1.0);
        self.last_prediction = self.last_prediction * (1.0 - alpha) + prediction * alpha;

        self.step_count += 1;
        self.last_prediction
    }

    /// Update predictor based on actual measurement
    pub fn learn(&mut self, actual: f32, _inputs: &[f32]) {
        if actual < self.config.output_min || actual > self.config.output_max {
            return; // Invalid measurement
        }

        let error = actual - self.last_prediction;
        
        // Update error EMA
        self.error_ema = 0.9 * self.error_ema + 0.1 * error.abs();

        // Normalize error to output range for learning
        let range = self.config.output_max - self.config.output_min;
        let normalized_error = error / range.max(1e-6);
        let lr = self.config.base.learning_rate;

        // Update output bias
        self.output_bias += lr * normalized_error;

        // Update output weights
        for (i, neuron) in self.hidden.iter().enumerate() {
            let state = neuron.state();
            self.output_weights[i] += lr * normalized_error * state * 0.1;
            self.output_weights[i] = self.output_weights[i].clamp(-2.0, 2.0);
        }
    }

    /// Get domain label
    pub fn label(&self) -> &str {
        &self.config.label
    }

    /// Get diagnostics
    pub fn diagnostics(&self, inputs: &[f32]) -> (f32, f32, f32, u64) {
        let avg_tau: f32 = self.hidden
            .iter()
            .map(|n| n.current_tau(inputs, self.config.base.tau_min, self.config.base.tau_max))
            .sum::<f32>() / self.hidden.len() as f32;

        (self.last_prediction, avg_tau, self.error_ema, self.step_count)
    }

    /// Reset predictor state
    pub fn reset(&mut self) {
        for neuron in &mut self.hidden {
            neuron.reset();
        }
        self.last_prediction = self.config.output_default;
        self.error_ema = 0.0;
    }

    /// Get last prediction
    pub fn last_prediction(&self) -> f32 {
        self.last_prediction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_neuron_dynamics() {
        let mut neuron = LtcNeuron::new(3, 2.0);

        // Step through time with constant input
        let inputs = [0.5, 0.5, 0.0]; // Rest state
        for _ in 0..100 {
            neuron.step_cfc(&inputs, 0.1, 0.5, 10.0);
        }

        // State should converge to some equilibrium
        let final_state = neuron.state();
        assert!(final_state.abs() < 1.0, "State should be bounded");
    }

    #[test]
    fn test_tau_adaptation() {
        let neuron = LtcNeuron::new(3, 2.0);

        // Rest state should have higher tau
        let rest_inputs = [0.3, 0.6, 0.0];
        let rest_tau = neuron.current_tau(&rest_inputs, 0.5, 10.0);

        // High motion should have lower tau
        let motion_inputs = [0.5, 0.3, 0.9];
        let motion_tau = neuron.current_tau(&motion_inputs, 0.5, 10.0);

        // Both should be in valid range
        assert!(rest_tau >= 0.5 && rest_tau <= 10.0);
        assert!(motion_tau >= 0.5 && motion_tau <= 10.0);
    }

    #[test]
    fn test_breath_predictor_convergence() {
        let mut predictor = LtcBreathPredictor::default_for_breath();

        // Simulate calm breathing session
        let calm_inputs = [0.3, 0.6, 0.0]; // Low HR, high HRV, no motion

        for _ in 0..100 {
            let _ = predictor.predict(&calm_inputs, 0.5);
        }

        let prediction = predictor.last_prediction();

        // Prediction should be in valid range
        assert!(
            prediction >= 4.0 && prediction <= 15.0,
            "Prediction {} should be in valid BPM range",
            prediction
        );
    }

    #[test]
    fn test_learning_from_measurement() {
        let mut predictor = LtcBreathPredictor::default_for_breath();

        let inputs = [0.35, 0.5, 0.1];

        // Get initial prediction
        let initial = predictor.predict(&inputs, 0.5);

        // Learn from actual measurement (lower than default)
        for _ in 0..20 {
            predictor.predict(&inputs, 0.5);
            predictor.learn_from_measurement(5.0, &inputs);
        }

        let after_learning = predictor.last_prediction();

        // Should have moved toward actual measurement
        assert!(
            (after_learning - 5.0).abs() < (initial - 5.0).abs(),
            "Should learn toward actual: initial={}, after={}, target=5.0",
            initial,
            after_learning
        );
    }

    #[test]
    fn test_irregular_sampling() {
        let mut predictor = LtcBreathPredictor::default_for_breath();
        let inputs = [0.4, 0.5, 0.0];

        // Simulate irregular sampling
        let dts = [0.1, 0.5, 0.2, 1.0, 0.3, 2.0];

        for &dt in &dts {
            let pred = predictor.predict(&inputs, dt);
            // Should never NaN or explode
            assert!(pred.is_finite(), "Prediction must be finite for dt={}", dt);
            assert!(pred >= 4.0 && pred <= 15.0, "Prediction out of range");
        }
    }
}
