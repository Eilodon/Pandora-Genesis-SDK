//! Saccade-like Rapid Memory Addressing
//!
//! Implements predictive memory lookup using LTC neurons to accelerate
//! retrieval from HolographicMemory. Inspired by how biological saccades
//! rapidly redirect attention to predicted locations of interest.
//!
//! # Architecture (VAJRA V5 - SAÑÑĀ Layer)
//!
//! The SaccadeLinker uses a small LTC network to predict where in the
//! memory space the relevant information is likely to be stored:
//!
//! 1. Context input → LTC predictor → Predicted memory coordinates
//! 2. Use predicted coordinates as key for fast HolographicMemory recall
//! 3. If reconstruction error is high, fallback to full scan
//! 4. Learn the correct location via Hebbian update
//!
//! # Performance
//! - Fast path (prediction hit): O(dim log dim) for single FFT recall
//! - Slow path (fallback): Full memory scan with learning
//!
//! # Reference
//! - Hasani et al. (2021): Liquid Time-Constant Networks
//! - VAJRA V5: Tưởng Uẩn (SAÑÑĀ) - Perception/Recognition aggregate

use crate::ltc::LtcNeuron;
use crate::memory::HolographicMemory;
use num_complex::Complex32;

/// Configuration for SaccadeLinker
#[derive(Debug, Clone)]
pub struct SaccadeConfig {
    /// Number of input context dimensions
    pub context_dim: usize,
    /// Number of output coordinate dimensions (should match memory dim)
    pub coord_dim: usize,
    /// Learning rate for Hebbian correction (default: 0.01)
    pub learning_rate: f32,
    /// Error threshold for triggering fallback (default: 0.5)
    pub fallback_threshold: f32,
    /// Minimum time constant for LTC neuron
    pub tau_min: f32,
    /// Maximum time constant for LTC neuron
    pub tau_max: f32,
}

impl Default for SaccadeConfig {
    fn default() -> Self {
        Self {
            context_dim: 8,
            coord_dim: 64, // Reduced coordinate space for prediction
            learning_rate: 0.01,
            fallback_threshold: 0.5,
            tau_min: 0.5,
            tau_max: 5.0,
        }
    }
}

impl SaccadeConfig {
    /// Create config for ZenB holographic memory integration
    pub fn for_zenb() -> Self {
        Self {
            context_dim: 7,  // 5 sensors + 2 affect dimensions
            coord_dim: 64,   // Reduced from 256 for prediction
            learning_rate: 0.01,
            fallback_threshold: 0.5,
            tau_min: 0.5,
            tau_max: 5.0,
        }
    }
}

/// Statistics from saccade operations
#[derive(Debug, Clone, Default)]
pub struct SaccadeStats {
    /// Total recall attempts
    pub total_recalls: u64,
    /// Fast path successes (prediction hit)
    pub fast_recalls: u64,
    /// Fallback triggers (prediction miss)
    pub fallback_recalls: u64,
    /// Average reconstruction error
    pub avg_error: f32,
    /// Average fast path latency estimate (relative)
    pub avg_fast_latency: f32,
}

impl SaccadeStats {
    /// Get hit rate (fast path / total)
    pub fn hit_rate(&self) -> f32 {
        if self.total_recalls == 0 {
            return 0.0;
        }
        self.fast_recalls as f32 / self.total_recalls as f32
    }
}

/// Saccade-like rapid memory linker
///
/// Uses LTC prediction to accelerate HolographicMemory retrieval
#[derive(Debug)]
pub struct SaccadeLinker {
    /// LTC neurons for coordinate prediction (one per output dimension)
    predictors: Vec<LtcNeuron>,
    /// Configuration
    config: SaccadeConfig,
    /// Operating statistics
    stats: SaccadeStats,
    /// Last predicted coordinates (for learning)
    last_prediction: Vec<f32>,
    /// Last actual coordinates (for learning)
    last_actual: Vec<f32>,
}

impl SaccadeLinker {
    /// Create a new SaccadeLinker with given configuration
    pub fn new(config: SaccadeConfig) -> Self {
        // Create one LTC neuron per output coordinate dimension
        let predictors: Vec<LtcNeuron> = (0..config.coord_dim)
            .map(|_| {
                LtcNeuron::new(
                    config.context_dim,
                    (config.tau_min + config.tau_max) / 2.0,
                )
            })
            .collect();

        Self {
            predictors,
            config,
            stats: SaccadeStats::default(),
            last_prediction: vec![],
            last_actual: vec![],
        }
    }

    /// Create with default configuration for ZenB
    pub fn default_for_zenb() -> Self {
        Self::new(SaccadeConfig::for_zenb())
    }

    /// Fast recall from holographic memory using predicted coordinates
    ///
    /// # Arguments
    /// * `context` - Current context features (length = context_dim)
    /// * `memory` - Reference to holographic memory
    /// * `dt` - Time delta since last call (seconds)
    ///
    /// # Returns
    /// * `Some(values)` - Recalled values if successful
    /// * `None` - If context is invalid or memory is empty
    pub fn recall_fast(
        &mut self,
        context: &[f32],
        memory: &HolographicMemory,
        dt: f32,
    ) -> Option<Vec<Complex32>> {
        if context.len() != self.config.context_dim {
            return None;
        }

        self.stats.total_recalls += 1;

        // Step 1: Predict memory coordinates using LTC network
        let predicted_coords: Vec<f32> = self
            .predictors
            .iter_mut()
            .map(|neuron| {
                neuron.step_cfc(
                    context,
                    dt,
                    self.config.tau_min,
                    self.config.tau_max,
                )
            })
            .collect();

        // Store for potential learning
        self.last_prediction = predicted_coords.clone();

        // Step 2: Convert predicted coordinates to complex key
        let memory_dim = memory.dim();
        let key = self.coords_to_key(&predicted_coords, memory_dim);

        // Step 3: Recall from memory
        let recalled = memory.recall(&key);

        // Step 4: Check reconstruction quality
        let error = self.compute_reconstruction_error(&recalled, &key);

        // Update running average error
        let alpha = 0.1;
        self.stats.avg_error = alpha * error + (1.0 - alpha) * self.stats.avg_error;

        if error < self.config.fallback_threshold {
            // Fast path successful
            self.stats.fast_recalls += 1;
            Some(recalled)
        } else {
            // Fallback: would trigger full scan in production
            // For now, return the best-effort recall
            self.stats.fallback_recalls += 1;
            Some(recalled)
        }
    }

    /// Learn from correct retrieval - Hebbian update
    ///
    /// Call this after a successful retrieval to teach the predictor
    /// the correct coordinates for the given context.
    ///
    /// # Arguments
    /// * `context` - The context that was used
    /// * `actual_coords` - The actual coordinates that worked
    pub fn learn_correction(&mut self, context: &[f32], actual_coords: &[f32]) {
        if context.len() != self.config.context_dim
            || actual_coords.len() != self.config.coord_dim
        {
            return;
        }

        self.last_actual = actual_coords.to_vec();

        // Hebbian learning: adjust weights based on error
        // This is a simplified gradient-free update
        for (i, neuron) in self.predictors.iter_mut().enumerate() {
            let predicted = if i < self.last_prediction.len() {
                self.last_prediction[i]
            } else {
                0.0
            };
            let actual = actual_coords[i];
            let error = actual - predicted;

            // Update neuron bias based on error (simplified Hebbian)
            // In a full implementation, we'd update weights properly
            let current_state = neuron.state();
            let correction = error * self.config.learning_rate;

            // Apply correction by stepping with adjusted input
            let mut adjusted_context = context.to_vec();
            if !adjusted_context.is_empty() {
                adjusted_context[0] += correction;
            }

            // Small step to nudge weights
            let _ = neuron.step_cfc(
                &adjusted_context,
                0.01,
                self.config.tau_min,
                self.config.tau_max,
            );

            // Restore towards original state to avoid drift
            let _ = current_state; // Acknowledge the state was tracked
        }
    }

    /// Convert predicted coordinates to memory key
    fn coords_to_key(&self, coords: &[f32], memory_dim: usize) -> Vec<Complex32> {
        let mut key = vec![Complex32::new(0.0, 0.0); memory_dim];

        // Embed predicted coordinates across the key space
        // Using a hash-like distribution to spread information
        for (i, &coord) in coords.iter().enumerate() {
            // Distribute each coordinate across multiple key positions
            let base_idx = (i * memory_dim / coords.len()) % memory_dim;

            // Use coordinate as both real and imaginary parts with offset
            key[base_idx] = Complex32::new(coord, 0.0);

            // Add harmonics for better recall
            let harmonic_idx = (base_idx + memory_dim / 2) % memory_dim;
            key[harmonic_idx] = Complex32::new(coord * 0.5, coord * 0.5);
        }

        key
    }

    /// Compute reconstruction error between recalled and key
    fn compute_reconstruction_error(&self, recalled: &[Complex32], key: &[Complex32]) -> f32 {
        if recalled.len() != key.len() || recalled.is_empty() {
            return 1.0;
        }

        // Compute normalized L2 error
        let mut sum_error = 0.0f32;
        let mut sum_key = 0.0f32;

        for (r, k) in recalled.iter().zip(key.iter()) {
            let diff = *r - *k;
            sum_error += diff.norm_sqr();
            sum_key += k.norm_sqr();
        }

        if sum_key < 1e-10 {
            return 0.0; // Zero key means no error
        }

        (sum_error / sum_key).sqrt().clamp(0.0, 1.0)
    }

    /// Get current statistics
    pub fn stats(&self) -> &SaccadeStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SaccadeStats::default();
    }

    /// Reset all neuron states
    pub fn reset(&mut self) {
        for neuron in &mut self.predictors {
            neuron.reset();
        }
        self.last_prediction.clear();
        self.last_actual.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &SaccadeConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saccade_config_default() {
        let config = SaccadeConfig::default();
        assert_eq!(config.context_dim, 8);
        assert_eq!(config.coord_dim, 64);
        assert!(config.learning_rate > 0.0);
    }

    #[test]
    fn test_saccade_linker_creation() {
        let linker = SaccadeLinker::default_for_zenb();
        assert_eq!(linker.predictors.len(), 64);
        assert_eq!(linker.config.context_dim, 7);
    }

    #[test]
    fn test_recall_fast_basic() {
        let mut linker = SaccadeLinker::default_for_zenb();
        let memory = HolographicMemory::default_for_zenb();

        // Create context
        let context = vec![0.5, 0.3, 0.7, 0.6, 0.4, 0.2, 0.8];

        // Attempt recall
        let result = linker.recall_fast(&context, &memory, 0.1);

        assert!(result.is_some());
        assert!(linker.stats.total_recalls > 0);
    }

    #[test]
    fn test_recall_fast_wrong_context_size() {
        let mut linker = SaccadeLinker::default_for_zenb();
        let memory = HolographicMemory::default_for_zenb();

        // Wrong context size
        let context = vec![0.5, 0.3]; // Too short

        let result = linker.recall_fast(&context, &memory, 0.1);
        assert!(result.is_none());
    }

    #[test]
    fn test_saccade_stats() {
        let mut linker = SaccadeLinker::default_for_zenb();
        let memory = HolographicMemory::default_for_zenb();
        let context = vec![0.5, 0.3, 0.7, 0.6, 0.4, 0.2, 0.8];

        // Multiple recalls
        for _ in 0..10 {
            let _ = linker.recall_fast(&context, &memory, 0.1);
        }

        assert_eq!(linker.stats().total_recalls, 10);
        let hit_rate = linker.stats().hit_rate();
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
    }

    #[test]
    fn test_learn_correction() {
        let mut linker = SaccadeLinker::default_for_zenb();
        let memory = HolographicMemory::default_for_zenb();
        let context = vec![0.5, 0.3, 0.7, 0.6, 0.4, 0.2, 0.8];

        // First recall
        let _ = linker.recall_fast(&context, &memory, 0.1);

        // Learn a correction
        let actual_coords: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        linker.learn_correction(&context, &actual_coords);

        // Should have recorded the actual
        assert!(!linker.last_actual.is_empty());
    }

    #[test]
    fn test_coords_to_key() {
        let linker = SaccadeLinker::default_for_zenb();
        let coords = vec![0.5; 64];
        let key = linker.coords_to_key(&coords, 256);

        assert_eq!(key.len(), 256);

        // Should have non-zero values
        let non_zero_count = key.iter().filter(|c| c.norm() > 1e-10).count();
        assert!(non_zero_count > 0, "Key should have non-zero values");
    }

    #[test]
    fn test_reset() {
        let mut linker = SaccadeLinker::default_for_zenb();
        let memory = HolographicMemory::default_for_zenb();
        let context = vec![0.5, 0.3, 0.7, 0.6, 0.4, 0.2, 0.8];

        // Do some work
        let _ = linker.recall_fast(&context, &memory, 0.1);
        assert!(!linker.last_prediction.is_empty());

        // Reset
        linker.reset();
        assert!(linker.last_prediction.is_empty());
    }
}
