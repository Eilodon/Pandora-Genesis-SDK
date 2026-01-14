//! Uncertainty-Aware Memory Retrieval (ZENITH Tier 4)
//!
//! Provides ensemble-based uncertainty quantification for memory retrieval operations.
//! Based on 2025-2026 research on epistemic vs aleatoric uncertainty separation.
//!
//! # Key Features
//! - **Ensemble retrieval**: Multiple queries with perturbations for variance estimation
//! - **Epistemic uncertainty**: Model/parameter uncertainty (reducible with more data)
//! - **Aleatoric uncertainty**: Data noise (irreducible)
//! - **Confidence thresholding**: Automatic flagging of uncertain predictions
//!
//! # Example
//! ```ignore
//! use zenb_core::memory::UncertaintyAwareRetrieval;
//!
//! let uq = UncertaintyAwareRetrieval::default();
//! let result = uq.retrieve_with_uncertainty(&query, |q| memory.retrieve(q));
//!
//! if result.epistemic_uncertainty > 0.5 {
//!     println!("High model uncertainty - need more data!");
//! }
//! ```

use crate::uncertain::Uncertain;

/// Configuration for uncertainty-aware retrieval
#[derive(Debug, Clone)]
pub struct UncertaintyRetrievalConfig {
    /// Number of ensemble members (perturbations)
    pub ensemble_size: usize,
    /// Standard deviation of query perturbations
    pub noise_std: f32,
    /// Minimum confidence to return a result
    pub min_confidence: f32,
}

impl Default for UncertaintyRetrievalConfig {
    fn default() -> Self {
        Self {
            ensemble_size: 5,
            noise_std: 0.01,
            min_confidence: 0.6,
        }
    }
}

/// Result of uncertainty-aware retrieval
#[derive(Debug, Clone)]
pub struct UncertainRetrievalResult<T> {
    /// The retrieved value (mean of ensemble)
    pub value: T,
    /// Overall confidence score
    pub confidence: f32,
    /// Epistemic uncertainty (model uncertainty, from ensemble variance)
    pub epistemic_uncertainty: f32,
    /// Aleatoric uncertainty (data noise, from confidence spread)
    pub aleatoric_uncertainty: f32,
}

impl<T> UncertainRetrievalResult<T> {
    /// Check if result is reliable for autonomous action
    pub fn is_reliable(&self, confidence_threshold: f32, epistemic_threshold: f32) -> bool {
        self.confidence >= confidence_threshold && self.epistemic_uncertainty <= epistemic_threshold
    }

    /// Total uncertainty
    pub fn total_uncertainty(&self) -> f32 {
        (self.epistemic_uncertainty + self.aleatoric_uncertainty).min(1.0)
    }

    /// Convert to Uncertain<T> type
    pub fn into_uncertain(self, source: impl Into<String>) -> Uncertain<T> {
        Uncertain::with_uncertainty(
            self.value,
            self.confidence,
            self.epistemic_uncertainty,
            self.aleatoric_uncertainty,
            source,
        )
    }
}

/// Uncertainty-aware retrieval wrapper
///
/// Wraps any retrieval function to add uncertainty quantification
/// via ensemble perturbation.
#[derive(Debug, Clone)]
pub struct UncertaintyAwareRetrieval {
    config: UncertaintyRetrievalConfig,
}

impl Default for UncertaintyAwareRetrieval {
    fn default() -> Self {
        Self::new(UncertaintyRetrievalConfig::default())
    }
}

impl UncertaintyAwareRetrieval {
    /// Create with custom configuration
    pub fn new(config: UncertaintyRetrievalConfig) -> Self {
        Self { config }
    }

    /// Retrieve with uncertainty estimation
    ///
    /// Performs multiple retrievals with query perturbations and computes:
    /// - Mean value from ensemble
    /// - Epistemic uncertainty from value variance
    /// - Aleatoric uncertainty from confidence variance
    ///
    /// # Arguments
    /// * `query` - Query vector (f32 slice)
    /// * `retrieval_fn` - Function that performs actual retrieval
    ///
    /// # Returns
    /// `Some(UncertainRetrievalResult)` if above confidence threshold, `None` otherwise
    pub fn retrieve_with_uncertainty<F>(
        &self,
        query: &[f32],
        mut retrieval_fn: F,
    ) -> Option<UncertainRetrievalResult<Vec<f32>>>
    where
        F: FnMut(&[f32]) -> Option<(Vec<f32>, f32)>,
    {
        let mut results: Vec<(Vec<f32>, f32)> = Vec::with_capacity(self.config.ensemble_size);

        // Perform ensemble retrievals
        for i in 0..self.config.ensemble_size {
            let perturbed = add_noise(query, self.config.noise_std, i as u64);
            if let Some((value, confidence)) = retrieval_fn(&perturbed) {
                results.push((value, confidence));
            }
        }

        if results.is_empty() {
            return None;
        }

        // Compute mean value
        let dim = results[0].0.len();
        let mut mean_value = vec![0.0f32; dim];
        for (value, _) in &results {
            for (i, &v) in value.iter().enumerate() {
                if i < dim {
                    mean_value[i] += v;
                }
            }
        }
        let n = results.len() as f32;
        for v in &mut mean_value {
            *v /= n;
        }

        // Compute epistemic uncertainty (variance of values)
        let mut variance_sum = 0.0f32;
        for (value, _) in &results {
            for (i, &v) in value.iter().enumerate() {
                if i < dim {
                    let diff = v - mean_value[i];
                    variance_sum += diff * diff;
                }
            }
        }
        let epistemic_uncertainty = (variance_sum / (dim as f32 * n)).sqrt();

        // Compute mean confidence
        let mean_confidence: f32 = results.iter().map(|(_, c)| c).sum::<f32>() / n;

        // Compute aleatoric uncertainty (variance of confidences)
        let confidence_variance: f32 = results
            .iter()
            .map(|(_, c)| (c - mean_confidence).powi(2))
            .sum::<f32>()
            / n;
        let aleatoric_uncertainty = confidence_variance.sqrt();

        // Check threshold
        if mean_confidence >= self.config.min_confidence {
            Some(UncertainRetrievalResult {
                value: mean_value,
                confidence: mean_confidence,
                epistemic_uncertainty: epistemic_uncertainty.min(1.0),
                aleatoric_uncertainty: aleatoric_uncertainty.min(1.0),
            })
        } else {
            None
        }
    }

    /// Get configuration
    pub fn config(&self) -> &UncertaintyRetrievalConfig {
        &self.config
    }
}

/// Add deterministic noise to a vector
fn add_noise(vec: &[f32], std_dev: f32, seed: u64) -> Vec<f32> {
    // Simple xorshift-based deterministic noise
    let mut state = seed.wrapping_add(12345);
    vec.iter()
        .map(|&x| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-0.5, 0.5] and scale by std_dev
            let noise = ((state % 1000) as f32 / 1000.0 - 0.5) * std_dev * 2.0;
            x + noise
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_retrieval_basic() {
        let uq = UncertaintyAwareRetrieval::default();

        // Mock retrieval that always returns same value
        let retrieval_fn = |_q: &[f32]| -> Option<(Vec<f32>, f32)> {
            Some((vec![0.5, 0.6, 0.7], 0.9))
        };

        let query = vec![1.0, 2.0, 3.0];
        let result = uq.retrieve_with_uncertainty(&query, retrieval_fn);

        assert!(result.is_some());
        let r = result.unwrap();
        assert!(r.confidence > 0.8);
        // Low epistemic since values are consistent
        assert!(r.epistemic_uncertainty < 0.1);
        // Low aleatoric since confidence is consistent
        assert!(r.aleatoric_uncertainty < 0.1);
    }

    #[test]
    fn test_uncertainty_retrieval_high_variance() {
        let uq = UncertaintyAwareRetrieval::new(UncertaintyRetrievalConfig {
            ensemble_size: 5,
            noise_std: 0.5, // High noise for testing
            min_confidence: 0.3,
        });

        // Mock retrieval that varies based on query
        let retrieval_fn = |q: &[f32]| -> Option<(Vec<f32>, f32)> {
            // Confidence varies with query
            let conf = (q[0] * 10.0).fract() * 0.5 + 0.4;
            Some((q.to_vec(), conf))
        };

        let query = vec![0.5, 0.5, 0.5];
        let result = uq.retrieve_with_uncertainty(&query, retrieval_fn);

        assert!(result.is_some());
        let r = result.unwrap();
        // With high noise, expect higher epistemic uncertainty
        assert!(r.epistemic_uncertainty > 0.0);
    }

    #[test]
    fn test_uncertainty_below_threshold() {
        let uq = UncertaintyAwareRetrieval::new(UncertaintyRetrievalConfig {
            ensemble_size: 3,
            noise_std: 0.01,
            min_confidence: 0.9, // High threshold
        });

        // Low confidence retrieval
        let retrieval_fn = |_q: &[f32]| -> Option<(Vec<f32>, f32)> {
            Some((vec![0.5], 0.5)) // Below threshold
        };

        let query = vec![1.0];
        let result = uq.retrieve_with_uncertainty(&query, retrieval_fn);

        assert!(result.is_none(), "Should return None below threshold");
    }

    #[test]
    fn test_is_reliable() {
        let result = UncertainRetrievalResult {
            value: vec![1.0],
            confidence: 0.9,
            epistemic_uncertainty: 0.1,
            aleatoric_uncertainty: 0.05,
        };

        assert!(result.is_reliable(0.8, 0.2));
        assert!(!result.is_reliable(0.95, 0.2)); // Confidence too low
        assert!(!result.is_reliable(0.8, 0.05)); // Epistemic too high
    }

    #[test]
    fn test_into_uncertain() {
        let result = UncertainRetrievalResult {
            value: 42.0,
            confidence: 0.85,
            epistemic_uncertainty: 0.15,
            aleatoric_uncertainty: 0.05,
        };

        let uncertain = result.into_uncertain("memory_retrieval");
        assert_eq!(uncertain.value, 42.0);
        assert_eq!(uncertain.confidence, 0.85);
        assert_eq!(uncertain.epistemic_uncertainty, 0.15);
        assert_eq!(uncertain.aleatoric_uncertainty, 0.05);
    }

    #[test]
    fn test_total_uncertainty() {
        let result = UncertainRetrievalResult {
            value: 1.0,
            confidence: 0.8,
            epistemic_uncertainty: 0.3,
            aleatoric_uncertainty: 0.4,
        };

        assert!((result.total_uncertainty() - 0.7).abs() < 0.001);

        // Test capping at 1.0
        let high_result = UncertainRetrievalResult {
            value: 1.0,
            confidence: 0.5,
            epistemic_uncertainty: 0.8,
            aleatoric_uncertainty: 0.8,
        };
        assert_eq!(high_result.total_uncertainty(), 1.0);
    }
}
