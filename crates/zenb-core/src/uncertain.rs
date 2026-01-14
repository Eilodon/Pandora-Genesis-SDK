//! Uncertainty quantification types
//!
//! Provides types for tracking uncertainty and confidence throughout
//! the causal reasoning pipeline. Inspired by DeepCausality's uncertainty
//! handling but tailored for AGOLOS use cases.
//!
//! # ZENITH Enhancement (Tier 4)
//! Extended with epistemic (model) and aleatoric (data) uncertainty tracking
//! to enable "know what you don't know" capabilities.

use serde::{Deserialize, Serialize};

/// Uncertain value with confidence score and provenance tracking.
///
/// # Design Philosophy
/// In safety-critical systems, knowing the confidence of a prediction
/// is as important as the prediction itself. This type makes uncertainty
/// explicit and forces downstream code to handle it.
///
/// # ZENITH Enhancement: Dual Uncertainty
/// - **Epistemic**: Model uncertainty (reducible with more data)
/// - **Aleatoric**: Data noise (irreducible)
///
/// # Examples
/// ```
/// use zenb_core::uncertain::Uncertain;
///
/// let prediction = Uncertain::new(0.7, 0.85, "PC Algorithm");
///
/// if prediction.confidence > 0.8 && prediction.epistemic_uncertainty < 0.2 {
///     // High confidence, low model uncertainty - safe to act
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uncertain<T> {
    /// The predicted/estimated value
    pub value: T,

    /// Confidence score in [0.0, 1.0]
    /// - 1.0 = completely certain (e.g., from prior knowledge)
    /// - 0.5 = neutral (cold start, no information)
    /// - 0.0 = completely uncertain (contradictory evidence)
    pub confidence: f32,

    /// Source of this value (for audit trail)
    /// Examples: "PC Algorithm", "Prior Knowledge", "UKF Estimation"
    pub source: String,

    // =========================================================================
    // ZENITH Tier 4: Dual Uncertainty Tracking
    // =========================================================================
    
    /// Epistemic uncertainty (model/parameter uncertainty)
    /// 
    /// Measures "uncertainty about the model itself":
    /// - High epistemic → "I'm not sure, need to learn more"
    /// - Reducible with more training data or better model selection
    /// - Default: 0.0 (for backward compatibility)
    #[serde(default)]
    pub epistemic_uncertainty: f32,

    /// Aleatoric uncertainty (data/observation noise)
    /// 
    /// Measures "inherent randomness in the data":
    /// - High aleatoric → "Data is inherently noisy"
    /// - Irreducible - fundamental limit of prediction
    /// - Default: 0.0 (for backward compatibility)
    #[serde(default)]
    pub aleatoric_uncertainty: f32,
}

impl<T> Uncertain<T> {
    /// Create a new uncertain value (backward compatible)
    pub fn new(value: T, confidence: f32, source: impl Into<String>) -> Self {
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
            source: source.into(),
            epistemic_uncertainty: 0.0,
            aleatoric_uncertainty: 0.0,
        }
    }

    /// Create uncertain value with full uncertainty decomposition (ZENITH)
    pub fn with_uncertainty(
        value: T,
        confidence: f32,
        epistemic: f32,
        aleatoric: f32,
        source: impl Into<String>,
    ) -> Self {
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
            source: source.into(),
            epistemic_uncertainty: epistemic.clamp(0.0, 1.0),
            aleatoric_uncertainty: aleatoric.clamp(0.0, 1.0),
        }
    }

    /// Create a certain value (confidence = 1.0, no uncertainty)
    pub fn certain(value: T, source: impl Into<String>) -> Self {
        Self::new(value, 1.0, source)
    }

    /// Create a neutral value (confidence = 0.5)
    pub fn neutral(value: T, source: impl Into<String>) -> Self {
        Self::new(value, 0.5, source)
    }

    /// Create a highly uncertain value (for ensemble disagreement)
    pub fn highly_uncertain(value: T, epistemic: f32, source: impl Into<String>) -> Self {
        Self::with_uncertainty(value, 0.5, epistemic, 0.0, source)
    }

    /// Map the value while preserving uncertainty metadata
    pub fn map<U, F>(self, f: F) -> Uncertain<U>
    where
        F: FnOnce(T) -> U,
    {
        Uncertain {
            value: f(self.value),
            confidence: self.confidence,
            source: self.source,
            epistemic_uncertainty: self.epistemic_uncertainty,
            aleatoric_uncertainty: self.aleatoric_uncertainty,
        }
    }

    /// Check if confidence is above threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Check if both confident AND low epistemic uncertainty
    /// 
    /// This is the gold standard for safe autonomous action:
    /// - High confidence alone might be "confidently wrong"
    /// - Low epistemic means the model is reliable
    pub fn is_reliably_confident(&self, confidence_threshold: f32, epistemic_threshold: f32) -> bool {
        self.confidence >= confidence_threshold && self.epistemic_uncertainty <= epistemic_threshold
    }

    /// Total uncertainty (epistemic + aleatoric)
    pub fn total_uncertainty(&self) -> f32 {
        (self.epistemic_uncertainty + self.aleatoric_uncertainty).min(1.0)
    }

    /// Combine two uncertain values using weighted average
    /// Weights are based on confidence scores
    pub fn combine_with<F>(self, other: Uncertain<T>, combiner: F) -> Uncertain<T>
    where
        F: FnOnce(T, T, f32, f32) -> T,
    {
        let total_confidence = self.confidence + other.confidence;
        let weight_self = if total_confidence > 0.0 {
            self.confidence / total_confidence
        } else {
            0.5
        };
        let weight_other = 1.0 - weight_self;

        Uncertain {
            value: combiner(self.value, other.value, weight_self, weight_other),
            confidence: (self.confidence + other.confidence) / 2.0,
            source: format!("{}+{}", self.source, other.source),
            // Combine uncertainties (average weighted by confidence)
            epistemic_uncertainty: weight_self * self.epistemic_uncertainty 
                                 + weight_other * other.epistemic_uncertainty,
            aleatoric_uncertainty: weight_self * self.aleatoric_uncertainty
                                 + weight_other * other.aleatoric_uncertainty,
        }
    }
}


impl<T: Clone> Uncertain<T> {
    /// Extract value, discarding uncertainty metadata
    /// Use with caution - prefer keeping uncertainty information
    pub fn into_value(self) -> T {
        self.value
    }

    /// Get reference to value
    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Maybe uncertain - either a certain value or an uncertain one
///
/// This is useful when some values are known with certainty
/// (e.g., sensor readings) while others are predictions.
#[derive(Debug, Clone)]
pub enum MaybeUncertain<T> {
    /// Value is certain (no uncertainty)
    Certain(T),

    /// Value has associated uncertainty
    Uncertain(Uncertain<T>),
}

impl<T> MaybeUncertain<T> {
    /// Get the value, discarding uncertainty if present
    pub fn into_value(self) -> T {
        match self {
            MaybeUncertain::Certain(v) => v,
            MaybeUncertain::Uncertain(u) => u.value,
        }
    }

    /// Get confidence (1.0 for certain values)
    pub fn confidence(&self) -> f32 {
        match self {
            MaybeUncertain::Certain(_) => 1.0,
            MaybeUncertain::Uncertain(u) => u.confidence,
        }
    }

    /// Convert to Uncertain (certain values get confidence = 1.0)
    pub fn into_uncertain(self, default_source: impl Into<String>) -> Uncertain<T> {
        match self {
            MaybeUncertain::Certain(v) => Uncertain::certain(v, default_source),
            MaybeUncertain::Uncertain(u) => u,
        }
    }
}

impl<T> From<T> for MaybeUncertain<T> {
    fn from(value: T) -> Self {
        MaybeUncertain::Certain(value)
    }
}

impl<T> From<Uncertain<T>> for MaybeUncertain<T> {
    fn from(uncertain: Uncertain<T>) -> Self {
        MaybeUncertain::Uncertain(uncertain)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertain_creation() {
        let u = Uncertain::new(42, 0.8, "test");
        assert_eq!(u.value, 42);
        assert_eq!(u.confidence, 0.8);
        assert_eq!(u.source, "test");
    }

    #[test]
    fn test_confidence_clamping() {
        let u1 = Uncertain::new(1, 1.5, "test");
        assert_eq!(u1.confidence, 1.0);

        let u2 = Uncertain::new(2, -0.5, "test");
        assert_eq!(u2.confidence, 0.0);
    }

    #[test]
    fn test_map() {
        let u = Uncertain::new(10, 0.9, "test");
        let mapped = u.map(|x| x * 2);

        assert_eq!(mapped.value, 20);
        assert_eq!(mapped.confidence, 0.9);
        assert_eq!(mapped.source, "test");
    }

    #[test]
    fn test_is_confident() {
        let u = Uncertain::new(1, 0.85, "test");
        assert!(u.is_confident(0.8));
        assert!(!u.is_confident(0.9));
    }

    #[test]
    fn test_combine() {
        let u1 = Uncertain::new(10.0, 0.8, "source1");
        let u2 = Uncertain::new(20.0, 0.6, "source2");

        let combined = u1.combine_with(u2, |v1, v2, w1, w2| v1 * w1 + v2 * w2);

        // Should be weighted average: 10*0.571 + 20*0.429 = 14.29
        assert!((combined.value - 14.29).abs() < 0.1);
        assert!((combined.confidence - 0.7).abs() < 0.01); // (0.8 + 0.6) / 2
    }

    #[test]
    fn test_maybe_uncertain() {
        let certain = MaybeUncertain::Certain(42);
        assert_eq!(certain.confidence(), 1.0);
        assert_eq!(certain.into_value(), 42);

        let uncertain = MaybeUncertain::Uncertain(Uncertain::new(10, 0.5, "test"));
        assert_eq!(uncertain.confidence(), 0.5);
    }

    #[test]
    fn test_from_conversions() {
        let _m1: MaybeUncertain<i32> = 42.into();
        let _m2: MaybeUncertain<i32> = Uncertain::new(10, 0.8, "test").into();
    }

    // ========================================================================
    // ZENITH Enhanced Uncertainty Tests
    // ========================================================================

    #[test]
    fn test_with_uncertainty() {
        let u = Uncertain::with_uncertainty(0.5, 0.9, 0.15, 0.05, "ensemble");
        
        assert_eq!(u.value, 0.5);
        assert_eq!(u.confidence, 0.9);
        assert!((u.epistemic_uncertainty - 0.15).abs() < 0.001);
        assert!((u.aleatoric_uncertainty - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_backward_compatible_new() {
        // Using new() should default epistemic/aleatoric to 0.0
        let u = Uncertain::new(42, 0.8, "test");
        
        assert_eq!(u.epistemic_uncertainty, 0.0);
        assert_eq!(u.aleatoric_uncertainty, 0.0);
    }

    #[test]
    fn test_reliably_confident() {
        // High confidence, low epistemic = reliable
        let reliable = Uncertain::with_uncertainty(1.0, 0.9, 0.1, 0.2, "good_model");
        assert!(reliable.is_reliably_confident(0.8, 0.2));
        
        // High confidence but high epistemic = NOT reliable
        let unreliable = Uncertain::with_uncertainty(1.0, 0.9, 0.5, 0.1, "bad_model");
        assert!(!unreliable.is_reliably_confident(0.8, 0.2));
        
        // Low confidence = NOT reliable
        let low_conf = Uncertain::with_uncertainty(1.0, 0.5, 0.1, 0.1, "weak");
        assert!(!low_conf.is_reliably_confident(0.8, 0.2));
    }

    #[test]
    fn test_total_uncertainty() {
        let u = Uncertain::with_uncertainty(1.0, 0.8, 0.3, 0.2, "test");
        assert!((u.total_uncertainty() - 0.5).abs() < 0.001);
        
        // Should cap at 1.0
        let high_u = Uncertain::with_uncertainty(1.0, 0.5, 0.8, 0.8, "noisy");
        assert_eq!(high_u.total_uncertainty(), 1.0);
    }

    #[test]
    fn test_map_preserves_uncertainty() {
        let u = Uncertain::with_uncertainty(10, 0.8, 0.2, 0.1, "original");
        let mapped = u.map(|x| x * 2);
        
        assert_eq!(mapped.value, 20);
        assert_eq!(mapped.epistemic_uncertainty, 0.2);
        assert_eq!(mapped.aleatoric_uncertainty, 0.1);
    }

    #[test]
    fn test_highly_uncertain() {
        let u = Uncertain::highly_uncertain(0.5, 0.7, "disagreement");
        
        assert_eq!(u.confidence, 0.5);
        assert_eq!(u.epistemic_uncertainty, 0.7);
        assert_eq!(u.aleatoric_uncertainty, 0.0);
    }

    #[test]
    fn test_combine_with_uncertainty() {
        let u1 = Uncertain::with_uncertainty(10.0, 0.8, 0.2, 0.1, "model1");
        let u2 = Uncertain::with_uncertainty(20.0, 0.6, 0.4, 0.2, "model2");
        
        let combined = u1.combine_with(u2, |v1, v2, w1, w2| v1 * w1 + v2 * w2);
        
        // Uncertainties should be weighted average
        // weight_self = 0.8 / 1.4 = 0.571, weight_other = 0.429
        let expected_epistemic = 0.571 * 0.2 + 0.429 * 0.4;
        assert!((combined.epistemic_uncertainty - expected_epistemic).abs() < 0.01);
    }
}

