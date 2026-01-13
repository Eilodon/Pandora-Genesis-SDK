//! Uncertainty quantification types
//!
//! Provides types for tracking uncertainty and confidence throughout
//! the causal reasoning pipeline. Inspired by DeepCausality's uncertainty
//! handling but tailored for AGOLOS use cases.

use serde::{Deserialize, Serialize};

/// Uncertain value with confidence score and provenance tracking.
///
/// # Design Philosophy
/// In safety-critical systems, knowing the confidence of a prediction
/// is as important as the prediction itself. This type makes uncertainty
/// explicit and forces downstream code to handle it.
///
/// # Examples
/// ```
/// use zenb_core::uncertain::Uncertain;
///
/// let prediction = Uncertain {
///     value: 0.7,
///     confidence: 0.85,
///     source: "PC Algorithm".to_string(),
/// };
///
/// if prediction.confidence > 0.8 {
///     // High confidence - can act on this
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
}

impl<T> Uncertain<T> {
    /// Create a new uncertain value
    pub fn new(value: T, confidence: f32, source: impl Into<String>) -> Self {
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
            source: source.into(),
        }
    }

    /// Create a certain value (confidence = 1.0)
    pub fn certain(value: T, source: impl Into<String>) -> Self {
        Self::new(value, 1.0, source)
    }

    /// Create a neutral value (confidence = 0.5)
    pub fn neutral(value: T, source: impl Into<String>) -> Self {
        Self::new(value, 0.5, source)
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
        }
    }

    /// Check if confidence is above threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
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
}
