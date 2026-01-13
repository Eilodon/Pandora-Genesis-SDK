//! Dharma Filter: Phase-based Ethical Action Filtering
//!
//! Implements physics-based morality using complex number phase alignment.
//! Actions that are "in phase" with the ethical reference (Dharma Key) are amplified;
//! actions that are "out of phase" are attenuated or vetoed.
//!
//! # Mathematical Foundation (Hành Uẩn - Saṃskāra-skandha)
//! In Buddhist philosophy, Hành Uẩn represents the aggregate of volitional formations.
//! This module embodies intentional action with moral direction encoded in phase space.
//!
//! # Key Insight
//! Phase-based ethics is **physically intrinsic** - harmful actions literally cannot
//! constructively interfere with beneficial intentions. This is NOT an if-else check;
//! it's a mathematical property of complex interference patterns.

use num_complex::Complex32;

/// Dharma Filter: Phase-based ethical action filtering
///
/// # Mathematical Model
/// Given a "Dharma Key" (reference vector representing "good" direction),
/// actions are evaluated by their phase alignment:
///
/// alignment = Re(<action | dharma>) / (|action| * |dharma|) = cos(θ)
///
/// where θ is the angle between action and dharma vectors in complex space.
///
/// # Alignment Interpretation
/// - alignment = 1.0: Perfect alignment (action reinforced)
/// - alignment = 0.0: Orthogonal (neutral, can be scaled down)
/// - alignment < 0.0: Misaligned (action VETOED - destructive interference)
///
/// # Safety Guarantee
/// This creates a **physically intrinsic** safety mechanism:
/// harmful actions literally cannot pass the filter because they
/// destructively interfere with the Dharma reference.
#[derive(Debug)]
pub struct DharmaFilter {
    /// Reference vector defining "good" direction in action space.
    /// All actions are evaluated against this ethical north star.
    dharma_key: Complex32,

    /// Soft threshold: actions below this alignment get scaled down
    soft_threshold: f32,

    /// Hard threshold: actions below this alignment get vetoed
    hard_threshold: f32,

    /// TIER 5: 11D Dharma key for multi-dimensional alignment
    dharma_key_11d: crate::safety::ConsciousnessVector,

    /// Feature flag to enable 11D mode
    use_11d_mode: bool,
}

impl DharmaFilter {
    /// Create filter with given ethical reference.
    ///
    /// # Arguments
    /// * `dharma_key` - The reference vector defining ethical direction
    ///
    /// # Panics
    /// Panics if dharma_key has zero magnitude.
    pub fn new(dharma_key: Complex32) -> Self {
        assert!(
            dharma_key.norm() > 1e-10,
            "Dharma key must have non-zero magnitude"
        );
        Self {
            dharma_key,
            soft_threshold: 0.5, // Actions below 50% alignment get scaled
            hard_threshold: 0.0, // Actions opposite to dharma get vetoed
            dharma_key_11d: crate::safety::ConsciousnessVector::default(),
            use_11d_mode: false,
        }
    }

    /// Create with default Dharma key optimized for ZenB.
    ///
    /// # Dharma Key Design for ZenB
    /// The key is chosen to represent "calming, restorative, safe" actions:
    /// - Phase 0 (positive real): Direct calming (baseline maintenance)
    /// - Small positive imaginary: Gentle upward modulation (energizing)
    ///
    /// This means:
    /// - Actions that reduce arousal toward baseline → high alignment
    /// - Actions that gently energize → moderate alignment  
    /// - Actions that cause stress/harm → negative alignment (vetoed)
    ///
    /// The key is normalized to unit magnitude for consistent scaling.
    pub fn default_for_zenb() -> Self {
        // Dharma key: primarily calming (real=1) with slight energizing potential (imag=0.3)
        // This encodes: "prefer homeostasis, allow gentle activation, reject harm"
        let raw_key = Complex32::new(1.0, 0.3);
        let normalized = raw_key / raw_key.norm();

        Self::new(normalized)
    }

    /// Create a purely conservative filter (only accepts pure calming actions).
    pub fn strict_calming() -> Self {
        Self::new(Complex32::new(1.0, 0.0))
    }

    /// Compute alignment score between action and Dharma.
    ///
    /// # Returns
    /// cos(θ) where θ is angle between vectors:
    /// - 1.0 = perfectly aligned (good)
    /// - 0.0 = orthogonal (neutral)
    /// - -1.0 = opposite (harmful)
    ///
    /// # Mathematical Formula
    /// alignment = Re(action * conj(dharma)) / (|action| * |dharma|)
    pub fn check_alignment(&self, action: Complex32) -> f32 {
        let magnitude = action.norm() * self.dharma_key.norm();
        if magnitude < 1e-10 {
            return 0.0; // Zero action is neutral
        }

        // Hermitian inner product: <action | dharma> = action * conj(dharma)
        let dot = action * self.dharma_key.conj();

        // cos(θ) = Re(dot) / |action| |dharma|
        dot.re / magnitude
    }

    /// Sanction (filter) an action based on Dharma alignment.
    ///
    /// This is the core safety mechanism. Actions are processed as follows:
    /// 1. Compute alignment with Dharma key
    /// 2. If alignment < hard_threshold (default 0.0): VETO (return None)
    /// 3. If alignment < soft_threshold: scale down by alignment
    /// 4. Otherwise: pass through (optionally scaled)
    ///
    /// # Returns
    /// - `Some(scaled_action)` if action is permitted
    /// - `None` if action is VETOED due to phase misalignment
    ///
    /// # Safety Note
    /// A vetoed action means the proposed action is fundamentally
    /// incompatible with the system's ethical framework. This is a
    /// hard stop, not a soft suggestion.
    pub fn sanction(&self, action: Complex32) -> Option<Complex32> {
        let alignment = self.check_alignment(action);

        // Hard veto: action is opposite to Dharma (destructive interference)
        if alignment < self.hard_threshold {
            log::warn!(
                "DharmaVeto: Action REJECTED (alignment={:.3}, phase_diff={:.2}rad, magnitude={:.3})",
                alignment,
                (action.arg() - self.dharma_key.arg()).abs(),
                action.norm()
            );
            return None;
        }

        // Soft scaling: reduce poorly-aligned actions
        let scale = if alignment < self.soft_threshold {
            alignment.max(0.1) // Never scale below 10% to avoid division issues
        } else {
            alignment
        };

        Some(action * scale)
    }

    /// Soft sanction: never veto, but always scale by alignment.
    ///
    /// Use this for non-critical actions where complete blocking
    /// would be too disruptive, but you still want alignment incentive.
    pub fn soft_sanction(&self, action: Complex32) -> Complex32 {
        let alignment = self.check_alignment(action).max(0.0);
        action * alignment
    }

    /// Project an action onto the Dharma-aligned subspace.
    ///
    /// This transforms any action into its "ethical component" by
    /// projecting onto the Dharma key direction.
    ///
    /// # Mathematical Operation
    /// projected = (action · dharma) * dharma / |dharma|²
    pub fn project_onto_dharma(&self, action: Complex32) -> Complex32 {
        let dot = action * self.dharma_key.conj();
        let dharma_mag_sq = self.dharma_key.norm_sqr();

        self.dharma_key * (dot.re / dharma_mag_sq)
    }

    /// Update Dharma key (for adaptive ethics).
    ///
    /// # EIDOLON FIX 1.3: DEPRECATED
    /// This method is deprecated due to ethical drift risk. If the dharma key
    /// is updated based on outcome feedback, adversarial users can manipulate
    /// the system into approving harmful actions by repeatedly rewarding them.
    ///
    /// # Warning
    /// Changing the Dharma key changes the ethical reference frame.
    /// This should only be done deliberately and with full understanding
    /// of the implications. DO NOT wire this into automated learning loops.
    ///
    /// # Panics
    /// Panics if new_key has zero magnitude.
    ///
    /// # Compile-Time Protection
    /// This method is only available in tests or with `allow_dharma_mutation` feature.
    /// Production builds will fail to compile if this is used.
    #[cfg(any(test, feature = "allow_dharma_mutation"))]
    #[deprecated(
        since = "0.2.0",
        note = "Ethical drift risk: Do not update dharma key based on user feedback. Use immutable constructor instead."
    )]
    pub fn update_dharma(&mut self, new_key: Complex32) {
        assert!(
            new_key.norm() > 1e-10,
            "Dharma key must have non-zero magnitude"
        );
        log::warn!(
            "DharmaFilter::update_dharma called - this is deprecated due to ethical drift risk!"
        );
        self.dharma_key = new_key / new_key.norm();
    }

    /// Get the current Dharma key.
    pub fn dharma_key(&self) -> Complex32 {
        self.dharma_key
    }

    /// Set the soft threshold for alignment scaling.
    pub fn set_soft_threshold(&mut self, threshold: f32) {
        self.soft_threshold = threshold;
    }

    /// Set the hard threshold for veto.
    pub fn set_hard_threshold(&mut self, threshold: f32) {
        self.hard_threshold = threshold;
    }

    /// Get alignment statistics for diagnostics.
    pub fn alignment_category(&self, action: Complex32) -> AlignmentCategory {
        let alignment = self.check_alignment(action);

        if alignment >= 0.8 {
            AlignmentCategory::Excellent
        } else if alignment >= 0.5 {
            AlignmentCategory::Good
        } else if alignment >= 0.0 {
            AlignmentCategory::Marginal
        } else if alignment >= -0.5 {
            AlignmentCategory::Poor
        } else {
            AlignmentCategory::Harmful
        }
    }
}

impl Default for DharmaFilter {
    fn default() -> Self {
        Self::default_for_zenb()
    }
}

/// Category of alignment with Dharma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentCategory {
    /// Alignment >= 0.8: Action strongly aligned with ethics
    Excellent,
    /// Alignment >= 0.5: Action reasonably aligned
    Good,
    /// Alignment >= 0.0: Action neutral to slightly positive
    Marginal,
    /// Alignment >= -0.5: Action somewhat misaligned (scaled down)
    Poor,
    /// Alignment < -0.5: Action strongly misaligned (vetoed)
    Harmful,
}

// ============================================================================
// COMPLEX DECISION TYPE
// ============================================================================

/// Complex-valued control decision with embedded moral intent.
///
/// # Interpretation
/// - Magnitude (|z|) = Action intensity (e.g., BPM deviation from baseline)
/// - Phase (arg(z)) = Moral/intentional direction in action space
///
/// This representation enables the DharmaFilter to evaluate both
/// the intensity AND the ethical direction of an action in one number.
#[derive(Debug, Clone, Copy)]
pub struct ComplexDecision {
    /// The complex vector encoding intensity and intent
    pub vector: Complex32,
}

impl ComplexDecision {
    /// Create from Cartesian coordinates (real = calming, imag = energizing).
    pub fn new(real: f32, imag: f32) -> Self {
        Self {
            vector: Complex32::new(real, imag),
        }
    }

    /// Create from polar coordinates (intensity, intent_phase).
    ///
    /// # Arguments
    /// * `magnitude` - Intensity of the action
    /// * `phase` - Intentional direction (radians, 0 = pure calming)
    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            vector: Complex32::from_polar(magnitude, phase),
        }
    }

    /// Create from a simple BPM target.
    ///
    /// Maps BPM deviation from baseline (6.0 BPM) to complex space:
    /// - Negative deviation (slowing) → positive real (calming)
    /// - Positive deviation (speeding) → positive imaginary (energizing)
    pub fn from_bpm_target(target_bpm: f32, baseline_bpm: f32) -> Self {
        let deviation = target_bpm - baseline_bpm;

        // Map deviation to complex plane
        // Slowing: more calming (positive real)
        // Speeding: more energizing (positive imaginary)
        let (real, imag) = if deviation < 0.0 {
            (-deviation, 0.0) // Slowing is calming
        } else {
            (0.0, deviation) // Speeding is energizing
        };

        Self::new(real, imag)
    }

    /// Convert to BPM target.
    ///
    /// # Arguments
    /// * `baseline` - Baseline BPM (typically 6.0 for breath guidance)
    ///
    /// # Returns
    /// Target BPM incorporating the action's intent.
    pub fn to_bpm(&self, baseline: f32) -> f32 {
        // Real component slows down, imaginary speeds up
        baseline - self.vector.re + self.vector.im
    }

    /// Get the magnitude (intensity) of the action.
    pub fn magnitude(&self) -> f32 {
        self.vector.norm()
    }

    /// Get the phase (intentional direction) of the action.
    pub fn phase(&self) -> f32 {
        self.vector.arg()
    }

    /// Apply a filter to this decision, returning filtered result.
    pub fn filter_with(&self, filter: &DharmaFilter) -> Option<ComplexDecision> {
        filter
            .sanction(self.vector)
            .map(|v| ComplexDecision { vector: v })
    }
}

// ============================================================================
// TIER 5: 11D Consciousness Vector Extension
// ============================================================================

impl DharmaFilter {
    /// Check alignment in 11-dimensional consciousness space.
    pub fn check_alignment_11d(&self, vec: &crate::safety::ConsciousnessVector) -> f32 {
        let alignment = vec.alignment(&self.dharma_key_11d);
        // Map from [-1, 1] to [0, 1]
        (alignment + 1.0) / 2.0
    }

    /// Sanction action in 11D space.
    pub fn sanction_11d(
        &self,
        vec: &crate::safety::ConsciousnessVector,
    ) -> Option<crate::safety::ConsciousnessVector> {
        let alignment = vec.alignment(&self.dharma_key_11d);

        if alignment < -0.5 {
            return None;
        }

        if alignment < 0.5 {
            let projected = vec.project_onto(&self.dharma_key_11d);
            Some(projected)
        } else {
            Some(*vec)
        }
    }

    /// Enable or disable 11D mode
    pub fn set_11d_mode(&mut self, enabled: bool) {
        self.use_11d_mode = enabled;
    }

    /// Set the 11D Dharma key
    pub fn set_dharma_key_11d(&mut self, key: crate::safety::ConsciousnessVector) {
        self.dharma_key_11d = key;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_new_filter() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));
        assert!(filter.dharma_key().norm() > 0.0);
    }

    #[test]
    fn test_default_zenb() {
        let filter = DharmaFilter::default_for_zenb();

        // Default key should be normalized
        assert!(approx_eq(filter.dharma_key().norm(), 1.0, 1e-6));

        // Should have positive real (calming) and positive imag (energizing)
        assert!(filter.dharma_key().re > 0.0);
        assert!(filter.dharma_key().im > 0.0);
    }

    #[test]
    fn test_perfect_alignment() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // Action in same direction should have alignment = 1.0
        let action = Complex32::new(2.0, 0.0);
        let alignment = filter.check_alignment(action);

        assert!(
            approx_eq(alignment, 1.0, 1e-6),
            "Perfect alignment expected: got {}",
            alignment
        );
    }

    #[test]
    fn test_orthogonal_alignment() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0)); // Pure real

        // Pure imaginary action should be orthogonal
        let action = Complex32::new(0.0, 1.0);
        let alignment = filter.check_alignment(action);

        assert!(
            approx_eq(alignment, 0.0, 1e-6),
            "Zero alignment expected: got {}",
            alignment
        );
    }

    #[test]
    fn test_opposite_alignment() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // Opposite direction should have alignment = -1.0
        let action = Complex32::new(-1.0, 0.0);
        let alignment = filter.check_alignment(action);

        assert!(
            approx_eq(alignment, -1.0, 1e-6),
            "Opposite alignment expected: got {}",
            alignment
        );
    }

    #[test]
    fn test_veto_opposite_action() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // DoD Test: Actions opposite to dharma_key should be vetoed
        let harmful = Complex32::new(-1.0, 0.0);
        let result = filter.sanction(harmful);

        assert!(result.is_none(), "Opposite action should be vetoed");
    }

    #[test]
    fn test_pass_aligned_action() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // Aligned action should pass
        let good = Complex32::new(1.0, 0.0);
        let result = filter.sanction(good);

        assert!(result.is_some(), "Aligned action should pass");

        // Should be scaled by alignment (1.0)
        let sanctioned = result.unwrap();
        assert!(approx_eq(sanctioned.re, 1.0, 0.1));
    }

    #[test]
    fn test_soft_sanction() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // Even opposite actions get soft-scaled (not vetoed)
        let harmful = Complex32::new(-1.0, 0.0);
        let result = filter.soft_sanction(harmful);

        // Alignment is -1.0, clamped to 0.0, so result should be zero
        assert!(
            result.norm() < 1e-6,
            "Soft sanction of opposite should be near zero"
        );
    }

    #[test]
    fn test_project_onto_dharma() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        // Project a diagonal vector onto dharma (real axis)
        let diagonal = Complex32::new(1.0, 1.0);
        let projected = filter.project_onto_dharma(diagonal);

        // Should only have real component
        assert!(approx_eq(projected.re, 1.0, 1e-6));
        assert!(approx_eq(projected.im, 0.0, 1e-6));
    }

    #[test]
    fn test_complex_decision_creation() {
        let decision = ComplexDecision::from_polar(2.0, PI / 4.0);

        assert!(approx_eq(decision.magnitude(), 2.0, 1e-6));
        assert!(approx_eq(decision.phase(), PI / 4.0, 1e-6));
    }

    #[test]
    fn test_complex_decision_bpm_conversion() {
        let baseline = 6.0;

        // Calming decision (target 5 BPM)
        let calming = ComplexDecision::from_bpm_target(5.0, baseline);
        assert!(calming.vector.re > 0.0, "Calming should have positive real");

        // Energizing decision (target 7 BPM)
        let energizing = ComplexDecision::from_bpm_target(7.0, baseline);
        assert!(
            energizing.vector.im > 0.0,
            "Energizing should have positive imaginary"
        );
    }

    #[test]
    fn test_alignment_categories() {
        let filter = DharmaFilter::new(Complex32::new(1.0, 0.0));

        assert_eq!(
            filter.alignment_category(Complex32::new(1.0, 0.0)),
            AlignmentCategory::Excellent
        );
        assert_eq!(
            filter.alignment_category(Complex32::new(-1.0, 0.0)),
            AlignmentCategory::Harmful
        );
        assert_eq!(
            filter.alignment_category(Complex32::new(0.0, 1.0)),
            AlignmentCategory::Marginal
        );
    }

    #[test]
    fn test_zenb_realistic_scenario() {
        let filter = DharmaFilter::default_for_zenb();

        // Scenario 1: Gentle calming breath (6 -> 5 BPM)
        let calming = ComplexDecision::from_bpm_target(5.0, 6.0);
        let result = calming.filter_with(&filter);
        assert!(result.is_some(), "Gentle calming should be allowed");

        // Scenario 2: Moderate energizing (6 -> 7 BPM)
        // ZenB dharma allows slight energizing
        let energizing = ComplexDecision::from_bpm_target(7.0, 6.0);
        let result = energizing.filter_with(&filter);
        assert!(result.is_some(), "Moderate energizing should be allowed");

        // Scenario 3: Extreme opposite action
        // This would be something like target -10 BPM (nonsensical)
        let extreme = Complex32::new(-5.0, -5.0); // Opposite quadrant
        let result = filter.sanction(extreme);
        // Might or might not be vetoed depending on angle, but should be severely scaled
    }
}
