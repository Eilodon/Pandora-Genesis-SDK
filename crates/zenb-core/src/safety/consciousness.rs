//! 11D Consciousness Vector for Multi-Dimensional Ethical Alignment
//!
//! Extends the DharmaFilter from 2D complex phase to 11-dimensional consciousness space.
//!
//! # Dimensional Breakdown
//! - **Dimensions 0-4**: Five Skandhas (Rūpa, Vedanā, Saññā, Saṅkhāra, Viññāṇa)
//! - **Dimensions 5-7**: Three Belief States (Bio, Cognitive, Social)
//! - **Dimensions 8-10**: Three Context Factors (Circadian, Environment, Interaction)
//!
//! # Mathematical Model
//! Alignment is computed via dot product in 11D space:
//! ```text
//! alignment(v, key) = (v · key) / (|v| * |key|)
//! ```

use serde::{Deserialize, Serialize};

/// 11-dimensional consciousness vector representing the complete state
/// of a system's awareness across multiple modalities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConsciousnessVector {
    // === Skandha Dimensions (0-4) ===
    /// Rūpa (Form): Energy in sensory processing
    pub rupa_energy: f32,

    /// Vedanā (Feeling): Affective valence (-1 to +1)
    pub vedana_valence: f32,

    /// Saññā (Perception): Pattern match quality (0 to 1)
    pub sanna_similarity: f32,

    /// Saṅkhāra (Formation): Ethical intent alignment (0 to 1)
    pub sankhara_alignment: f32,

    /// Viññāṇa (Consciousness): Synthesis confidence (0 to 1)
    pub vinnana_confidence: f32,

    // === Belief State Dimensions (5-7) ===
    /// Biological arousal state (0 calm to 1 aroused)
    pub bio_arousal: f32,

    /// Cognitive focus level (0 distracted to 1 focused)
    pub cognitive_focus: f32,

    /// Social interaction load (0 solitary to 1 overwhelmed)
    pub social_load: f32,

    // === Context Dimensions (8-10) ===
    /// Circadian phase (0 to 1, mapped from hour of day)
    pub circadian_phase: f32,

    /// Environmental stress level (0 to 1)
    pub environment_stress: f32,

    /// Digital interaction intensity (0 to 1)
    pub interaction_intensity: f32,
}

impl Default for ConsciousnessVector {
    fn default() -> Self {
        Self {
            rupa_energy: 0.5,
            vedana_valence: 0.0,
            sanna_similarity: 0.5,
            sankhara_alignment: 1.0, // Default to aligned
            vinnana_confidence: 0.5,
            bio_arousal: 0.5,
            cognitive_focus: 0.5,
            social_load: 0.0,
            circadian_phase: 0.5,
            environment_stress: 0.0,
            interaction_intensity: 0.0,
        }
    }
}

impl ConsciousnessVector {
    /// Create a consciousness vector from Skandha pipeline output
    pub fn from_synthesized_state(state: &crate::skandha::SynthesizedState) -> Self {
        Self {
            rupa_energy: 0.5,        // Not directly available, placeholder
            vedana_valence: 0.0,     // Would come from affect if available
            sanna_similarity: 0.5,   // Would come from pattern if available
            sankhara_alignment: 1.0, // Would come from intent if available
            vinnana_confidence: state.confidence,
            bio_arousal: 0.5,           // From belief state
            cognitive_focus: 0.5,       // From belief state
            social_load: 0.0,           // From belief state
            circadian_phase: 0.5,       // From context
            environment_stress: 0.0,    // From context
            interaction_intensity: 0.0, // From context
        }
    }

    /// Create from a raw 11D array
    pub fn from_array(arr: [f32; 11]) -> Self {
        Self {
            rupa_energy: arr[0],
            vedana_valence: arr[1],
            sanna_similarity: arr[2],
            sankhara_alignment: arr[3],
            vinnana_confidence: arr[4],
            bio_arousal: arr[5],
            cognitive_focus: arr[6],
            social_load: arr[7],
            circadian_phase: arr[8],
            environment_stress: arr[9],
            interaction_intensity: arr[10],
        }
    }

    /// Convert to array for linear algebra operations
    pub fn to_array(&self) -> [f32; 11] {
        [
            self.rupa_energy,
            self.vedana_valence,
            self.sanna_similarity,
            self.sankhara_alignment,
            self.vinnana_confidence,
            self.bio_arousal,
            self.cognitive_focus,
            self.social_load,
            self.circadian_phase,
            self.environment_stress,
            self.interaction_intensity,
        ]
    }

    /// Compute the magnitude (L2 norm) of the consciousness vector
    pub fn magnitude(&self) -> f32 {
        let arr = self.to_array();
        arr.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Compute dot product with another consciousness vector
    pub fn dot(&self, other: &Self) -> f32 {
        let a = self.to_array();
        let b = other.to_array();
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Compute cosine similarity (normalized alignment) with another vector
    /// Returns value in [-1, 1] where 1 = perfectly aligned, -1 = opposite, 0 = orthogonal
    pub fn alignment(&self, other: &Self) -> f32 {
        let dot = self.dot(other);
        let mag_self = self.magnitude();
        let mag_other = other.magnitude();

        if mag_self < 1e-6 || mag_other < 1e-6 {
            return 0.0; // Avoid division by zero
        }

        (dot / (mag_self * mag_other)).clamp(-1.0, 1.0)
    }

    /// Project this vector onto another (returns component in direction of `other`)
    pub fn project_onto(&self, other: &Self) -> Self {
        let alignment = self.dot(other);
        let other_mag_sq = other.magnitude().powi(2);

        if other_mag_sq < 1e-6 {
            return Self::default();
        }

        let scalar = alignment / other_mag_sq;
        let other_arr = other.to_array();
        let projected_arr: [f32; 11] = std::array::from_fn(|i| other_arr[i] * scalar);

        Self::from_array(projected_arr)
    }

    /// Validate that all components are in valid ranges
    pub fn is_valid(&self) -> bool {
        let arr = self.to_array();
        arr.iter().all(|&x| x.is_finite())
    }
}

// ============================================================================
// PHASE CONSCIOUSNESS VECTOR (VAJRA-VOID MCCS Enhancement)
// ============================================================================

use num_complex::Complex32;

/// Phase-enhanced 11D consciousness vector using Complex32 components.
///
/// # VAJRA-VOID: MCCS Phase-Based Ethics
///
/// The original ConsciousnessVector uses real f32 values. This extension uses
/// Complex32 to encode both magnitude AND phase for each dimension.
///
/// ## Key Innovation: Dharma Phase Check
/// The phase (argument) of dimension 3 (sankhara/ethical intent) serves as
/// a "moral compass". Actions with phase within ±π/2 of the ethical reference
/// are permitted; others are vetoed.
///
/// ## Mathematical Model
/// ```text
/// dharma_phase = arg(dimensions[3])
/// ethical = |dharma_phase| <= π/2
/// ```
///
/// ## Dimension Layout
/// - [0-4]: Five Skandhas (Rūpa, Vedanā, Saññā, Saṅkhāra, Viññāṇa)
/// - [5-7]: Three Belief States (Bio, Cognitive, Social)
/// - [8-10]: Three Context Factors (Circadian, Environment, Interaction)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseConsciousnessVector {
    /// 11D complex consciousness vector
    pub dimensions: [Complex32; 11],
}

impl Default for PhaseConsciousnessVector {
    fn default() -> Self {
        Self {
            dimensions: [
                Complex32::new(0.5, 0.0),  // rupa_energy
                Complex32::new(0.0, 0.0),  // vedana_valence
                Complex32::new(0.5, 0.0),  // sanna_similarity
                Complex32::new(1.0, 0.0),  // sankhara_alignment (ethical, real positive = aligned)
                Complex32::new(0.5, 0.0),  // vinnana_confidence
                Complex32::new(0.5, 0.0),  // bio_arousal
                Complex32::new(0.5, 0.0),  // cognitive_focus
                Complex32::new(0.0, 0.0),  // social_load
                Complex32::new(0.5, 0.0),  // circadian_phase
                Complex32::new(0.0, 0.0),  // environment_stress
                Complex32::new(0.0, 0.0),  // interaction_intensity
            ],
        }
    }
}

impl PhaseConsciousnessVector {
    /// Dimension index for ethical intent (Saṅkhāra)
    pub const SANKHARA_IDX: usize = 3;

    /// Create from raw Complex32 array
    pub fn from_array(dimensions: [Complex32; 11]) -> Self {
        Self { dimensions }
    }

    /// Create from real ConsciousnessVector (phase = 0 for all dimensions)
    pub fn from_real(real: &ConsciousnessVector) -> Self {
        let arr = real.to_array();
        let dimensions: [Complex32; 11] = std::array::from_fn(|i| Complex32::new(arr[i], 0.0));
        Self { dimensions }
    }

    /// Convert to real ConsciousnessVector (discards phase information)
    pub fn to_real(&self) -> ConsciousnessVector {
        let arr: [f32; 11] = std::array::from_fn(|i| self.dimensions[i].norm());
        ConsciousnessVector::from_array(arr)
    }

    /// Get the dharma (ethical intent) phase angle.
    ///
    /// # Returns
    /// Phase angle in radians, range [-π, π]
    ///
    /// # Interpretation
    /// - Phase ≈ 0: Aligned with ethical reference (good)
    /// - Phase ≈ ±π/2: Neutral/uncertain
    /// - Phase ≈ ±π: Opposed to ethical reference (harmful)
    pub fn dharma_phase(&self) -> f32 {
        self.dimensions[Self::SANKHARA_IDX].arg()
    }

    /// Get the dharma magnitude (strength of ethical intent).
    pub fn dharma_magnitude(&self) -> f32 {
        self.dimensions[Self::SANKHARA_IDX].norm()
    }

    /// Check if the current state passes the phase-based ethics check.
    ///
    /// # MCCS Ethics Gate
    /// Actions are only permitted when the dharma phase is within ±π/2
    /// of the ethical reference (positive real axis).
    ///
    /// # Returns
    /// `true` if ethical, `false` if action should be vetoed
    pub fn check_phase_ethics(&self) -> bool {
        self.dharma_phase().abs() <= std::f32::consts::FRAC_PI_2
    }

    /// Check ethics with custom tolerance.
    ///
    /// # Arguments
    /// * `tolerance` - Maximum allowed deviation from aligned (in radians)
    pub fn check_phase_ethics_with_tolerance(&self, tolerance: f32) -> bool {
        self.dharma_phase().abs() <= tolerance
    }

    /// Rotate the dharma phase by the given angle.
    ///
    /// # Use Case
    /// Apply intentional drift or correction to ethical alignment.
    pub fn rotate_dharma(&mut self, angle: f32) {
        let rotation = Complex32::from_polar(1.0, angle);
        self.dimensions[Self::SANKHARA_IDX] *= rotation;
    }

    /// Compute total consciousness magnitude (L2 norm across all dimensions)
    pub fn magnitude(&self) -> f32 {
        self.dimensions.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt()
    }

    /// Compute alignment with another phase vector
    ///
    /// Uses complex inner product: alignment = Re(⟨u, v⟩) / (|u| × |v|)
    pub fn alignment(&self, other: &Self) -> f32 {
        let inner: Complex32 = self
            .dimensions
            .iter()
            .zip(other.dimensions.iter())
            .map(|(a, b)| a * b.conj())
            .sum();

        let mag_self = self.magnitude();
        let mag_other = other.magnitude();

        if mag_self < 1e-6 || mag_other < 1e-6 {
            return 0.0;
        }

        (inner.re / (mag_self * mag_other)).clamp(-1.0, 1.0)
    }

    /// Validate that all components are finite
    pub fn is_valid(&self) -> bool {
        self.dimensions.iter().all(|c| c.re.is_finite() && c.im.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_consciousness_vector() {
        let vec = ConsciousnessVector::default();
        assert!(vec.is_valid());
        assert!(vec.magnitude() > 0.0);
    }

    #[test]
    fn test_magnitude() {
        let vec = ConsciousnessVector::from_array([1.0; 11]);
        let expected = (11.0_f32).sqrt();
        assert!((vec.magnitude() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_dot_product() {
        let v1 = ConsciousnessVector::from_array([1.0; 11]);
        let v2 = ConsciousnessVector::from_array([2.0; 11]);
        assert_eq!(v1.dot(&v2), 22.0); // 11 * 1 * 2
    }

    #[test]
    fn test_alignment_parallel() {
        let v1 = ConsciousnessVector::from_array([1.0; 11]);
        let v2 = ConsciousnessVector::from_array([2.0; 11]);
        assert!((v1.alignment(&v2) - 1.0).abs() < 1e-5); // Parallel vectors
    }

    #[test]
    fn test_alignment_orthogonal() {
        let mut arr1 = [0.0; 11];
        arr1[0] = 1.0;
        let mut arr2 = [0.0; 11];
        arr2[1] = 1.0;

        let v1 = ConsciousnessVector::from_array(arr1);
        let v2 = ConsciousnessVector::from_array(arr2);
        assert!((v1.alignment(&v2) - 0.0).abs() < 1e-5); // Orthogonal vectors
    }

    #[test]
    fn test_projection() {
        let v = ConsciousnessVector::from_array([
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let onto = ConsciousnessVector::from_array([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);

        let proj = v.project_onto(&onto);
        assert!((proj.rupa_energy - 1.0).abs() < 1e-5);
        assert!((proj.vedana_valence - 0.0).abs() < 1e-5);
    }

    // ========================================================================
    // PhaseConsciousnessVector Tests
    // ========================================================================

    #[test]
    fn test_phase_default() {
        let pv = PhaseConsciousnessVector::default();
        assert!(pv.is_valid());
        // Default dharma phase should be 0 (aligned)
        assert!(pv.dharma_phase().abs() < 1e-5);
        // Should pass ethics check
        assert!(pv.check_phase_ethics());
    }

    #[test]
    fn test_phase_from_real() {
        let real = ConsciousnessVector::default();
        let phase = PhaseConsciousnessVector::from_real(&real);
        
        // All imaginary parts should be 0
        for c in phase.dimensions.iter() {
            assert!(c.im.abs() < 1e-5);
        }
        
        // Dharma phase should be 0 (aligned)
        assert!(phase.check_phase_ethics());
    }

    #[test]
    fn test_phase_ethics_aligned() {
        // Phase = 0 (positive real): Perfectly aligned
        let mut pv = PhaseConsciousnessVector::default();
        pv.dimensions[PhaseConsciousnessVector::SANKHARA_IDX] = Complex32::new(1.0, 0.0);
        assert!(pv.check_phase_ethics());
        assert!(pv.dharma_phase().abs() < 1e-5);
    }

    #[test]
    fn test_phase_ethics_neutral() {
        // Phase = π/4: Within tolerance
        let mut pv = PhaseConsciousnessVector::default();
        pv.dimensions[PhaseConsciousnessVector::SANKHARA_IDX] = 
            Complex32::from_polar(1.0, std::f32::consts::FRAC_PI_4);
        assert!(pv.check_phase_ethics()); // π/4 < π/2
    }

    #[test]
    fn test_phase_ethics_boundary() {
        // Phase = π/2: At boundary (should pass)
        let mut pv = PhaseConsciousnessVector::default();
        pv.dimensions[PhaseConsciousnessVector::SANKHARA_IDX] = 
            Complex32::from_polar(1.0, std::f32::consts::FRAC_PI_2);
        assert!(pv.check_phase_ethics()); // π/2 == π/2 (boundary case)
    }

    #[test]
    fn test_phase_ethics_misaligned() {
        // Phase = 3π/4: Beyond tolerance (should veto)
        let mut pv = PhaseConsciousnessVector::default();
        pv.dimensions[PhaseConsciousnessVector::SANKHARA_IDX] = 
            Complex32::from_polar(1.0, 3.0 * std::f32::consts::FRAC_PI_4);
        assert!(!pv.check_phase_ethics()); // 3π/4 > π/2
    }

    #[test]
    fn test_phase_ethics_opposite() {
        // Phase = π: Completely opposed (harmful)
        let mut pv = PhaseConsciousnessVector::default();
        pv.dimensions[PhaseConsciousnessVector::SANKHARA_IDX] = Complex32::new(-1.0, 0.0);
        assert!(!pv.check_phase_ethics()); // π > π/2
    }

    #[test]
    fn test_rotate_dharma() {
        let mut pv = PhaseConsciousnessVector::default();
        // Start aligned
        assert!(pv.check_phase_ethics());
        
        // Rotate by π/4 (still within tolerance)
        pv.rotate_dharma(std::f32::consts::FRAC_PI_4);
        assert!(pv.check_phase_ethics());
        
        // Rotate by another π/2 (now beyond tolerance)
        pv.rotate_dharma(std::f32::consts::FRAC_PI_2);
        assert!(!pv.check_phase_ethics());
    }

    #[test]
    fn test_phase_roundtrip() {
        let real = ConsciousnessVector::from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]);
        let phase = PhaseConsciousnessVector::from_real(&real);
        let back = phase.to_real();
        
        // Magnitudes should match (phase = 0, so magnitude == value)
        for (r, b) in real.to_array().iter().zip(back.to_array().iter()) {
            assert!((r - b).abs() < 1e-5);
        }
    }
}

