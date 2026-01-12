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
            rupa_energy: 0.5, // Not directly available, placeholder
            vedana_valence: 0.0, // Would come from affect if available
            sanna_similarity: 0.5, // Would come from pattern if available
            sankhara_alignment: 1.0, // Would come from intent if available
            vinnana_confidence: state.confidence,
            bio_arousal: 0.5, // From belief state
            cognitive_focus: 0.5, // From belief state
            social_load: 0.0, // From belief state
            circadian_phase: 0.5, // From context
            environment_stress: 0.0, // From context
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
        let v = ConsciousnessVector::from_array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let onto = ConsciousnessVector::from_array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        
        let proj = v.project_onto(&onto);
        assert!((proj.rupa_energy - 1.0).abs() < 1e-5);
        assert!((proj.vedana_valence - 0.0).abs() < 1e-5);
    }
}
