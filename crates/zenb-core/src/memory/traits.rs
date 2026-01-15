//! Memory Backend Trait System
//!
//! Unified trait-based interface for all memory implementations.
//! Enables pluggable backends and standardized memory operations.
//!
//! # Core Traits
//! - [`MemoryBackend`] - Base trait for all memory systems
//! - [`UncertaintyAwareBackend`] - Extension for uncertainty quantification
//! - [`TieredBackend`] - Extension for tiered memory systems
//!
//! # Example
//! ```ignore
//! use zenb_core::memory::{MemoryBackend, HolographicMemory};
//!
//! let mut mem: Box<dyn MemoryBackend> = Box::new(HolographicMemory::new(256));
//! mem.store(&key, &value, None);
//! let result = mem.retrieve(&key);
//! ```

use std::fmt::Debug;

use super::uncertainty::UncertainRetrievalResult;

// ============================================================================
// Core MemoryBackend Trait
// ============================================================================

/// Unified memory backend trait.
///
/// All memory implementations (Holographic, HDC, Tiered, etc.) implement this.
/// Provides a standardized interface for memory operations.
///
/// # Thread Safety
/// Implementors should be `Send + Sync` where possible for concurrent access.
/// Use `SharedMemory<T>` wrapper for non-thread-safe implementations.
///
/// # Performance
/// - `store`: O(dim) to O(dim log dim) depending on implementation
/// - `retrieve`: O(dim log dim) for holographic, O(n * dim) for linear search
/// - `decay`: O(dim) uniform decay, O(n * dim) for item-wise
pub trait MemoryBackend: Send + Sync + Debug {
    /// Store key-value pair with optional context.
    ///
    /// # Arguments
    /// * `key` - Query vector for later retrieval
    /// * `value` - Associated value to store
    /// * `context` - Optional contextual features (e.g., time, location)
    fn store(&mut self, key: &[f32], value: &[f32], context: Option<&[f32]>);

    /// Retrieve value by key.
    ///
    /// # Returns
    /// `Some((value, similarity))` if found, `None` if no match.
    /// Similarity is in range [0, 1], higher = better match.
    fn retrieve(&mut self, key: &[f32]) -> Option<(Vec<f32>, f32)>;

    /// Apply decay/forgetting mechanism.
    ///
    /// # Arguments
    /// * `factor` - Decay multiplier (0.0 = forget all, 1.0 = keep all)
    fn decay(&mut self, factor: f32);

    /// Clear all stored memories.
    fn clear(&mut self);

    /// Get backend name for logging/routing.
    fn backend_name(&self) -> &'static str;

    /// Get current item count (approximate for holographic).
    fn item_count(&self) -> usize;

    /// Get memory dimension.
    fn dimension(&self) -> usize;

    // ==========================================
    // Default implementations
    // ==========================================

    /// Check if memory is empty.
    fn is_empty(&self) -> bool {
        self.item_count() == 0
    }

    /// Store without context (convenience method).
    fn store_simple(&mut self, key: &[f32], value: &[f32]) {
        self.store(key, value, None);
    }

    /// Get capacity utilization (0.0 - 1.0).
    /// Default: returns 0.5 (unknown).
    fn utilization(&self) -> f32 {
        0.5
    }
}

// ============================================================================
// UncertaintyAwareBackend Extension
// ============================================================================

/// Extension trait for uncertainty-aware memory backends.
///
/// Provides uncertainty quantification for retrieved values,
/// distinguishing between:
/// - **Epistemic uncertainty**: Model/knowledge uncertainty (reducible)
/// - **Aleatoric uncertainty**: Data/noise uncertainty (irreducible)
pub trait UncertaintyAwareBackend: MemoryBackend {
    /// Retrieve with full uncertainty quantification.
    ///
    /// # Returns
    /// `Some(result)` with value, mean, variance, and uncertainty decomposition.
    fn retrieve_uncertain(&mut self, key: &[f32]) -> Option<UncertainRetrievalResult<Vec<f32>>>;

    /// Retrieve with simple confidence score.
    ///
    /// Default implementation uses retrieve() similarity as confidence proxy.
    fn retrieve_with_confidence(&mut self, key: &[f32]) -> Option<(Vec<f32>, f32)> {
        self.retrieve(key)
    }
}

// ============================================================================
// TieredBackend Extension
// ============================================================================

/// Statistics for tiered memory systems.
#[derive(Debug, Clone, Default)]
pub struct TierStats {
    /// Items in working memory (fast, ephemeral)
    pub working_count: usize,
    /// Items in long-term memory (slow, persistent)
    pub long_term_count: usize,
    /// Total consolidations performed
    pub consolidation_count: u64,
    /// Total evictions from working memory
    pub eviction_count: u64,
}

/// Extension trait for tiered memory systems.
///
/// Supports the LifeHD pattern:
/// - Working memory: Fast, limited capacity
/// - Long-term memory: Slower, larger capacity, anti-forgetting
pub trait TieredBackend: MemoryBackend {
    /// Check if pattern is consolidated to long-term memory.
    fn is_consolidated(&self, key: &[f32]) -> bool;

    /// Force consolidation of a pattern to long-term memory.
    ///
    /// # Returns
    /// `true` if pattern was found and consolidated.
    fn consolidate(&mut self, key: &[f32]) -> bool;

    /// Get tier-specific statistics.
    fn tier_stats(&self) -> TierStats;

    /// Get working memory capacity.
    fn working_capacity(&self) -> usize;

    /// Get long-term memory capacity.
    fn long_term_capacity(&self) -> usize;
}

// ============================================================================
// BackendType Enum (for routing)
// ============================================================================

/// Backend type enum for routing decisions.
///
/// This replaces the old `MemoryBackend` enum and is used by
/// `AdaptiveMemoryRouter` to select backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// FFT-based holographic memory
    Holographic,
    /// Binary HDC (NPU-accelerated)
    BinaryHDC,
    /// Two-tier HDC with consolidation
    TieredHDC,
    /// Weighted combination of backends
    Hybrid { holo_weight: u8 },
}

impl BackendType {
    /// Get hybrid with equal weights.
    pub fn hybrid_equal() -> Self {
        Self::Hybrid { holo_weight: 50 }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Holographic => "Holographic",
            Self::BinaryHDC => "BinaryHDC",
            Self::TieredHDC => "TieredHDC",
            Self::Hybrid { .. } => "Hybrid",
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock backend for testing
    #[derive(Debug)]
    struct MockBackend {
        items: Vec<(Vec<f32>, Vec<f32>)>,
        dim: usize,
    }

    impl MockBackend {
        fn new(dim: usize) -> Self {
            Self { items: vec![], dim }
        }
    }

    impl MemoryBackend for MockBackend {
        fn store(&mut self, key: &[f32], value: &[f32], _context: Option<&[f32]>) {
            self.items.push((key.to_vec(), value.to_vec()));
        }

        fn retrieve(&mut self, key: &[f32]) -> Option<(Vec<f32>, f32)> {
            // Simple linear search with cosine similarity
            self.items
                .iter()
                .map(|(k, v)| {
                    let dot: f32 = k.iter().zip(key.iter()).map(|(a, b)| a * b).sum();
                    let norm_k: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let norm_q: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let sim = if norm_k > 0.0 && norm_q > 0.0 {
                        dot / (norm_k * norm_q)
                    } else {
                        0.0
                    };
                    (v.clone(), sim)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        }

        fn decay(&mut self, _factor: f32) {
            // Mock: no-op
        }

        fn clear(&mut self) {
            self.items.clear();
        }

        fn backend_name(&self) -> &'static str {
            "Mock"
        }

        fn item_count(&self) -> usize {
            self.items.len()
        }

        fn dimension(&self) -> usize {
            self.dim
        }
    }

    #[test]
    fn test_memory_backend_trait() {
        let mut backend: Box<dyn MemoryBackend> = Box::new(MockBackend::new(4));

        assert!(backend.is_empty());
        assert_eq!(backend.backend_name(), "Mock");
        assert_eq!(backend.dimension(), 4);

        backend.store_simple(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(backend.item_count(), 1);

        let result = backend.retrieve(&[1.0, 0.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (value, sim) = result.unwrap();
        assert_eq!(value, vec![0.0, 1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_backend_type_enum() {
        let holo = BackendType::Holographic;
        assert_eq!(holo.name(), "Holographic");

        let hybrid = BackendType::hybrid_equal();
        assert_eq!(hybrid, BackendType::Hybrid { holo_weight: 50 });
    }
}
