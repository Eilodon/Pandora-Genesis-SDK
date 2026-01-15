//! Unified ZENITH Memory System
//!
//! Top-level orchestration layer that combines all ZENITH tiers into a single,
//! intelligent memory interface.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      ZenithMemory                           │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │               AdaptiveMemoryRouter                   │   │
//! │  │  (Tier 1: Task-based backend selection)              │   │
//! │  └──────────────────────┬──────────────────────────────┘   │
//! │                         ▼                                   │
//! │  ┌────────────────────────────────────────────────────────┐│
//! │  │  ┌──────────┐  ┌───────────────────────────────────┐  ││
//! │  │  │ HDC/     │  │  TieredHDC (Context-aware)        │  ││
//! │  │  │ SparseHDC│  │  Primary backend for all tasks    │  ││
//! │  │  └──────────┘  └───────────────────────────────────┘  ││
//! │  │              Backend Pool                              ││
//! │  └────────────────────────────────────────────────────────┘│
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │         UncertaintyAwareRetrieval                    │   │
//! │  │  (Tier 4: Epistemic/Aleatoric quantification)        │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use super::{
    router::{AdaptiveMemoryRouter, MemoryBackend, RouterConfig, TaskType},
    uncertainty::{UncertaintyAwareRetrieval, UncertainRetrievalResult},
    HdcMemory, TieredHdcMemory,
};

/// Configuration for ZenithMemory
#[derive(Debug, Clone)]
pub struct ZenithConfig {
    /// Enable router-based backend selection
    pub enable_routing: bool,
    /// Enable uncertainty quantification on retrieval
    pub enable_uncertainty: bool,
    /// Context weight for context-aware retrieval (0.0-1.0)
    pub context_weight: f32,
}

impl Default for ZenithConfig {
    fn default() -> Self {
        Self {
            enable_routing: true,
            enable_uncertainty: true,
            context_weight: 0.3,
        }
    }
}

/// Unified ZENITH Memory System
///
/// Orchestrates all ZENITH tiers for intelligent, uncertainty-aware memory operations.
#[derive(Debug)]
pub struct ZenithMemory {
    /// Tier 1: Adaptive routing
    router: AdaptiveMemoryRouter,
    /// Backend: HDC memory (symbolic data)
    hdc: HdcMemory,
    /// Backend: Tiered HDC with context (episodic data, primary)
    tiered: TieredHdcMemory,
    /// Tier 4: Uncertainty quantification
    uncertainty: UncertaintyAwareRetrieval,
    /// Configuration
    config: ZenithConfig,
    /// Statistics
    total_stores: u64,
    total_retrieves: u64,
}

impl Default for ZenithMemory {
    fn default() -> Self {
        Self::new(ZenithConfig::default())
    }
}

impl ZenithMemory {
    /// Create with custom configuration
    pub fn new(config: ZenithConfig) -> Self {
        Self {
            router: AdaptiveMemoryRouter::new(RouterConfig::default()),
            hdc: HdcMemory::for_zenb(),
            tiered: TieredHdcMemory::for_zenb(),
            uncertainty: UncertaintyAwareRetrieval::default(),
            config,
            total_stores: 0,
            total_retrieves: 0,
        }
    }

    /// Store value with automatic backend selection
    ///
    /// Uses context to determine optimal backend via router.
    pub fn store(&mut self, key: &[f32], value: &[f32], context: &[f32]) {
        self.total_stores += 1;

        if self.config.enable_routing {
            let (backend, _) = self.router.route(context);
            self.store_to_backend(backend, key, value, context);
        } else {
            // Default to tiered HDC
            self.store_to_backend(MemoryBackend::TieredHDC, key, value, context);
        }
    }

    /// Store to specific backend
    fn store_to_backend(
        &mut self,
        backend: MemoryBackend,
        key: &[f32],
        value: &[f32],
        context: &[f32],
    ) {
        match backend {
            MemoryBackend::BinaryHDC => {
                let key_vec = self.hdc.encode_features(key);
                let val_vec = self.hdc.encode_features(value);
                self.hdc.store(key_vec, val_vec);
            }
            MemoryBackend::TieredHDC | MemoryBackend::Holographic => {
                // TieredHDC is primary backend, also used for Holographic tasks
                let key_vec = self.tiered.encode_features(key);
                let val_vec = self.tiered.encode_features(value);
                self.tiered.store_with_context(key_vec, val_vec, context);
            }
            MemoryBackend::Hybrid { .. } => {
                // Store to both HDC and Tiered
                let key_hdc = self.hdc.encode_features(key);
                let val_hdc = self.hdc.encode_features(value);
                self.hdc.store(key_hdc, val_hdc);

                let key_tiered = self.tiered.encode_features(key);
                let val_tiered = self.tiered.encode_features(value);
                self.tiered.store_with_context(key_tiered, val_tiered, context);
            }
        }
    }

    /// Retrieve with optional uncertainty quantification
    pub fn retrieve(
        &mut self,
        query: &[f32],
        context: &[f32],
    ) -> Option<ZenithRetrievalResult> {
        self.total_retrieves += 1;

        let (backend, routing_confidence) = if self.config.enable_routing {
            self.router.route(context)
        } else {
            (MemoryBackend::TieredHDC, 0.5)
        };

        let result = self.retrieve_from_backend(backend, query, context);

        result.map(|(value, similarity)| ZenithRetrievalResult {
            value,
            similarity,
            backend,
            routing_confidence,
            epistemic_uncertainty: None,
            aleatoric_uncertainty: None,
        })
    }

    /// Retrieve from specific backend
    fn retrieve_from_backend(
        &mut self,
        backend: MemoryBackend,
        query: &[f32],
        context: &[f32],
    ) -> Option<(Vec<f32>, f32)> {
        match backend {
            MemoryBackend::BinaryHDC => {
                let query_vec = self.hdc.encode_features(query);
                self.hdc.retrieve(&query_vec).map(|(val, sim)| {
                    // Convert HDC vector back to features (simplified extraction)
                    let mut features = vec![0.0f32; query.len().min(16)];
                    for (i, &word) in val.as_slice().iter().enumerate().take(features.len()) {
                        features[i] = (word as f32 / u64::MAX as f32).clamp(0.0, 1.0);
                    }
                    (features, sim)
                })
            }
            MemoryBackend::TieredHDC | MemoryBackend::Holographic => {
                let query_vec = self.tiered.encode_features(query);
                self.tiered
                    .retrieve_context_aware(&query_vec, context, self.config.context_weight)
                    .map(|(sim, _tier)| {
                        // Return query as placeholder (real impl would decode value)
                        (query.to_vec(), sim)
                    })
            }
            MemoryBackend::Hybrid { holo_weight } => {
                // Try tiered first if holographic weight is high
                if holo_weight >= 50 {
                    let query_vec = self.tiered.encode_features(query);
                    self.tiered
                        .retrieve_context_aware(&query_vec, context, self.config.context_weight)
                        .map(|(sim, _)| (query.to_vec(), sim))
                } else {
                    let query_vec = self.hdc.encode_features(query);
                    self.hdc.retrieve(&query_vec).map(|(val, sim)| {
                        let mut features = vec![0.0f32; query.len().min(16)];
                        for (i, &word) in val.as_slice().iter().enumerate().take(features.len()) {
                            features[i] = (word as f32 / u64::MAX as f32).clamp(0.0, 1.0);
                        }
                        (features, sim)
                    })
                }
            }
        }
    }

    /// Retrieve with full uncertainty quantification (ZENITH Tier 4)
    ///
    /// Performs ensemble retrieval to estimate epistemic and aleatoric uncertainty.
    pub fn retrieve_uncertain(
        &mut self,
        query: &[f32],
        context: &[f32],
    ) -> Option<UncertainRetrievalResult<Vec<f32>>> {
        if !self.config.enable_uncertainty {
            // Fall back to non-uncertainty retrieval
            return self.retrieve(query, context).map(|r| UncertainRetrievalResult {
                value: r.value,
                confidence: r.similarity,
                epistemic_uncertainty: 0.0,
                aleatoric_uncertainty: 0.0,
            });
        }

        // Use uncertainty module with ensemble retrieval on tiered memory
        let ctx_weight = self.config.context_weight;
        let context_vec = context.to_vec();

        self.uncertainty.retrieve_with_uncertainty(query, |q| {
            let query_vec = self.tiered.encode_features(q);
            self.tiered
                .retrieve_context_aware(&query_vec, &context_vec, ctx_weight)
                .map(|(sim, _tier)| (q.to_vec(), sim))
        })
    }

    /// Update router performance based on retrieval outcome
    pub fn report_outcome(&mut self, task_type: TaskType, backend: MemoryBackend, success: bool) {
        self.router.update_performance(task_type, backend, success, 1000.0);
    }

    /// Get statistics
    pub fn stats(&self) -> ZenithStats {
        let router_stats = self.router.stats();
        ZenithStats {
            total_stores: self.total_stores,
            total_retrieves: self.total_retrieves,
            routing_decisions: router_stats.total_routes,
            routing_enabled: self.config.enable_routing,
            uncertainty_enabled: self.config.enable_uncertainty,
        }
    }
}

/// Result of ZENITH retrieval
#[derive(Debug, Clone)]
pub struct ZenithRetrievalResult {
    /// Retrieved value
    pub value: Vec<f32>,
    /// Similarity score
    pub similarity: f32,
    /// Backend used
    pub backend: MemoryBackend,
    /// Router confidence in backend selection
    pub routing_confidence: f32,
    /// Epistemic uncertainty (if computed)
    pub epistemic_uncertainty: Option<f32>,
    /// Aleatoric uncertainty (if computed)
    pub aleatoric_uncertainty: Option<f32>,
}

/// ZENITH system statistics
#[derive(Debug, Clone)]
pub struct ZenithStats {
    pub total_stores: u64,
    pub total_retrieves: u64,
    pub routing_decisions: u64,
    pub routing_enabled: bool,
    pub uncertainty_enabled: bool,
}

// ============================================================================
// SannaSkandha Integration (Tưởng Uẩn - Perception/Memory)
// ============================================================================

use crate::skandha::{AffectiveState, PerceivedPattern, ProcessedForm, SannaSkandha};

/// Implement SannaSkandha for ZenithMemory.
/// 
/// This bridges the ZENITH memory system into the Ngũ Uẩn pipeline,
/// making ZenithMemory the primary Perception (Tưởng) processing stage.
/// 
/// # Tưởng Uẩn Processing
/// - Uses ProcessedForm values as query for memory retrieval
/// - Uses AffectiveState (arousal, karma_weight) as context
/// - Returns PerceivedPattern with recalled pattern ID and similarity
impl SannaSkandha for ZenithMemory {
    /// Perceive patterns from processed sensory form.
    /// 
    /// # Tưởng Uẩn Flow
    /// 1. Convert ProcessedForm values to query vector
    /// 2. Convert AffectiveState to context vector 
    /// 3. Retrieve from ZENITH memory with uncertainty
    /// 4. Map retrieval result to PerceivedPattern
    fn perceive(&mut self, form: &ProcessedForm, affect: &AffectiveState) -> PerceivedPattern {
        // Build query from ProcessedForm values
        let query: Vec<f32> = form.values.to_vec();
        
        // Build context from AffectiveState
        // Context: [valence, arousal, confidence, karma_weight, energy]
        let context = vec![
            affect.valence,
            affect.arousal,
            affect.confidence,
            affect.karma_weight,
            form.energy,
        ];
        
        // Retrieve from memory
        match self.retrieve(&query, &context) {
            Some(result) => {
                // Generate pattern_id from similarity hash
                let pattern_id = (result.similarity * 1_000_000.0) as u64;
                
                // Check for trauma association (low similarity + high arousal)
                let is_trauma = result.similarity < 0.3 && affect.arousal > 0.7;
                
                PerceivedPattern {
                    pattern_id,
                    similarity: result.similarity,
                    context: [
                        context[0],
                        context[1],
                        context[2],
                        context[3],
                        context[4],
                    ],
                    is_trauma_associated: is_trauma,
                }
            }
            None => {
                // No pattern found - return default with low similarity
                PerceivedPattern {
                    pattern_id: 0,
                    similarity: 0.0,
                    context: [
                        affect.valence,
                        affect.arousal,
                        affect.confidence,
                        affect.karma_weight,
                        form.energy,
                    ],
                    is_trauma_associated: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zenith_basic() {
        let mut zenith = ZenithMemory::default();

        let key = vec![0.5, 0.6, 0.7];
        let value = vec![1.0, 0.8, 0.2];
        let context = vec![0.9, 0.1, 0.0, 0.0]; // Sensory context

        zenith.store(&key, &value, &context);

        let stats = zenith.stats();
        assert_eq!(stats.total_stores, 1);
    }

    #[test]
    fn test_zenith_retrieve() {
        let mut zenith = ZenithMemory::default();

        let key = vec![0.5, 0.6, 0.7, 0.8];
        let value = vec![1.0, 0.8, 0.2, 0.1];
        let context = vec![0.9, 0.1, 0.0, 0.0];

        zenith.store(&key, &value, &context);
        let _result = zenith.retrieve(&key, &context);

        let stats = zenith.stats();
        assert_eq!(stats.total_retrieves, 1);
    }

    #[test]
    fn test_zenith_routing_disabled() {
        let config = ZenithConfig {
            enable_routing: false,
            enable_uncertainty: false,
            context_weight: 0.3,
        };
        let mut zenith = ZenithMemory::new(config);

        let key = vec![0.5, 0.6];
        let value = vec![1.0, 0.8];
        let context = vec![0.5, 0.5];

        zenith.store(&key, &value, &context);

        let stats = zenith.stats();
        assert!(!stats.routing_enabled);
    }

    #[test]
    fn test_zenith_stats() {
        let zenith = ZenithMemory::default();
        let stats = zenith.stats();

        assert_eq!(stats.total_stores, 0);
        assert_eq!(stats.total_retrieves, 0);
        assert!(stats.routing_enabled);
        assert!(stats.uncertainty_enabled);
    }
}
