//! Tiered Hyperdimensional Computing Memory (LifeHD Pattern)
//!
//! Implementation of a two-tier HDC memory system inspired by LifeHD research
//! (USENIX 2024), achieving 74.8% accuracy improvement with 34x energy efficiency.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                   TieredHdcMemory                       │
//! │  ┌───────────────────┐    ┌────────────────────────┐   │
//! │  │  Working Memory   │───▶│   Long-Term Memory     │   │
//! │  │  (fast, limited)  │    │   (large, persistent)  │   │
//! │  │  256 patterns     │    │   2048 patterns        │   │
//! │  └───────────────────┘    └────────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//! - **Dual-tier storage**: Fast working memory + persistent long-term memory
//! - **Automatic consolidation**: Frequently accessed patterns migrate to long-term
//! - **Importance-weighted eviction**: Long-term uses smart forgetting, not FIFO
//! - **Access tracking**: Patterns "strengthen" with repeated retrieval
//!
//! # Mathematical Foundation
//! Based on LifeHD's hierarchical hypervector organization:
//! - Working memory: Recent patterns with access counters
//! - Long-term memory: Consolidated patterns with importance scores
//! - Migration criterion: access_count >= consolidation_threshold
//!
//! # Research Citation
//! LifeHD (USENIX 2024): "Lifelong Learning for Edge Device with Hyperdimensional Computing"
//! - 74.8% accuracy improvement over neural baselines
//! - 34x energy efficiency on edge hardware

use super::hdc::{HdcConfig, HdcMemory, HdcVector};

/// Configuration for Tiered HDC Memory
#[derive(Debug, Clone)]
pub struct TieredHdcConfig {
    /// Configuration for working memory (fast, limited capacity)
    pub working_config: HdcConfig,
    /// Configuration for long-term memory (large, persistent)
    pub long_term_config: HdcConfig,
    /// Number of accesses required before consolidation to long-term
    pub consolidation_threshold: u32,
    /// Enable automatic consolidation after each retrieve
    pub auto_consolidate: bool,
    /// Decay rate for working memory (applied periodically)
    pub working_decay_rate: f32,
    /// Decay rate for long-term memory (applied less frequently)
    pub long_term_decay_rate: f32,
}

impl Default for TieredHdcConfig {
    fn default() -> Self {
        Self {
            working_config: HdcConfig {
                dimension: 4096,
                max_patterns: 256, // Fast, limited
                similarity_threshold: 0.6,
            },
            long_term_config: HdcConfig {
                dimension: 4096,
                max_patterns: 2048, // Large, persistent
                similarity_threshold: 0.55, // Slightly more lenient
            },
            consolidation_threshold: 5, // Access 5 times → consolidate
            auto_consolidate: true,
            working_decay_rate: 0.95,
            long_term_decay_rate: 0.99, // Much slower decay
        }
    }
}

impl TieredHdcConfig {
    /// Configuration optimized for AGOLOS/ZenB
    pub fn for_zenb() -> Self {
        Self {
            working_config: HdcConfig {
                dimension: 4096,
                max_patterns: 128, // Even faster for real-time
                similarity_threshold: 0.6,
            },
            long_term_config: HdcConfig {
                dimension: 4096,
                max_patterns: 1024,
                similarity_threshold: 0.55,
            },
            consolidation_threshold: 3, // Faster consolidation for responsive learning
            auto_consolidate: true,
            working_decay_rate: 0.9,
            long_term_decay_rate: 0.98,
        }
    }

    /// Minimal configuration for testing
    pub fn minimal() -> Self {
        Self {
            working_config: HdcConfig {
                dimension: 1024,
                max_patterns: 32,
                similarity_threshold: 0.5,
            },
            long_term_config: HdcConfig {
                dimension: 1024,
                max_patterns: 128,
                similarity_threshold: 0.5,
            },
            consolidation_threshold: 2,
            auto_consolidate: true,
            working_decay_rate: 0.9,
            long_term_decay_rate: 0.95,
        }
    }
}

/// Entry in working memory with access tracking
#[derive(Debug, Clone)]
struct WorkingEntry {
    key: HdcVector,
    value: HdcVector,
    access_count: u32,
    importance: f32, // Computed from access patterns
}

/// Tiered Hyperdimensional Computing Memory
///
/// Two-tier memory system implementing the LifeHD pattern for efficient
/// lifelong learning with anti-forgetting properties.
///
/// # Example
/// ```ignore
/// use zenb_core::memory::TieredHdcMemory;
///
/// let mut mem = TieredHdcMemory::for_zenb();
///
/// // Store pattern
/// let key = mem.encode_features(&[0.5, 0.6, 0.7]);
/// let value = mem.encode_features(&[1.0, 0.8, 0.2]);
/// mem.store(key.clone(), value);
///
/// // Access multiple times → automatic consolidation to long-term
/// for _ in 0..5 {
///     let _ = mem.retrieve(&key);
/// }
///
/// // Pattern is now in long-term memory, resistant to forgetting
/// assert!(mem.is_in_long_term(&key));
/// ```
#[derive(Debug)]
pub struct TieredHdcMemory {
    /// Working memory entries with access tracking
    working_entries: Vec<WorkingEntry>,
    /// Long-term memory (uses standard HdcMemory)
    long_term: HdcMemory,
    /// Configuration
    config: TieredHdcConfig,
    /// Total retrievals across both tiers
    total_retrievals: u64,
    /// Successful retrievals
    successful_retrievals: u64,
    /// Items consolidated to long-term
    consolidation_count: u64,
}

impl TieredHdcMemory {
    /// Create new tiered memory with given configuration
    pub fn new(config: TieredHdcConfig) -> Self {
        Self {
            working_entries: Vec::with_capacity(config.working_config.max_patterns),
            long_term: HdcMemory::new(config.long_term_config.clone()),
            config,
            total_retrievals: 0,
            successful_retrievals: 0,
            consolidation_count: 0,
        }
    }

    /// Create with ZenB-optimized configuration
    pub fn for_zenb() -> Self {
        Self::new(TieredHdcConfig::for_zenb())
    }

    /// Create with minimal configuration for testing
    pub fn minimal() -> Self {
        Self::new(TieredHdcConfig::minimal())
    }

    /// Store a key-value pair (enters working memory first)
    pub fn store(&mut self, key: HdcVector, value: HdcVector) {
        // Check if already in working memory (update if so)
        for entry in &mut self.working_entries {
            if entry.key.similarity(&key) > 0.95 {
                // Update existing entry
                entry.value = value;
                entry.access_count += 1;
                // Inline importance computation to avoid borrow conflict
                entry.importance = 1.0 + (entry.access_count as f32).ln();
                return;
            }
        }

        // Check capacity and evict if needed
        if self.working_entries.len() >= self.config.working_config.max_patterns {
            self.evict_working_memory();
        }

        // Add new entry
        self.working_entries.push(WorkingEntry {
            key,
            value,
            access_count: 1,
            importance: 1.0,
        });
    }

    /// Store from feature vectors (convenience wrapper)
    pub fn store_features(&mut self, key_features: &[f32], value_features: &[f32]) {
        let key = self.encode_features(key_features);
        let value = self.encode_features(value_features);
        self.store(key, value);
    }

    /// Retrieve value for given key
    ///
    /// Searches both tiers: working memory first (fast path), then long-term.
    /// Increments access count on hit, triggering consolidation if threshold met.
    /// 
    /// Returns (similarity, tier) only. Use `retrieve_value` if you need the actual value.
    pub fn retrieve(&mut self, query: &HdcVector) -> Option<(f32, MemoryTier)> {
        self.total_retrievals += 1;

        // Search working memory first (fast path)
        let mut best_working_idx: Option<usize> = None;
        let mut best_working_sim = 0.0f32;

        for (i, entry) in self.working_entries.iter().enumerate() {
            let sim = query.similarity(&entry.key);
            if sim > best_working_sim && sim >= self.config.working_config.similarity_threshold {
                best_working_sim = sim;
                best_working_idx = Some(i);
            }
        }

        // Check long-term memory (extract similarity only, not reference)
        let long_term_sim = self.long_term.retrieve(query).map(|(_, sim)| sim);

        // Determine best match across both tiers
        match (best_working_idx, long_term_sim) {
            (Some(idx), Some(lt_sim)) => {
                if best_working_sim >= lt_sim {
                    // Working memory wins
                    self.handle_working_hit(idx);
                    self.successful_retrievals += 1;
                    Some((best_working_sim, MemoryTier::Working))
                } else {
                    // Long-term wins
                    self.successful_retrievals += 1;
                    Some((lt_sim, MemoryTier::LongTerm))
                }
            }
            (Some(idx), None) => {
                self.handle_working_hit(idx);
                self.successful_retrievals += 1;
                Some((best_working_sim, MemoryTier::Working))
            }
            (None, Some(lt_sim)) => {
                self.successful_retrievals += 1;
                Some((lt_sim, MemoryTier::LongTerm))
            }
            (None, None) => None,
        }
    }

    /// Retrieve the actual value for a given key (clones the value)
    /// 
    /// Use this when you need the stored value. Returns cloned HdcVector.
    pub fn retrieve_value(&mut self, query: &HdcVector) -> Option<(HdcVector, f32, MemoryTier)> {
        self.total_retrievals += 1;

        // Search working memory
        let mut best_working: Option<(HdcVector, f32)> = None;
        let mut best_working_idx: Option<usize> = None;

        for (i, entry) in self.working_entries.iter().enumerate() {
            let sim = query.similarity(&entry.key);
            if sim >= self.config.working_config.similarity_threshold {
                if best_working.as_ref().map(|(_, s)| sim > *s).unwrap_or(true) {
                    best_working = Some((entry.value.clone(), sim));
                    best_working_idx = Some(i);
                }
            }
        }

        // Search long-term memory
        let long_term_result = self.long_term.retrieve(query).map(|(v, s)| (v.clone(), s));

        // Determine best match
        match (best_working, long_term_result) {
            (Some((wk_val, wk_sim)), Some((lt_val, lt_sim))) => {
                if wk_sim >= lt_sim {
                    if let Some(idx) = best_working_idx {
                        self.handle_working_hit(idx);
                    }
                    self.successful_retrievals += 1;
                    Some((wk_val, wk_sim, MemoryTier::Working))
                } else {
                    self.successful_retrievals += 1;
                    Some((lt_val, lt_sim, MemoryTier::LongTerm))
                }
            }
            (Some((wk_val, wk_sim)), None) => {
                if let Some(idx) = best_working_idx {
                    self.handle_working_hit(idx);
                }
                self.successful_retrievals += 1;
                Some((wk_val, wk_sim, MemoryTier::Working))
            }
            (None, Some((lt_val, lt_sim))) => {
                self.successful_retrievals += 1;
                Some((lt_val, lt_sim, MemoryTier::LongTerm))
            }
            (None, None) => None,
        }
    }

    /// Retrieve using feature vector (returns similarity and tier only)
    pub fn retrieve_features(&mut self, query_features: &[f32]) -> Option<(f32, MemoryTier)> {
        let query = self.encode_features(query_features);
        self.retrieve(&query)
    }

    /// Retrieve using feature vector (returns cloned value)
    pub fn retrieve_features_value(&mut self, query_features: &[f32]) -> Option<(HdcVector, f32, MemoryTier)> {
        let query = self.encode_features(query_features);
        self.retrieve_value(&query)
    }

    /// Handle a hit in working memory (increment access, check consolidation)
    fn handle_working_hit(&mut self, idx: usize) {
        self.working_entries[idx].access_count += 1;
        self.working_entries[idx].importance =
            self.compute_importance(self.working_entries[idx].access_count);

        // Check for consolidation
        if self.config.auto_consolidate
            && self.working_entries[idx].access_count >= self.config.consolidation_threshold
        {
            self.consolidate_entry(idx);
        }
    }

    /// Consolidate a specific entry to long-term memory
    fn consolidate_entry(&mut self, idx: usize) {
        if idx >= self.working_entries.len() {
            return;
        }

        let entry = self.working_entries.remove(idx);
        self.long_term.store(entry.key, entry.value);
        self.consolidation_count += 1;

        log::debug!(
            "TieredHdcMemory: Consolidated entry to long-term (access_count={})",
            entry.access_count
        );
    }

    /// Manually trigger consolidation of all qualifying entries
    pub fn consolidate(&mut self) {
        let threshold = self.config.consolidation_threshold;
        let mut to_consolidate: Vec<usize> = self
            .working_entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.access_count >= threshold)
            .map(|(i, _)| i)
            .collect();

        // Remove in reverse order to preserve indices
        to_consolidate.sort_by(|a, b| b.cmp(a));
        for idx in to_consolidate {
            self.consolidate_entry(idx);
        }
    }

    /// Check if a key is in long-term memory
    pub fn is_in_long_term(&mut self, key: &HdcVector) -> bool {
        self.long_term
            .retrieve(key)
            .map(|(_, sim)| sim > 0.9)
            .unwrap_or(false)
    }

    /// Check if a key is in working memory
    pub fn is_in_working(&self, key: &HdcVector) -> bool {
        self.working_entries
            .iter()
            .any(|e| e.key.similarity(key) > 0.9)
    }

    /// Evict least important entry from working memory
    fn evict_working_memory(&mut self) {
        if self.working_entries.is_empty() {
            return;
        }

        // Find entry with lowest importance
        let min_idx = self
            .working_entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.importance.partial_cmp(&b.importance).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.working_entries.remove(min_idx);
    }

    /// Compute importance score from access count
    fn compute_importance(&self, access_count: u32) -> f32 {
        // Logarithmic scaling: diminishing returns for high access counts
        1.0 + (access_count as f32).ln()
    }

    /// Encode features to HDC vector (uses long-term memory's encoder)
    pub fn encode_features(&self, features: &[f32]) -> HdcVector {
        self.long_term.encode_features(features)
    }

    /// Apply decay to both memory tiers
    pub fn decay(&mut self, rate: f32) {
        // Decay working memory importance scores
        for entry in &mut self.working_entries {
            entry.importance *= rate;
        }

        // Decay long-term memory
        self.long_term.decay(rate);
    }

    /// Apply tier-specific decay rates
    pub fn decay_tiered(&mut self) {
        // Working memory decays faster
        for entry in &mut self.working_entries {
            entry.importance *= self.config.working_decay_rate;
        }

        // Long-term memory decays slower
        self.long_term.decay(self.config.long_term_decay_rate);
    }

    /// Clear all memory
    pub fn clear(&mut self) {
        self.working_entries.clear();
        self.long_term.clear();
        self.total_retrievals = 0;
        self.successful_retrievals = 0;
        self.consolidation_count = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> TieredHdcStats {
        let (lt_count, lt_retrievals, lt_successful, lt_rate) = self.long_term.stats();
        TieredHdcStats {
            working_count: self.working_entries.len(),
            long_term_count: lt_count,
            total_retrievals: self.total_retrievals,
            successful_retrievals: self.successful_retrievals,
            consolidation_count: self.consolidation_count,
            recall_rate: if self.total_retrievals > 0 {
                self.successful_retrievals as f32 / self.total_retrievals as f32
            } else {
                0.0
            },
            long_term_recall_rate: lt_rate,
            // Internal long-term stats for diagnostics
            _lt_internal_retrievals: lt_retrievals,
            _lt_internal_successful: lt_successful,
        }
    }

    /// Get dimension
    #[inline]
    pub fn dim(&self) -> usize {
        self.config.working_config.dimension
    }
}

/// Which memory tier a result came from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    /// Fast, short-term working memory
    Working,
    /// Persistent long-term memory
    LongTerm,
}

/// Statistics for tiered memory
#[derive(Debug, Clone)]
pub struct TieredHdcStats {
    /// Number of patterns in working memory
    pub working_count: usize,
    /// Number of patterns in long-term memory
    pub long_term_count: usize,
    /// Total retrieval attempts
    pub total_retrievals: u64,
    /// Successful retrievals
    pub successful_retrievals: u64,
    /// Patterns consolidated to long-term
    pub consolidation_count: u64,
    /// Overall recall rate
    pub recall_rate: f32,
    /// Long-term memory recall rate
    pub long_term_recall_rate: f32,
    // Internal tracking
    _lt_internal_retrievals: u64,
    _lt_internal_successful: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tiered_memory() {
        let mem = TieredHdcMemory::for_zenb();
        let stats = mem.stats();
        assert_eq!(stats.working_count, 0);
        assert_eq!(stats.long_term_count, 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut mem = TieredHdcMemory::minimal();

        let key = HdcVector::random(1024);
        let value = HdcVector::random(1024);

        mem.store(key.clone(), value.clone());

        let result = mem.retrieve(&key);
        assert!(result.is_some(), "Should retrieve stored pattern");

        let (sim, tier) = result.unwrap();
        assert!(sim > 0.99, "Exact match should have high similarity");
        assert_eq!(tier, MemoryTier::Working, "New pattern should be in working memory");
    }

    #[test]
    fn test_consolidation_to_long_term() {
        let config = TieredHdcConfig {
            consolidation_threshold: 3, // Consolidate after 3 accesses
            ..TieredHdcConfig::minimal()
        };
        let mut mem = TieredHdcMemory::new(config);

        let key = HdcVector::random(1024);
        let value = HdcVector::random(1024);
        mem.store(key.clone(), value);

        // Access 3 times to trigger consolidation
        for _ in 0..3 {
            let _ = mem.retrieve(&key);
        }

        // Should now be in long-term
        assert!(
            mem.is_in_long_term(&key),
            "Pattern should be consolidated to long-term after threshold accesses"
        );

        // Should no longer be in working
        assert!(
            !mem.is_in_working(&key),
            "Pattern should be removed from working after consolidation"
        );

        let stats = mem.stats();
        assert_eq!(stats.consolidation_count, 1);
    }

    #[test]
    fn test_working_memory_eviction() {
        let config = TieredHdcConfig {
            working_config: HdcConfig {
                dimension: 1024,
                max_patterns: 5, // Very small for testing
                similarity_threshold: 0.5,
            },
            auto_consolidate: false, // Disable auto-consolidation
            ..TieredHdcConfig::minimal()
        };
        let mut mem = TieredHdcMemory::new(config);

        // Store 10 patterns (exceeds capacity of 5)
        for i in 0..10 {
            let key = HdcVector::from_seed(1024, i as u64 * 12345);
            let value = HdcVector::from_seed(1024, i as u64 * 67890);
            mem.store(key, value);
        }

        // Should only have max_patterns in working memory
        let stats = mem.stats();
        assert!(
            stats.working_count <= 5,
            "Working memory should evict to stay at capacity"
        );
    }

    #[test]
    fn test_manual_consolidation() {
        let config = TieredHdcConfig {
            consolidation_threshold: 2,
            auto_consolidate: false, // Manual only
            ..TieredHdcConfig::minimal()
        };
        let mut mem = TieredHdcMemory::new(config);

        let key = HdcVector::random(1024);
        let value = HdcVector::random(1024);
        mem.store(key.clone(), value);

        // Access to reach threshold
        mem.retrieve(&key);
        mem.retrieve(&key);

        // Not yet consolidated (auto_consolidate is false)
        assert!(mem.is_in_working(&key));

        // Manual consolidation
        mem.consolidate();

        // Now should be in long-term
        assert!(mem.is_in_long_term(&key));
    }

    #[test]
    fn test_dual_tier_retrieval_priority() {
        let mut mem = TieredHdcMemory::minimal();

        // Store pattern and consolidate to long-term
        let key = HdcVector::random(1024);
        let value_lt = HdcVector::from_seed(1024, 111);
        mem.store(key.clone(), value_lt.clone());

        // Force consolidation
        for _ in 0..5 {
            mem.retrieve(&key);
        }
        mem.consolidate();

        // Store same key with different value in working (simulating update)
        let value_wk = HdcVector::from_seed(1024, 222);
        mem.store(key.clone(), value_wk.clone());

        // Retrieve should return working memory version (more recent)
        let result = mem.retrieve(&key);
        assert!(result.is_some());
        let (_, tier) = result.unwrap();
        assert_eq!(tier, MemoryTier::Working, "Working memory should have priority");
    }

    #[test]
    fn test_feature_encoding() {
        let mem = TieredHdcMemory::minimal();

        let f1 = [0.5, 0.5, 0.5];
        let f2 = [0.52, 0.48, 0.51]; // Similar
        let f3 = [0.1, 0.9, 0.2]; // Different

        let e1 = mem.encode_features(&f1);
        let e2 = mem.encode_features(&f2);
        let e3 = mem.encode_features(&f3);

        let sim_12 = e1.similarity(&e2);
        let sim_13 = e1.similarity(&e3);

        assert!(
            sim_12 > sim_13,
            "Similar features should encode similarly: {} vs {}",
            sim_12,
            sim_13
        );
    }

    #[test]
    fn test_decay_tiered() {
        let mut mem = TieredHdcMemory::minimal();

        let key = HdcVector::random(1024);
        let value = HdcVector::random(1024);
        mem.store(key, value);

        // Get initial importance
        let initial_importance = mem.working_entries[0].importance;

        // Apply tiered decay
        mem.decay_tiered();

        // Importance should decrease
        assert!(
            mem.working_entries[0].importance < initial_importance,
            "Importance should decrease after decay"
        );
    }
}
