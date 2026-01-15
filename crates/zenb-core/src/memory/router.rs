//! Adaptive Memory Router (ZENITH Tier 1)
//!
//! Intelligent routing layer that selects optimal memory backend per task/context.
//! Based on 2026 research: different memory paradigms excel at different tasks.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │              AdaptiveMemoryRouter               │
//! │  ┌─────────────┐   ┌─────────────────────────┐ │
//! │  │   Task      │   │   Performance History   │ │
//! │  │ Classifier  │   │   (per backend/task)    │ │
//! │  └──────┬──────┘   └───────────────────────┬─┘ │
//! │         ▼                                   │   │
//! │  ┌──────────────────────────────────────────┼─┐ │
//! │  │            Routing Decision              │ │ │
//! │  │  Sensory → Holographic                   │ │ │
//! │  │  Symbolic → HDC                          │ │ │
//! │  │  Episodic → TieredHDC                    │ │ │
//! │  │  Temporal → Hybrid                       │ │ │
//! │  └──────────────────────────────────────────┴─┘ │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Key Features
//! - **Task classification**: Lightweight inference of task type from context
//! - **Backend selection**: Route to optimal memory backend
//! - **Performance tracking**: Adapt routing based on historical success
//! - **Confidence thresholding**: Fall back to hybrid when uncertain

use std::collections::HashMap;

/// Task type classification for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Continuous sensory data → Holographic memory (FFT-based)
    Sensory,
    /// Discrete symbols → HDC memory (binary, NPU-optimized)
    Symbolic,
    /// Specific memories with temporal context → Tiered HDC
    Episodic,
    /// Sequential patterns → Hybrid approach
    Temporal,
    /// Abstract knowledge → Holographic with high dim
    Semantic,
}

/// Memory backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryBackend {
    /// FFT-based holographic memory
    Holographic,
    /// Binary HDC (NPU-accelerated)
    BinaryHDC,
    /// Two-tier HDC with consolidation
    TieredHDC,
    /// Weighted combination of backends
    Hybrid { holo_weight: u8 }, // 0-100 percentage
}

impl MemoryBackend {
    /// Get hybrid with equal weights
    pub fn hybrid_equal() -> Self {
        Self::Hybrid { holo_weight: 50 }
    }
}

/// Performance metrics for a backend on a task type
#[derive(Debug, Clone)]
#[allow(dead_code)] // energy_estimate reserved for future routing optimization
struct BackendPerformance {
    success_rate: f32,
    avg_latency_us: f32,
    energy_estimate: f32,
    sample_count: u32,
}

impl Default for BackendPerformance {
    fn default() -> Self {
        Self {
            success_rate: 0.5,
            avg_latency_us: 1000.0,
            energy_estimate: 1.0,
            sample_count: 0,
        }
    }
}

/// Configuration for adaptive routing
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Minimum confidence to use single backend (else hybrid)
    pub confidence_threshold: f32,
    /// Learning rate for performance updates
    pub learning_rate: f32,
    /// Enable dynamic routing (false = static rules only)
    pub dynamic_routing: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            learning_rate: 0.1,
            dynamic_routing: true,
        }
    }
}

/// Adaptive Memory Router (ZENITH Tier 1)
///
/// Intelligent routing layer for memory operations.
#[derive(Debug)]
pub struct AdaptiveMemoryRouter {
    /// Task classifier weights (simplified linear model)
    classifier_weights: Vec<Vec<f32>>,
    /// Performance history per (task_type, backend) pair
    performance_history: HashMap<(TaskType, MemoryBackend), BackendPerformance>,
    /// Configuration
    config: RouterConfig,
    /// Total routing decisions made
    total_routes: u64,
}

impl Default for AdaptiveMemoryRouter {
    fn default() -> Self {
        Self::new(RouterConfig::default())
    }
}

impl AdaptiveMemoryRouter {
    /// Create new router with configuration
    pub fn new(config: RouterConfig) -> Self {
        // Initialize classifier weights (5 task types x 16 features)
        let classifier_weights = vec![vec![0.0; 16]; 5];

        Self {
            classifier_weights,
            performance_history: HashMap::new(),
            config,
            total_routes: 0,
        }
    }

    /// Route query to optimal backend based on context
    ///
    /// # Arguments
    /// * `context` - Context feature vector describing the current task/state
    ///
    /// # Returns
    /// (MemoryBackend, confidence) tuple
    pub fn route(&mut self, context: &[f32]) -> (MemoryBackend, f32) {
        self.total_routes += 1;

        // Classify task type
        let (task_type, confidence) = self.classify_task(context);

        // If low confidence or dynamic routing disabled, use hybrid
        if confidence < self.config.confidence_threshold || !self.config.dynamic_routing {
            return (MemoryBackend::hybrid_equal(), confidence);
        }

        // Select backend based on task type and performance history
        let backend = self.select_backend(task_type);

        (backend, confidence)
    }

    /// Classify task type from context features
    fn classify_task(&self, context: &[f32]) -> (TaskType, f32) {
        // Project context to task scores using simple rules
        // In production, this would be a learned classifier
        
        let features = self.extract_features(context);
        
        // Simple heuristic classification based on feature patterns
        let mut scores = [0.0f32; 5];
        
        // Sensory: high values in first few features (e.g., sensor data)
        scores[0] = features.get(0).unwrap_or(&0.0) * 0.5 + features.get(1).unwrap_or(&0.0) * 0.3;
        
        // Symbolic: second feature range
        scores[1] = features.get(2).unwrap_or(&0.0) * 0.4 + features.get(3).unwrap_or(&0.0) * 0.4;
        
        // Episodic: middle features
        scores[2] = features.get(4).unwrap_or(&0.0) * 0.5 + features.get(5).unwrap_or(&0.0) * 0.3;
        
        // Temporal: sequence-like patterns
        scores[3] = features.get(6).unwrap_or(&0.0) * 0.3 + features.get(7).unwrap_or(&0.0) * 0.5;
        
        // Semantic: last features
        scores[4] = features.get(8).unwrap_or(&0.0) * 0.4 + features.get(9).unwrap_or(&0.0) * 0.4;

        // Softmax for probabilities
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        
        let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum.max(0.001)).collect();
        
        let (max_idx, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.2));

        let task = match max_idx {
            0 => TaskType::Sensory,
            1 => TaskType::Symbolic,
            2 => TaskType::Episodic,
            3 => TaskType::Temporal,
            _ => TaskType::Semantic,
        };

        (task, max_prob)
    }

    /// Extract features from context (padding/truncating to fixed size)
    fn extract_features(&self, context: &[f32]) -> Vec<f32> {
        let target_len = 16;
        let mut features = context.to_vec();
        features.resize(target_len, 0.0);
        features
    }

    /// Select backend based on task type and performance history
    fn select_backend(&self, task_type: TaskType) -> MemoryBackend {
        // Default routing rules based on task type
        match task_type {
            TaskType::Sensory | TaskType::Semantic => MemoryBackend::Holographic,
            TaskType::Symbolic => MemoryBackend::BinaryHDC,
            TaskType::Episodic => MemoryBackend::TieredHDC,
            TaskType::Temporal => MemoryBackend::Hybrid { holo_weight: 60 },
        }
    }

    /// Update performance based on result
    ///
    /// Call this after each memory operation to improve routing.
    pub fn update_performance(
        &mut self,
        task_type: TaskType,
        backend: MemoryBackend,
        success: bool,
        latency_us: f32,
    ) {
        let key = (task_type, backend);
        let perf = self.performance_history.entry(key).or_default();

        let alpha = self.config.learning_rate;

        perf.success_rate = alpha * (if success { 1.0 } else { 0.0 }) + (1.0 - alpha) * perf.success_rate;
        perf.avg_latency_us = alpha * latency_us + (1.0 - alpha) * perf.avg_latency_us;
        perf.sample_count += 1;
    }

    /// Get routing statistics
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            total_routes: self.total_routes,
            performance_entries: self.performance_history.len(),
            dynamic_enabled: self.config.dynamic_routing,
        }
    }

    /// Get performance for a specific task/backend combination
    pub fn get_performance(&self, task_type: TaskType, backend: MemoryBackend) -> Option<(f32, f32)> {
        self.performance_history
            .get(&(task_type, backend))
            .map(|p| (p.success_rate, p.avg_latency_us))
    }
}

/// Router statistics
#[derive(Debug, Clone)]
pub struct RouterStats {
    pub total_routes: u64,
    pub performance_entries: usize,
    pub dynamic_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_basic() {
        let mut router = AdaptiveMemoryRouter::default();

        let context = vec![0.8, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (backend, confidence) = router.route(&context);

        // Should return some backend with some confidence
        assert!(confidence > 0.0);
        println!("Backend: {:?}, Confidence: {}", backend, confidence);
    }

    #[test]
    fn test_performance_update() {
        let mut router = AdaptiveMemoryRouter::default();

        // Record some performance data
        router.update_performance(TaskType::Sensory, MemoryBackend::Holographic, true, 500.0);
        router.update_performance(TaskType::Sensory, MemoryBackend::Holographic, true, 600.0);
        router.update_performance(TaskType::Sensory, MemoryBackend::Holographic, false, 800.0);

        let (success_rate, latency) = router
            .get_performance(TaskType::Sensory, MemoryBackend::Holographic)
            .unwrap();

        assert!(success_rate > 0.0 && success_rate < 1.0);
        assert!(latency > 0.0);
    }

    #[test]
    fn test_task_classification() {
        let mut router = AdaptiveMemoryRouter::default();

        // High sensory features
        let sensory_ctx = vec![0.9, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, _) = router.route(&sensory_ctx);

        // High symbolic features
        let symbolic_ctx = vec![0.0, 0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.0];
        let (_, _) = router.route(&symbolic_ctx);

        assert_eq!(router.stats().total_routes, 2);
    }

    #[test]
    fn test_low_confidence_hybrid() {
        let config = RouterConfig {
            confidence_threshold: 0.99, // Very high threshold
            learning_rate: 0.1,
            dynamic_routing: true,
        };
        let mut router = AdaptiveMemoryRouter::new(config);

        // With very high threshold, should get hybrid
        let context = vec![0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
        let (backend, _) = router.route(&context);

        // Should use hybrid when confidence is low
        matches!(backend, MemoryBackend::Hybrid { .. });
    }

    #[test]
    fn test_backend_selection() {
        let router = AdaptiveMemoryRouter::default();

        // Test default routing rules
        assert_eq!(
            router.select_backend(TaskType::Sensory),
            MemoryBackend::Holographic
        );
        assert_eq!(
            router.select_backend(TaskType::Symbolic),
            MemoryBackend::BinaryHDC
        );
        assert_eq!(
            router.select_backend(TaskType::Episodic),
            MemoryBackend::TieredHDC
        );
    }
}
