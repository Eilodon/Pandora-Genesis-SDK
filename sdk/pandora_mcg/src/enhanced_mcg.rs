// 🎯 Enhanced Meta-Cognitive Governor Implementation
// File: sdk/pandora_mcg/src/enhanced_mcg.rs

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uncertainty: f32,
    pub compression_reward: f64,
    pub novelty_score: f32,
    pub performance: f32,
    pub resource_usage: ResourceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub timestamp: u64,
    pub metrics: SystemMetrics,
    pub decision: Option<ActionTrigger>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionTrigger {
    TriggerSelfImprovementLevel1 {
        reason: String,
        target_component: String,
        confidence: f32,
    },
    TriggerSelfImprovementLevel2 {
        reason: String,
        target_component: String,
        confidence: f32,
    },
    TriggerSelfImprovementLevel3 {
        reason: String,
        target_component: String,
        confidence: f32,
    },
    RequestMoreInformation {
        reason: String,
        priority: Priority,
    },
    OptimizeResources {
        reason: String,
        target: String,
    },
    NoAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Adaptive threshold that adjusts based on historical performance
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    base_value: f32,
    current_value: f32,
    learning_rate: f32,
    min_value: f32,
    max_value: f32,
}

impl AdaptiveThreshold {
    pub fn new(base: f32, min: f32, max: f32, learning_rate: f32) -> Self {
        Self {
            base_value: base,
            current_value: base,
            learning_rate,
            min_value: min,
            max_value: max,
        }
    }
    
    pub fn get(&self) -> f32 {
        self.current_value
    }
    
    /// Adapt threshold based on recent performance
    pub fn adapt(&mut self, performance_delta: f32) {
        // If performance is good (positive delta), we can be more aggressive (lower threshold)
        // If performance is bad (negative delta), be more conservative (higher threshold)
        let adjustment = -performance_delta * self.learning_rate;
        self.current_value = (self.current_value + adjustment)
            .clamp(self.min_value, self.max_value);
        
        info!(
            "MCG: Adapted threshold from {:.4} to {:.4} (delta: {:.4})",
            self.base_value, self.current_value, adjustment
        );
    }
    
    pub fn reset(&mut self) {
        self.current_value = self.base_value;
    }
}

/// Detects anomalies in system behavior
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    window_size: usize,
    history: VecDeque<f32>,
    threshold_std: f32,
}

impl AnomalyDetector {
    pub fn new(window_size: usize, threshold_std: f32) -> Self {
        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
            threshold_std,
        }
    }
    
    /// Add a new observation and return anomaly score
    pub fn score(&mut self, value: f32) -> f32 {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(value);
        
        if self.history.len() < 3 {
            return 0.0; // Not enough data
        }
        
        // Calculate mean and std
        let mean: f32 = self.history.iter().sum::<f32>() / self.history.len() as f32;
        let variance: f32 = self.history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.history.len() as f32;
        let std = variance.sqrt();
        
        if std < 1e-6 {
            return 0.0; // No variance
        }
        
        // Z-score: how many standard deviations away from mean
        let z_score = ((value - mean) / std).abs();
        
        // Anomaly if beyond threshold
        if z_score > self.threshold_std {
            warn!(
                "MCG: Anomaly detected! Value: {:.4}, Mean: {:.4}, Std: {:.4}, Z-score: {:.4}",
                value, mean, std, z_score
            );
            z_score / self.threshold_std // Normalized anomaly score
        } else {
            0.0
        }
    }
}

/// Tracks confidence in decisions
#[derive(Debug, Clone)]
pub struct ConfidenceTracker {
    success_count: usize,
    total_count: usize,
    recent_successes: VecDeque<bool>,
    window_size: usize,
}

impl ConfidenceTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            success_count: 0,
            total_count: 0,
            recent_successes: VecDeque::with_capacity(window_size),
            window_size,
        }
    }
    
    /// Compute confidence based on historical success rate
    pub fn compute(&self, anomaly_score: f32) -> f32 {
        let base_confidence = if self.total_count > 0 {
            self.success_count as f32 / self.total_count as f32
        } else {
            0.5 // Default moderate confidence
        };
        
        // Reduce confidence if anomaly detected
        let anomaly_penalty = anomaly_score * 0.3;
        let confidence = (base_confidence - anomaly_penalty).clamp(0.0, 1.0);
        
        confidence
    }
    
    /// Update with outcome of a decision
    pub fn update(&mut self, success: bool) {
        if self.recent_successes.len() >= self.window_size {
            if let Some(old) = self.recent_successes.pop_front() {
                if old {
                    self.success_count = self.success_count.saturating_sub(1);
                }
                self.total_count = self.total_count.saturating_sub(1);
            }
        }
        
        self.recent_successes.push_back(success);
        if success {
            self.success_count += 1;
        }
        self.total_count += 1;
    }
    
    pub fn success_rate(&self) -> f32 {
        if self.total_count > 0 {
            self.success_count as f32 / self.total_count as f32
        } else {
            0.5
        }
    }
}

/// Buffer for storing observations before causal discovery
#[derive(Debug, Clone)]
pub struct ObservationBuffer {
    data: Vec<Vec<f32>>,
    capacity: usize,
    min_samples_for_discovery: usize,
}

impl ObservationBuffer {
    pub fn new(capacity: usize, min_samples: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            min_samples_for_discovery: min_samples,
        }
    }

    pub fn add(&mut self, observation: Vec<f32>) {
        if self.data.len() >= self.capacity {
            self.data.remove(0);
        }
        self.data.push(observation);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_ready_for_discovery(&self) -> bool {
        self.data.len() >= self.min_samples_for_discovery
    }

    pub fn get_data_and_clear(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.data)
    }

    pub fn get_data(&self) -> &[Vec<f32>] {
        &self.data
    }
}

/// Enhanced Meta-Cognitive Governor with adaptive thresholds and anomaly detection
pub struct EnhancedMetaCognitiveGovernor {
    // Adaptive thresholds for different metrics
    uncertainty_threshold: AdaptiveThreshold,
    compression_threshold: AdaptiveThreshold,
    novelty_threshold: AdaptiveThreshold,
    
    // Anomaly detection for each metric
    uncertainty_detector: AnomalyDetector,
    compression_detector: AnomalyDetector,
    novelty_detector: AnomalyDetector,
    
    // Confidence tracking
    confidence_tracker: ConfidenceTracker,
    
    // Historical state
    state_history: VecDeque<SystemState>,
    max_history: usize,
    
    // Configuration
    min_samples_for_adaptation: usize,
}

impl EnhancedMetaCognitiveGovernor {
    pub fn new() -> Self {
        Self {
            uncertainty_threshold: AdaptiveThreshold::new(0.5, 0.2, 0.8, 0.05),
            compression_threshold: AdaptiveThreshold::new(0.7, 0.3, 0.9, 0.05),
            novelty_threshold: AdaptiveThreshold::new(0.6, 0.3, 0.9, 0.05),
            
            uncertainty_detector: AnomalyDetector::new(50, 2.5),
            compression_detector: AnomalyDetector::new(50, 2.5),
            novelty_detector: AnomalyDetector::new(50, 2.5),
            
            confidence_tracker: ConfidenceTracker::new(100),
            
            state_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            
            min_samples_for_adaptation: 20,
        }
    }

    /// Compatibility constructor for tests that use CausalDiscoveryConfig
    /// This allows old tests to work with the new MCG implementation
    pub fn new_with_discovery_config(_config: crate::causal_discovery::CausalDiscoveryConfig) -> Self {
        // For now, we ignore the config and use default settings
        // In the future, we could map config parameters to MCG thresholds
        Self::new()
    }
    
    /// Monitor system and make decisions
    pub fn monitor_comprehensive(&mut self, metrics: &SystemMetrics) -> DecisionWithConfidence {
        info!("\n=== Meta-Cognitive Governor - Comprehensive Monitoring ===");
        
        // 1. Detect anomalies
        let uncertainty_anomaly = self.uncertainty_detector.score(metrics.uncertainty);
        let compression_anomaly = self.compression_detector.score(metrics.compression_reward as f32);
        let novelty_anomaly = self.novelty_detector.score(metrics.novelty_score);
        
        let max_anomaly = uncertainty_anomaly.max(compression_anomaly).max(novelty_anomaly);
        
        if max_anomaly > 0.0 {
            warn!("MCG: Anomaly detected! Score: {:.4}", max_anomaly);
        }
        
        // 2. Check thresholds
        let mut triggers = Vec::new();
        
        // Uncertainty check
        if metrics.uncertainty > self.uncertainty_threshold.get() {
            info!(
                "MCG: High uncertainty detected: {:.4} > {:.4}",
                metrics.uncertainty,
                self.uncertainty_threshold.get()
            );
            triggers.push((
                "uncertainty",
                ActionTrigger::TriggerSelfImprovementLevel1 {
                    reason: format!(
                        "Uncertainty {:.4} exceeds threshold {:.4}",
                        metrics.uncertainty,
                        self.uncertainty_threshold.get()
                    ),
                    target_component: "world_model".to_string(),
                    confidence: 0.0, // Will be filled later
                },
            ));
        }
        
        // Compression reward check
        if metrics.compression_reward > self.compression_threshold.get() as f64 {
            info!(
                "MCG: High compression reward: {:.4} > {:.4}",
                metrics.compression_reward,
                self.compression_threshold.get()
            );
            triggers.push((
                "compression",
                ActionTrigger::TriggerSelfImprovementLevel2 {
                    reason: format!(
                        "Compression reward {:.4} indicates learning opportunity",
                        metrics.compression_reward
                    ),
                    target_component: "learning_engine".to_string(),
                    confidence: 0.0,
                },
            ));
        }
        
        // Novelty check
        if metrics.novelty_score > self.novelty_threshold.get() {
            info!(
                "MCG: High novelty detected: {:.4} > {:.4}",
                metrics.novelty_score,
                self.novelty_threshold.get()
            );
            triggers.push((
                "novelty",
                ActionTrigger::TriggerSelfImprovementLevel3 {
                    reason: format!(
                        "Novelty score {:.4} suggests new pattern",
                        metrics.novelty_score
                    ),
                    target_component: "pattern_recognizer".to_string(),
                    confidence: 0.0,
                },
            ));
        }
        
        // Resource optimization check
        if metrics.resource_usage.cpu_usage > 0.9 || metrics.resource_usage.memory_usage > 0.9 {
            warn!(
                "MCG: High resource usage - CPU: {:.2}%, Memory: {:.2}%",
                metrics.resource_usage.cpu_usage * 100.0,
                metrics.resource_usage.memory_usage * 100.0
            );
            triggers.push((
                "resources",
                ActionTrigger::OptimizeResources {
                    reason: "High resource utilization".to_string(),
                    target: "system".to_string(),
                },
            ));
        }
        
        // 3. Select best action and compute confidence
        let (decision, confidence) = if triggers.is_empty() {
            info!("MCG: System stable. No action needed.");
            (ActionTrigger::NoAction, 1.0)
        } else {
            // Select highest priority trigger
            let (_, mut action) = triggers[0].clone();
            
            // Compute confidence
            let confidence = self.confidence_tracker.compute(max_anomaly);
            
            // Update action with confidence
            action = match action {
                ActionTrigger::TriggerSelfImprovementLevel1 { reason, target_component, .. } => {
                    ActionTrigger::TriggerSelfImprovementLevel1 {
                        reason,
                        target_component,
                        confidence,
                    }
                }
                ActionTrigger::TriggerSelfImprovementLevel2 { reason, target_component, .. } => {
                    ActionTrigger::TriggerSelfImprovementLevel2 {
                        reason,
                        target_component,
                        confidence,
                    }
                }
                ActionTrigger::TriggerSelfImprovementLevel3 { reason, target_component, .. } => {
                    ActionTrigger::TriggerSelfImprovementLevel3 {
                        reason,
                        target_component,
                        confidence,
                    }
                }
                other => other,
            };
            
            (action, confidence)
        };
        
        // 4. Store state
        self.add_state(SystemState {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics: metrics.clone(),
            decision: if matches!(decision, ActionTrigger::NoAction) {
                None
            } else {
                Some(decision.clone())
            },
        });
        
        // 5. Adapt thresholds if enough history
        if self.state_history.len() >= self.min_samples_for_adaptation {
            self.adapt_thresholds();
        }
        
        info!("=== Meta-Cognitive Governor - Decision Complete ===\n");
        
        DecisionWithConfidence { decision, confidence }
    }
    
    fn add_state(&mut self, state: SystemState) {
        if self.state_history.len() >= self.max_history {
            self.state_history.pop_front();
        }
        self.state_history.push_back(state);
    }
    
    fn adapt_thresholds(&mut self) {
        if self.state_history.len() < 2 {
            return;
        }
        
        // Calculate performance delta (e.g., improvement in metrics)
        let recent = &self.state_history[self.state_history.len() - 1];
        let previous = &self.state_history[self.state_history.len() - 2];
        
        let performance_delta = recent.metrics.performance - previous.metrics.performance;
        
        // Adapt each threshold
        self.uncertainty_threshold.adapt(performance_delta);
        self.compression_threshold.adapt(performance_delta);
        self.novelty_threshold.adapt(performance_delta);
    }
    
    /// Update confidence based on action outcome
    pub fn report_outcome(&mut self, success: bool) {
        self.confidence_tracker.update(success);
        info!(
            "MCG: Outcome reported. Success rate: {:.2}%",
            self.confidence_tracker.success_rate() * 100.0
        );
    }
    
    /// Get system health summary
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            confidence: self.confidence_tracker.success_rate(),
            state_history_size: self.state_history.len(),
            current_thresholds: ThresholdSummary {
                uncertainty: self.uncertainty_threshold.get(),
                compression: self.compression_threshold.get(),
                novelty: self.novelty_threshold.get(),
            },
        }
    }
}

impl Default for EnhancedMetaCognitiveGovernor {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionWithConfidence {
    pub decision: ActionTrigger,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    pub confidence: f32,
    pub state_history_size: usize,
    pub current_thresholds: ThresholdSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdSummary {
    pub uncertainty: f32,
    pub compression: f32,
    pub novelty: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_threshold() {
        let mut threshold = AdaptiveThreshold::new(0.5, 0.2, 0.8, 0.1);
        
        // Good performance should lower threshold
        threshold.adapt(0.1);
        assert!(threshold.get() < 0.5);
        
        // Bad performance should raise threshold
        let old_value = threshold.get();
        threshold.adapt(-0.1);
        assert!(threshold.get() > old_value);
    }
    
    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(10, 2.0);
        
        // Normal values
        for _ in 0..10 {
            assert_eq!(detector.score(5.0), 0.0);
        }
        
        // Anomalous value
        let score = detector.score(50.0);
        assert!(score > 0.0);
    }
    
    #[test]
    fn test_confidence_tracker() {
        let mut tracker = ConfidenceTracker::new(10);
        
        // Record some successes
        for _ in 0..7 {
            tracker.update(true);
        }
        for _ in 0..3 {
            tracker.update(false);
        }
        
        assert_eq!(tracker.success_rate(), 0.7);
    }
}