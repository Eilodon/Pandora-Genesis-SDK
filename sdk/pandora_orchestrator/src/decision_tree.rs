// sdk/pandora_orchestrator/src/decision_tree.rs
// Decision Tree Engine Implementation theo Neural Skills Specifications

use crate::*;
use pandora_core::ontology::{TaskType, SkillId, TaskId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{Timelike, Datelike};

#[derive(Debug, Error)]
pub enum DecisionTreeError {
    #[error("Decision tree traversal failed: {0}")]
    TraversalFailed(String),
    #[error("Feature evaluation failed: {0}")]
    FeatureEvaluationFailed(String),
    #[error("Confidence threshold not met: {0}")]
    ConfidenceThresholdNotMet(f32),
    #[error("Invalid decision node: {0}")]
    InvalidDecisionNode(String),
}

// ===== 2.2 Decision Tree Engine Specifications =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: DecisionNode,
    pub confidence_threshold: f32,
    pub max_depth: usize,
    pub pruning_strategy: PruningStrategy,
    pub feature_importance: HashMap<String, f32>,
    pub performance_metrics: DecisionTreeMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionNode {
    Internal {
        feature: FeatureType,
        threshold: f32,
        left: Box<DecisionNode>,
        right: Box<DecisionNode>,
        samples: usize,
        confidence: f32,
        split_quality: f32,
    },
    Leaf {
        action: Action,
        confidence: f32,
        samples: usize,
        reasoning: Vec<String>,
        success_rate: f32,
        average_duration: Duration,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    TaskComplexity,
    AvailableResources { resource: ResourceType },
    HistoricalPerformance { skill_id: SkillId },
    UserContext { context_key: String },
    SystemState { metric: String },
    TimeOfDay,
    DayOfWeek,
    Custom { name: String, evaluator: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    CPU,
    Memory,
    Battery,
    Network,
    Storage,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    RouteToSkill(SkillId),
    ComposePipeline(Vec<SkillId>),
    RequestMoreInfo(InfoRequest),
    TriggerSelfCorrection,
    EscalateToHuman,
    WaitForResources { resource: ResourceType, timeout: Duration },
    FallbackToSimpleMode,
    RetryWithDifferentStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfoRequest {
    pub request_type: InfoRequestType,
    pub message: String,
    pub required_fields: Vec<String>,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InfoRequestType {
    UserInput,
    SystemConfiguration,
    ResourceStatus,
    PerformanceData,
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningStrategy {
    None,
    PrePruning { max_depth: usize, min_samples: usize, min_impurity: f32 },
    PostPruning { confidence_threshold: f32, complexity_penalty: f32 },
    CostComplexity { alpha: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeMetrics {
    pub total_decisions: u64,
    pub successful_decisions: u64,
    pub average_confidence: f32,
    pub average_duration: Duration,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

// ===== Decision Tree Implementation =====

impl DecisionTree {
    pub fn new(
        root: DecisionNode,
        confidence_threshold: f32,
        max_depth: usize,
        pruning_strategy: PruningStrategy,
    ) -> Self {
        Self {
            root,
            confidence_threshold,
            max_depth,
            pruning_strategy,
            feature_importance: HashMap::new(),
            performance_metrics: DecisionTreeMetrics::new(),
        }
    }

    /// Thực hiện quyết định dựa trên context
    pub async fn decide(
        &self,
        context: &DecisionContext,
    ) -> Result<DecisionResult, DecisionTreeError> {
        let start_time = std::time::Instant::now();
        
        // 1. Traverse decision tree
        let decision = self.traverse_tree(&self.root, context)?;
        
        // 2. Validate confidence threshold
        if decision.confidence < self.confidence_threshold {
            return Err(DecisionTreeError::ConfidenceThresholdNotMet(decision.confidence));
        }
        
        // 3. Update performance metrics
        let duration = start_time.elapsed();
        self.update_metrics(decision.success, duration).await;
        
        Ok(decision)
    }

    /// Traverse decision tree từ root đến leaf
    fn traverse_tree(
        &self,
        node: &DecisionNode,
        context: &DecisionContext,
    ) -> Result<DecisionResult, DecisionTreeError> {
        match node {
            DecisionNode::Internal {
                feature,
                threshold,
                left,
                right,
                confidence: _,
                ..
            } => {
                // Evaluate feature
                let feature_value = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(self.evaluate_feature(feature, context))
                })?;
                
                // Choose branch based on threshold
                let next_node = if feature_value <= *threshold {
                    left.as_ref()
                } else {
                    right.as_ref()
                };
                
                // Recursively traverse
                self.traverse_tree(next_node, context)
            }
            DecisionNode::Leaf {
                action,
                confidence,
                reasoning,
                success_rate,
                average_duration,
                samples: _,
            } => {
                Ok(DecisionResult {
                    action: action.clone(),
                    confidence: *confidence,
                    reasoning: reasoning.clone(),
                    success_rate: *success_rate,
                    estimated_duration: *average_duration,
                    feature_values: tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.extract_feature_values(context))
                    }),
                    decision_path: tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(self.get_decision_path(node))
                    }),
                    success: *confidence >= self.confidence_threshold,
                })
            }
        }
    }

    /// Evaluate feature value từ context
    async fn evaluate_feature(
        &self,
        feature: &FeatureType,
        context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        match feature {
            FeatureType::TaskComplexity => {
                Ok(self.calculate_task_complexity(&context.task_type, &context.input).await)
            }
            FeatureType::AvailableResources { resource } => {
                self.get_resource_availability(resource, context).await
            }
            FeatureType::HistoricalPerformance { skill_id } => {
                self.get_skill_performance(skill_id.clone(), context).await
            }
            FeatureType::UserContext { context_key } => {
                self.get_user_context_value(context_key, context).await
            }
            FeatureType::SystemState { metric } => {
                self.get_system_metric(metric, context).await
            }
            FeatureType::TimeOfDay => {
                Ok(self.get_time_of_day_feature().await)
            }
            FeatureType::DayOfWeek => {
                Ok(self.get_day_of_week_feature().await)
            }
            FeatureType::Custom { name, evaluator } => {
                self.evaluate_custom_feature(name, evaluator, context).await
            }
        }
    }

    /// Calculate task complexity score
    async fn calculate_task_complexity(
        &self,
        task_type: &TaskType,
        input: &serde_json::Value,
    ) -> f32 {
        // Logic để tính complexity score dựa trên task type và input
        match task_type {
            TaskType::Arithmetic => {
                // Dựa trên độ phức tạp của biểu thức
                let input_str = input.as_str().unwrap_or("");
                self.calculate_arithmetic_complexity(input_str).await
            }
            TaskType::InformationRetrieval => {
                // Dựa trên query length và complexity
                let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
                self.calculate_ir_complexity(query).await
            }
            TaskType::PatternMatching => {
                // Dựa trên pattern length và data size
                let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                self.calculate_pattern_complexity(pattern).await
            }
            TaskType::LogicalReasoning => {
                // Dựa trên số lượng rules và facts
                let rules_count = input.get("rules").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                self.calculate_logic_complexity(rules_count).await
            }
            TaskType::AnalogyReasoning => {
                // Dựa trên độ phức tạp của analogy
                let analogy = input.get("analogy").and_then(|v| v.as_str()).unwrap_or("");
                self.calculate_analogy_complexity(analogy).await
            }
            TaskType::SelfCorrection | TaskType::MetaAnalysis => {
                // Default complexity for advanced task types
                0.8
            }
        }
    }

    async fn calculate_arithmetic_complexity(&self, expression: &str) -> f32 {
        // Simple complexity calculation based on expression features
        let length = expression.len() as f32;
        let operator_count = expression.chars().filter(|c| "+-*/^()".contains(*c)).count() as f32;
        let function_count = expression.matches("sin cos tan log sqrt").count() as f32;
        
        // Normalize to 0.0-1.0 range
        (length * 0.1 + operator_count * 0.3 + function_count * 0.6).min(1.0)
    }

    async fn calculate_ir_complexity(&self, query: &str) -> f32 {
        let length = query.len() as f32;
        let word_count = query.split_whitespace().count() as f32;
        let complexity = (length * 0.01 + word_count * 0.1).min(1.0);
        complexity
    }

    async fn calculate_pattern_complexity(&self, pattern: &str) -> f32 {
        let length = pattern.len() as f32;
        let regex_chars = pattern.chars().filter(|c| ".*+?^${}[]|\\".contains(*c)).count() as f32;
        (length * 0.02 + regex_chars * 0.1).min(1.0)
    }

    async fn calculate_logic_complexity(&self, rules_count: usize) -> f32 {
        (rules_count as f32 * 0.1).min(1.0)
    }

    async fn calculate_analogy_complexity(&self, analogy: &str) -> f32 {
        let length = analogy.len() as f32;
        let word_count = analogy.split_whitespace().count() as f32;
        (length * 0.01 + word_count * 0.05).min(1.0)
    }

    /// Get resource availability
    async fn get_resource_availability(
        &self,
        resource: &ResourceType,
        context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        match resource {
            ResourceType::CPU => Ok(context.available_resources.cpu_cores),
            ResourceType::Memory => Ok(context.available_resources.memory_mb as f32 / 1000.0), // Normalize to 0-1
            ResourceType::Battery => Ok(context.available_resources.battery_percent / 100.0),
            ResourceType::Network => Ok(context.available_resources.network_bandwidth_mbps / 100.0), // Normalize
            ResourceType::Storage => Ok(context.available_resources.storage_mb as f32 / 1000.0), // Normalize
            ResourceType::Custom { name } => {
                Ok(context.custom_resources.get(name).copied().unwrap_or(0.0))
            }
        }
    }

    /// Get skill performance from context
    async fn get_skill_performance(
        &self,
        skill_id: SkillId,
        context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        context.skill_performance
            .get(&skill_id)
            .copied()
            .ok_or_else(|| DecisionTreeError::FeatureEvaluationFailed(
                format!("No performance data for skill: {}", skill_id)
            ))
    }

    /// Get user context value
    async fn get_user_context_value(
        &self,
        context_key: &str,
        context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        context.user_context
            .get(context_key)
            .copied()
            .ok_or_else(|| DecisionTreeError::FeatureEvaluationFailed(
                format!("No user context for key: {}", context_key)
            ))
    }

    /// Get system metric
    async fn get_system_metric(
        &self,
        metric: &str,
        context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        context.system_metrics
            .get(metric)
            .copied()
            .ok_or_else(|| DecisionTreeError::FeatureEvaluationFailed(
                format!("No system metric for: {}", metric)
            ))
    }

    /// Get time of day feature (0.0 = midnight, 1.0 = 11:59 PM)
    async fn get_time_of_day_feature(&self) -> f32 {
        let now = chrono::Utc::now();
        let hour = now.hour() as f32;
        hour / 24.0
    }

    /// Get day of week feature (0.0 = Monday, 1.0 = Sunday)
    async fn get_day_of_week_feature(&self) -> f32 {
        let now = chrono::Utc::now();
        let weekday = now.weekday().num_days_from_monday() as f32;
        weekday / 7.0
    }

    /// Evaluate custom feature
    async fn evaluate_custom_feature(
        &self,
        _name: &str,
        _evaluator: &str,
        _context: &DecisionContext,
    ) -> Result<f32, DecisionTreeError> {
        // Placeholder for custom feature evaluation
        // In real implementation, this would call the specified evaluator
        Ok(0.5) // Default neutral value
    }

    /// Extract all feature values for debugging
    async fn extract_feature_values(&self, context: &DecisionContext) -> HashMap<String, f32> {
        let mut values = HashMap::new();
        
        // Extract common features
        values.insert("task_complexity".to_string(), 
            self.calculate_task_complexity(&context.task_type, &context.input).await);
        values.insert("cpu_availability".to_string(), 
            context.available_resources.cpu_cores);
        values.insert("memory_availability".to_string(), 
            context.available_resources.memory_mb as f32 / 1000.0);
        values.insert("battery_level".to_string(), 
            context.available_resources.battery_percent / 100.0);
        values.insert("time_of_day".to_string(), 
            self.get_time_of_day_feature().await);
        values.insert("day_of_week".to_string(), 
            self.get_day_of_week_feature().await);
        
        values
    }

    /// Get decision path for debugging
    async fn get_decision_path(&self, _node: &DecisionNode) -> Vec<String> {
        // Placeholder for decision path extraction
        vec!["root".to_string(), "leaf".to_string()]
    }

    /// Update performance metrics
    async fn update_metrics(&self, _success: bool, _duration: Duration) {
        // This would update the performance metrics
        // For now, it's a placeholder
    }

    /// Add training data for learning
    pub async fn add_training_data(
        &mut self,
        _context: DecisionContext,
        _action: Action,
        _success: bool,
        _duration: Duration,
    ) -> Result<(), DecisionTreeError> {
        // Placeholder for adding training data
        // In real implementation, this would update the tree structure
        Ok(())
    }

    /// Retrain decision tree with new data
    pub async fn retrain(&mut self) -> Result<(), DecisionTreeError> {
        // Placeholder for retraining logic
        // In real implementation, this would rebuild the tree
        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &DecisionTreeMetrics {
        &self.performance_metrics
    }
}

// ===== Supporting Types =====

#[derive(Debug, Clone)]
pub struct DecisionContext {
    pub task_type: TaskType,
    pub input: serde_json::Value,
    pub user_id: Option<Uuid>,
    pub session_id: Uuid,
    pub priority: Priority,
    pub available_resources: ResourceProfile,
    pub skill_performance: HashMap<SkillId, f32>,
    pub user_context: HashMap<String, f32>,
    pub system_metrics: HashMap<String, f32>,
    pub custom_resources: HashMap<String, f32>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ResourceProfile {
    pub cpu_cores: f32,
    pub memory_mb: usize,
    pub battery_percent: f32,
    pub network_bandwidth_mbps: f32,
    pub storage_mb: usize,
}

#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub action: Action,
    pub confidence: f32,
    pub reasoning: Vec<String>,
    pub success_rate: f32,
    pub estimated_duration: Duration,
    pub feature_values: HashMap<String, f32>,
    pub decision_path: Vec<String>,
    pub success: bool,
}

// ===== Default Implementations =====

impl Default for DecisionTree {
    fn default() -> Self {
        Self::new(
            DecisionNode::Leaf {
                action: Action::EscalateToHuman,
                confidence: 0.5,
                samples: 0,
                reasoning: vec!["Default fallback action".to_string()],
                success_rate: 0.5,
                average_duration: Duration::from_millis(1000),
            },
            0.7, // confidence threshold
            10,  // max depth
            PruningStrategy::PrePruning {
                max_depth: 10,
                min_samples: 5,
                min_impurity: 0.1,
            },
        )
    }
}

impl Default for DecisionTreeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTreeMetrics {
    pub fn new() -> Self {
        Self {
            total_decisions: 0,
            successful_decisions: 0,
            average_confidence: 0.0,
            average_duration: Duration::from_millis(0),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            last_updated: chrono::Utc::now(),
        }
    }
}

// ===== Decision Tree Builder =====

pub struct DecisionTreeBuilder {
    root: Option<DecisionNode>,
    confidence_threshold: f32,
    max_depth: usize,
    pruning_strategy: PruningStrategy,
}

impl DecisionTreeBuilder {
    pub fn new() -> Self {
        Self {
            root: None,
            confidence_threshold: 0.7,
            max_depth: 10,
            pruning_strategy: PruningStrategy::PrePruning {
                max_depth: 10,
                min_samples: 5,
                min_impurity: 0.1,
            },
        }
    }

    pub fn with_root(mut self, root: DecisionNode) -> Self {
        self.root = Some(root);
        self
    }

    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_pruning_strategy(mut self, strategy: PruningStrategy) -> Self {
        self.pruning_strategy = strategy;
        self
    }

    pub fn build(self) -> DecisionTree {
        DecisionTree::new(
            self.root.unwrap_or_else(|| DecisionNode::Leaf {
                action: Action::EscalateToHuman,
                confidence: 0.5,
                samples: 0,
                reasoning: vec!["Default fallback action".to_string()],
                success_rate: 0.5,
                average_duration: Duration::from_millis(1000),
            }),
            self.confidence_threshold,
            self.max_depth,
            self.pruning_strategy,
        )
    }
}

impl Default for DecisionTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
