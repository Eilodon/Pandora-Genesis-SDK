// sdk/pandora_mcg/src/lib.rs

#![allow(clippy::all)]
use chrono::Duration;
use pandora_core::ontology::TaskType;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum McgError {
    #[error("Không thể truy cập SelfModel")]
    SelfModelInaccessible,
}

// ===== 5. Meta-Cognitive Controller Specifications =====

// --- 5.1 Self-Model ---

#[derive(Debug, Clone, Default)]
pub struct CapabilityProfile {
    pub accuracy: f32,
    pub speed: f32,       // a.k.a latency in ms
    pub reliability: f32, // 0.0 to 1.0
    pub resource_efficiency: f32,
}

#[derive(Debug, Clone, Default)]
pub struct SelfModel {
    // Đánh giá năng lực
    pub capabilities: HashMap<TaskType, CapabilityProfile>,
    pub strengths: HashMap<String, f32>,
    pub weaknesses: HashMap<String, f32>,

    // Các mẫu hình học tập
    pub learning_efficiency: HashMap<String, f32>, // Domain -> efficiency
    pub adaptation_speed: f32,

    // Các chỉ số tự nhận thức
    pub metacognitive_accuracy: f32, // Khả năng dự đoán đúng hiệu năng của chính mình
}

// --- 5.2 Reflection Engine ---

pub struct PerformanceAnalyzer;
pub struct ErrorAnalyzer;
pub struct InsightGenerator;

#[derive(Debug, Clone)]
pub enum ReflectionTrigger {
    PerformanceDrop { threshold: f32 },
    ErrorSpike { count: usize, timeframe: Duration },
    UserFeedback { rating: f32 },
    ResourceExhaustion,
    NewTaskType,
    ScheduledReflection { interval: Duration },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReflectionType {
    PerformanceReflection,
    ErrorReflection,
    LearningReflection,
    CapabilityReflection,
}

// Định nghĩa tạm thời Action tại đây để tránh phụ thuộc chéo
#[derive(Debug, Clone)]
pub enum Action {
    RouteToSkill(String),
    ComposePipeline(Vec<String>),
    RequestMoreInfo,
    TriggerSelfCorrection,
    EscalateToHuman,
}

#[derive(Debug, Clone)]
pub struct ReflectionResult {
    pub insights: Vec<String>,
    pub recommended_actions: Vec<Action>,
}

#[allow(dead_code)]
pub struct ReflectionEngine {
    performance_analyzer: PerformanceAnalyzer,
    error_analyzer: ErrorAnalyzer,
    insight_generator: InsightGenerator,
    reflection_triggers: Vec<ReflectionTrigger>,
}

impl Default for ReflectionEngine {
    fn default() -> Self {
        Self {
            performance_analyzer: PerformanceAnalyzer,
            error_analyzer: ErrorAnalyzer,
            insight_generator: InsightGenerator,
            reflection_triggers: Vec::new(),
        }
    }
}

// --- Meta-Cognitive Controller ---

pub struct MetaCognitiveController {
    pub self_model: Arc<RwLock<SelfModel>>,
    pub reflection_engine: ReflectionEngine,
    // ... các thành phần khác như ConsciousnessMonitor, MetaLearningSystem
}

impl MetaCognitiveController {
    /// Giám sát trạng thái hệ thống và thực hiện phản tư.
    pub async fn monitor_and_reflect(&self) -> Result<ReflectionResult, McgError> {
        let _self_model = self
            .self_model
            .read()
            .map_err(|_| McgError::SelfModelInaccessible)?;

        // Logic phản tư:
        // 1. Dùng `reflection_engine.performance_analyzer` để phân tích lịch sử hiệu năng.
        // 2. Dùng `reflection_engine.error_analyzer` để tìm nguyên nhân gốc của các lỗi.
        // 3. Dựa trên các `reflection_triggers`, quyết định có cần phản tư sâu hơn không.
        // 4. Nếu có, dùng `insight_generator` để tạo ra các "insight" (ví dụ: "Skill X yếu ở domain Y").
        // 5. Tạo ra các hành động đề xuất (ví dụ: "Kích hoạt EvolutionEngine cho Skill X", "Tăng tài nguyên cho Skill Z").

        // TODO: Hiện thực hóa logic phản tư chi tiết.

        // Trả về kết quả giả
        Ok(ReflectionResult {
            insights: vec!["Hiệu năng ổn định.".to_string()],
            recommended_actions: vec![],
        })
    }
}

impl MetaCognitiveController {
    pub fn new() -> Self {
        Self {
            self_model: Arc::new(RwLock::new(SelfModel::default())),
            reflection_engine: ReflectionEngine::default(),
        }
    }
}

// ====== Enhanced MCG Implementation ======
pub mod enhanced_mcg;

pub mod causal_discovery {
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct CausalHypothesis {
        pub from_node_index: usize,
        pub to_node_index: usize,
        pub strength: f32,
        pub confidence: f32,
        pub edge_type: CausalEdgeType,
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum CausalEdgeType {
        Direct,
        Indirect,
        Conditional,
        Inhibitory,
    }
    impl Default for CausalEdgeType {
        fn default() -> Self {
            CausalEdgeType::Direct
        }
    }
    #[derive(Clone, Default)]
    pub struct DiscoveryConfig {
        pub min_strength_threshold: f32,
        pub min_confidence_threshold: f32,
        pub max_hypotheses: usize,
        pub algorithm: CausalAlgorithm,
    }
    pub type CausalDiscoveryConfig = DiscoveryConfig;
    #[derive(Clone, Copy, Debug)]
    pub enum CausalAlgorithm {
        Greedy,
        Exhaustive,
        DirectLiNGAM,
        PC,
        GES,
    }
    impl Default for CausalAlgorithm {
        fn default() -> Self {
            CausalAlgorithm::Greedy
        }
    }
    pub fn validate_hypothesis(h: &CausalHypothesis, data: &Vec<Vec<f32>>) -> bool {
        if data.len() < 3 { return false; }
        (h.strength >= 0.1) && (h.confidence >= 0.2)
    }
    pub fn discover_causal_links<T>(_data: T, _cfg: &DiscoveryConfig) -> Result<Vec<CausalHypothesis>, String> { Ok(vec![]) }
}

// ====== Legacy compatibility shims for older tests ======
pub mod legacy_shims {
    #[derive(Clone, Debug)]
    pub enum ActionTrigger {
        TriggerSelfImprovementLevel1 {
            reason: String,
            target_component: String,
        },
    }

    #[derive(Clone, Debug)]
    pub enum MetaRule {
        IfCompressionRewardExceeds {
            threshold: f64,
            action: ActionTrigger,
        },
    }

    #[derive(Clone, Debug)]
    pub struct RuleEngine {
        pub rules: Vec<MetaRule>,
    }

    impl RuleEngine {
        pub fn new(rules: Vec<MetaRule>) -> Self {
            Self { rules }
        }
    }

    #[derive(Clone, Debug)]
    pub struct MetaCognitiveGovernor {
        pub rule_engine: RuleEngine,
    }

    impl MetaCognitiveGovernor {
        pub fn new(rule_engine: RuleEngine) -> Self {
            Self { rule_engine }
        }
    }
}

pub use legacy_shims::{MetaRule, RuleEngine};
pub use enhanced_mcg::{
    EnhancedMetaCognitiveGovernor as MetaCognitiveGovernor, 
    ActionTrigger, 
    SystemMetrics, 
    DecisionWithConfidence,
    ObservationBuffer,
};
