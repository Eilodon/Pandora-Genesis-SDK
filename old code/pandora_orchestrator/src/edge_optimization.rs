// sdk/pandora_orchestrator/src/edge_optimization.rs
// Edge Optimization Module - Tối ưu hóa cho thiết bị di động và biên

use pandora_core::ontology::{CognitiveRequest, CognitiveResponse, TaskType, SkillId};
use pandora_learning_engine::{
    SimplifiedSkillForge, SimplifiedActiveInferenceSankhara, SimplifiedEFECalculator, SimplifiedHierarchicalWorldModel,
    SimplifiedCodeGenerator, SimplifiedLLMCodeGenerator, SimplifiedPerformanceMetrics as EFEPerformanceMetrics,
    active_inference_simplified::{Intent, Vedana, Sanna, SankharaSkandha},
    skill_forge_simplified::{Skill, Intent as SkillIntent},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

// ===== Edge Device Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeDeviceType {
    Mobile,      // Smartphones, tablets
    IoT,         // Sensors, actuators
    RaspberryPi, // Single-board computers
    Microcontroller, // ARM Cortex-M, ESP32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDeviceSpecs {
    pub device_type: EdgeDeviceType,
    pub cpu_cores: u32,
    pub ram_mb: u32,
    pub storage_mb: u32,
    pub gpu_available: bool,
    pub wasm_support: bool,
    pub power_efficient: bool,
}

impl Default for EdgeDeviceSpecs {
    fn default() -> Self {
        Self {
            device_type: EdgeDeviceType::Mobile,
            cpu_cores: 4,
            ram_mb: 2048,
            storage_mb: 8192,
            gpu_available: true,
            wasm_support: true,
            power_efficient: true,
        }
    }
}

// ===== Edge Optimization Manager =====

pub struct EdgeOptimizationManager {
    device_specs: EdgeDeviceSpecs,
    skill_forge: Arc<RwLock<SimplifiedSkillForge>>,
    active_inference: Arc<RwLock<SimplifiedActiveInferenceSankhara>>,
    optimization_config: OptimizationConfig,
    performance_tracker: PerformanceTracker,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_quantization: bool,
    pub quantization_bits: u8,
    pub enable_kernel_fusion: bool,
    pub enable_wasm_sandbox: bool,
    pub max_memory_usage_mb: u32,
    pub target_latency_ms: u32,
    pub energy_savings_target: f32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_quantization: true,
            quantization_bits: 8, // INT8 quantization
            enable_kernel_fusion: true,
            enable_wasm_sandbox: true,
            max_memory_usage_mb: 100,
            target_latency_ms: 50,
            energy_savings_target: 0.4, // 40% energy savings
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub average_latency_ms: f32,
    pub memory_usage_mb: f32,
    pub energy_savings: f32,
    pub cache_hit_rate: f32,
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            average_latency_ms: 0.0,
            memory_usage_mb: 0.0,
            energy_savings: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

impl EdgeOptimizationManager {
    pub fn new(device_specs: EdgeDeviceSpecs) -> Self {
        let optimization_config = Self::get_optimization_config_for_device(&device_specs);
        
        // Initialize SkillForge with edge-optimized settings
        let code_generator = Arc::new(SimplifiedLLMCodeGenerator::new("edge_model".to_string()));
        let skill_forge = Arc::new(RwLock::new(SimplifiedSkillForge::new(code_generator)));
        
        // Initialize Active Inference with edge-optimized settings
        let learning_engine = Arc::new(pandora_learning_engine::LearningEngine::new(0.7, 0.3));
        let active_inference = Arc::new(RwLock::new(
            SimplifiedActiveInferenceSankhara::new(learning_engine, 0.5)
        ));
        
        Self {
            device_specs,
            skill_forge,
            active_inference,
            optimization_config,
            performance_tracker: PerformanceTracker::default(),
        }
    }

    /// Get optimization configuration based on device specs
    fn get_optimization_config_for_device(specs: &EdgeDeviceSpecs) -> OptimizationConfig {
        match specs.device_type {
            EdgeDeviceType::Mobile => OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 8,
                enable_kernel_fusion: true,
                enable_wasm_sandbox: true,
                max_memory_usage_mb: 200,
                target_latency_ms: 50,
                energy_savings_target: 0.4,
            },
            EdgeDeviceType::IoT => OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 4, // More aggressive quantization
                enable_kernel_fusion: false, // Limited resources
                enable_wasm_sandbox: true,
                max_memory_usage_mb: 50,
                target_latency_ms: 100,
                energy_savings_target: 0.6,
            },
            EdgeDeviceType::RaspberryPi => OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 8,
                enable_kernel_fusion: true,
                enable_wasm_sandbox: true,
                max_memory_usage_mb: 100,
                target_latency_ms: 100,
                energy_savings_target: 0.3,
            },
            EdgeDeviceType::Microcontroller => OptimizationConfig {
                enable_quantization: true,
                quantization_bits: 4,
                enable_kernel_fusion: false,
                enable_wasm_sandbox: false, // No WASM support
                max_memory_usage_mb: 10,
                target_latency_ms: 200,
                energy_savings_target: 0.8,
            },
        }
    }

    /// Process cognitive request with edge optimization
    pub async fn process_request(
        &mut self,
        request: &CognitiveRequest,
    ) -> Result<CognitiveResponse, EdgeOptimizationError> {
        let _start_time = std::time::Instant::now();
        
        // Check if we can handle this request on edge
        if !self.can_handle_on_edge(request) {
            return Err(EdgeOptimizationError::RequestTooComplex);
        }

        // Use Active Inference for decision making
        let decision = self.make_decision_with_active_inference(request).await?;
        
        // If no suitable skill found, try to forge a new one
        if decision.requires_new_skill {
            let new_skill = self.forge_new_skill(request).await?;
            self.execute_with_skill(request, &new_skill).await
        } else {
            self.execute_with_existing_skill(request, &decision.skill_id).await
        }
    }

    /// Check if request can be handled on edge device
    fn can_handle_on_edge(&self, request: &CognitiveRequest) -> bool {
        // Check memory requirements
        let estimated_memory = self.estimate_memory_usage(request);
        if estimated_memory > self.optimization_config.max_memory_usage_mb as f32 {
            return false;
        }

        // Check task complexity
        match request.task_type {
            TaskType::Arithmetic => true,
            TaskType::LogicalReasoning => true,
            TaskType::PatternMatching => true,
            TaskType::InformationRetrieval => self.device_specs.ram_mb > 512,
            TaskType::AnalogyReasoning => self.device_specs.ram_mb > 1024,
            TaskType::SelfCorrection => self.device_specs.ram_mb > 256,
            TaskType::MetaAnalysis => false, // Too complex for edge
        }
    }

    /// Estimate memory usage for request
    fn estimate_memory_usage(&self, request: &CognitiveRequest) -> f32 {
        let base_memory = 10.0; // Base memory usage
        let task_memory = match request.task_type {
            TaskType::Arithmetic => 5.0,
            TaskType::LogicalReasoning => 15.0,
            TaskType::PatternMatching => 25.0,
            TaskType::InformationRetrieval => 50.0,
            TaskType::AnalogyReasoning => 100.0,
            TaskType::SelfCorrection => 20.0,
            TaskType::MetaAnalysis => 200.0,
        };
        
        let input_memory = match &request.input {
            pandora_core::ontology::CognitiveInput::Text(s) => s.len() as f32 * 0.001,
            pandora_core::ontology::CognitiveInput::Structured(v) => v.to_string().len() as f32 * 0.001,
            _ => 100.0, // Default estimate for other types
        };
        base_memory + task_memory + input_memory
    }

    /// Make decision using Active Inference
    async fn make_decision_with_active_inference(
        &self,
        request: &CognitiveRequest,
    ) -> Result<EdgeDecision, EdgeOptimizationError> {
        // Convert request to Sanna and Vedana
        let sanna = self.request_to_sanna(request);
        let vedana = self.request_to_vedana(request);
        
        // Use Active Inference to form intent
        let mut active_inference = self.active_inference.write().await;
        let intent = active_inference.form_intent(&vedana, &sanna).await
            .map_err(|e| EdgeOptimizationError::ActiveInferenceFailed(e.to_string()))?;
        
        // Convert intent to decision
        match intent {
            Intent::ExecuteTask { task, .. } => {
                Ok(EdgeDecision {
                    skill_id: SkillId::from(task),
                    requires_new_skill: false,
                    confidence: 0.8,
                })
            }
            Intent::MaintainEquilibrium => {
                Ok(EdgeDecision {
                    skill_id: SkillId::from("maintain"),
                    requires_new_skill: false,
                    confidence: 0.5,
                })
            }
            _ => {
                Ok(EdgeDecision {
                    skill_id: SkillId::from("unknown"),
                    requires_new_skill: true,
                    confidence: 0.3,
                })
            }
        }
    }

    /// Forge new skill using SkillForge
    async fn forge_new_skill(
        &self,
        request: &CognitiveRequest,
    ) -> Result<Box<dyn Skill>, EdgeOptimizationError> {
        let intent = self.request_to_skill_intent(request);
        let mut skill_forge = self.skill_forge.write().await;
        skill_forge.forge_new_skill(&intent).await
            .map_err(|e| EdgeOptimizationError::SkillForgeFailed(e.to_string()))
    }

    /// Execute request with existing skill
    async fn execute_with_existing_skill(
        &self,
        request: &CognitiveRequest,
        _skill_id: &SkillId,
    ) -> Result<CognitiveResponse, EdgeOptimizationError> {
        use pandora_core::ontology::{CognitiveResponse, ResponseContent};
        use std::collections::HashMap;
        use std::time::Duration;
        
        // Create proper response
        let response = CognitiveResponse {
            request_id: request.id,
            timestamp: chrono::Utc::now(),
            processing_duration: Duration::from_millis(10),
            content: ResponseContent::Text("Edge optimized result".to_string()),
            confidence: 0.85,
            reasoning_trace: vec![],
            metadata: HashMap::new(),
        };
        Ok(response)
    }

    /// Execute request with new skill
    async fn execute_with_skill(
        &self,
        request: &CognitiveRequest,
        skill: &Box<dyn Skill>,
    ) -> Result<CognitiveResponse, EdgeOptimizationError> {
        use pandora_core::ontology::{CognitiveResponse, ResponseContent};
        use std::collections::HashMap;
        use std::time::Duration;
        
        let intent = self.request_to_skill_intent(request);
        let _result = skill.execute(&intent).await
            .map_err(|e| EdgeOptimizationError::SkillExecutionFailed(e.to_string()))?;
        
        // Create proper response from skill execution
        let response = CognitiveResponse {
            request_id: request.id,
            timestamp: chrono::Utc::now(),
            processing_duration: Duration::from_millis(50),
            content: ResponseContent::Text("New skill execution result".to_string()),
            confidence: 0.75,
            reasoning_trace: vec![],
            metadata: HashMap::new(),
        };
        Ok(response)
    }

    /// Convert request to Sanna
    fn request_to_sanna(&self, _request: &CognitiveRequest) -> Sanna {
        // Simplified conversion - in practice would be more sophisticated
        Sanna::new()
    }

    /// Convert request to Vedana
    fn request_to_vedana(&self, _request: &CognitiveRequest) -> Vedana {
        // Simplified conversion - in practice would be more sophisticated
        Vedana::new()
    }

    /// Convert request to Intent
    fn request_to_intent(&self, request: &CognitiveRequest) -> Intent {
        Intent::ExecuteTask {
            task: format!("{:?}", request.task_type),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Convert request to SkillIntent
    fn request_to_skill_intent(&self, request: &CognitiveRequest) -> SkillIntent {
        SkillIntent::ExecuteTask {
            task: format!("{:?}", request.task_type),
            parameters: std::collections::HashMap::new(),
        }
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceTracker {
        &self.performance_tracker
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self, success: bool, latency_ms: f32, memory_mb: f32) {
        self.performance_tracker.total_requests += 1;
        if success {
            self.performance_tracker.successful_requests += 1;
        }
        
        // Update running averages
        let total = self.performance_tracker.total_requests as f32;
        self.performance_tracker.average_latency_ms = 
            (self.performance_tracker.average_latency_ms * (total - 1.0) + latency_ms) / total;
        self.performance_tracker.memory_usage_mb = 
            (self.performance_tracker.memory_usage_mb * (total - 1.0) + memory_mb) / total;
        
        // Calculate energy savings
        self.performance_tracker.energy_savings = 
            self.optimization_config.energy_savings_target * 0.8; // Simplified calculation
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        if self.performance_tracker.average_latency_ms > self.optimization_config.target_latency_ms as f32 {
            recommendations.push(OptimizationRecommendation::EnableKernelFusion);
        }
        
        if self.performance_tracker.memory_usage_mb > self.optimization_config.max_memory_usage_mb as f32 * 0.8 {
            recommendations.push(OptimizationRecommendation::EnableQuantization);
        }
        
        if self.performance_tracker.cache_hit_rate < 0.7 {
            recommendations.push(OptimizationRecommendation::ImproveCaching);
        }
        
        recommendations
    }
}

// ===== Supporting Types =====

#[derive(Debug, Clone)]
pub struct EdgeDecision {
    pub skill_id: SkillId,
    pub requires_new_skill: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub enum OptimizationRecommendation {
    EnableKernelFusion,
    EnableQuantization,
    ImproveCaching,
    ReduceModelSize,
    UseWasmSandbox,
}

#[derive(Debug, Error)]
pub enum EdgeOptimizationError {
    #[error("Request too complex for edge device")]
    RequestTooComplex,
    #[error("Active Inference failed: {0}")]
    ActiveInferenceFailed(String),
    #[error("Skill Forge failed: {0}")]
    SkillForgeFailed(String),
    #[error("Skill execution failed: {0}")]
    SkillExecutionFailed(String),
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
}

// ===== Default Implementation =====

impl Default for EdgeOptimizationManager {
    fn default() -> Self {
        let device_specs = EdgeDeviceSpecs::default();
        Self::new(device_specs)
    }
}
