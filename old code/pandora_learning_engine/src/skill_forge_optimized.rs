// Optimized SkillForge Implementation with Candle and dfdx
// Tối ưu hóa cho thiết bị di động và biên với QueST Encoder

use async_trait::async_trait;
use candle_core::{Device, Tensor, DType, Result as CandleResult};
use dfdx::prelude::*;
use pandora_core::error::PandoraError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// ===== Enhanced Types =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Intent {
    ExecuteTask { 
        task: String, 
        parameters: HashMap<String, String>,
        complexity: f32,
        priority: u8,
    },
    MaintainEquilibrium {
        current_state: HashMap<String, f32>,
        target_state: HashMap<String, f32>,
    },
    ExecutePlan { 
        plan: Vec<String>,
        estimated_duration: u64,
        resource_requirements: ResourceRequirements,
    },
    LearnSkill {
        skill_type: String,
        training_data: Vec<f32>,
        target_performance: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u8,
    pub memory_mb: u32,
    pub gpu_memory_mb: u32,
    pub network_bandwidth_mbps: u32,
    pub storage_mb: u32,
}

#[async_trait]
pub trait Skill {
    async fn execute(&self, intent: &Intent) -> Result<Intent, PandoraError>;
    fn get_skill_id(&self) -> String;
    fn get_performance_metrics(&self) -> SkillPerformanceMetrics;
    fn can_handle(&self, intent: &Intent) -> bool;
    fn get_resource_requirements(&self) -> ResourceRequirements;
}

// ===== QueST Encoder with Candle =====

pub struct QueSTEncoder {
    device: Device,
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    sequence_length: usize,
    quantized: bool,
}

impl QueSTEncoder {
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        sequence_length: usize,
        quantized: bool,
    ) -> Self {
        let device = Device::Cpu; // Có thể mở rộng cho GPU
        
        Self {
            device,
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            sequence_length,
            quantized,
        }
    }

    /// Encode input sequence thành latent representation
    pub fn encode(&self, input_ids: &[u32]) -> CandleResult<Tensor> {
        let batch_size = 1;
        let seq_len = input_ids.len().min(self.sequence_length);
        
        // Create input tensor
        let input_tensor = Tensor::new(input_ids, &self.device)?
            .reshape((batch_size, seq_len))?;
        
        // Apply quantization nếu cần
        let encoded = if self.quantized {
            self.quantize_tensor(input_tensor)?
        } else {
            input_tensor
        };
        
        // Apply transformer layers
        let mut hidden_states = self.embed_tokens(&encoded)?;
        
        for layer_idx in 0..self.num_layers {
            hidden_states = self.transformer_layer(&hidden_states, layer_idx)?;
        }
        
        // Pooling để tạo sequence representation
        self.pool_sequence(&hidden_states)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        // Token embedding layer
        let vocab_embedding = Tensor::randn(
            0.0,
            0.02,
            (self.vocab_size, self.hidden_size),
            DType::F32,
            &self.device,
        )?;
        
        let position_embedding = Tensor::randn(
            0.0,
            0.02,
            (self.sequence_length, self.hidden_size),
            DType::F32,
            &self.device,
        )?;
        
        // Gather token embeddings
        let token_embeds = input_ids.matmul(&vocab_embedding)?;
        
        // Add position embeddings
        let seq_len = input_ids.dim(1)?;
        let pos_embeds = position_embedding.narrow(1, 0, seq_len)?;
        
        token_embeds.broadcast_add(&pos_embeds)
    }

    fn transformer_layer(&self, hidden_states: &Tensor, layer_idx: usize) -> CandleResult<Tensor> {
        // Multi-head attention
        let attention_output = self.multi_head_attention(hidden_states)?;
        
        // Layer normalization
        let norm1_output = self.layer_norm(&attention_output.broadcast_add(hidden_states)?)?;
        
        // Feed-forward network
        let ff_output = self.feed_forward(&norm1_output)?;
        
        // Final layer normalization
        self.layer_norm(&ff_output.broadcast_add(&norm1_output)?)
    }

    fn multi_head_attention(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        let seq_len = hidden_states.dim(1)?;
        
        // Linear projections for Q, K, V
        let q_proj = Tensor::randn(0.0, 0.02, (self.hidden_size, self.hidden_size), DType::F32, &self.device)?;
        let k_proj = Tensor::randn(0.0, 0.02, (self.hidden_size, self.hidden_size), DType::F32, &self.device)?;
        let v_proj = Tensor::randn(0.0, 0.02, (self.hidden_size, self.hidden_size), DType::F32, &self.device)?;
        
        let q = hidden_states.matmul(&q_proj)?;
        let k = hidden_states.matmul(&k_proj)?;
        let v = hidden_states.matmul(&v_proj)?;
        
        // Reshape for multi-head attention
        let head_dim = self.hidden_size / self.num_heads;
        let q_reshaped = q.reshape((batch_size, seq_len, self.num_heads, head_dim))?;
        let k_reshaped = k.reshape((batch_size, seq_len, self.num_heads, head_dim))?;
        let v_reshaped = v.reshape((batch_size, seq_len, self.num_heads, head_dim))?;
        
        // Scaled dot-product attention
        let attention_scores = self.scaled_dot_product_attention(&q_reshaped, &k_reshaped, &v_reshaped)?;
        
        // Reshape back
        attention_scores.reshape((batch_size, seq_len, self.hidden_size))
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> CandleResult<Tensor> {
        let head_dim = self.hidden_size / self.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        // Compute attention scores
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scaled_scores = scores.broadcast_mul(&Tensor::new(&[scale], &self.device)?)?;
        
        // Apply softmax
        let attention_weights = self.softmax(&scaled_scores)?;
        
        // Apply attention to values
        attention_weights.matmul(v)
    }

    fn softmax(&self, input: &Tensor) -> CandleResult<Tensor> {
        let max_vals = input.max_keepdim(3)?;
        let shifted = input.broadcast_sub(&max_vals)?;
        let exp_vals = shifted.exp()?;
        let sum_vals = exp_vals.sum_keepdim(3)?;
        exp_vals.broadcast_div(&sum_vals)
    }

    fn feed_forward(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        let intermediate_size = self.hidden_size * 4;
        
        // First linear layer
        let w1 = Tensor::randn(0.0, 0.02, (self.hidden_size, intermediate_size), DType::F32, &self.device)?;
        let b1 = Tensor::zeros((intermediate_size,), DType::F32, &self.device)?;
        let intermediate = hidden_states.matmul(&w1)?.broadcast_add(&b1)?;
        
        // GELU activation
        let activated = self.gelu(&intermediate)?;
        
        // Second linear layer
        let w2 = Tensor::randn(0.0, 0.02, (intermediate_size, self.hidden_size), DType::F32, &self.device)?;
        let b2 = Tensor::zeros((self.hidden_size,), DType::F32, &self.device)?;
        
        activated.matmul(&w2)?.broadcast_add(&b2)
    }

    fn gelu(&self, input: &Tensor) -> CandleResult<Tensor> {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = input.powf(3.0)?;
        let inner = input.broadcast_mul(&Tensor::new(&[0.044715], &self.device)?)?.broadcast_add(&x_cubed)?;
        let inner = inner.broadcast_mul(&Tensor::new(&[0.7978845608], &self.device)?)?; // sqrt(2/π)
        let tanh_inner = inner.tanh()?;
        let one = Tensor::new(&[1.0], &self.device)?;
        let tanh_part = tanh_inner.broadcast_add(&one)?;
        let half = Tensor::new(&[0.5], &self.device)?;
        input.broadcast_mul(&tanh_part)?.broadcast_mul(&half)
    }

    fn layer_norm(&self, input: &Tensor) -> CandleResult<Tensor> {
        let epsilon = 1e-5;
        let mean = input.mean_keepdim(2)?;
        let variance = input.var_keepdim(2)?;
        let normalized = input.broadcast_sub(&mean)?.broadcast_div(&(variance.broadcast_add(&Tensor::new(&[epsilon], &self.device)?)?.sqrt()?)?;
        
        // Apply learnable parameters (simplified)
        let gamma = Tensor::ones((self.hidden_size,), DType::F32, &self.device)?;
        let beta = Tensor::zeros((self.hidden_size,), DType::F32, &self.device)?;
        
        normalized.broadcast_mul(&gamma)?.broadcast_add(&beta)
    }

    fn pool_sequence(&self, hidden_states: &Tensor) -> CandleResult<Tensor> {
        // Mean pooling
        hidden_states.mean_keepdim(1)
    }

    fn quantize_tensor(&self, tensor: Tensor) -> CandleResult<Tensor> {
        if !self.quantized {
            return Ok(tensor);
        }
        
        // INT8 quantization
        let scale = 127.0;
        let quantized = tensor.broadcast_mul(&Tensor::new(&[scale], &self.device)?)?;
        let clamped = quantized.clamp(-128.0, 127.0)?;
        clamped.round()?.to_dtype(DType::I8)
    }
}

// ===== Vector Quantizer =====

pub struct VectorQuantizer {
    codebook_size: usize,
    codebook_dim: usize,
    commitment_cost: f32,
    device: Device,
}

impl VectorQuantizer {
    pub fn new(codebook_size: usize, codebook_dim: usize, commitment_cost: f32) -> Self {
        Self {
            codebook_size,
            codebook_dim,
            commitment_cost,
            device: Device::Cpu,
        }
    }

    pub fn quantize(&self, inputs: &Tensor) -> CandleResult<(Tensor, Tensor, f32)> {
        let input_shape = inputs.shape();
        let flat_inputs = inputs.reshape((input_shape[0] * input_shape[1], input_shape[2]))?;
        
        // Initialize codebook
        let codebook = Tensor::randn(
            0.0,
            0.02,
            (self.codebook_size, self.codebook_dim),
            DType::F32,
            &self.device,
        )?;
        
        // Compute distances
        let distances = self.compute_distances(&flat_inputs, &codebook)?;
        
        // Find closest codes
        let encoding_indices = distances.argmax(1)?;
        
        // Quantize
        let quantized = self.quantize_codes(&encoding_indices, &codebook)?;
        
        // Compute commitment loss
        let commitment_loss = self.compute_commitment_loss(&flat_inputs, &quantized)?;
        
        // Straight-through estimator
        let quantized_st = flat_inputs.broadcast_add(&quantized.broadcast_sub(&flat_inputs)?)?;
        
        Ok((
            quantized_st.reshape(input_shape)?,
            encoding_indices,
            commitment_loss,
        ))
    }

    fn compute_distances(&self, inputs: &Tensor, codebook: &Tensor) -> CandleResult<Tensor> {
        let input_norm = inputs.powf(2.0)?.sum_keepdim(1)?;
        let codebook_norm = codebook.powf(2.0)?.sum_keepdim(1)?;
        let distances = input_norm.broadcast_add(&codebook_norm.transpose(0, 1)?)?;
        
        // Compute 2 * <inputs, codebook>
        let dot_product = inputs.matmul(&codebook.transpose(0, 1)?)?;
        let double_dot = dot_product.broadcast_mul(&Tensor::new(&[2.0], &self.device)?)?;
        
        distances.broadcast_sub(&double_dot)
    }

    fn quantize_codes(&self, encoding_indices: &Tensor, codebook: &Tensor) -> CandleResult<Tensor> {
        // One-hot encoding
        let batch_size = encoding_indices.dim(0)?;
        let mut quantized = Vec::new();
        
        for i in 0..batch_size {
            let idx = encoding_indices.get(i)?;
            let code = codebook.get(idx.to_scalar::<u32>()? as usize)?;
            quantized.push(code);
        }
        
        Tensor::stack(&quantized, 0)
    }

    fn compute_commitment_loss(&self, inputs: &Tensor, quantized: &Tensor) -> CandleResult<f32> {
        let diff = inputs.broadcast_sub(quantized)?;
        let mse = diff.powf(2.0)?.mean_all()?;
        Ok(mse.to_scalar::<f32>()? * self.commitment_cost)
    }
}

// ===== Code Generator =====

#[async_trait]
pub trait CodeGenerator {
    async fn generate_skill_code(&self, intent: &Intent, context: &SkillContext) -> Result<String, PandoraError>;
    async fn optimize_code(&self, code: &str, performance_metrics: &SkillPerformanceMetrics) -> Result<String, PandoraError>;
}

pub struct LLMCodeGenerator {
    model_path: String,
    device: Device,
    max_length: usize,
    temperature: f32,
}

impl LLMCodeGenerator {
    pub fn new(model_path: String, max_length: usize, temperature: f32) -> Self {
        Self {
            model_path,
            device: Device::Cpu,
            max_length,
            temperature,
        }
    }
}

#[async_trait]
impl CodeGenerator for LLMCodeGenerator {
    async fn generate_skill_code(&self, intent: &Intent, context: &SkillContext) -> Result<String, PandoraError> {
        // Simplified code generation - trong thực tế sẽ sử dụng LLM
        let prompt = self.create_prompt(intent, context);
        
        // Placeholder implementation
        let generated_code = match intent {
            Intent::ExecuteTask { task, parameters, .. } => {
                format!(
                    "async fn execute_task() -> Result<(), Error> {{
                        // Task: {}
                        // Parameters: {:?}
                        // Implementation here
                        Ok(())
                    }}",
                    task,
                    parameters
                )
            }
            Intent::MaintainEquilibrium { current_state, target_state } => {
                format!(
                    "async fn maintain_equilibrium() -> Result<(), Error> {{
                        // Current: {:?}
                        // Target: {:?}
                        // Balance logic here
                        Ok(())
                    }}",
                    current_state,
                    target_state
                )
            }
            Intent::ExecutePlan { plan, .. } => {
                format!(
                    "async fn execute_plan() -> Result<(), Error> {{
                        // Plan: {:?}
                        // Sequential execution here
                        Ok(())
                    }}",
                    plan
                )
            }
            Intent::LearnSkill { skill_type, .. } => {
                format!(
                    "async fn learn_skill() -> Result<(), Error> {{
                        // Skill type: {}
                        // Learning algorithm here
                        Ok(())
                    }}",
                    skill_type
                )
            }
        };
        
        Ok(generated_code)
    }

    async fn optimize_code(&self, code: &str, metrics: &SkillPerformanceMetrics) -> Result<String, PandoraError> {
        // Code optimization based on performance metrics
        let mut optimized = code.to_string();
        
        // Simple optimizations
        if metrics.execution_time_ms > 1000.0 {
            optimized = optimized.replace("// Implementation here", "// Optimized implementation");
        }
        
        if metrics.memory_usage_mb > 100.0 {
            optimized = optimized.replace("// Sequential execution here", "// Memory-efficient execution");
        }
        
        Ok(optimized)
    }
}

impl LLMCodeGenerator {
    fn create_prompt(&self, intent: &Intent, context: &SkillContext) -> String {
        format!(
            "Generate Rust code for the following intent:\n\
            Intent: {:?}\n\
            Context: {:?}\n\
            Requirements: Performance: {}, Memory: {}MB, CPU: {} cores\n\
            Generate efficient, safe Rust code:",
            intent,
            context,
            context.performance_target,
            context.memory_limit_mb,
            context.cpu_cores
        )
    }
}

// ===== Skill Context =====

#[derive(Debug, Clone)]
pub struct SkillContext {
    pub performance_target: f32,
    pub memory_limit_mb: u32,
    pub cpu_cores: u8,
    pub gpu_available: bool,
    pub network_bandwidth_mbps: u32,
    pub latency_requirement_ms: u32,
}

// ===== Performance Metrics =====

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPerformanceMetrics {
    pub execution_time_ms: f32,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    pub gpu_usage_percent: f32,
    pub accuracy: f32,
    pub throughput: f32,
    pub energy_consumption_j: f32,
    pub cache_hit_rate: f32,
}

// ===== Generated Skill =====

pub struct GeneratedSkill {
    pub skill_id: String,
    pub code: String,
    pub performance_metrics: SkillPerformanceMetrics,
    pub resource_requirements: ResourceRequirements,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
    pub usage_count: u32,
}

// ===== SkillForge Main Implementation =====

pub struct SkillForge {
    quest_encoder: QueSTEncoder,
    vector_quantizer: VectorQuantizer,
    code_generator: Box<dyn CodeGenerator + Send + Sync>,
    skills: HashMap<String, GeneratedSkill>,
    performance_tracker: PerformanceTracker,
}

pub struct PerformanceTracker {
    metrics: HashMap<String, Vec<SkillPerformanceMetrics>>,
    thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_execution_time_ms: f32,
    pub max_memory_usage_mb: f32,
    pub min_accuracy: f32,
    pub min_throughput: f32,
    pub max_energy_consumption_j: f32,
}

impl SkillForge {
    pub fn new(
        quest_encoder: QueSTEncoder,
        vector_quantizer: VectorQuantizer,
        code_generator: Box<dyn CodeGenerator + Send + Sync>,
    ) -> Self {
        Self {
            quest_encoder,
            vector_quantizer,
            code_generator,
            skills: HashMap::new(),
            performance_tracker: PerformanceTracker::new(),
        }
    }

    /// Forge a new skill based on intent
    pub async fn forge_skill(&mut self, intent: &Intent, context: &SkillContext) -> Result<String, PandoraError> {
        // Encode intent using QueST
        let intent_encoding = self.encode_intent(intent)?;
        
        // Quantize the encoding
        let (quantized, encoding_indices, commitment_loss) = self.vector_quantizer.quantize(&intent_encoding)?;
        
        // Generate code using LLM
        let generated_code = self.code_generator.generate_skill_code(intent, context).await?;
        
        // Create skill ID
        let skill_id = Uuid::new_v4().to_string();
        
        // Create generated skill
        let skill = GeneratedSkill {
            skill_id: skill_id.clone(),
            code: generated_code,
            performance_metrics: SkillPerformanceMetrics::default(),
            resource_requirements: self.estimate_resource_requirements(intent, context),
            created_at: chrono::Utc::now(),
            last_used: None,
            usage_count: 0,
        };
        
        // Store skill
        self.skills.insert(skill_id.clone(), skill);
        
        Ok(skill_id)
    }

    /// Execute a skill
    pub async fn execute_skill(&mut self, skill_id: &str, intent: &Intent) -> Result<Intent, PandoraError> {
        let skill = self.skills.get_mut(skill_id)
            .ok_or_else(|| PandoraError::SkillNotFound(skill_id.to_string()))?;
        
        // Update usage
        skill.last_used = Some(chrono::Utc::now());
        skill.usage_count += 1;
        
        // Execute skill (simplified)
        let start_time = std::time::Instant::now();
        
        // In real implementation, this would execute the generated code
        let result = self.simulate_skill_execution(intent).await?;
        
        let execution_time = start_time.elapsed().as_millis() as f32;
        
        // Update performance metrics
        skill.performance_metrics.execution_time_ms = execution_time;
        skill.performance_metrics.memory_usage_mb = self.estimate_memory_usage(intent);
        skill.performance_metrics.cpu_usage_percent = self.estimate_cpu_usage(intent);
        
        // Track performance
        self.performance_tracker.record_metrics(skill_id, &skill.performance_metrics);
        
        Ok(result)
    }

    fn encode_intent(&self, intent: &Intent) -> Result<Tensor, PandoraError> {
        // Convert intent to token IDs (simplified)
        let tokens = self.intent_to_tokens(intent);
        self.quest_encoder.encode(&tokens)
            .map_err(|e| PandoraError::EncodingError(e.to_string()))
    }

    fn intent_to_tokens(&self, intent: &Intent) -> Vec<u32> {
        // Simplified tokenization
        match intent {
            Intent::ExecuteTask { task, .. } => {
                vec![1, 2, 3] // Placeholder tokens
            }
            Intent::MaintainEquilibrium { .. } => {
                vec![4, 5, 6]
            }
            Intent::ExecutePlan { .. } => {
                vec![7, 8, 9]
            }
            Intent::LearnSkill { .. } => {
                vec![10, 11, 12]
            }
        }
    }

    fn estimate_resource_requirements(&self, intent: &Intent, context: &SkillContext) -> ResourceRequirements {
        match intent {
            Intent::ExecuteTask { complexity, .. } => {
                ResourceRequirements {
                    cpu_cores: (*complexity * 2.0) as u8,
                    memory_mb: (*complexity * 100.0) as u32,
                    gpu_memory_mb: if *complexity > 0.7 { 512 } else { 0 },
                    network_bandwidth_mbps: (*complexity * 50.0) as u32,
                    storage_mb: (*complexity * 200.0) as u32,
                }
            }
            Intent::MaintainEquilibrium { .. } => {
                ResourceRequirements {
                    cpu_cores: 1,
                    memory_mb: 50,
                    gpu_memory_mb: 0,
                    network_bandwidth_mbps: 10,
                    storage_mb: 20,
                }
            }
            Intent::ExecutePlan { .. } => {
                ResourceRequirements {
                    cpu_cores: 2,
                    memory_mb: 150,
                    gpu_memory_mb: 0,
                    network_bandwidth_mbps: 25,
                    storage_mb: 100,
                }
            }
            Intent::LearnSkill { .. } => {
                ResourceRequirements {
                    cpu_cores: 4,
                    memory_mb: 500,
                    gpu_memory_mb: 1024,
                    network_bandwidth_mbps: 100,
                    storage_mb: 1000,
                }
            }
        }
    }

    async fn simulate_skill_execution(&self, intent: &Intent) -> Result<Intent, PandoraError> {
        // Simulate skill execution
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        match intent {
            Intent::ExecuteTask { task, .. } => {
                Ok(Intent::ExecuteTask {
                    task: format!("Completed: {}", task),
                    parameters: HashMap::new(),
                    complexity: 0.0,
                    priority: 0,
                })
            }
            Intent::MaintainEquilibrium { current_state, target_state } => {
                Ok(Intent::MaintainEquilibrium {
                    current_state: target_state.clone(),
                    target_state: current_state.clone(),
                })
            }
            Intent::ExecutePlan { plan, .. } => {
                Ok(Intent::ExecutePlan {
                    plan: vec!["Completed".to_string()],
                    estimated_duration: 0,
                    resource_requirements: ResourceRequirements {
                        cpu_cores: 0,
                        memory_mb: 0,
                        gpu_memory_mb: 0,
                        network_bandwidth_mbps: 0,
                        storage_mb: 0,
                    },
                })
            }
            Intent::LearnSkill { skill_type, .. } => {
                Ok(Intent::LearnSkill {
                    skill_type: format!("Learned: {}", skill_type),
                    training_data: vec![],
                    target_performance: 1.0,
                })
            }
        }
    }

    fn estimate_memory_usage(&self, intent: &Intent) -> f32 {
        match intent {
            Intent::ExecuteTask { complexity, .. } => *complexity * 100.0,
            Intent::MaintainEquilibrium { .. } => 50.0,
            Intent::ExecutePlan { .. } => 150.0,
            Intent::LearnSkill { .. } => 500.0,
        }
    }

    fn estimate_cpu_usage(&self, intent: &Intent) -> f32 {
        match intent {
            Intent::ExecuteTask { complexity, .. } => *complexity * 80.0,
            Intent::MaintainEquilibrium { .. } => 20.0,
            Intent::ExecutePlan { .. } => 60.0,
            Intent::LearnSkill { .. } => 90.0,
        }
    }

    /// Get skill by ID
    pub fn get_skill(&self, skill_id: &str) -> Option<&GeneratedSkill> {
        self.skills.get(skill_id)
    }

    /// List all skills
    pub fn list_skills(&self) -> Vec<&GeneratedSkill> {
        self.skills.values().collect()
    }

    /// Remove skill
    pub fn remove_skill(&mut self, skill_id: &str) -> Option<GeneratedSkill> {
        self.skills.remove(skill_id)
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        
        if !self.skills.is_empty() {
            let total_execution_time: f32 = self.skills.values()
                .map(|s| s.performance_metrics.execution_time_ms)
                .sum();
            let avg_execution_time = total_execution_time / self.skills.len() as f32;
            
            stats.insert("avg_execution_time_ms".to_string(), avg_execution_time);
            stats.insert("total_skills".to_string(), self.skills.len() as f32);
        }
        
        stats
    }
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            thresholds: PerformanceThresholds {
                max_execution_time_ms: 1000.0,
                max_memory_usage_mb: 500.0,
                min_accuracy: 0.8,
                min_throughput: 10.0,
                max_energy_consumption_j: 1000.0,
            },
        }
    }

    pub fn record_metrics(&mut self, skill_id: &str, metrics: &SkillPerformanceMetrics) {
        self.metrics.entry(skill_id.to_string())
            .or_insert_with(Vec::new)
            .push(metrics.clone());
    }

    pub fn get_skill_performance(&self, skill_id: &str) -> Option<&Vec<SkillPerformanceMetrics>> {
        self.metrics.get(skill_id)
    }

    pub fn is_performance_acceptable(&self, metrics: &SkillPerformanceMetrics) -> bool {
        metrics.execution_time_ms <= self.thresholds.max_execution_time_ms
            && metrics.memory_usage_mb <= self.thresholds.max_memory_usage_mb
            && metrics.accuracy >= self.thresholds.min_accuracy
            && metrics.throughput >= self.thresholds.min_throughput
            && metrics.energy_consumption_j <= self.thresholds.max_energy_consumption_j
    }
}

impl Default for SkillPerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            gpu_usage_percent: 0.0,
            accuracy: 1.0,
            throughput: 0.0,
            energy_consumption_j: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

// ===== Error Extensions =====

impl From<candle_core::Error> for PandoraError {
    fn from(err: candle_core::Error) -> Self {
        PandoraError::PredictionFailed(format!("Candle error: {}", err))
    }
}

// Add new error variants to PandoraError
impl PandoraError {
    pub fn SkillNotFound(skill_id: String) -> Self {
        PandoraError::SkillVerificationFailed(format!("Skill not found: {}", skill_id))
    }
    
    pub fn EncodingError(msg: String) -> Self {
        PandoraError::PredictionFailed(format!("Encoding error: {}", msg))
    }
}
