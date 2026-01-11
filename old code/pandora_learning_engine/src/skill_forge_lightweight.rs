// Lightweight SkillForge Implementation with ndarray
// Tối ưu hóa cho thiết bị di động và biên với QueST Encoder

use async_trait::async_trait;
use ndarray::{Array2, Array1, Array3, s};
use pandora_core::error::PandoraError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
// use std::sync::Arc;  // Not used
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

// ===== QueST Encoder with ndarray =====

pub struct QueSTEncoder {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    sequence_length: usize,
    quantized: bool,
    // Pre-computed embeddings
    vocab_embedding: Array2<f32>,
    position_embedding: Array2<f32>,
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
        // Initialize embeddings with random values
        let vocab_embedding = Array2::from_shape_fn((vocab_size, hidden_size), |_| {
            fastrand::f32() * 0.02 - 0.01
        });
        
        let position_embedding = Array2::from_shape_fn((sequence_length, hidden_size), |_| {
            fastrand::f32() * 0.02 - 0.01
        });
        
        Self {
            vocab_size,
            hidden_size,
            num_layers,
            num_heads,
            sequence_length,
            quantized,
            vocab_embedding,
            position_embedding,
        }
    }

    /// Encode input sequence thành latent representation
    pub fn encode(&self, input_ids: &[u32]) -> Result<Array2<f32>, PandoraError> {
        let batch_size = 1;
        let seq_len = input_ids.len().min(self.sequence_length);
        
        // Create input tensor
        let mut input_tensor = Array2::zeros((batch_size, seq_len));
        for (i, &id) in input_ids.iter().enumerate() {
            input_tensor[[0, i]] = id as f32;
        }
        
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

    fn embed_tokens(&self, input_ids: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let (batch_size, seq_len) = input_ids.dim();
        
        // Gather token embeddings
        let mut token_embeds = Array3::zeros((batch_size, seq_len, self.hidden_size));
        
        for i in 0..batch_size {
            for j in 0..seq_len {
                let token_id = input_ids[[i, j]] as usize;
                if token_id < self.vocab_size {
                    token_embeds.slice_mut(s![i, j, ..]).assign(&self.vocab_embedding.slice(s![token_id, ..]));
                }
            }
        }
        
        // Add position embeddings
        let pos_embeds = self.position_embedding.slice(s![..seq_len, ..]);
        let mut result = Array2::zeros((batch_size, self.hidden_size));
        
        for i in 0..batch_size {
            let token_embeds_slice = token_embeds.slice(s![i, .., ..]);
            let summed = token_embeds_slice.sum_axis(ndarray::Axis(0));
            let with_pos = &summed + &pos_embeds.sum_axis(ndarray::Axis(0));
            result.slice_mut(s![i, ..]).assign(&with_pos);
        }
        
        Ok(result)
    }

    fn transformer_layer(&self, hidden_states: &Array2<f32>, _layer_idx: usize) -> Result<Array2<f32>, PandoraError> {
        // Multi-head attention
        let attention_output = self.multi_head_attention(hidden_states)?;
        
        // Layer normalization
        let norm1_output = self.layer_norm(&(&attention_output + hidden_states))?;
        
        // Feed-forward network
        let ff_output = self.feed_forward(&norm1_output)?;
        
        // Final layer normalization
        self.layer_norm(&(&ff_output + &norm1_output))
    }

    fn multi_head_attention(&self, hidden_states: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let (_batch_size, hidden_dim) = hidden_states.dim();
        
        // Linear projections for Q, K, V
        let q_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| fastrand::f32() * 0.02 - 0.01);
        let k_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| fastrand::f32() * 0.02 - 0.01);
        let v_proj = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| fastrand::f32() * 0.02 - 0.01);
        
        let q = hidden_states.dot(&q_proj);
        let k = hidden_states.dot(&k_proj);
        let v = hidden_states.dot(&v_proj);
        
        // Simplified attention for 2D arrays
        let attention_scores = self.scaled_dot_product_attention(&q, &k, &v)?;
        Ok(attention_scores)
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> Result<Array2<f32>, PandoraError> {
        let head_dim = self.hidden_size / self.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt() as f32;
        
        // Compute attention scores
        let scores = q.dot(&k.t()) * scale;
        
        // Apply softmax
        let attention_weights = self.softmax(&scores)?;
        
        // Apply attention to values
        Ok(attention_weights.dot(v))
    }

    fn softmax(&self, input: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let max_vals = input.map_axis(ndarray::Axis(1), |row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let shifted = input - &max_vals.insert_axis(ndarray::Axis(1));
        let exp_vals = shifted.mapv(|x| x.exp());
        let sum_vals = exp_vals.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
        Ok(exp_vals / sum_vals)
    }

    fn feed_forward(&self, hidden_states: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let intermediate_size = self.hidden_size * 4;
        
        // First linear layer
        let w1 = Array2::from_shape_fn((self.hidden_size, intermediate_size), |_| fastrand::f32() * 0.02 - 0.01);
        let b1: Array1<f32> = Array1::zeros(intermediate_size);
        let intermediate = hidden_states.dot(&w1) + &b1;
        
        // GELU activation
        let activated = self.gelu(&intermediate)?;
        
        // Second linear layer
        let w2 = Array2::from_shape_fn((intermediate_size, self.hidden_size), |_| fastrand::f32() * 0.02 - 0.01);
        let b2: Array1<f32> = Array1::zeros(self.hidden_size);
        
        Ok(activated.dot(&w2) + &b2)
    }

    fn gelu(&self, input: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = input.mapv(|x| x.powi(3));
        let inner = input + &x_cubed * 0.044715;
        let inner = &inner * 0.7978845608; // sqrt(2/π)
        let tanh_inner = inner.mapv(|x| x.tanh());
        let tanh_part = &tanh_inner + 1.0;
        let half = 0.5;
        Ok(input * &tanh_part * half)
    }

    fn layer_norm(&self, input: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let epsilon = 1e-5;
        let mean = input.mean_axis(ndarray::Axis(1)).unwrap().insert_axis(ndarray::Axis(1));
        let variance = input.var_axis(ndarray::Axis(1), 1.0).insert_axis(ndarray::Axis(1));
        let normalized = (input - &mean) / (variance + epsilon).mapv(|x| x.sqrt());
        
        // Apply learnable parameters (simplified)
        let gamma: Array1<f32> = Array1::ones(self.hidden_size);
        let beta: Array1<f32> = Array1::zeros(self.hidden_size);
        
        Ok(&normalized * &gamma + &beta)
    }

    fn pool_sequence(&self, hidden_states: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        // Mean pooling
        Ok(hidden_states.mean_axis(ndarray::Axis(0)).unwrap().insert_axis(ndarray::Axis(0)))
    }

    fn quantize_tensor(&self, tensor: Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        if !self.quantized {
            return Ok(tensor);
        }
        
        // INT8 quantization
        let scale = 127.0;
        let quantized = &tensor * scale;
        let clamped = quantized.mapv(|x| x.clamp(-128.0, 127.0));
        Ok(clamped.mapv(|x| x.round()))
    }
}

// ===== Vector Quantizer =====

pub struct VectorQuantizer {
    codebook_size: usize,
    codebook_dim: usize,
    commitment_cost: f32,
    codebook: Array2<f32>,
}

impl VectorQuantizer {
    pub fn new(codebook_size: usize, codebook_dim: usize, commitment_cost: f32) -> Self {
        let codebook = Array2::from_shape_fn((codebook_size, codebook_dim), |_| {
            fastrand::f32() * 0.02 - 0.01
        });
        
        Self {
            codebook_size,
            codebook_dim,
            commitment_cost,
            codebook,
        }
    }

    pub fn quantize(&self, inputs: &Array2<f32>) -> Result<(Array2<f32>, Array1<usize>, f32), PandoraError> {
        let (batch_size, seq_len) = inputs.dim();
        let hidden_dim = self.codebook.ncols();
        let flat_inputs = inputs.clone().into_shape((batch_size * seq_len, hidden_dim)).unwrap();
        
        // Compute distances
        let distances = self.compute_distances(&flat_inputs)?;
        
        // Find closest codes
        let encoding_indices = distances.map_axis(ndarray::Axis(1), |row| {
            row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
        });
        
        // Quantize
        let quantized = self.quantize_codes(&encoding_indices)?;
        
        // Compute commitment loss
        let commitment_loss = self.compute_commitment_loss(&flat_inputs, &quantized)?;
        
        // Straight-through estimator
        let quantized_st = &flat_inputs + &(&quantized - &flat_inputs);
        
        Ok((
            quantized_st,
            encoding_indices,
            commitment_loss,
        ))
    }

    fn compute_distances(&self, inputs: &Array2<f32>) -> Result<Array2<f32>, PandoraError> {
        let input_norm = inputs.map_axis(ndarray::Axis(1), |row| row.iter().map(|x| x * x).sum::<f32>());
        let codebook_norm = self.codebook.map_axis(ndarray::Axis(1), |row| row.iter().map(|x| x * x).sum::<f32>());
        
        let distances = input_norm.insert_axis(ndarray::Axis(1)) + &codebook_norm.insert_axis(ndarray::Axis(0));
        
        // Compute 2 * <inputs, codebook>
        let dot_product = inputs.dot(&self.codebook.t());
        let double_dot = &dot_product * 2.0;
        
        Ok(distances - double_dot)
    }

    fn quantize_codes(&self, encoding_indices: &Array1<usize>) -> Result<Array2<f32>, PandoraError> {
        let mut quantized = Array2::zeros((encoding_indices.len(), self.codebook_dim));
        
        for (i, &idx) in encoding_indices.iter().enumerate() {
            if idx < self.codebook_size {
                quantized.slice_mut(s![i, ..]).assign(&self.codebook.slice(s![idx, ..]));
            }
        }
        
        Ok(quantized)
    }

    fn compute_commitment_loss(&self, inputs: &Array2<f32>, quantized: &Array2<f32>) -> Result<f32, PandoraError> {
        let diff = inputs - quantized;
        let mse = diff.mapv(|x| x * x).mean().unwrap();
        Ok(mse * self.commitment_cost)
    }
}

// ===== Code Generator =====

#[async_trait]
pub trait CodeGenerator {
    async fn generate_skill_code(&self, intent: &Intent, context: &SkillContext) -> Result<String, PandoraError>;
    async fn optimize_code(&self, code: &str, performance_metrics: &SkillPerformanceMetrics) -> Result<String, PandoraError>;
}

pub struct LLMCodeGenerator {
    #[allow(dead_code)]
    model_path: String,
    #[allow(dead_code)]
    max_length: usize,
    #[allow(dead_code)]
    temperature: f32,
}

impl LLMCodeGenerator {
    pub fn new(model_path: String, max_length: usize, temperature: f32) -> Self {
        Self {
            model_path,
            max_length,
            temperature,
        }
    }
}

#[async_trait]
impl CodeGenerator for LLMCodeGenerator {
    async fn generate_skill_code(&self, intent: &Intent, context: &SkillContext) -> Result<String, PandoraError> {
        // Simplified code generation - trong thực tế sẽ sử dụng LLM
        let _prompt = self.create_prompt(intent, context);
        
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
            Intent::ExecutePlan { .. } => {
                "async fn execute_plan() -> Result<(), Error> {
                    // Plan execution
                    // Sequential execution here
                    Ok(())
                }".to_string()
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
        let (_quantized, _encoding_indices, _commitment_loss) = self.vector_quantizer.quantize(&intent_encoding)?;
        
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
        // Pre-compute metrics to avoid borrowing issues
        let memory_usage = self.estimate_memory_usage(intent);
        let cpu_usage = self.estimate_cpu_usage(intent);
        
        // Execute skill (simplified)
        let start_time = std::time::Instant::now();
        let result = self.simulate_skill_execution(intent).await?;
        let execution_time = start_time.elapsed().as_millis() as f32;
        
        // Update skill after execution
        if let Some(skill) = self.skills.get_mut(skill_id) {
            skill.last_used = Some(chrono::Utc::now());
            skill.usage_count += 1;
            
            // Update performance metrics
            skill.performance_metrics.execution_time_ms = execution_time;
            skill.performance_metrics.memory_usage_mb = memory_usage;
            skill.performance_metrics.cpu_usage_percent = cpu_usage;
            
            // Track performance
            self.performance_tracker.record_metrics(skill_id, &skill.performance_metrics);
        } else {
            return Err(PandoraError::PredictionFailed(format!("Skill not found: {}", skill_id)));
        }
        
        Ok(result)
    }

    fn encode_intent(&self, intent: &Intent) -> Result<Array2<f32>, PandoraError> {
        // Convert intent to token IDs (simplified)
        let tokens = self.intent_to_tokens(intent);
        self.quest_encoder.encode(&tokens)
            .map_err(|e| PandoraError::EncodingError(e.to_string()))
    }

    fn intent_to_tokens(&self, intent: &Intent) -> Vec<u32> {
        // Simplified tokenization
        match intent {
            Intent::ExecuteTask { .. } => {
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

    fn estimate_resource_requirements(&self, intent: &Intent, _context: &SkillContext) -> ResourceRequirements {
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
            Intent::ExecuteTask { .. } => {
                Ok(Intent::ExecuteTask {
                    task: "Completed task".to_string(),
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
            Intent::ExecutePlan { .. } => {
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

// Error extensions are now handled in pandora_core::error
