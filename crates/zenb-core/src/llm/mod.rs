//! LLM Provider Module - Backend-agnostic generative model integration.
//!
//! This module provides a trait-based abstraction for integrating Large Language Models
//! (LLMs) into the AGOLOS cognitive architecture. LLMs act as capability providers
//! within the Skandha pipeline, particularly for:
//!
//! - **Sanna (Pattern Recognition)**: Deep semantic pattern matching
//! - **Sankhara (Intent Formation)**: Complex reasoning for intent generation
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                  LlmProvider Trait                   │
//! │  ┌─────────┐  ┌──────────┐  ┌──────────┐           │
//! │  │ OpenAI  │  │ Llama.cpp│  │   Mock   │           │
//! │  │  API    │  │   GGUF   │  │  (Test)  │           │
//! │  └─────────┘  └──────────┘  └──────────┘           │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Feature Flags
//!
//! - `llm-openai`: Enable OpenAI API provider (requires `async-openai`)
//! - `llm-llama`: Enable llama.cpp local provider (requires `llama_cpp`)
//!
//! # Example
//!
//! ```rust,ignore
//! use zenb_core::llm::{LlmProvider, MockProvider, GenerationConfig};
//!
//! let provider = MockProvider::new();
//! let config = GenerationConfig::default();
//!
//! let result = provider.complete_sync("Hello, world!", &config)?;
//! println!("Response: {}", result);
//! ```

use std::fmt::Debug;
use std::sync::Arc;

pub mod providers;

pub use providers::mock::MockProvider;

// Re-export providers based on feature flags
#[cfg(feature = "llm-ollama")]
pub use providers::ollama::OllamaProvider;

#[cfg(feature = "llm-openai")]
pub use providers::openai::OpenAiProvider;

#[cfg(feature = "llm-llama")]
pub use providers::llama_cpp::LlamaCppProvider;

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during LLM operations.
#[derive(Debug, Clone)]
pub enum LlmError {
    /// Provider is not available (e.g., API key missing, model not loaded).
    NotAvailable(String),
    
    /// Request timed out.
    Timeout { elapsed_ms: u64 },
    
    /// Rate limited by provider.
    RateLimited { retry_after_ms: Option<u64> },
    
    /// Invalid response from provider.
    InvalidResponse(String),
    
    /// Backend-specific error.
    Backend(String),
    
    /// Generation was blocked by safety filters.
    SafetyBlocked(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable(msg) => write!(f, "LLM not available: {}", msg),
            Self::Timeout { elapsed_ms } => write!(f, "LLM timeout after {}ms", elapsed_ms),
            Self::RateLimited { retry_after_ms } => {
                if let Some(ms) = retry_after_ms {
                    write!(f, "Rate limited, retry after {}ms", ms)
                } else {
                    write!(f, "Rate limited")
                }
            }
            Self::InvalidResponse(msg) => write!(f, "Invalid LLM response: {}", msg),
            Self::Backend(msg) => write!(f, "LLM backend error: {}", msg),
            Self::SafetyBlocked(msg) => write!(f, "Blocked by safety filter: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    
    /// Temperature for sampling (0.0 = deterministic, 2.0 = very random).
    pub temperature: f32,
    
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    
    /// Stop sequences that terminate generation.
    pub stop_sequences: Vec<String>,
    
    /// Timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
    
    /// System prompt to prepend.
    pub system_prompt: Option<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: vec![],
            timeout_ms: 30_000, // 30 seconds
            system_prompt: None,
        }
    }
}

impl GenerationConfig {
    /// Create a deterministic config (temperature = 0).
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
    }
    
    /// Create a config optimized for short, focused answers.
    pub fn short_answer() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.3,
            stop_sequences: vec!["\n".to_string(), ".".to_string()],
            ..Default::default()
        }
    }
    
    /// Create a config for creative generation.
    pub fn creative() -> Self {
        Self {
            max_tokens: 512,
            temperature: 1.0,
            top_p: 0.95,
            ..Default::default()
        }
    }
}

/// Configuration for text embedding.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Model to use for embedding (provider-specific).
    pub model: Option<String>,
    
    /// Normalize output vector to unit length.
    pub normalize: bool,
    
    /// Truncate input if too long.
    pub truncate: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: None,
            normalize: true,
            truncate: true,
        }
    }
}

// ============================================================================
// LLM PROVIDER TRAIT
// ============================================================================

/// Trait for LLM inference providers.
///
/// This trait defines the interface for integrating various LLM backends
/// into AGOLOS. Implementations can be:
/// - Cloud APIs (OpenAI, Anthropic, Google)
/// - Local inference (llama.cpp, Ollama)
/// - Mock providers for testing
///
/// # Thread Safety
/// All providers must be `Send + Sync` for use in the Engine.
///
/// # Async vs Sync
/// The trait provides both async and sync interfaces. Sync methods use
/// internal blocking for compatibility with the Skandha pipeline.
pub trait LlmProvider: Send + Sync + Debug {
    /// Complete a prompt with text generation.
    ///
    /// # Arguments
    /// * `prompt` - The input prompt to complete
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Generated text on success, or an error.
    fn complete(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError>;
    
    /// Embed text into a vector representation.
    ///
    /// # Arguments
    /// * `text` - Text to embed
    /// * `config` - Embedding configuration
    ///
    /// # Returns
    /// Vector of f32 values representing the text embedding.
    fn embed(&self, text: &str, config: &EmbeddingConfig) -> Result<Vec<f32>, LlmError>;
    
    /// Check if the provider is available and ready.
    fn is_available(&self) -> bool;
    
    /// Get the provider name for diagnostics.
    fn name(&self) -> &'static str;
    
    /// Get the model name being used.
    fn model(&self) -> &str;
    
    /// Get embedding dimension (if embedding is supported).
    fn embedding_dim(&self) -> Option<usize> {
        None
    }
}

// ============================================================================
// PROVIDER REGISTRY
// ============================================================================

/// Registry for managing multiple LLM providers.
///
/// Allows switching between providers at runtime and provides fallback logic.
#[derive(Debug)]
pub struct ProviderRegistry {
    providers: Vec<Arc<dyn LlmProvider>>,
    active_index: usize,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            active_index: 0,
        }
    }
    
    /// Create a registry with a single provider.
    pub fn with_provider<P: LlmProvider + 'static>(provider: P) -> Self {
        let mut registry = Self::new();
        registry.add(provider);
        registry
    }
    
    /// Add a provider to the registry.
    pub fn add<P: LlmProvider + 'static>(&mut self, provider: P) {
        self.providers.push(Arc::new(provider));
    }
    
    /// Get the active provider.
    pub fn active(&self) -> Option<Arc<dyn LlmProvider>> {
        self.providers.get(self.active_index).cloned()
    }
    
    /// Set the active provider by index.
    pub fn set_active(&mut self, index: usize) -> bool {
        if index < self.providers.len() {
            self.active_index = index;
            true
        } else {
            false
        }
    }
    
    /// Find first available provider.
    pub fn first_available(&self) -> Option<Arc<dyn LlmProvider>> {
        self.providers.iter().find(|p| p.is_available()).cloned()
    }
    
    /// Complete with fallback to other providers on failure.
    pub fn complete_with_fallback(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<String, LlmError> {
        // Try active provider first
        if let Some(provider) = self.active() {
            if provider.is_available() {
                match provider.complete(prompt, config) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        log::warn!("Active provider {} failed: {}, trying fallbacks", provider.name(), e);
                    }
                }
            }
        }
        
        // Try fallbacks
        for (i, provider) in self.providers.iter().enumerate() {
            if i == self.active_index {
                continue; // Already tried
            }
            if provider.is_available() {
                match provider.complete(prompt, config) {
                    Ok(result) => {
                        log::info!("Fallback provider {} succeeded", provider.name());
                        return Ok(result);
                    }
                    Err(e) => {
                        log::warn!("Fallback provider {} failed: {}", provider.name(), e);
                    }
                }
            }
        }
        
        Err(LlmError::NotAvailable("All providers failed".to_string()))
    }
    
    /// List all registered providers.
    pub fn list(&self) -> Vec<&str> {
        self.providers.iter().map(|p| p.name()).collect()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < 0.01);
    }
    
    #[test]
    fn test_generation_config_deterministic() {
        let config = GenerationConfig::deterministic();
        assert!((config.temperature).abs() < 0.01);
    }
    
    #[test]
    fn test_provider_registry() {
        let mut registry = ProviderRegistry::new();
        assert!(registry.active().is_none());
        
        registry.add(MockProvider::new());
        assert!(registry.active().is_some());
        assert_eq!(registry.list(), vec!["mock"]);
    }
    
    #[test]
    fn test_mock_provider_complete() {
        let provider = MockProvider::new();
        let config = GenerationConfig::default();
        
        let result = provider.complete("Hello", &config);
        assert!(result.is_ok());
    }
}
