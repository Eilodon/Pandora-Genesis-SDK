//! Ollama LLM Provider - Local inference with Ollama API.
//!
//! Supports DeepSeek R1 distill and other GGUF models via Ollama REST API.
//!
//! # Setup
//! ```bash
//! # Install Ollama
//! curl -fsSL https://ollama.com/install.sh | sh
//!
//! # Pull DeepSeek R1 distill (recommended)
//! ollama pull deepseek-r1:8b   # 8B for speed
//! ollama pull deepseek-r1:32b  # 32B for quality
//!
//! # Start server (if not running)
//! ollama serve
//! ```
//!
//! # Example
//! ```rust,ignore
//! use zenb_core::llm::{LlmProvider, GenerationConfig};
//! use zenb_core::llm::providers::OllamaProvider;
//!
//! let provider = OllamaProvider::new("http://localhost:11434", "deepseek-r1:8b");
//! let result = provider.complete("Explain entropy in 50 words", &GenerationConfig::default())?;
//! ```

use crate::llm::{EmbeddingConfig, GenerationConfig, LlmError, LlmProvider};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Ollama provider for local LLM inference.
#[derive(Debug)]
pub struct OllamaProvider {
    /// Base URL for Ollama API (default: http://localhost:11434)
    base_url: String,
    
    /// Model name (e.g., "deepseek-r1:8b", "llama3:8b")
    model_name: String,
    
    /// Timeout for generation requests
    timeout: Duration,
    
    /// Cached availability status
    available: AtomicBool,
    
    /// Embedding dimension (model-dependent)
    embed_dim: Option<usize>,
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new("http://localhost:11434", "deepseek-r1:8b")
    }
}

impl OllamaProvider {
    /// Create a new Ollama provider.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model_name: model.into(),
            timeout: Duration::from_secs(120),
            available: AtomicBool::new(false),
            embed_dim: Some(4096), // DeepSeek default
        }
    }
    
    /// Create provider for DeepSeek R1 distill 8B (recommended for speed).
    pub fn deepseek_r1_8b() -> Self {
        Self::new("http://localhost:11434", "deepseek-r1:8b")
    }
    
    /// Create provider for DeepSeek R1 distill 32B (recommended for quality).
    pub fn deepseek_r1_32b() -> Self {
        Self::new("http://localhost:11434", "deepseek-r1:32b")
    }
    
    /// Set custom timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Check if Ollama server is reachable.
    fn check_health(&self) -> bool {
        // Synchronous HTTP check using ureq (minimal dependency)
        // For full async, user should use ollama-rs crate
        match ureq::get(&format!("{}/api/tags", self.base_url))
            .timeout(Duration::from_secs(5))
            .call()
        {
            Ok(resp) => {
                let healthy = resp.status() == 200;
                self.available.store(healthy, Ordering::SeqCst);
                healthy
            }
            Err(_) => {
                self.available.store(false, Ordering::SeqCst);
                false
            }
        }
    }
    
    /// Make a generation request to Ollama.
    fn do_generate(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        let url = format!("{}/api/generate", self.base_url);
        
        let body = serde_json::json!({
            "model": self.model_name,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
            }
        });
        
        let response = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, _) => {
                    LlmError::Backend(format!("Ollama returned status {}", code))
                }
                ureq::Error::Transport(t) => {
                    let msg = t.to_string();
                    if msg.contains("timeout") || msg.contains("timed out") {
                        LlmError::Timeout { elapsed_ms: self.timeout.as_millis() as u64 }
                    } else {
                        LlmError::NotAvailable(format!("Transport error: {}", msg))
                    }
                }
            })?;
        
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        
        json.get("response")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::InvalidResponse("Missing 'response' field".to_string()))
    }
    
    /// Make an embedding request to Ollama.
    fn do_embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let url = format!("{}/api/embeddings", self.base_url);
        
        let body = serde_json::json!({
            "model": self.model_name,
            "prompt": text,
        });
        
        let response = ureq::post(&url)
            .timeout(self.timeout)
            .send_json(&body)
            .map_err(|e| LlmError::Backend(format!("Ollama embed error: {}", e)))?;
        
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        
        json.get("embedding")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .ok_or_else(|| LlmError::InvalidResponse("Missing 'embedding' field".to_string()))
    }
}

impl LlmProvider for OllamaProvider {
    fn complete(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        if !self.is_available() && !self.check_health() {
            return Err(LlmError::NotAvailable(
                format!("Ollama not reachable at {}", self.base_url)
            ));
        }
        
        self.do_generate(prompt, config)
    }
    
    fn embed(&self, text: &str, _config: &EmbeddingConfig) -> Result<Vec<f32>, LlmError> {
        if !self.is_available() && !self.check_health() {
            return Err(LlmError::NotAvailable(
                format!("Ollama not reachable at {}", self.base_url)
            ));
        }
        
        self.do_embed(text)
    }
    
    fn is_available(&self) -> bool {
        self.available.load(Ordering::SeqCst)
    }
    
    fn name(&self) -> &'static str {
        "ollama"
    }
    
    fn model(&self) -> &str {
        &self.model_name
    }
    
    fn embedding_dim(&self) -> Option<usize> {
        self.embed_dim
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ollama_provider_creation() {
        let provider = OllamaProvider::new("http://localhost:11434", "deepseek-r1:8b");
        assert_eq!(provider.name(), "ollama");
        assert_eq!(provider.model(), "deepseek-r1:8b");
    }
    
    #[test]
    fn test_ollama_provider_defaults() {
        let provider = OllamaProvider::default();
        assert_eq!(provider.model(), "deepseek-r1:8b");
        assert_eq!(provider.base_url, "http://localhost:11434");
    }
    
    #[test]
    fn test_ollama_provider_with_timeout() {
        let provider = OllamaProvider::default()
            .with_timeout(Duration::from_secs(60));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }
    
    #[test]
    fn test_ollama_r1_presets() {
        let p8b = OllamaProvider::deepseek_r1_8b();
        assert_eq!(p8b.model(), "deepseek-r1:8b");
        
        let p32b = OllamaProvider::deepseek_r1_32b();
        assert_eq!(p32b.model(), "deepseek-r1:32b");
    }
    
    // Integration test - requires Ollama running
    #[test]
    #[ignore] // Run with: cargo test --features llm-ollama -- --ignored
    fn test_ollama_integration() {
        let provider = OllamaProvider::deepseek_r1_8b();
        
        // Check availability
        if !provider.check_health() {
            eprintln!("Ollama not running, skipping integration test");
            return;
        }
        
        let config = GenerationConfig::short_answer();
        let result = provider.complete("What is 2+2? Answer with just the number.", &config);
        
        assert!(result.is_ok(), "Generation failed: {:?}", result);
        let response = result.unwrap();
        assert!(response.contains("4"), "Expected '4' in response: {}", response);
    }
}
