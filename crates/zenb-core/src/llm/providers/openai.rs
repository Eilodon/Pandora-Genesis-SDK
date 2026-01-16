//! OpenAI-compatible API Provider
//!
//! Supports any OpenAI-compatible API endpoint including:
//! - OpenAI official API
//! - Azure OpenAI
//! - Local servers (vLLM, text-generation-inference, llama.cpp server)
//! - DeepSeek API
//!
//! # Example
//! ```rust,ignore
//! use zenb_core::llm::providers::OpenAiProvider;
//!
//! // OpenAI official
//! let openai = OpenAiProvider::new("https://api.openai.com/v1", "sk-...", "gpt-4");
//!
//! // Local vLLM server
//! let local = OpenAiProvider::new("http://localhost:8000/v1", "", "Qwen/Qwen2.5-7B");
//!
//! // DeepSeek
//! let deepseek = OpenAiProvider::deepseek("sk-...");
//! ```

use crate::llm::{EmbeddingConfig, GenerationConfig, LlmError, LlmProvider};
use std::time::Duration;

/// OpenAI-compatible API provider.
#[derive(Debug)]
pub struct OpenAiProvider {
    /// Base URL for API (e.g., "https://api.openai.com/v1")
    base_url: String,
    
    /// API key (can be empty for local servers)
    api_key: String,
    
    /// Model name
    model_name: String,
    
    /// Request timeout
    timeout: Duration,
    
    /// Embedding model (may differ from chat model)
    embedding_model: Option<String>,
}

impl OpenAiProvider {
    /// Create a new OpenAI-compatible provider.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model_name: model.into(),
            timeout: Duration::from_secs(120),
            embedding_model: None,
        }
    }
    
    /// Create provider for OpenAI official API.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1", api_key, model)
    }
    
    /// Create provider for DeepSeek API.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::new("https://api.deepseek.com/v1", api_key, "deepseek-reasoner")
    }
    
    /// Create provider for local server (no API key required).
    pub fn local(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(base_url, "", model)
    }
    
    /// Set custom timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Set embedding model (if different from chat model).
    pub fn with_embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }
    
    /// Make a chat completion request.
    fn do_chat(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        
        let body = serde_json::json!({
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        });
        
        let mut req = ureq::post(&url).timeout(self.timeout);
        
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        
        let response = req
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    LlmError::Backend(format!("API error {}: {}", code, body))
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
        
        // Extract content from OpenAI response format
        json.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::InvalidResponse("Missing content in response".to_string()))
    }
    
    /// Make an embedding request.
    fn do_embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let url = format!("{}/embeddings", self.base_url);
        let model = self.embedding_model.as_ref().unwrap_or(&self.model_name);
        
        let body = serde_json::json!({
            "model": model,
            "input": text,
        });
        
        let mut req = ureq::post(&url).timeout(self.timeout);
        
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        
        let response = req
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| LlmError::Backend(format!("Embedding API error: {}", e)))?;
        
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        
        json.get("data")
            .and_then(|d| d.get(0))
            .and_then(|d| d.get("embedding"))
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .ok_or_else(|| LlmError::InvalidResponse("Missing embedding in response".to_string()))
    }
    
    /// Check if API is reachable.
    fn check_health(&self) -> bool {
        let url = format!("{}/models", self.base_url);
        
        let mut req = ureq::get(&url).timeout(Duration::from_secs(5));
        if !self.api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", self.api_key));
        }
        
        req.call().is_ok()
    }
}

impl LlmProvider for OpenAiProvider {
    fn complete(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        self.do_chat(prompt, config)
    }
    
    fn embed(&self, text: &str, _config: &EmbeddingConfig) -> Result<Vec<f32>, LlmError> {
        self.do_embed(text)
    }
    
    fn is_available(&self) -> bool {
        self.check_health()
    }
    
    fn name(&self) -> &'static str {
        "openai"
    }
    
    fn model(&self) -> &str {
        &self.model_name
    }
    
    fn embedding_dim(&self) -> Option<usize> {
        // Common embedding dimensions
        match self.embedding_model.as_deref().unwrap_or(&self.model_name) {
            m if m.contains("ada") => Some(1536),
            m if m.contains("3-small") => Some(1536),
            m if m.contains("3-large") => Some(3072),
            _ => None,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_openai_provider_creation() {
        let provider = OpenAiProvider::new("https://api.example.com/v1", "sk-test", "gpt-4");
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), "gpt-4");
    }
    
    #[test]
    fn test_openai_presets() {
        let openai = OpenAiProvider::openai("sk-test", "gpt-4");
        assert!(openai.base_url.contains("openai.com"));
        
        let deepseek = OpenAiProvider::deepseek("sk-test");
        assert!(deepseek.base_url.contains("deepseek.com"));
        assert_eq!(deepseek.model(), "deepseek-reasoner");
        
        let local = OpenAiProvider::local("http://localhost:8000/v1", "mistral");
        assert!(local.api_key.is_empty());
    }
    
    #[test]
    fn test_openai_with_options() {
        let provider = OpenAiProvider::openai("sk-test", "gpt-4")
            .with_timeout(Duration::from_secs(60))
            .with_embedding_model("text-embedding-ada-002");
        
        assert_eq!(provider.timeout, Duration::from_secs(60));
        assert_eq!(provider.embedding_model, Some("text-embedding-ada-002".to_string()));
        assert_eq!(provider.embedding_dim(), Some(1536));
    }
}
