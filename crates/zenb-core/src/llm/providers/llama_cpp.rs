//! llama.cpp Server Provider
//!
//! Connects to llama.cpp's built-in HTTP server for local GGUF model inference.
//! This is lighter than embedding llama.cpp directly as a library.
//!
//! # Setup
//! ```bash
//! # Download and run llama.cpp server
//! ./llama-server -m models/deepseek-r1-distill-qwen-7b-Q4_K_M.gguf --port 8080
//! ```
//!
//! # Example
//! ```rust,ignore
//! use zenb_core::llm::providers::LlamaCppProvider;
//!
//! let provider = LlamaCppProvider::new("http://localhost:8080");
//! let result = provider.complete("Explain entropy", &config)?;
//! ```

use crate::llm::{EmbeddingConfig, GenerationConfig, LlmError, LlmProvider};
use std::time::Duration;

/// llama.cpp server provider for local GGUF model inference.
#[derive(Debug)]
pub struct LlamaCppProvider {
    /// Server URL (e.g., "http://localhost:8080")
    base_url: String,
    
    /// Request timeout
    timeout: Duration,
    
    /// Cached model name from server
    model_name: String,
    
    /// Server available flag
    available: bool,
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        Self::new("http://localhost:8080")
    }
}

impl LlamaCppProvider {
    /// Create a new llama.cpp server provider.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        
        // Try to get model name from server
        let (model_name, available) = Self::probe_server(&base_url);
        
        Self {
            base_url,
            timeout: Duration::from_secs(300), // GGUF models can be slow
            model_name,
            available,
        }
    }
    
    /// Set custom timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Probe server for model info.
    fn probe_server(base_url: &str) -> (String, bool) {
        let url = format!("{}/v1/models", base_url);
        
        match ureq::get(&url)
            .timeout(Duration::from_secs(5))
            .call()
        {
            Ok(resp) => {
                if let Ok(json) = resp.into_json::<serde_json::Value>() {
                    let model = json
                        .get("data")
                        .and_then(|d| d.get(0))
                        .and_then(|m| m.get("id"))
                        .and_then(|id| id.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    log::info!("llama.cpp server connected, model: {}", model);
                    (model, true)
                } else {
                    ("unknown".to_string(), true)
                }
            }
            Err(_) => {
                log::warn!("llama.cpp server not reachable at {}", base_url);
                ("unknown".to_string(), false)
            }
        }
    }
    
    /// Make a completion request (uses llama.cpp native format).
    fn do_completion(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        // llama.cpp server supports both /completion and /v1/chat/completions
        // Use native /completion for better control
        let url = format!("{}/completion", self.base_url);
        
        let body = serde_json::json!({
            "prompt": prompt,
            "n_predict": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": false,
        });
        
        let response = ureq::post(&url)
            .timeout(self.timeout)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    LlmError::Backend(format!("llama.cpp error {}: {}", code, body))
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
        
        json.get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| LlmError::InvalidResponse("Missing 'content' field".to_string()))
    }
    
    /// Make an embedding request.
    fn do_embed(&self, text: &str) -> Result<Vec<f32>, LlmError> {
        let url = format!("{}/embedding", self.base_url);
        
        let body = serde_json::json!({
            "content": text,
        });
        
        let response = ureq::post(&url)
            .timeout(self.timeout)
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| LlmError::Backend(format!("llama.cpp embedding error: {}", e)))?;
        
        let json: serde_json::Value = response
            .into_json()
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        
        json.get("embedding")
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .ok_or_else(|| LlmError::InvalidResponse("Missing 'embedding' field".to_string()))
    }
}

impl LlmProvider for LlamaCppProvider {
    fn complete(&self, prompt: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        if !self.available {
            return Err(LlmError::NotAvailable(
                format!("llama.cpp server not reachable at {}", self.base_url)
            ));
        }
        
        self.do_completion(prompt, config)
    }
    
    fn embed(&self, text: &str, _config: &EmbeddingConfig) -> Result<Vec<f32>, LlmError> {
        if !self.available {
            return Err(LlmError::NotAvailable(
                format!("llama.cpp server not reachable at {}", self.base_url)
            ));
        }
        
        self.do_embed(text)
    }
    
    fn is_available(&self) -> bool {
        self.available
    }
    
    fn name(&self) -> &'static str {
        "llama.cpp"
    }
    
    fn model(&self) -> &str {
        &self.model_name
    }
    
    fn embedding_dim(&self) -> Option<usize> {
        // Model-dependent, typically 4096 for 7B models
        Some(4096)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_llamacpp_provider_creation() {
        let provider = LlamaCppProvider::new("http://localhost:8080");
        assert_eq!(provider.name(), "llama.cpp");
        // Server not running in test, so available should be false
        assert!(!provider.is_available());
    }
    
    #[test]
    fn test_llamacpp_with_timeout() {
        let provider = LlamaCppProvider::default()
            .with_timeout(Duration::from_secs(600));
        assert_eq!(provider.timeout, Duration::from_secs(600));
    }
    
    #[test]
    fn test_llamacpp_default() {
        let provider = LlamaCppProvider::default();
        assert_eq!(provider.base_url, "http://localhost:8080");
    }
}
