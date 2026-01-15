//! Mock LLM Provider for testing.
//!
//! This provider returns deterministic responses for testing the LLM integration
//! without requiring actual API calls or model loading.

use crate::llm::{EmbeddingConfig, GenerationConfig, LlmError, LlmProvider};
use std::collections::HashMap;
use std::sync::RwLock;

/// Mock LLM provider for testing.
///
/// Supports:
/// - Canned responses for specific prompts
/// - Pattern-based response matching
/// - Configurable embedding dimensions
/// - Simulated errors for testing error handling
#[derive(Debug)]
pub struct MockProvider {
    /// Canned responses: prompt -> response
    responses: RwLock<HashMap<String, String>>,
    
    /// Default response when no match found
    default_response: String,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Simulate unavailability
    is_available: RwLock<bool>,
    
    /// Simulate errors
    simulate_error: RwLock<Option<LlmError>>,
    
    /// Call counter for testing
    call_count: RwLock<usize>,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MockProvider {
    /// Create a new mock provider with default settings.
    pub fn new() -> Self {
        Self {
            responses: RwLock::new(HashMap::new()),
            default_response: "Mock response".to_string(),
            embedding_dim: 384, // Sentence-transformer default
            is_available: RwLock::new(true),
            simulate_error: RwLock::new(None),
            call_count: RwLock::new(0),
        }
    }
    
    /// Create with a specific default response.
    pub fn with_default_response(response: impl Into<String>) -> Self {
        Self {
            default_response: response.into(),
            ..Self::new()
        }
    }
    
    /// Add a canned response for a specific prompt.
    pub fn add_response(&self, prompt: impl Into<String>, response: impl Into<String>) {
        self.responses
            .write()
            .unwrap()
            .insert(prompt.into(), response.into());
    }
    
    /// Set availability status.
    pub fn set_available(&self, available: bool) {
        *self.is_available.write().unwrap() = available;
    }
    
    /// Simulate an error on the next call.
    pub fn simulate_error(&self, error: LlmError) {
        *self.simulate_error.write().unwrap() = Some(error);
    }
    
    /// Clear simulated error.
    pub fn clear_error(&self) {
        *self.simulate_error.write().unwrap() = None;
    }
    
    /// Get the number of calls made.
    pub fn call_count(&self) -> usize {
        *self.call_count.read().unwrap()
    }
    
    /// Reset call counter.
    pub fn reset_call_count(&self) {
        *self.call_count.write().unwrap() = 0;
    }
    
    /// Create a mock provider configured for character counting tests.
    pub fn for_counting_tests() -> Self {
        let provider = Self::new();
        
        // Add responses for character counting
        provider.add_response(
            "Count the letter 'r' in 'raspberry'",
            "3"
        );
        provider.add_response(
            "Count the letter 's' in 'raspberry'",
            "1"
        );
        provider.add_response(
            "Count the letter 'a' in 'banana'",
            "3"
        );
        provider.add_response(
            "Count the letter 'e' in 'excellence'",
            "4"
        );
        provider.add_response(
            "Count the letter 'z' in 'raspberry'",
            "0"
        );
        
        provider
    }
}

impl LlmProvider for MockProvider {
    fn complete(&self, prompt: &str, _config: &GenerationConfig) -> Result<String, LlmError> {
        // Increment call counter
        *self.call_count.write().unwrap() += 1;
        
        // Check for simulated error
        if let Some(error) = self.simulate_error.read().unwrap().clone() {
            return Err(error);
        }
        
        // Check availability
        if !*self.is_available.read().unwrap() {
            return Err(LlmError::NotAvailable("Mock provider disabled".to_string()));
        }
        
        // Look for exact match
        if let Some(response) = self.responses.read().unwrap().get(prompt) {
            return Ok(response.clone());
        }
        
        // Look for partial match (prompt contains key)
        for (key, response) in self.responses.read().unwrap().iter() {
            if prompt.contains(key) {
                return Ok(response.clone());
            }
        }
        
        // Return default response
        Ok(self.default_response.clone())
    }
    
    fn embed(&self, _text: &str, _config: &EmbeddingConfig) -> Result<Vec<f32>, LlmError> {
        // Check for simulated error
        if let Some(error) = self.simulate_error.read().unwrap().clone() {
            return Err(error);
        }
        
        // Check availability
        if !*self.is_available.read().unwrap() {
            return Err(LlmError::NotAvailable("Mock provider disabled".to_string()));
        }
        
        // Return a deterministic embedding (all 0.5s for simplicity)
        Ok(vec![0.5; self.embedding_dim])
    }
    
    fn is_available(&self) -> bool {
        *self.is_available.read().unwrap()
    }
    
    fn name(&self) -> &'static str {
        "mock"
    }
    
    fn model(&self) -> &str {
        "mock-model-v1"
    }
    
    fn embedding_dim(&self) -> Option<usize> {
        Some(self.embedding_dim)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_provider_default_response() {
        let provider = MockProvider::new();
        let config = GenerationConfig::default();
        
        let result = provider.complete("Any prompt", &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Mock response");
    }
    
    #[test]
    fn test_mock_provider_canned_response() {
        let provider = MockProvider::new();
        provider.add_response("Hello", "World");
        
        let config = GenerationConfig::default();
        let result = provider.complete("Hello", &config);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "World");
    }
    
    #[test]
    fn test_mock_provider_counting() {
        let provider = MockProvider::for_counting_tests();
        let config = GenerationConfig::default();
        
        let result = provider.complete("Count the letter 'r' in 'raspberry'", &config);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "3");
    }
    
    #[test]
    fn test_mock_provider_unavailable() {
        let provider = MockProvider::new();
        provider.set_available(false);
        
        let config = GenerationConfig::default();
        let result = provider.complete("Hello", &config);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::NotAvailable(_)));
    }
    
    #[test]
    fn test_mock_provider_simulated_error() {
        let provider = MockProvider::new();
        provider.simulate_error(LlmError::Timeout { elapsed_ms: 5000 });
        
        let config = GenerationConfig::default();
        let result = provider.complete("Hello", &config);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LlmError::Timeout { .. }));
    }
    
    #[test]
    fn test_mock_provider_call_count() {
        let provider = MockProvider::new();
        let config = GenerationConfig::default();
        
        assert_eq!(provider.call_count(), 0);
        
        let _ = provider.complete("Test 1", &config);
        assert_eq!(provider.call_count(), 1);
        
        let _ = provider.complete("Test 2", &config);
        assert_eq!(provider.call_count(), 2);
        
        provider.reset_call_count();
        assert_eq!(provider.call_count(), 0);
    }
    
    #[test]
    fn test_mock_provider_embedding() {
        let provider = MockProvider::new();
        let config = EmbeddingConfig::default();
        
        let result = provider.embed("Test text", &config);
        assert!(result.is_ok());
        
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&x| (x - 0.5).abs() < 0.01));
    }
}
