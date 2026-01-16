//! LLM Provider Implementations
//!
//! Available providers:
//! - `MockProvider` - Always available for testing
//! - `OllamaProvider` - Local inference via Ollama (feature: llm-ollama)
//! - `OpenAiProvider` - OpenAI-compatible APIs (feature: llm-openai)
//! - `LlamaCppProvider` - llama.cpp server (feature: llm-llama)

pub mod mock;

#[cfg(feature = "llm-ollama")]
pub mod ollama;

#[cfg(feature = "llm-openai")]
pub mod openai;

#[cfg(feature = "llm-llama")]
pub mod llama_cpp;

// Re-exports
pub use mock::MockProvider;

#[cfg(feature = "llm-ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "llm-openai")]
pub use openai::OpenAiProvider;

#[cfg(feature = "llm-llama")]
pub use llama_cpp::LlamaCppProvider;
