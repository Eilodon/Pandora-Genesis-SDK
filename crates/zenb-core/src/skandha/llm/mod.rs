//! LLM-Augmented Skandha implementations for text processing.
//!
//! This module provides Skandha trait implementations that leverage LLM capabilities
//! for text understanding and generation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    LLM Skandha Pipeline                      │
//! │  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
//! │  │ LlmRupa  │→ │ LlmVedana │→ │ LlmSanna  │→ │LlmSankhara│ │
//! │  │(Tokenize)│  │(Sentiment)│  │(Patterns) │  │ (Intent)  │ │
//! │  └──────────┘  └───────────┘  └───────────┘  └───────────┘ │
//! │         ↓                            ↓                      │
//! │      ┌──────────────────────────────────────────────────┐  │
//! │      │                    LlmVinnana                     │  │
//! │      │                    (Synthesis)                    │  │
//! │      └──────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//! ```

mod llm_pipeline;
mod llm_rupa;
mod llm_sanna;

pub use llm_pipeline::{LlmPipeline, LlmPipelineConfig};
pub use llm_rupa::{LlmRupa, TaskType, TextForm};
pub use llm_sanna::{LlmSanna, PatternType, TextPerceivedPattern};
