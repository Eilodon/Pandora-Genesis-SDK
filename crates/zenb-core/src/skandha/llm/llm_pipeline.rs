//! LLM Pipeline: Complete Skandha pipeline for text processing.

use crate::domains::text::{TextAction, TextInput, TextMode};
use crate::llm::LlmProvider;
use crate::skandha::llm::llm_rupa::{LlmRupa, TaskType, TextForm};
use crate::skandha::llm::llm_sanna::{LlmSanna, PatternType, TextPerceivedPattern};
use crate::skandha::AffectiveState;
use std::sync::Arc;

/// Configuration for the LLM pipeline.
#[derive(Debug, Clone)]
pub struct LlmPipelineConfig {
    /// Enable LLM fallback for unknown patterns.
    pub enable_llm_fallback: bool,
    
    /// Memory dimension for holographic memory.
    pub memory_dim: usize,
    
    /// Enable detailed reasoning traces.
    pub enable_reasoning: bool,
}

impl Default for LlmPipelineConfig {
    fn default() -> Self {
        Self {
            enable_llm_fallback: true,
            memory_dim: 256,
            enable_reasoning: true,
        }
    }
}

/// Complete LLM pipeline for text processing.
///
/// Integrates Rupa (tokenization) and Sanna (pattern recognition) stages
/// with optional LLM augmentation.
#[derive(Debug)]
pub struct LlmPipeline<P: LlmProvider> {
    /// Rupa stage: text tokenization.
    rupa: LlmRupa,
    
    /// Sanna stage: pattern recognition.
    sanna: LlmSanna<P>,
    
    /// Pipeline configuration.
    #[allow(dead_code)] // Reserved for future configuration access
    config: LlmPipelineConfig,
    
    /// Processing counter.
    process_count: u64,
}

impl<P: LlmProvider + 'static> LlmPipeline<P> {
    /// Create a new LLM pipeline with the given provider.
    pub fn new(provider: Arc<P>) -> Self {
        Self::with_config(provider, LlmPipelineConfig::default())
    }
    
    /// Create with custom configuration.
    pub fn with_config(provider: Arc<P>, config: LlmPipelineConfig) -> Self {
        let sanna = if config.enable_llm_fallback {
            LlmSanna::with_memory_dim(provider, config.memory_dim)
        } else {
            LlmSanna::with_memory_dim(provider, config.memory_dim).disable_llm_fallback()
        };
        
        Self {
            rupa: LlmRupa::new(),
            sanna,
            config,
            process_count: 0,
        }
    }
    
    /// Process text input through the complete pipeline.
    pub fn process(&mut self, input: &TextInput) -> LlmPipelineResult {
        self.process_count += 1;
        
        // Stage 1: Rupa (Form) - Tokenize and parse
        let form = self.rupa.process_text(input);
        
        // Stage 2: Vedana (Feeling) - Simple affect extraction
        let affect = self.extract_affect(&form);
        
        // Stage 3: Sanna (Perception) - Pattern recognition
        let pattern = self.sanna.perceive_text(&form, &affect);
        
        // Stage 4: Sankhara (Formation) - Intent formation
        let intent = self.form_intent(&pattern, &affect);
        
        // Stage 5: Vinnana (Consciousness) - Synthesis
        let action = self.synthesize(&pattern, &intent);
        
        LlmPipelineResult {
            form,
            affect,
            pattern,
            mode: intent.mode,
            action,
            process_id: self.process_count,
        }
    }
    
    /// Extract affective state from text form.
    fn extract_affect(&self, form: &TextForm) -> AffectiveState {
        // Simple heuristic affect extraction
        // Could be enhanced with sentiment analysis LLM call
        
        let valence = if form.task_type == Some(TaskType::Counting) {
            0.0 // Neutral for procedural tasks
        } else {
            0.5 // Default neutral
        };
        
        let arousal = if form.word_count > 50 {
            0.7 // Higher arousal for longer texts
        } else {
            0.3
        };
        
        AffectiveState {
            valence,
            arousal,
            confidence: form.quality,
            karma_weight: 1.0,
            is_karmic_debt: false,
        }
    }
    
    /// Form intent from pattern and affect.
    fn form_intent(&self, pattern: &TextPerceivedPattern, _affect: &AffectiveState) -> TextIntent {
        let mode = match pattern.pattern_type {
            PatternType::Counting => TextMode::Procedural,
            PatternType::QuestionAnswer => TextMode::Informative,
            PatternType::Summarization => TextMode::Analytical,
            PatternType::Translation => TextMode::Analytical,
            PatternType::Unknown => TextMode::Unknown,
        };
        
        let should_respond = pattern.computed_answer.is_some();
        
        TextIntent {
            mode,
            should_respond,
            confidence: pattern.similarity,
        }
    }
    
    /// Synthesize final action from pattern and intent.
    fn synthesize(&self, pattern: &TextPerceivedPattern, intent: &TextIntent) -> TextAction {
        if let Some(answer) = &pattern.computed_answer {
            TextAction::Respond {
                text: answer.clone(),
                confidence: intent.confidence,
            }
        } else if pattern.requires_llm {
            TextAction::Clarify {
                question: "I need more context to answer this.".to_string(),
            }
        } else {
            TextAction::Observe
        }
    }
    
    /// Get processing statistics.
    pub fn stats(&self) -> LlmPipelineStats {
        LlmPipelineStats {
            process_count: self.process_count,
        }
    }
}

/// Intent formed during Sankhara stage.
#[derive(Debug, Clone)]
pub struct TextIntent {
    /// Detected text mode.
    pub mode: TextMode,
    
    /// Whether to generate a response.
    #[allow(dead_code)] // Reserved for response generation logic
    pub should_respond: bool,
    
    /// Confidence in intent.
    pub confidence: f32,
}

/// Result of LLM pipeline processing.
#[derive(Debug)]
pub struct LlmPipelineResult {
    /// Tokenized form from Rupa stage.
    pub form: TextForm,
    
    /// Affective state from Vedana stage.
    pub affect: AffectiveState,
    
    /// Perceived pattern from Sanna stage.
    pub pattern: TextPerceivedPattern,
    
    /// Detected mode.
    pub mode: TextMode,
    
    /// Final action.
    pub action: TextAction,
    
    /// Process ID for tracking.
    pub process_id: u64,
}

impl LlmPipelineResult {
    /// Get the answer if one was computed.
    pub fn answer(&self) -> Option<&str> {
        self.pattern.computed_answer.as_deref()
    }
    
    /// Check if processing required LLM.
    pub fn used_llm(&self) -> bool {
        self.pattern.requires_llm && self.pattern.computed_answer.is_some()
    }
    
    /// Get reasoning trace.
    pub fn reasoning(&self) -> &str {
        &self.pattern.reasoning
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone)]
pub struct LlmPipelineStats {
    /// Total inputs processed.
    pub process_count: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::providers::mock::MockProvider;

    #[test]
    fn test_pipeline_counting() {
        let provider = Arc::new(MockProvider::new());
        let mut pipeline = LlmPipeline::new(provider);
        
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let result = pipeline.process(&input);
        
        assert_eq!(result.answer(), Some("3"));
        assert!(!result.used_llm()); // Procedural - no LLM needed
    }

    #[test]
    fn test_pipeline_action_respond() {
        let provider = Arc::new(MockProvider::new());
        let mut pipeline = LlmPipeline::new(provider);
        
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let result = pipeline.process(&input);
        
        match result.action {
            TextAction::Respond { text, confidence } => {
                assert_eq!(text, "3");
                assert!(confidence > 0.5);
            }
            _ => panic!("Expected Respond action"),
        }
    }

    #[test]
    fn test_pipeline_mode_procedural() {
        let provider = Arc::new(MockProvider::new());
        let mut pipeline = LlmPipeline::new(provider);
        
        let input = TextInput::new("Count the letter 'a' in 'banana'");
        let result = pipeline.process(&input);
        
        assert_eq!(result.mode, TextMode::Procedural);
    }

    #[test]
    fn test_pipeline_original_test_proposal() {
        // This tests the EXACT Level 0 test from the original proposal
        let provider = Arc::new(MockProvider::new());
        let mut pipeline = LlmPipeline::new(provider);
        
        // Level 0: Count 'r' in "raspberry"
        let result = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
        assert_eq!(result.answer(), Some("3"), "Level 0: 'r' in raspberry = 3");
        
        // Level 1: Generalization tests
        let result = pipeline.process(&TextInput::new("Count the letter 's' in 'raspberry'"));
        assert_eq!(result.answer(), Some("1"), "Level 1: 's' in raspberry = 1");
        
        let result = pipeline.process(&TextInput::new("Count the letter 'a' in 'banana'"));
        assert_eq!(result.answer(), Some("3"), "Level 1: 'a' in banana = 3");
        
        let result = pipeline.process(&TextInput::new("Count the letter 'e' in 'excellence'"));
        assert_eq!(result.answer(), Some("4"), "Level 1: 'e' in excellence = 4");
        
        let result = pipeline.process(&TextInput::new("Count the letter 'z' in 'raspberry'"));
        assert_eq!(result.answer(), Some("0"), "Level 1: 'z' in raspberry = 0 (edge case!)");
    }

    #[test]
    fn test_pipeline_has_reasoning() {
        let provider = Arc::new(MockProvider::new());
        let mut pipeline = LlmPipeline::new(provider);
        
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let result = pipeline.process(&input);
        
        // Explainability: reasoning trace available
        assert!(result.reasoning().contains("3 occurrences"));
    }
}
