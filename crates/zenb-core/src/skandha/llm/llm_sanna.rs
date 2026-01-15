//! LlmSanna: LLM-powered pattern recognition for the Sanna (Perception) stage.

use crate::llm::{GenerationConfig, LlmError, LlmProvider};
use crate::memory::HolographicMemory;
use crate::skandha::llm::llm_rupa::{TaskType, TextForm};
use crate::skandha::{AffectiveState, PerceivedPattern, ProcessedForm, SannaSkandha};
use std::sync::Arc;

/// LLM-powered Sanna for deep pattern recognition.
///
/// This stage:
/// 1. Uses procedural knowledge for known task types (counting, etc.)
/// 2. Falls back to LLM for complex pattern matching
/// 3. Maintains associative memory for pattern recall
#[derive(Debug)]
pub struct LlmSanna<P: LlmProvider> {
    /// LLM provider for complex reasoning.
    provider: Arc<P>,
    
    /// Holographic memory for pattern storage.
    memory: HolographicMemory,
    
    /// Enable LLM fallback for unknown patterns.
    enable_llm_fallback: bool,
    
    /// Pattern count tracker.
    pattern_count: u64,
}

impl<P: LlmProvider> LlmSanna<P> {
    /// Create a new LlmSanna with the given provider.
    pub fn new(provider: Arc<P>) -> Self {
        Self {
            provider,
            memory: HolographicMemory::new(256),
            enable_llm_fallback: true,
            pattern_count: 0,
        }
    }
    
    /// Create with custom memory dimension.
    pub fn with_memory_dim(provider: Arc<P>, dim: usize) -> Self {
        Self {
            provider,
            memory: HolographicMemory::new(dim),
            enable_llm_fallback: true,
            pattern_count: 0,
        }
    }
    
    /// Disable LLM fallback (use only procedural knowledge).
    pub fn disable_llm_fallback(mut self) -> Self {
        self.enable_llm_fallback = false;
        self
    }
    
    /// Process a TextForm to recognize patterns and derive answers.
    pub fn perceive_text(&mut self, form: &TextForm, _affect: &AffectiveState) -> TextPerceivedPattern {
        self.pattern_count += 1;
        
        match form.task_type {
            Some(TaskType::Counting) => self.handle_counting_task(form),
            Some(TaskType::QuestionAnswering) => self.handle_qa_task(form),
            Some(TaskType::Summarization) => self.handle_summarization_task(form),
            _ => self.handle_unknown_task(form),
        }
    }
    
    /// Handle counting tasks using procedural knowledge.
    fn handle_counting_task(&self, form: &TextForm) -> TextPerceivedPattern {
        if let (Some(target), Some(sequence)) = (form.target, form.sequence.as_ref()) {
            // PROCEDURAL COUNTING - no LLM needed!
            let count = sequence
                .chars()
                .filter(|&c| c.to_ascii_lowercase() == target.to_ascii_lowercase())
                .count();
            
            TextPerceivedPattern {
                pattern_id: self.pattern_count,
                pattern_type: PatternType::Counting,
                similarity: 1.0, // Exact match - we know this pattern
                computed_answer: Some(count.to_string()),
                reasoning: format!(
                    "Counted '{}' in '{}': found {} occurrences",
                    target, sequence, count
                ),
                requires_llm: false,
                is_trauma_associated: false,
            }
        } else {
            // Couldn't parse counting parameters - may need LLM
            TextPerceivedPattern {
                pattern_id: self.pattern_count,
                pattern_type: PatternType::Unknown,
                similarity: 0.3,
                computed_answer: None,
                reasoning: "Could not parse counting parameters".to_string(),
                requires_llm: self.enable_llm_fallback,
                is_trauma_associated: false,
            }
        }
    }
    
    /// Handle Q&A tasks (may use LLM).
    fn handle_qa_task(&mut self, form: &TextForm) -> TextPerceivedPattern {
        if self.enable_llm_fallback && self.provider.is_available() {
            match self.query_llm(form) {
                Ok(answer) => TextPerceivedPattern {
                    pattern_id: self.pattern_count,
                    pattern_type: PatternType::QuestionAnswer,
                    similarity: 0.8,
                    computed_answer: Some(answer.clone()),
                    reasoning: format!("LLM answered: {}", answer),
                    requires_llm: true,
                    is_trauma_associated: false,
                },
                Err(e) => TextPerceivedPattern {
                    pattern_id: self.pattern_count,
                    pattern_type: PatternType::Unknown,
                    similarity: 0.2,
                    computed_answer: None,
                    reasoning: format!("LLM error: {}", e),
                    requires_llm: false,
                    is_trauma_associated: false,
                },
            }
        } else {
            TextPerceivedPattern {
                pattern_id: self.pattern_count,
                pattern_type: PatternType::QuestionAnswer,
                similarity: 0.5,
                computed_answer: None,
                reasoning: "Q&A detected but LLM not available".to_string(),
                requires_llm: true,
                is_trauma_associated: false,
            }
        }
    }
    
    /// Handle summarization tasks.
    fn handle_summarization_task(&self, _form: &TextForm) -> TextPerceivedPattern {
        TextPerceivedPattern {
            pattern_id: self.pattern_count,
            pattern_type: PatternType::Summarization,
            similarity: 0.7,
            computed_answer: None,
            reasoning: "Summarization task detected".to_string(),
            requires_llm: true,
            is_trauma_associated: false,
        }
    }
    
    /// Handle unknown tasks.
    fn handle_unknown_task(&self, _form: &TextForm) -> TextPerceivedPattern {
        TextPerceivedPattern {
            pattern_id: self.pattern_count,
            pattern_type: PatternType::Unknown,
            similarity: 0.1,
            computed_answer: None,
            reasoning: "Unknown task type".to_string(),
            requires_llm: self.enable_llm_fallback,
            is_trauma_associated: false,
        }
    }
    
    /// Query the LLM for an answer.
    fn query_llm(&self, form: &TextForm) -> Result<String, LlmError> {
        let config = GenerationConfig::short_answer();
        self.provider.complete(&form.text, &config)
    }
}

// Implement SannaSkandha for compatibility
impl<P: LlmProvider + 'static> SannaSkandha for LlmSanna<P> {
    fn perceive(&mut self, form: &ProcessedForm, _affect: &AffectiveState) -> PerceivedPattern {
        // Compatibility shim - convert ProcessedForm to pattern
        PerceivedPattern {
            pattern_id: self.pattern_count,
            similarity: form.energy,
            context: form.values,
            is_trauma_associated: form.anomaly_score > 0.8,
        }
    }
}

/// Extended perceived pattern for text processing.
#[derive(Debug, Clone)]
pub struct TextPerceivedPattern {
    /// Unique pattern ID.
    pub pattern_id: u64,
    
    /// Type of pattern recognized.
    pub pattern_type: PatternType,
    
    /// Similarity/confidence score (0-1).
    pub similarity: f32,
    
    /// Computed answer (if task is solvable).
    pub computed_answer: Option<String>,
    
    /// Reasoning trace for explainability.
    pub reasoning: String,
    
    /// Whether LLM was/would be needed.
    pub requires_llm: bool,
    
    /// Trauma association flag.
    pub is_trauma_associated: bool,
}

/// Types of patterns recognized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    /// Counting pattern (solved procedurally).
    Counting,
    
    /// Question-answer pattern.
    QuestionAnswer,
    
    /// Summarization pattern.
    Summarization,
    
    /// Translation pattern.
    Translation,
    
    /// Unknown pattern.
    Unknown,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::text::TextInput;
    use crate::llm::providers::mock::MockProvider;
    use crate::skandha::llm::llm_rupa::LlmRupa;

    #[test]
    fn test_counting_pattern_recognition() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.pattern_type, PatternType::Counting);
        assert_eq!(pattern.computed_answer, Some("3".to_string()));
        assert!(!pattern.requires_llm); // Solved procedurally!
    }

    #[test]
    fn test_counting_s_in_raspberry() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 's' in 'raspberry'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.computed_answer, Some("1".to_string()));
    }

    #[test]
    fn test_counting_a_in_banana() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 'a' in 'banana'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.computed_answer, Some("3".to_string()));
    }

    #[test]
    fn test_counting_e_in_excellence() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 'e' in 'excellence'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.computed_answer, Some("4".to_string()));
    }

    #[test]
    fn test_counting_z_in_raspberry_edge_case() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 'z' in 'raspberry'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.computed_answer, Some("0".to_string())); // Edge case: no z's
    }

    #[test]
    fn test_qa_uses_llm() {
        let provider = Arc::new(MockProvider::for_counting_tests());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("What is the capital of France?");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        assert_eq!(pattern.pattern_type, PatternType::QuestionAnswer);
        // Q&A typically requires LLM
    }

    #[test]
    fn test_pattern_has_reasoning() {
        let provider = Arc::new(MockProvider::new());
        let mut sanna = LlmSanna::new(provider);
        let rupa = LlmRupa::new();
        
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let form = rupa.process_text(&input);
        let affect = AffectiveState::default();
        
        let pattern = sanna.perceive_text(&form, &affect);
        
        // Should have reasoning for explainability
        assert!(!pattern.reasoning.is_empty());
        assert!(pattern.reasoning.contains("3 occurrences"));
    }
}
