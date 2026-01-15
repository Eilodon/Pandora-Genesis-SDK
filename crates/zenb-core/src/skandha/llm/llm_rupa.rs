//! LlmRupa: Text tokenization and parsing for the Rupa (Form) stage.

use crate::domains::text::TextInput;
use crate::skandha::{ProcessedForm, RupaSkandha, SensorInput};
use std::collections::HashMap;

/// Tokenized text form output from LlmRupa.
#[derive(Debug, Clone, Default)]
pub struct TextForm {
    /// Original text.
    pub text: String,
    
    /// Character array.
    pub chars: Vec<char>,
    
    /// Word tokens.
    pub words: Vec<String>,
    
    /// Character count.
    pub char_count: usize,
    
    /// Word count.
    pub word_count: usize,
    
    /// Character frequency map.
    pub char_frequency: HashMap<char, usize>,
    
    /// Detected task type (if any).
    pub task_type: Option<TaskType>,
    
    /// Extracted target (for counting tasks).
    pub target: Option<char>,
    
    /// Extracted sequence (for counting tasks).
    pub sequence: Option<String>,
    
    /// Signal quality (based on input clarity).
    pub quality: f32,
    
    /// Anomaly score (unusual patterns).
    pub anomaly_score: f32,
}

/// Types of tasks detected from input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    /// Count occurrences (character, word, etc.)
    Counting,
    
    /// Question answering.
    QuestionAnswering,
    
    /// Summarization.
    Summarization,
    
    /// Translation.
    Translation,
    
    /// Code generation.
    CodeGeneration,
    
    /// General conversation.
    Conversation,
    
    /// Unknown task type.
    Unknown,
}

/// LLM-aware Rupa for text processing.
///
/// This stage:
/// 1. Tokenizes text into characters and words
/// 2. Counts character frequencies
/// 3. Detects task type from input patterns
/// 4. Extracts relevant targets for specific tasks (e.g., letter to count)
#[derive(Debug, Clone)]
pub struct LlmRupa {
    /// Patterns for task detection.
    task_patterns: Vec<(TaskType, Vec<&'static str>)>,
}

impl Default for LlmRupa {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmRupa {
    /// Create a new LlmRupa processor.
    pub fn new() -> Self {
        let task_patterns = vec![
            (TaskType::Counting, vec![
                "count", "how many", "occurrences", "frequency",
            ]),
            (TaskType::QuestionAnswering, vec![
                "what", "who", "when", "where", "why", "how",
            ]),
            (TaskType::Summarization, vec![
                "summarize", "summary", "brief", "tldr", "shorten",
            ]),
            (TaskType::Translation, vec![
                "translate", "translation", "convert to",
            ]),
            (TaskType::CodeGeneration, vec![
                "code", "function", "implement", "program", "script",
            ]),
            (TaskType::Conversation, vec![
                "hello", "hi", "hey", "thanks", "thank you",
            ]),
        ];
        
        Self { task_patterns }
    }
    
    /// Process raw text input into structured form.
    pub fn process_text(&self, input: &TextInput) -> TextForm {
        let text = &input.text;
        let chars: Vec<char> = text.chars().collect();
        let words: Vec<String> = text.split_whitespace().map(String::from).collect();
        
        // Build character frequency map
        let mut char_frequency: HashMap<char, usize> = HashMap::new();
        for &c in &chars {
            *char_frequency.entry(c.to_ascii_lowercase()).or_insert(0) += 1;
        }
        
        // Detect task type
        let task_type = self.detect_task_type(text);
        
        // Extract target and sequence for counting tasks
        let (target, sequence) = if task_type == Some(TaskType::Counting) {
            self.extract_counting_params(text)
        } else {
            (None, None)
        };
        
        // Calculate quality based on input clarity
        let quality = self.calculate_quality(text);
        
        // Calculate anomaly score (unusual character patterns)
        let anomaly_score = self.calculate_anomaly(text, &char_frequency);
        
        TextForm {
            text: text.clone(),
            chars,
            words,
            char_count: text.chars().count(),
            word_count: text.split_whitespace().count(),
            char_frequency,
            task_type,
            target,
            sequence,
            quality,
            anomaly_score,
        }
    }
    
    /// Detect task type from text patterns.
    fn detect_task_type(&self, text: &str) -> Option<TaskType> {
        let lower = text.to_lowercase();
        
        for (task_type, patterns) in &self.task_patterns {
            for pattern in patterns {
                if lower.contains(pattern) {
                    return Some(*task_type);
                }
            }
        }
        
        None
    }
    
    /// Extract counting parameters from a counting task.
    ///
    /// Recognizes patterns like:
    /// - "Count the letter 'r' in 'raspberry'"
    /// - "How many 'e's in 'excellence'"
    fn extract_counting_params(&self, text: &str) -> (Option<char>, Option<String>) {
        // Pattern: 'X' in 'WORD' or "X" in "WORD"
        let parts: Vec<&str> = text.split(|c| c == '\'' || c == '"').collect();
        
        if parts.len() >= 4 {
            // Expecting: ["Count the letter ", "r", " in ", "raspberry", ""]
            let target_str = parts[1];
            let sequence = parts[3].to_string();
            
            if target_str.len() == 1 {
                return (target_str.chars().next(), Some(sequence));
            }
        }
        
        // Fallback: try to find single character mentions
        for part in &parts {
            if part.len() == 1 {
                let c = part.chars().next().unwrap();
                if c.is_alphabetic() {
                    // Found a single letter, now look for the word
                    for potential_word in &parts {
                        if potential_word.len() > 1 && potential_word.chars().all(|c| c.is_alphabetic()) {
                            return (Some(c), Some(potential_word.to_string()));
                        }
                    }
                }
            }
        }
        
        (None, None)
    }
    
    /// Calculate input quality score.
    fn calculate_quality(&self, text: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        
        // Quality factors:
        // - Not too short
        // - Has alphanumeric content
        // - Reasonable punctuation
        
        let alpha_ratio = text.chars().filter(|c| c.is_alphanumeric()).count() as f32 
            / text.len() as f32;
        
        let has_content = text.len() >= 3;
        let reasonable_length = text.len() <= 10000;
        
        let mut quality = alpha_ratio;
        if !has_content {
            quality *= 0.5;
        }
        if !reasonable_length {
            quality *= 0.8;
        }
        
        quality.clamp(0.0, 1.0)
    }
    
    /// Calculate anomaly score.
    fn calculate_anomaly(&self, text: &str, _freq: &HashMap<char, usize>) -> f32 {
        // Simple anomaly detection:
        // - Very short text
        // - Only special characters
        // - Excessive repetition
        
        if text.is_empty() {
            return 1.0;
        }
        
        let special_ratio = text.chars().filter(|c| !c.is_alphanumeric() && !c.is_whitespace()).count() as f32
            / text.len() as f32;
        
        if special_ratio > 0.5 {
            return 0.8;
        }
        
        if text.len() < 3 {
            return 0.5;
        }
        
        0.0
    }
}

// Implement RupaSkandha for compatibility with existing pipeline
impl RupaSkandha for LlmRupa {
    fn process_form(&mut self, input: &SensorInput) -> ProcessedForm {
        // Convert SensorInput to ProcessedForm
        // This is a compatibility shim for the existing Skandha interface
        let values = [
            input.hr_bpm.unwrap_or(0.0) / 200.0,
            input.hrv_rmssd.unwrap_or(0.0) / 100.0,
            input.rr_bpm.unwrap_or(0.0) / 20.0,
            input.quality,
            input.motion,
        ];
        
        ProcessedForm {
            values,
            anomaly_score: if input.quality < 0.3 { 1.0 } else { 0.0 },
            energy: values.iter().sum::<f32>() / 5.0,
            is_reliable: input.quality > 0.3,
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
    fn test_task_detection_counting() {
        let rupa = LlmRupa::new();
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let form = rupa.process_text(&input);
        
        assert_eq!(form.task_type, Some(TaskType::Counting));
    }

    #[test]
    fn test_counting_param_extraction() {
        let rupa = LlmRupa::new();
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let form = rupa.process_text(&input);
        
        assert_eq!(form.target, Some('r'));
        assert_eq!(form.sequence, Some("raspberry".to_string()));
    }

    #[test]
    fn test_char_frequency() {
        let rupa = LlmRupa::new();
        let input = TextInput::new("aabbc");
        let form = rupa.process_text(&input);
        
        assert_eq!(form.char_frequency.get(&'a'), Some(&2));
        assert_eq!(form.char_frequency.get(&'b'), Some(&2));
        assert_eq!(form.char_frequency.get(&'c'), Some(&1));
    }

    #[test]
    fn test_quality_calculation() {
        let rupa = LlmRupa::new();
        
        let good = rupa.calculate_quality("Hello world");
        let bad = rupa.calculate_quality("!!!");
        
        assert!(good > bad);
    }
    
    #[test]
    fn test_count_r_in_raspberry() {
        let rupa = LlmRupa::new();
        let input = TextInput::new("Count the letter 'r' in 'raspberry'");
        let form = rupa.process_text(&input);
        
        // Extract and count manually
        let target = form.target.unwrap();
        let sequence = form.sequence.as_ref().unwrap();
        let count = sequence.chars().filter(|&c| c.to_ascii_lowercase() == target.to_ascii_lowercase()).count();
        
        assert_eq!(count, 3); // r-a-s-p-b-e-r-r-y has 3 'r's
    }
}
