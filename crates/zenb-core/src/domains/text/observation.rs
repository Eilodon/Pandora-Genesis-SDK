//! Text observation/input type.

use crate::core::DomainObservation;

/// Raw text input for the text domain.
///
/// This is the observation type that enters the Skandha pipeline.
#[derive(Clone, Debug, Default)]
pub struct TextInput {
    /// The raw text content.
    pub text: String,
    
    /// Timestamp in microseconds.
    pub timestamp_us: i64,
    
    // === Pre-computed features (optional, for efficiency) ===
    
    /// Pre-computed sentiment score (0-1, where 0.5 = neutral).
    pub sentiment: Option<f32>,
    
    /// Pre-computed complexity score (0-1).
    pub complexity: Option<f32>,
    
    /// Pre-computed coherence score (0-1).
    pub coherence: Option<f32>,
    
    /// Pre-computed emotional load (0-1).
    pub emotional_load: Option<f32>,
    
    /// Pre-computed factual density (0-1).
    pub factual_density: Option<f32>,
    
    /// Pre-computed ambiguity (0-1).
    pub ambiguity: Option<f32>,
    
    /// Pre-computed intent clarity (0-1).
    pub intent_clarity: Option<f32>,
    
    /// Context from previous interactions.
    pub context: Option<String>,
    
    /// Quality/confidence of input (0-1).
    pub quality: f32,
}

impl TextInput {
    /// Create a new text input with just the text content.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            timestamp_us: 0,
            quality: 1.0,
            ..Default::default()
        }
    }
    
    /// Create with timestamp.
    pub fn with_timestamp(text: impl Into<String>, timestamp_us: i64) -> Self {
        Self {
            text: text.into(),
            timestamp_us,
            quality: 1.0,
            ..Default::default()
        }
    }
    
    /// Create from a counting task prompt.
    pub fn counting_task(target: char, word: &str) -> Self {
        Self::new(format!("Count the letter '{}' in '{}'", target, word))
    }
    
    /// Character count of the text.
    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }
    
    /// Word count of the text.
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
    
    /// Extract individual characters.
    pub fn chars(&self) -> Vec<char> {
        self.text.chars().collect()
    }
}

impl DomainObservation for TextInput {
    fn timestamp_us(&self) -> i64 {
        self.timestamp_us
    }
    
    fn quality(&self) -> f32 {
        self.quality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_input_new() {
        let input = TextInput::new("Hello, world!");
        assert_eq!(input.text, "Hello, world!");
        assert_eq!(input.char_count(), 13);
        assert_eq!(input.word_count(), 2);
    }

    #[test]
    fn test_counting_task() {
        let input = TextInput::counting_task('r', "raspberry");
        assert!(input.text.contains("raspberry"));
        assert!(input.text.contains("'r'"));
    }

    #[test]
    fn test_chars() {
        let input = TextInput::new("abc");
        assert_eq!(input.chars(), vec!['a', 'b', 'c']);
    }
}
