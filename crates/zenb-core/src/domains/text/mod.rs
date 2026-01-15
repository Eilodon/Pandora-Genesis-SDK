//! Text Domain: Language processing and generative AI integration.
//!
//! This domain enables AGOLOS to process natural language through the Skandha
//! pipeline, with LLM-powered pattern recognition and intent formation.
//!
//! # Components
//!
//! - [`TextVariable`]: Language signal variables (Sentiment, Complexity, etc.)
//! - [`TextMode`]: Text understanding belief states
//! - [`TextAction`]: Text processing actions (Respond, Clarify, etc.)
//! - [`TextInput`]: Observation input type
//! - [`TextDomain`]: Complete domain implementation

mod actions;
mod modes;
mod observation;
mod variables;

pub use actions::TextAction;
pub use modes::TextMode;
pub use observation::TextInput;
pub use variables::TextVariable;

use crate::core::{Domain, GenericCausalGraph, SignalVariable};

/// Type alias for text-specific causal graph.
pub type TextCausalGraph = GenericCausalGraph<TextVariable>;

/// Text processing domain for natural language understanding.
///
/// # Causal Priors
///
/// The text domain encodes linguistic knowledge:
/// - Sentiment → EmotionalLoad correlation
/// - Complexity → Ambiguity correlation
/// - Coherence → FactualDensity inverse correlation
pub struct TextDomain;

impl Domain for TextDomain {
    type Config = TextConfig;
    type Variable = TextVariable;
    type Action = TextAction;
    type Mode = TextMode;
    type Observation = TextInput;

    fn name() -> &'static str {
        "text"
    }

    fn default_priors() -> fn(cause: usize, effect: usize) -> f32 {
        |cause, effect| {
            // Index mapping from TextVariable:
            // 0: Sentiment, 1: Complexity, 2: Coherence,
            // 3: EmotionalLoad, 4: FactualDensity, 5: Ambiguity, 6: Intent
            match (cause, effect) {
                // Sentiment influences emotional load
                (0, 3) => 0.7, // Sentiment → EmotionalLoad

                // Complexity increases ambiguity
                (1, 5) => 0.6, // Complexity → Ambiguity

                // Coherence reduces ambiguity
                (2, 5) => -0.5, // Coherence → Ambiguity (negative)

                // Factual density affects coherence
                (4, 2) => 0.4, // FactualDensity → Coherence

                // Emotional load affects intent
                (3, 6) => 0.5, // EmotionalLoad → Intent

                _ => 0.0,
            }
        }
    }

    fn mode_to_default_action(mode: Self::Mode) -> Option<Self::Action> {
        Some(match mode {
            TextMode::Analytical => TextAction::Analyze,
            TextMode::Emotional => TextAction::Empathize,
            TextMode::Persuasive => TextAction::Counter,
            TextMode::Informative => TextAction::Summarize,
            TextMode::Conversational => TextAction::Respond { 
                text: String::new(),
                confidence: 0.5,
            },
            TextMode::Procedural => TextAction::Execute,
            TextMode::Unknown => TextAction::Clarify {
                question: "Could you please elaborate?".to_string(),
            },
        })
    }
    
    fn extract_variables(obs: &Self::Observation) -> Vec<f32> {
        // Extract normalized variables from text observation
        let mut vars = vec![0.5f32; TextVariable::COUNT];
        
        vars[TextVariable::Sentiment.index()] = obs.sentiment.unwrap_or(0.5);
        vars[TextVariable::Complexity.index()] = obs.complexity.unwrap_or(0.5);
        vars[TextVariable::Coherence.index()] = obs.coherence.unwrap_or(0.5);
        vars[TextVariable::EmotionalLoad.index()] = obs.emotional_load.unwrap_or(0.5);
        vars[TextVariable::FactualDensity.index()] = obs.factual_density.unwrap_or(0.5);
        vars[TextVariable::Ambiguity.index()] = obs.ambiguity.unwrap_or(0.5);
        vars[TextVariable::Intent.index()] = obs.intent_clarity.unwrap_or(0.5);
        
        vars
    }
}

/// Text processing configuration.
#[derive(Debug, Clone, Default)]
pub struct TextConfig {
    /// Target response tokens per minute (for pacing).
    pub tokens_per_minute: f32,
    
    /// Maximum context window size.
    pub max_context_tokens: usize,
    
    /// Enable streaming output.
    pub streaming: bool,
}

impl crate::core::OscillatorConfig for TextConfig {
    fn target_frequency(&self) -> f32 {
        self.tokens_per_minute / 60.0 // Convert to per-second
    }

    fn set_target_frequency(&mut self, freq: f32) {
        self.tokens_per_minute = freq * 60.0;
    }

    fn min_frequency(&self) -> f32 {
        0.1 // Very slow (for careful reasoning)
    }

    fn max_frequency(&self) -> f32 {
        1000.0 // High-speed for batch processing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BeliefMode, SignalVariable};

    #[test]
    fn test_domain_name() {
        assert_eq!(TextDomain::name(), "text");
    }

    #[test]
    fn test_text_causal_graph() {
        let graph = TextCausalGraph::with_priors(TextDomain::default_priors());

        // Sentiment → EmotionalLoad relationship should exist
        let effect = graph.get_effect(TextVariable::Sentiment, TextVariable::EmotionalLoad);
        assert!(effect > 0.6, "Sentiment should affect EmotionalLoad");
    }

    #[test]
    fn test_mode_action_mapping() {
        let action = TextDomain::mode_to_default_action(TextMode::Analytical);
        assert!(matches!(action, Some(TextAction::Analyze)));
    }

    #[test]
    fn test_variable_count() {
        assert_eq!(TextVariable::count(), 7);
    }

    #[test]
    fn test_mode_count() {
        assert_eq!(TextMode::count(), 7);
    }
}
