//! Text actions for response generation.

use crate::core::ActionKind;
use serde::{Deserialize, Serialize};

/// Actions the text domain can perform.
///
/// These are the outputs of the policy selection process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TextAction {
    /// Generate a direct response.
    Respond {
        /// The generated response text.
        text: String,
        /// Confidence in the response (0-1).
        confidence: f32,
    },
    
    /// Ask a clarifying question.
    Clarify {
        /// The clarifying question to ask.
        question: String,
    },
    
    /// Summarize the input.
    Summarize,
    
    /// Perform deep analysis.
    Analyze,
    
    /// Provide an empathetic response.
    Empathize,
    
    /// Counter an argument.
    Counter,
    
    /// Execute a procedure/command.
    Execute,
    
    /// Escalate to human review.
    Escalate {
        /// Reason for escalation.
        reason: String,
    },
    
    /// Take no action (observe).
    Observe,
}

impl ActionKind for TextAction {
    fn description(&self) -> String {
        match self {
            Self::Respond { text, confidence } => {
                format!("Respond ({:.0}% confidence): {}", confidence * 100.0, 
                    if text.len() > 50 { &text[..50] } else { text })
            }
            Self::Clarify { question } => format!("Clarify: {}", question),
            Self::Summarize => "Summarize input".to_string(),
            Self::Analyze => "Perform deep analysis".to_string(),
            Self::Empathize => "Provide empathetic response".to_string(),
            Self::Counter => "Counter argument".to_string(),
            Self::Execute => "Execute procedure".to_string(),
            Self::Escalate { reason } => format!("Escalate: {}", reason),
            Self::Observe => "Observe (no action)".to_string(),
        }
    }

    fn intrusiveness(&self) -> f32 {
        match self {
            Self::Observe => 0.0,
            Self::Summarize => 0.2,
            Self::Analyze => 0.3,
            Self::Clarify { .. } => 0.4,
            Self::Respond { .. } => 0.5,
            Self::Empathize => 0.5,
            Self::Counter => 0.6,
            Self::Execute => 0.8, // High - executes commands
            Self::Escalate { .. } => 0.9, // Very high - requires human
        }
    }

    fn requires_permission(&self) -> bool {
        matches!(self, Self::Execute | Self::Escalate { .. })
    }

    fn type_id(&self) -> String {
        match self {
            Self::Respond { .. } => "respond".to_string(),
            Self::Clarify { .. } => "clarify".to_string(),
            Self::Summarize => "summarize".to_string(),
            Self::Analyze => "analyze".to_string(),
            Self::Empathize => "empathize".to_string(),
            Self::Counter => "counter".to_string(),
            Self::Execute => "execute".to_string(),
            Self::Escalate { .. } => "escalate".to_string(),
            Self::Observe => "observe".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_intrusiveness() {
        assert!(TextAction::Observe.intrusiveness() < 0.1);
        assert!(TextAction::Execute.intrusiveness() > 0.7);
    }

    #[test]
    fn test_action_permission() {
        assert!(!TextAction::Observe.requires_permission());
        assert!(TextAction::Execute.requires_permission());
    }

    #[test]
    fn test_action_description() {
        let action = TextAction::Respond {
            text: "Hello world".to_string(),
            confidence: 0.85,
        };
        let desc = action.description();
        assert!(desc.contains("85%"));
    }
}
