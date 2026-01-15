//! Text signal variables for causal modeling.

use crate::core::SignalVariable;
use serde::{Deserialize, Serialize};

/// Text signal variables for natural language processing.
///
/// These variables represent measurable aspects of text that the
/// causal graph can model relationships between.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum TextVariable {
    /// Overall sentiment score (-1 to 1 normalized to 0-1).
    Sentiment,
    
    /// Text complexity (Flesch-Kincaid or similar, normalized).
    Complexity,
    
    /// Logical coherence/flow of ideas.
    Coherence,
    
    /// Emotional intensity/load.
    EmotionalLoad,
    
    /// Information/factual density.
    FactualDensity,
    
    /// Semantic ambiguity level.
    Ambiguity,
    
    /// Clarity of intent/purpose.
    Intent,
}

impl TextVariable {
    /// Total number of text variables.
    pub const COUNT: usize = 7;
    
    /// All variables as a static array.
    pub const ALL: [TextVariable; 7] = [
        Self::Sentiment,
        Self::Complexity,
        Self::Coherence,
        Self::EmotionalLoad,
        Self::FactualDensity,
        Self::Ambiguity,
        Self::Intent,
    ];
}

impl SignalVariable for TextVariable {
    fn index(&self) -> usize {
        match self {
            Self::Sentiment => 0,
            Self::Complexity => 1,
            Self::Coherence => 2,
            Self::EmotionalLoad => 3,
            Self::FactualDensity => 4,
            Self::Ambiguity => 5,
            Self::Intent => 6,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Sentiment),
            1 => Some(Self::Complexity),
            2 => Some(Self::Coherence),
            3 => Some(Self::EmotionalLoad),
            4 => Some(Self::FactualDensity),
            5 => Some(Self::Ambiguity),
            6 => Some(Self::Intent),
            _ => None,
        }
    }

    fn count() -> usize {
        Self::COUNT
    }

    fn all() -> &'static [Self] {
        &Self::ALL
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Sentiment => "Sentiment",
            Self::Complexity => "Complexity",
            Self::Coherence => "Coherence",
            Self::EmotionalLoad => "EmotionalLoad",
            Self::FactualDensity => "FactualDensity",
            Self::Ambiguity => "Ambiguity",
            Self::Intent => "Intent",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_index_roundtrip() {
        for var in TextVariable::all() {
            let idx = var.index();
            let recovered = TextVariable::from_index(idx);
            assert_eq!(Some(*var), recovered);
        }
    }

    #[test]
    fn test_variable_count() {
        assert_eq!(TextVariable::count(), 7);
        assert_eq!(TextVariable::all().len(), 7);
    }
}
