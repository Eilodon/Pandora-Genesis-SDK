//! Text belief modes for state estimation.

use crate::core::BeliefMode;
use serde::{Deserialize, Serialize};

/// Belief modes for text understanding.
///
/// These modes represent the system's belief about the nature of the
/// text being processed, which influences action selection.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum TextMode {
    /// Logical, factual, analytical content.
    Analytical,
    
    /// Feeling-laden, emotional content.
    Emotional,
    
    /// Argumentative, persuasive content.
    Persuasive,
    
    /// Educational, explanatory content.
    Informative,
    
    /// Casual dialogue, social interaction.
    Conversational,
    
    /// Step-by-step instructions, commands.
    Procedural,
    
    /// Cannot determine text type (uncertain).
    Unknown,
}

impl TextMode {
    /// Total number of modes.
    pub const COUNT: usize = 7;
    
    /// All modes as a static array.
    pub const ALL: [TextMode; 7] = [
        Self::Analytical,
        Self::Emotional,
        Self::Persuasive,
        Self::Informative,
        Self::Conversational,
        Self::Procedural,
        Self::Unknown,
    ];
}

impl BeliefMode for TextMode {
    fn count() -> usize {
        Self::COUNT
    }

    fn index(&self) -> usize {
        match self {
            Self::Analytical => 0,
            Self::Emotional => 1,
            Self::Persuasive => 2,
            Self::Informative => 3,
            Self::Conversational => 4,
            Self::Procedural => 5,
            Self::Unknown => 6,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Analytical),
            1 => Some(Self::Emotional),
            2 => Some(Self::Persuasive),
            3 => Some(Self::Informative),
            4 => Some(Self::Conversational),
            5 => Some(Self::Procedural),
            6 => Some(Self::Unknown),
            _ => None,
        }
    }

    fn default_mode() -> Self {
        Self::Unknown // Start with uncertainty
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Analytical => "Analytical",
            Self::Emotional => "Emotional",
            Self::Persuasive => "Persuasive",
            Self::Informative => "Informative",
            Self::Conversational => "Conversational",
            Self::Procedural => "Procedural",
            Self::Unknown => "Unknown",
        }
    }

    fn all() -> &'static [Self] {
        &Self::ALL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mode_index_roundtrip() {
        for mode in TextMode::all() {
            let idx = mode.index();
            let recovered = TextMode::from_index(idx);
            assert_eq!(Some(*mode), recovered);
        }
    }

    #[test]
    fn test_mode_count() {
        assert_eq!(TextMode::count(), 7);
        assert_eq!(TextMode::all().len(), 7);
    }

    #[test]
    fn test_default_mode() {
        assert_eq!(TextMode::default_mode(), TextMode::Unknown);
    }
}
