//! Trading signal variables for causal modeling.

use crate::core::SignalVariable;
use serde::{Deserialize, Serialize};

/// Market signal variables for causal modeling.
///
/// These variables represent observable factors in financial markets
/// that the trading engine tracks and models relationships between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradingVariable {
    /// Asset price (normalized)
    Price,
    /// Trading volume
    Volume,
    /// Price volatility (e.g., ATR, std dev)
    Volatility,
    /// Price momentum (rate of change)
    Momentum,
    /// Market sentiment (from news, social media)
    Sentiment,
    /// Macro-economic factor (interest rates, GDP, etc.)
    MacroFactor,
    /// Portfolio-level risk exposure
    PortfolioRisk,
}

impl TradingVariable {
    pub const COUNT: usize = 7;

    const ALL: [TradingVariable; 7] = [
        TradingVariable::Price,
        TradingVariable::Volume,
        TradingVariable::Volatility,
        TradingVariable::Momentum,
        TradingVariable::Sentiment,
        TradingVariable::MacroFactor,
        TradingVariable::PortfolioRisk,
    ];
}

impl SignalVariable for TradingVariable {
    fn index(&self) -> usize {
        match self {
            TradingVariable::Price => 0,
            TradingVariable::Volume => 1,
            TradingVariable::Volatility => 2,
            TradingVariable::Momentum => 3,
            TradingVariable::Sentiment => 4,
            TradingVariable::MacroFactor => 5,
            TradingVariable::PortfolioRisk => 6,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(TradingVariable::Price),
            1 => Some(TradingVariable::Volume),
            2 => Some(TradingVariable::Volatility),
            3 => Some(TradingVariable::Momentum),
            4 => Some(TradingVariable::Sentiment),
            5 => Some(TradingVariable::MacroFactor),
            6 => Some(TradingVariable::PortfolioRisk),
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
            TradingVariable::Price => "Price",
            TradingVariable::Volume => "Volume",
            TradingVariable::Volatility => "Volatility",
            TradingVariable::Momentum => "Momentum",
            TradingVariable::Sentiment => "Sentiment",
            TradingVariable::MacroFactor => "MacroFactor",
            TradingVariable::PortfolioRisk => "PortfolioRisk",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_roundtrip() {
        for var in TradingVariable::all() {
            assert_eq!(TradingVariable::from_index(var.index()), Some(*var));
        }
    }
}
