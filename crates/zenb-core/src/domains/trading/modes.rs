//! Market belief modes for the trading domain.

use crate::core::BeliefMode;
use serde::{Deserialize, Serialize};

/// Market belief modes representing market regime classifications.
///
/// These modes capture the system's belief about current market conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum MarketMode {
    /// Bull market: rising prices, optimistic sentiment
    Bullish,
    /// Bear market: falling prices, pessimistic sentiment
    Bearish,
    /// Sideways/ranging market
    #[default]
    Sideways,
    /// High volatility regime (uncertainty)
    HighVolatility,
    /// Low volatility regime (stability)
    LowVolatility,
}

impl MarketMode {
    pub const COUNT: usize = 5;

    const ALL: [MarketMode; 5] = [
        MarketMode::Bullish,
        MarketMode::Bearish,
        MarketMode::Sideways,
        MarketMode::HighVolatility,
        MarketMode::LowVolatility,
    ];

    /// Risk level associated with this market mode [0, 1].
    pub fn risk_level(&self) -> f32 {
        match self {
            MarketMode::Bullish => 0.3,
            MarketMode::Bearish => 0.7,
            MarketMode::Sideways => 0.4,
            MarketMode::HighVolatility => 0.8,
            MarketMode::LowVolatility => 0.2,
        }
    }
}

impl BeliefMode for MarketMode {
    fn count() -> usize {
        Self::COUNT
    }

    fn index(&self) -> usize {
        match self {
            MarketMode::Bullish => 0,
            MarketMode::Bearish => 1,
            MarketMode::Sideways => 2,
            MarketMode::HighVolatility => 3,
            MarketMode::LowVolatility => 4,
        }
    }

    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(MarketMode::Bullish),
            1 => Some(MarketMode::Bearish),
            2 => Some(MarketMode::Sideways),
            3 => Some(MarketMode::HighVolatility),
            4 => Some(MarketMode::LowVolatility),
            _ => None,
        }
    }

    fn default_mode() -> Self {
        MarketMode::Sideways
    }

    fn name(&self) -> &'static str {
        match self {
            MarketMode::Bullish => "Bullish",
            MarketMode::Bearish => "Bearish",
            MarketMode::Sideways => "Sideways",
            MarketMode::HighVolatility => "HighVolatility",
            MarketMode::LowVolatility => "LowVolatility",
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
    fn test_default_is_sideways() {
        assert_eq!(MarketMode::default_mode(), MarketMode::Sideways);
    }

    #[test]
    fn test_risk_levels() {
        // High volatility should have highest risk
        assert!(MarketMode::HighVolatility.risk_level() > MarketMode::Bullish.risk_level());
    }
}
