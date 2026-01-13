//! Trading actions for portfolio management.

use crate::core::ActionKind;
use serde::{Deserialize, Serialize};

/// Trading actions the system can take.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingAction {
    /// Buy an asset.
    Buy {
        /// Asset symbol/identifier.
        symbol: String,
        /// Amount to buy (normalized or units).
        amount: f32,
    },
    /// Sell an asset.
    Sell {
        /// Asset symbol/identifier.
        symbol: String,
        /// Amount to sell.
        amount: f32,
    },
    /// Hedge portfolio with derivatives or inverse positions.
    Hedge {
        /// Hedge ratio (0.0 = no hedge, 1.0 = fully hedged).
        ratio: f32,
    },
    /// Rebalance portfolio to target allocations.
    Rebalance,
    /// Do nothing, maintain current positions.
    Hold,
}

impl ActionKind for TradingAction {
    fn description(&self) -> String {
        match self {
            TradingAction::Buy { symbol, amount } => {
                format!("Buy {} units of {}", amount, symbol)
            }
            TradingAction::Sell { symbol, amount } => {
                format!("Sell {} units of {}", amount, symbol)
            }
            TradingAction::Hedge { ratio } => {
                format!("Hedge at {:.0}% ratio", ratio * 100.0)
            }
            TradingAction::Rebalance => "Rebalance portfolio".to_string(),
            TradingAction::Hold => "Hold current positions".to_string(),
        }
    }

    fn intrusiveness(&self) -> f32 {
        match self {
            TradingAction::Buy { .. } | TradingAction::Sell { .. } => 0.8,
            TradingAction::Hedge { .. } => 0.6,
            TradingAction::Rebalance => 0.7,
            TradingAction::Hold => 0.0,
        }
    }

    fn requires_permission(&self) -> bool {
        // All trading actions except Hold require confirmation
        !matches!(self, TradingAction::Hold)
    }

    fn type_id(&self) -> String {
        match self {
            TradingAction::Buy { .. } => "buy".to_string(),
            TradingAction::Sell { .. } => "sell".to_string(),
            TradingAction::Hedge { .. } => "hedge".to_string(),
            TradingAction::Rebalance => "rebalance".to_string(),
            TradingAction::Hold => "hold".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_description() {
        let action = TradingAction::Buy {
            symbol: "AAPL".to_string(),
            amount: 100.0,
        };
        assert!(action.description().contains("AAPL"));
    }

    #[test]
    fn test_hold_is_not_intrusive() {
        assert_eq!(TradingAction::Hold.intrusiveness(), 0.0);
        assert!(!TradingAction::Hold.requires_permission());
    }
}
