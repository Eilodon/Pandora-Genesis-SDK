//! Trading Domain: Market analysis and algorithmic trading.
//!
//! This domain demonstrates AGOLOS applied to financial trading,
//! with causal modeling of market variables and action selection
//! for portfolio management.
//!
//! # Components
//!
//! - [`TradingVariable`]: Market signal variables (Price, Volume, Volatility, etc.)
//! - [`MarketMode`]: Market belief states (Bullish, Bearish, Volatile, etc.)
//! - [`TradingAction`]: Trading actions (Buy, Sell, Hedge, Hold)
//! - [`TradingDomain`]: Complete domain implementation

mod actions;
mod modes;
mod variables;

pub use actions::TradingAction;
pub use modes::MarketMode;
pub use variables::TradingVariable;

use crate::core::{Domain, GenericCausalGraph};

/// Type alias for trading-specific causal graph.
pub type TradingCausalGraph = GenericCausalGraph<TradingVariable>;

/// Trading domain for market analysis and algorithmic trading.
///
/// # Causal Priors
///
/// The trading domain encodes market knowledge:
/// - Volume → Price movement correlation
/// - Sentiment → Price dynamics
/// - Volatility → Risk indicators
/// - MacroFactor → Sector-wide effects
pub struct TradingDomain;

impl Domain for TradingDomain {
    type Config = TradingConfig;
    type Variable = TradingVariable;
    type Action = TradingAction;
    type Mode = MarketMode;

    fn name() -> &'static str {
        "trading"
    }

    fn default_priors() -> fn(cause: usize, effect: usize) -> f32 {
        |cause, effect| {
            // Index mapping from TradingVariable:
            // 0: Price, 1: Volume, 2: Volatility, 3: Momentum,
            // 4: Sentiment, 5: MacroFactor, 6: PortfolioRisk
            match (cause, effect) {
                // Volume often leads price movements
                (1, 0) => 0.5, // Volume → Price

                // Momentum is self-reinforcing (trend)
                (3, 0) => 0.4, // Momentum → Price

                // Sentiment affects price
                (4, 0) => 0.35, // Sentiment → Price

                // Macro factors affect everything
                (5, 0) => 0.3, // MacroFactor → Price
                (5, 2) => 0.4, // MacroFactor → Volatility

                // Volatility affects risk
                (2, 6) => 0.6, // Volatility → PortfolioRisk

                _ => 0.0,
            }
        }
    }

    fn mode_to_default_action(mode: Self::Mode) -> Option<Self::Action> {
        Some(match mode {
            MarketMode::Bullish => TradingAction::Hold, // Stay invested
            MarketMode::Bearish => TradingAction::Hedge { ratio: 0.3 },
            MarketMode::Sideways => TradingAction::Hold,
            MarketMode::HighVolatility => TradingAction::Hedge { ratio: 0.5 },
            MarketMode::LowVolatility => TradingAction::Hold,
        })
    }
}

/// Trading configuration for rebalancing frequency.
#[derive(Debug, Clone, Default)]
pub struct TradingConfig {
    /// Rebalancing frequency in times per day.
    pub rebalance_frequency: f32,
}

impl crate::core::OscillatorConfig for TradingConfig {
    fn target_frequency(&self) -> f32 {
        self.rebalance_frequency
    }

    fn set_target_frequency(&mut self, freq: f32) {
        self.rebalance_frequency = freq;
    }

    fn min_frequency(&self) -> f32 {
        0.1 // Once per 10 days
    }

    fn max_frequency(&self) -> f32 {
        100.0 // High-frequency (not recommended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BeliefMode, SignalVariable};

    #[test]
    fn test_domain_name() {
        assert_eq!(TradingDomain::name(), "trading");
    }

    #[test]
    fn test_trading_causal_graph() {
        let graph = TradingCausalGraph::with_priors(TradingDomain::default_priors());

        // Volume → Price relationship should exist
        let effect = graph.get_effect(TradingVariable::Volume, TradingVariable::Price);
        assert!(effect > 0.4, "Volume should affect Price");
    }

    #[test]
    fn test_mode_action_mapping() {
        let action = TradingDomain::mode_to_default_action(MarketMode::Bearish);
        assert!(matches!(action, Some(TradingAction::Hedge { .. })));
    }

    #[test]
    fn test_variable_count() {
        assert_eq!(TradingVariable::count(), 7);
    }

    #[test]
    fn test_mode_count() {
        assert_eq!(MarketMode::count(), 5);
    }
}
