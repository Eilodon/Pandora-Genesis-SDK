//! Observation types for the trading domain.

use crate::core::DomainObservation;
use serde::{Deserialize, Serialize};

/// Market tick observation for trading.
///
/// Represents a single market data update containing price,
/// volume, and other relevant trading metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    /// Current price (normalized or absolute).
    pub price: f32,
    
    /// Trading volume for this tick.
    pub volume: f32,
    
    /// Price volatility (e.g., ATR, realized vol).
    pub volatility: Option<f32>,
    
    /// Price momentum (rate of change).
    pub momentum: Option<f32>,
    
    /// Market sentiment score [-1, 1].
    pub sentiment: Option<f32>,
    
    /// Macro factor impact (interest rate, etc.).
    pub macro_factor: Option<f32>,
    
    /// Timestamp in microseconds.
    pub timestamp_us: i64,
    
    /// Data quality [0, 1].
    pub quality: f32,
}

impl Default for MarketTick {
    fn default() -> Self {
        Self {
            price: 0.0,
            volume: 0.0,
            volatility: None,
            momentum: None,
            sentiment: None,
            macro_factor: None,
            timestamp_us: 0,
            quality: 1.0,
        }
    }
}

impl MarketTick {
    /// Create a new market tick with price and volume.
    pub fn new(price: f32, volume: f32, timestamp_us: i64) -> Self {
        Self {
            price,
            volume,
            volatility: None,
            momentum: None,
            sentiment: None,
            macro_factor: None,
            timestamp_us,
            quality: 1.0,
        }
    }
    
    /// Builder pattern for adding volatility.
    pub fn with_volatility(mut self, vol: f32) -> Self {
        self.volatility = Some(vol);
        self
    }
    
    /// Builder pattern for adding momentum.
    pub fn with_momentum(mut self, mom: f32) -> Self {
        self.momentum = Some(mom);
        self
    }
    
    /// Builder pattern for adding sentiment.
    pub fn with_sentiment(mut self, sent: f32) -> Self {
        self.sentiment = Some(sent);
        self
    }
}

impl DomainObservation for MarketTick {
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
    fn test_market_tick_builder() {
        let tick = MarketTick::new(100.0, 1000.0, 5000)
            .with_volatility(0.2)
            .with_momentum(0.05)
            .with_sentiment(0.3);
        
        assert_eq!(tick.price, 100.0);
        assert_eq!(tick.volume, 1000.0);
        assert_eq!(tick.volatility, Some(0.2));
        assert_eq!(tick.momentum, Some(0.05));
        assert_eq!(tick.sentiment, Some(0.3));
    }
    
    #[test]
    fn test_domain_observation_trait() {
        let tick = MarketTick::new(50.0, 500.0, 10000);
        assert_eq!(tick.timestamp_us(), 10000);
        assert_eq!(tick.quality(), 1.0);
    }
}
