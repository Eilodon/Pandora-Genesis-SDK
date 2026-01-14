//! Timing Subsystem
//!
//! # EIDOLON FIX: Engine Decomposition (Phase 2)
//!
//! Extracts timestamp and timing-related fields from the Engine struct.
//! This subsystem handles:
//!
//! - Timestamp logging and validation
//! - Delta-time calculations
//! - Prediction timing for FEP
//!
//! # Invariants
//! - Timestamps are monotonically increasing within a session
//! - Delta-time is always positive

use crate::timestamp::TimestampLog;
use serde::{Deserialize, Serialize};

/// Timing Subsystem - unified timing state management.
///
/// # EIDOLON FIX: Engine Decomposition
/// Extracts timestamp and prediction timing from Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingSubsystem {
    /// Timestamp log for event tracking
    timestamp_log: TimestampLog,
    /// Last predicted context values (for FEP prediction error)
    last_predicted_context: Option<Vec<f32>>,
    /// Exponential moving average of prediction error
    prediction_error_ema: f32,
    /// EMA smoothing factor for prediction error
    prediction_ema_alpha: f32,
}

impl TimingSubsystem {
    /// Create new timing subsystem.
    pub fn new() -> Self {
        Self {
            timestamp_log: TimestampLog::new(),
            last_predicted_context: None,
            prediction_error_ema: 0.0,
            prediction_ema_alpha: 0.1,
        }
    }
    
    /// Create with custom prediction EMA alpha.
    pub fn with_prediction_alpha(alpha: f32) -> Self {
        Self {
            prediction_ema_alpha: alpha.clamp(0.01, 0.99),
            ..Self::new()
        }
    }
    
    /// Get timestamp log reference.
    pub fn timestamp_log(&self) -> &TimestampLog {
        &self.timestamp_log
    }
    
    /// Get mutable timestamp log reference.
    pub fn timestamp_log_mut(&mut self) -> &mut TimestampLog {
        &mut self.timestamp_log
    }
    
    /// Store predicted context for next tick comparison.
    pub fn set_predicted_context(&mut self, context: Vec<f32>) {
        self.last_predicted_context = Some(context);
    }
    
    /// Get last predicted context.
    pub fn last_predicted_context(&self) -> Option<&Vec<f32>> {
        self.last_predicted_context.as_ref()
    }
    
    /// Compute prediction error and update EMA.
    /// 
    /// # Arguments
    /// * `actual` - Actual observed context values
    /// 
    /// # Returns
    /// (instant_error, ema_error)
    pub fn compute_prediction_error(&mut self, actual: &[f32]) -> (f32, f32) {
        let instant_error = if let Some(predicted) = &self.last_predicted_context {
            if predicted.len() == actual.len() {
                predicted.iter()
                    .zip(actual.iter())
                    .map(|(p, a)| (p - a).powi(2))
                    .sum::<f32>()
                    .sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        self.prediction_error_ema = self.prediction_error_ema * (1.0 - self.prediction_ema_alpha)
            + instant_error * self.prediction_ema_alpha;
        
        (instant_error, self.prediction_error_ema)
    }
    
    /// Get current prediction error EMA.
    pub fn prediction_error_ema(&self) -> f32 {
        self.prediction_error_ema
    }
    
    /// Check if prediction error indicates high surprise (model needs update).
    pub fn is_high_surprise(&self, threshold: f32) -> bool {
        self.prediction_error_ema > threshold
    }
    
    /// Get diagnostics: (prediction_error_ema, has_prediction)
    pub fn diagnostics(&self) -> (f32, bool) {
        (self.prediction_error_ema, self.last_predicted_context.is_some())
    }
}

impl Default for TimingSubsystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timing_subsystem_creation() {
        let ts = TimingSubsystem::new();
        assert_eq!(ts.prediction_error_ema(), 0.0);
        assert!(ts.last_predicted_context().is_none());
    }
    
    #[test]
    fn test_prediction_error() {
        let mut ts = TimingSubsystem::with_prediction_alpha(0.5);
        
        // Set prediction
        ts.set_predicted_context(vec![1.0, 0.0, 0.0]);
        
        // Compute error with actual
        let (instant, ema) = ts.compute_prediction_error(&[0.0, 0.0, 0.0]);
        
        // Error should be sqrt((1-0)^2) = 1.0
        assert!((instant - 1.0).abs() < 0.01);
        // EMA should be 0.0 * 0.5 + 1.0 * 0.5 = 0.5
        assert!((ema - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_high_surprise() {
        let mut ts = TimingSubsystem::new();
        ts.prediction_error_ema = 0.5;
        
        assert!(ts.is_high_surprise(0.3));
        assert!(!ts.is_high_surprise(0.7));
    }
}
