//! Temporal Normalization and Temporal Difference
//!
//! Modern detrending methods for rPPG signals that are simpler and more
//! effective than spline-based approaches.
//!
//! # Methods
//!
//! - **Temporal Difference (TD)**: First derivative removes slow drift
//! - **Temporal Normalization (TN)**: Local mean/std normalization
//!
//! # Reference
//!
//! ME-rPPG: Temporal Normalization for Memory-Efficient rPPG (2025)

use ndarray::Array1;

/// Temporal Difference configuration
#[derive(Debug, Clone)]
pub struct TdConfig {
    /// Order of difference (1 = first derivative, 2 = second)
    pub order: usize,
}

impl Default for TdConfig {
    fn default() -> Self {
        Self { order: 1 }
    }
}

/// Temporal Normalization configuration
#[derive(Debug, Clone)]
pub struct TnConfig {
    /// Window size for local statistics (in samples)
    pub window_size: usize,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl Default for TnConfig {
    fn default() -> Self {
        Self {
            window_size: 15,   // ~0.5 sec at 30 fps
            epsilon: 1e-6,
        }
    }
}

/// Temporal Difference (first derivative detrending)
///
/// Computes s[i] = x[i] - x[i-1], effectively removing slow baseline drift.
///
/// # Arguments
/// * `signal` - Input signal
///
/// # Returns
/// Differentiated signal (length = input - 1)
pub fn temporal_difference(signal: &Array1<f32>) -> Array1<f32> {
    let n = signal.len();
    if n < 2 {
        return signal.clone();
    }
    
    let mut result = Array1::zeros(n - 1);
    for i in 1..n {
        result[i - 1] = signal[i] - signal[i - 1];
    }
    result
}

/// Second-order Temporal Difference
///
/// Computes second derivative for stronger detrending.
pub fn temporal_difference_order2(signal: &Array1<f32>) -> Array1<f32> {
    let first = temporal_difference(signal);
    temporal_difference(&first)
}

/// Generic Temporal Difference with configurable order
pub fn temporal_difference_n(signal: &Array1<f32>, config: &TdConfig) -> Array1<f32> {
    let mut result = signal.clone();
    for _ in 0..config.order {
        result = temporal_difference(&result);
    }
    result
}

/// Temporal Normalization (local mean/std normalization)
///
/// For each sample, computes: (x[i] - local_mean) / local_std
///
/// # Arguments
/// * `signal` - Input signal
/// * `window_size` - Size of the sliding window for local statistics
///
/// # Returns
/// Normalized signal (same length as input)
pub fn temporal_normalization(signal: &Array1<f32>, window_size: usize) -> Array1<f32> {
    temporal_normalization_with_config(signal, &TnConfig { window_size, ..Default::default() })
}

/// Temporal Normalization with full config
pub fn temporal_normalization_with_config(signal: &Array1<f32>, config: &TnConfig) -> Array1<f32> {
    let n = signal.len();
    if n < 2 {
        return signal.clone();
    }
    
    let half_win = config.window_size / 2;
    let mut result = Array1::zeros(n);
    
    for i in 0..n {
        // Determine window bounds (centered at i)
        let start = i.saturating_sub(half_win);
        let end = (i + half_win + 1).min(n);
        let win_len = end - start;
        
        if win_len < 2 {
            result[i] = 0.0;
            continue;
        }
        
        // Compute local mean
        let local_mean: f32 = signal.slice(ndarray::s![start..end]).sum() / win_len as f32;
        
        // Compute local std
        let local_var: f32 = signal
            .slice(ndarray::s![start..end])
            .iter()
            .map(|x| (x - local_mean).powi(2))
            .sum::<f32>()
            / win_len as f32;
        let local_std = local_var.sqrt().max(config.epsilon);
        
        result[i] = (signal[i] - local_mean) / local_std;
    }
    
    result
}

/// Combined TD + TN detrending (recommended for rPPG)
///
/// First applies temporal difference to remove slow drift,
/// then temporal normalization for amplitude stability.
pub fn combined_detrending(signal: &Array1<f32>, tn_window: usize) -> Array1<f32> {
    let td_signal = temporal_difference(signal);
    temporal_normalization(&td_signal, tn_window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_difference_removes_drift() {
        // Signal with linear drift + oscillation
        let n = 100;
        let signal: Array1<f32> = (0..n)
            .map(|i| {
                let t = i as f32;
                t * 0.1 + (t * 0.2).sin() * 5.0  // Linear drift + sine
            })
            .collect();
        
        let td = temporal_difference(&signal);
        
        assert_eq!(td.len(), n - 1);
        
        // Mean of TD should be close to slope (0.1)
        let mean = td.mean().unwrap();
        assert!((mean - 0.1).abs() < 0.5, "Mean {} should be close to 0.1", mean);
    }

    #[test]
    fn test_temporal_normalization_zero_mean() {
        let n = 90;
        let signal: Array1<f32> = (0..n)
            .map(|i| 100.0 + (i as f32 * 0.1).sin() * 10.0)
            .collect();
        
        let tn = temporal_normalization(&signal, 15);
        
        assert_eq!(tn.len(), n);
        
        // Middle portion should have mean ~0
        let middle: Vec<f32> = tn.slice(ndarray::s![15..75]).to_vec();
        let mean: f32 = middle.iter().sum::<f32>() / middle.len() as f32;
        assert!(mean.abs() < 0.5, "Mean {} should be close to 0", mean);
    }

    #[test]
    fn test_combined_detrending() {
        let n = 100;
        let signal: Array1<f32> = (0..n)
            .map(|i| i as f32 * 0.5 + (i as f32 * 0.15).sin() * 3.0)
            .collect();
        
        let detrended = combined_detrending(&signal, 10);
        
        // Result should be shorter by 1 due to TD
        assert_eq!(detrended.len(), n - 1);
        
        // All values should be finite
        assert!(detrended.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_td_short_signal() {
        let signal = Array1::from(vec![1.0]);
        let td = temporal_difference(&signal);
        assert_eq!(td.len(), 1); // Returns unchanged for len < 2
    }

    #[test]
    fn test_tn_window_sizes() {
        let n = 50;
        let signal: Array1<f32> = (0..n).map(|i| i as f32).collect();
        
        // Small window
        let tn_small = temporal_normalization(&signal, 5);
        assert_eq!(tn_small.len(), n);
        
        // Large window (entire signal)
        let tn_large = temporal_normalization(&signal, 100);
        assert_eq!(tn_large.len(), n);
    }
}
