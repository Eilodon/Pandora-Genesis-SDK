//! Adaptive Filters for rPPG signal processing
//!
//! Provides SNR-aware bandpass filtering that adjusts aggressiveness
//! based on signal quality.

use ndarray::Array1;

/// Adaptive filter configuration
#[derive(Debug, Clone)]
pub struct AdaptiveFilterConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Base high-pass alpha (0.9-0.99)
    pub base_hp_alpha: f32,
    /// Base low-pass alpha (0.1-0.3)
    pub base_lp_alpha: f32,
    /// SNR threshold for gentle filtering (dB)
    pub high_snr_threshold: f32,
    /// SNR threshold for aggressive filtering (dB)
    pub low_snr_threshold: f32,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            base_hp_alpha: 0.95,
            base_lp_alpha: 0.2,
            high_snr_threshold: 10.0,
            low_snr_threshold: 5.0,
        }
    }
}

/// Adaptive bandpass filter
pub struct AdaptiveFilter {
    config: AdaptiveFilterConfig,
}

impl AdaptiveFilter {
    pub fn new() -> Self {
        Self::with_config(AdaptiveFilterConfig::default())
    }

    pub fn with_config(config: AdaptiveFilterConfig) -> Self {
        Self { config }
    }

    /// Apply adaptive bandpass filter based on SNR
    pub fn filter(&self, signal: &Array1<f32>, snr: f32) -> Array1<f32> {
        let (hp_alpha, lp_alpha) = self.compute_adaptive_params(snr);

        let mut filtered = signal.to_vec();

        // High-pass filter
        let mut hp_prev = 0.0;
        for i in 1..filtered.len() {
            let new_val = hp_alpha * (hp_prev + filtered[i] - filtered[i - 1]);
            hp_prev = new_val;
            filtered[i] = new_val;
        }

        // Low-pass filter
        for i in 1..filtered.len() {
            filtered[i] = lp_alpha * filtered[i] + (1.0 - lp_alpha) * filtered[i - 1];
        }

        Array1::from(filtered)
    }

    fn compute_adaptive_params(&self, snr: f32) -> (f32, f32) {
        if snr > self.config.high_snr_threshold {
            // High SNR: gentle filtering
            ((self.config.base_hp_alpha + 0.03).min(0.99), (self.config.base_lp_alpha + 0.1).min(0.4))
        } else if snr > self.config.low_snr_threshold {
            // Medium SNR
            (self.config.base_hp_alpha, self.config.base_lp_alpha)
        } else {
            // Low SNR: aggressive filtering
            ((self.config.base_hp_alpha - 0.05).max(0.85), (self.config.base_lp_alpha - 0.1).max(0.1))
        }
    }
}

impl Default for AdaptiveFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy compatible function
pub fn adaptive_bandpass_filter(signal: &Array1<f32>, snr: f32) -> Array1<f32> {
    AdaptiveFilter::new().filter(signal, snr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_adaptive_filter_high_snr() {
        let filter = AdaptiveFilter::new();
        let n = 90;

        let signal: Array1<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 30.0;
                (2.0 * PI * t).sin()
            })
            .collect();

        let filtered = filter.filter(&signal, 15.0);
        assert_eq!(filtered.len(), n);
    }

    #[test]
    fn test_adaptive_filter_low_snr() {
        let filter = AdaptiveFilter::new();
        let n = 90;

        let signal: Array1<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 30.0;
                (2.0 * PI * t).sin() + (i as f32 * 0.5).sin() * 0.5
            })
            .collect();

        let filtered = filter.filter(&signal, 2.0);
        assert_eq!(filtered.len(), n);
    }
}
