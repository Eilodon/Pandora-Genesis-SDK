//! Signal Quality Assessment module
//!
//! Provides SNR computation and signal quality metrics for rPPG signals.

use ndarray::Array1;
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Signal quality assessment result
#[derive(Debug, Clone)]
pub struct SignalQuality {
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Whether motion artifacts were detected
    pub has_motion_artifact: bool,
    /// Overall signal validity
    pub is_valid: bool,
    /// Peak power in HR band
    pub peak_power: f32,
    /// Noise power
    pub noise_power: f32,
}

impl SignalQuality {
    /// Check if signal is good quality for HR extraction
    pub fn is_good(&self) -> bool {
        self.is_valid && self.snr > 5.0 && !self.has_motion_artifact
    }
}

/// Configuration for signal quality assessment
#[derive(Debug, Clone)]
pub struct SignalQualityConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Minimum HR frequency (Hz)
    pub min_freq: f32,
    /// Maximum HR frequency (Hz)
    pub max_freq: f32,
    /// Motion artifact detection threshold
    pub motion_threshold: f32,
    /// Minimum SNR for valid signal (dB)
    pub min_snr: f32,
}

impl Default for SignalQualityConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 0.67,
            max_freq: 3.0,
            motion_threshold: 5.0,
            min_snr: 2.0,
        }
    }
}

/// Signal Quality Analyzer
pub struct SignalQualityAnalyzer {
    config: SignalQualityConfig,
    fft_planner: FftPlanner<f32>,
}

impl SignalQualityAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self::with_config(SignalQualityConfig::default())
    }

    /// Create analyzer with custom config
    pub fn with_config(config: SignalQualityConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Analyze signal quality
    pub fn analyze(&mut self, signal: &Array1<f32>) -> SignalQuality {
        let n = signal.len();

        if n < 30 {
            return SignalQuality {
                snr: 0.0,
                has_motion_artifact: false,
                is_valid: false,
                peak_power: 0.0,
                noise_power: 0.0,
            };
        }

        let has_motion_artifact = self.detect_motion_artifact(signal);
        let (snr, peak_power, noise_power) = self.compute_snr_detailed(signal);
        let is_valid = snr > self.config.min_snr && n >= 60;

        SignalQuality {
            snr,
            has_motion_artifact,
            is_valid,
            peak_power,
            noise_power,
        }
    }

    /// Compute SNR with detailed power breakdown
    pub fn compute_snr_detailed(&mut self, signal: &Array1<f32>) -> (f32, f32, f32) {
        let n = signal.len();
        let fs = self.config.sample_rate;

        let windowed: Vec<Complex32> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
                Complex32::new(s * window, 0.0)
            })
            .collect();

        let fft = self.fft_planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        let half_n = n / 2;
        let bin_res = fs / n as f32;
        let min_bin = (self.config.min_freq / bin_res) as usize;
        let max_bin = (self.config.max_freq / bin_res).min(half_n as f32) as usize;

        let mut peak_power = 0.0f32;
        let mut total_power = 0.0f32;

        for i in min_bin..=max_bin.min(half_n - 1) {
            let power = buffer[i].norm_sqr();
            total_power += power;
            if power > peak_power {
                peak_power = power;
            }
        }

        let noise_power = total_power - peak_power;
        let snr = if noise_power > 0.0 {
            10.0 * (peak_power / noise_power).log10()
        } else {
            0.0
        };

        (snr, peak_power, noise_power)
    }

    fn detect_motion_artifact(&self, signal: &Array1<f32>) -> bool {
        let n = signal.len();
        if n < 10 {
            return false;
        }

        let mut diffs = Vec::with_capacity(n - 1);
        for i in 1..n {
            diffs.push((signal[i] - signal[i - 1]).abs());
        }

        let mean_diff: f32 = diffs.iter().sum::<f32>() / diffs.len() as f32;
        let variance: f32 = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f32>() / diffs.len() as f32;
        let std = variance.sqrt();

        let threshold = mean_diff + self.config.motion_threshold * std;
        let outliers: usize = diffs.iter().filter(|&&d| d > threshold).count();

        outliers as f32 / diffs.len() as f32 > 0.05
    }

    /// Simple SNR computation
    pub fn compute_snr(&mut self, signal: &Array1<f32>) -> f32 {
        self.compute_snr_detailed(signal).0
    }
}

impl Default for SignalQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_signal_quality() {
        let mut analyzer = SignalQualityAnalyzer::new();
        let fs = 30.0;
        let n = 90;

        let signal: Array1<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                (2.0 * PI * t).sin()
            })
            .collect();

        let quality = analyzer.analyze(&signal);
        assert!(quality.is_valid, "Clean signal should be valid");
        assert!(quality.snr > 0.0, "SNR should be positive");
    }

    #[test]
    fn test_insufficient_data() {
        let mut analyzer = SignalQualityAnalyzer::new();
        let signal = Array1::zeros(10);
        let quality = analyzer.analyze(&signal);
        assert!(!quality.is_valid, "Short signal should be invalid");
    }
}
