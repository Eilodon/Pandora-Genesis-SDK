//! PRISM Adaptive rPPG Algorithm
//!
//! State-of-the-art unsupervised rPPG extraction with adaptive parameter optimization.
//! Achieves MAE 0.66-0.77 bpm on UBFC-rPPG (comparable to supervised deep learning).
//!
//! # Algorithm
//!
//! PRISM combines adaptive color mixing with real-time signal quality monitoring:
//! 1. Grid search optimal α: Signal = α·G + (1-α)·(R-B)
//! 2. Adaptive temporal filtering based on SNR
//! 3. FFT-based heart rate extraction
//!
//! # Reference
//!
//! PRISM (2025): Adaptive Parameter Optimization for rPPG

use ndarray::Array1;
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

use super::apon::{AponNoiseEstimator, AponResult};
use crate::dsp::temporal_normalization;

/// PRISM processing result
#[derive(Debug, Clone)]
pub struct PrismResult {
    /// Heart rate in BPM
    pub bpm: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Optimal alpha used for color mixing
    pub optimal_alpha: f32,
}

/// PRISM Algorithm configuration
#[derive(Debug, Clone)]
pub struct PrismConfig {
    /// Sample rate in Hz
    pub sample_rate: f32,
    /// Number of alpha candidates for grid search
    pub alpha_steps: usize,
    /// Minimum HR frequency (Hz) - typically 40 BPM = 0.67 Hz
    pub min_freq: f32,
    /// Maximum HR frequency (Hz) - typically 180 BPM = 3.0 Hz
    pub max_freq: f32,
    /// Whether to use APON warm-start for grid search
    pub use_apon: bool,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            alpha_steps: 11, // α ∈ {0, 0.1, 0.2, ..., 1.0}
            min_freq: 0.67,  // 40 BPM
            max_freq: 3.0,   // 180 BPM
            use_apon: true,  // Enable APON by default
        }
    }
}

/// PRISM Processor for rPPG extraction
pub struct PrismProcessor {
    config: PrismConfig,
    /// Rolling history of optimal alphas for smoothing
    alpha_history: Vec<f32>,
    /// FFT planner (reused for efficiency)
    fft_planner: FftPlanner<f32>,
    /// APON noise estimator (for warm-start)
    apon: AponNoiseEstimator,
}

impl PrismProcessor {
    /// Create new PRISM processor with default config
    pub fn new() -> Self {
        Self::with_config(PrismConfig::default())
    }

    /// Create PRISM processor with custom config
    pub fn with_config(config: PrismConfig) -> Self {
        Self {
            config,
            alpha_history: Vec::with_capacity(10),
            fft_planner: FftPlanner::new(),
            apon: AponNoiseEstimator::new(),
        }
    }

    /// Process RGB signals and extract heart rate
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - RGB channel arrays (should be same length)
    ///
    /// # Returns
    /// PrismResult with BPM, confidence, SNR, and optimal alpha
    pub fn process(
        &mut self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
    ) -> Option<PrismResult> {
        let n = r.len();
        if n < 30 || g.len() != n || b.len() != n {
            return None;
        }

        // Normalize each channel
        let r_norm = Self::normalize(r);
        let g_norm = Self::normalize(g);
        let b_norm = Self::normalize(b);

        // Determine search range (APON warm-start or full range)
        let search_range = if self.config.use_apon {
            let apon_result = self.apon.estimate(&r_norm, &g_norm, &b_norm);
            if apon_result.is_reliable {
                Some(apon_result.alpha_range)
            } else {
                None // Fall back to full search
            }
        } else {
            None
        };

        // Grid search for optimal alpha
        let (best_alpha, _best_signal, best_snr) =
            self.grid_search_alpha_range(&r_norm, &g_norm, &b_norm, search_range);

        // Update alpha history for smoothing
        self.alpha_history.push(best_alpha);
        if self.alpha_history.len() > 10 {
            self.alpha_history.remove(0);
        }

        // Use smoothed alpha
        let smoothed_alpha: f32 =
            self.alpha_history.iter().sum::<f32>() / self.alpha_history.len() as f32;

        // Final signal with smoothed alpha
        let final_signal = self.mix_signal(&r_norm, &g_norm, &b_norm, smoothed_alpha);

        // Apply SOTA temporal normalization (Phase 3 optimization)
        // Adjust window size based on SNR: low SNR = larger window to smooth more
        let tn_window = if best_snr > 5.0 { 15 } else { 30 };
        let filtered_signal = temporal_normalization(&final_signal, tn_window);

        // Extract heart rate
        let (bpm, snr) = self.extract_heart_rate(&filtered_signal);

        // Compute confidence based on SNR
        let confidence = self.snr_to_confidence(snr);

        Some(PrismResult {
            bpm,
            confidence,
            snr,
            optimal_alpha: smoothed_alpha,
        })
    }

    /// Process with explicit APON result (for advanced usage)
    pub fn process_with_apon(
        &mut self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
        apon_result: &AponResult,
    ) -> Option<PrismResult> {
        let n = r.len();
        if n < 30 || g.len() != n || b.len() != n {
            return None;
        }

        let r_norm = Self::normalize(r);
        let g_norm = Self::normalize(g);
        let b_norm = Self::normalize(b);

        let search_range = if apon_result.is_reliable {
            Some(apon_result.alpha_range)
        } else {
            None
        };

        let (best_alpha, _, best_snr) =
            self.grid_search_alpha_range(&r_norm, &g_norm, &b_norm, search_range);

        self.alpha_history.push(best_alpha);
        if self.alpha_history.len() > 10 {
            self.alpha_history.remove(0);
        }

        let smoothed_alpha: f32 =
            self.alpha_history.iter().sum::<f32>() / self.alpha_history.len() as f32;

        let final_signal = self.mix_signal(&r_norm, &g_norm, &b_norm, smoothed_alpha);
        
        // SOTA temporal normalization
        let tn_window = if best_snr > 5.0 { 15 } else { 30 };
        let filtered_signal = temporal_normalization(&final_signal, tn_window);
        let (bpm, snr) = self.extract_heart_rate(&filtered_signal);
        let confidence = self.snr_to_confidence(snr);

        Some(PrismResult {
            bpm,
            confidence,
            snr,
            optimal_alpha: smoothed_alpha,
        })
    }

    /// Normalize array (zero-mean, unit variance)
    fn normalize(arr: &Array1<f32>) -> Array1<f32> {
        let mean = arr.mean().unwrap_or(0.0);
        let std = {
            let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
            variance.sqrt().max(1e-6)
        };
        arr.mapv(|x| (x - mean) / std)
    }

    /// Grid search for optimal alpha with optional warm-start range
    fn grid_search_alpha_range(
        &mut self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
        range: Option<(f32, f32)>,
    ) -> (f32, Array1<f32>, f32) {
        let (alpha_min, alpha_max) = range.unwrap_or((0.0, 1.0));
        
        let mut best_alpha = (alpha_min + alpha_max) / 2.0;
        let mut best_snr = f32::NEG_INFINITY;
        let mut best_signal = g.clone();

        // When using APON warm-start, use fewer steps in narrower range
        let steps = if range.is_some() {
            5  // Reduced from 11 to 5 steps
        } else {
            self.config.alpha_steps.max(1)  // Guard against alpha_steps=0
        };

        for i in 0..=steps {
            let alpha = alpha_min + (i as f32 / steps as f32) * (alpha_max - alpha_min);

            // Signal = α·G + (1-α)·(R-B)
            let signal = self.mix_signal(r, g, b, alpha);

            // Compute SNR for this alpha
            let snr = self.compute_snr(&signal);

            if snr > best_snr {
                best_snr = snr;
                best_alpha = alpha;
                best_signal = signal;
            }
        }

        (best_alpha, best_signal, best_snr)
    }

    /// Legacy grid search (full range) - kept for backward compatibility
    #[allow(dead_code)]
    fn grid_search_alpha(
        &mut self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
    ) -> (f32, Array1<f32>, f32) {
        self.grid_search_alpha_range(r, g, b, None)
    }

    /// Mix RGB signals with given alpha weight
    fn mix_signal(
        &self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
        alpha: f32,
    ) -> Array1<f32> {
        // Signal = α·G + (1-α)·(R-B)
        let r_minus_b = r - b;
        g * alpha + &r_minus_b * (1.0 - alpha)
    }

    /// Compute Signal-to-Noise Ratio in dB
    fn compute_snr(&mut self, signal: &Array1<f32>) -> f32 {
        let n = signal.len();
        let fs = self.config.sample_rate;

        // Apply Hamming window
        let windowed: Vec<Complex32> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
                Complex32::new(s * window, 0.0)
            })
            .collect();

        // FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        // Power spectrum (positive frequencies only)
        let half_n = n / 2;
        let bin_res = fs / n as f32;
        let min_bin = (self.config.min_freq / bin_res) as usize;
        let max_bin = (self.config.max_freq / bin_res).min(half_n as f32) as usize;

        // Find peak power and total noise
        let mut peak_power = 0.0f32;
        let mut total_power = 0.0f32;

        for i in min_bin..=max_bin.min(half_n - 1) {
            let power = buffer[i].norm_sqr();
            total_power += power;
            if power > peak_power {
                peak_power = power;
            }
        }

        // SNR = peak / noise (in dB)
        let noise_power = total_power - peak_power;
        if noise_power > 0.0 {
            10.0 * (peak_power / noise_power).log10()
        } else {
            0.0
        }
    }


    /// Extract heart rate from filtered signal
    fn extract_heart_rate(&mut self, signal: &Array1<f32>) -> (f32, f32) {
        let n = signal.len();
        let fs = self.config.sample_rate;

        // Apply Hamming window
        let windowed: Vec<Complex32> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let window = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
                Complex32::new(s * window, 0.0)
            })
            .collect();

        // FFT
        let fft = self.fft_planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        // Find peak in HR band
        let half_n = n / 2;
        let bin_res = fs / n as f32;
        let min_bin = (self.config.min_freq / bin_res) as usize;
        let max_bin = (self.config.max_freq / bin_res).min(half_n as f32) as usize;

        let mut peak_power = 0.0f32;
        let mut peak_bin = min_bin;
        let mut total_power = 0.0f32;

        for i in min_bin..=max_bin.min(half_n - 1) {
            let power = buffer[i].norm_sqr();
            total_power += power;
            if power > peak_power {
                peak_power = power;
                peak_bin = i;
            }
        }

        // Parabolic interpolation for sub-bin accuracy
        let peak_freq = if peak_bin > min_bin && peak_bin < max_bin.min(half_n - 1) {
            let p_left = buffer[peak_bin - 1].norm_sqr();
            let p_center = peak_power;
            let p_right = buffer[peak_bin + 1].norm_sqr();

            let delta = 0.5 * (p_left - p_right) / (p_left - 2.0 * p_center + p_right + 1e-10);
            (peak_bin as f32 + delta) * bin_res
        } else {
            peak_bin as f32 * bin_res
        };

        let bpm = peak_freq * 60.0;

        // SNR calculation
        let noise_power = total_power - peak_power;
        let snr = if noise_power > 0.0 {
            10.0 * (peak_power / noise_power).log10()
        } else {
            0.0
        };

        (bpm, snr)
    }

    /// Convert SNR to confidence score (0-1)
    fn snr_to_confidence(&self, snr: f32) -> f32 {
        // Sigmoid-like mapping
        // SNR < 0 dB → low confidence
        // SNR > 10 dB → high confidence
        let normalized = (snr + 5.0) / 15.0; // Shift and scale
        normalized.clamp(0.0, 1.0)
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.alpha_history.clear();
    }
}

impl Default for PrismProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prism_synthetic_60bpm() {
        let mut processor = PrismProcessor::new();
        let fs = 30.0;
        let duration = 3.0;
        let n = (fs * duration) as usize;
        let target_bpm = 60.0; // 1 Hz

        // Generate synthetic RGB with 60 BPM pulse
        let mut r = Array1::zeros(n);
        let mut g = Array1::zeros(n);
        let mut b = Array1::zeros(n);

        for i in 0..n {
            let t = i as f32 / fs;
            let pulse = (2.0 * PI * (target_bpm / 60.0) * t).sin();

            // Green channel has strongest pulse (realistic)
            g[i] = 128.0 + pulse * 2.0; // PPG signal on green
            r[i] = 128.0 + pulse * 0.5; // Weaker on red
            b[i] = 128.0 + pulse * 0.3; // Weakest on blue
        }

        let result = processor.process(&r, &g, &b).expect("Should process");

        assert!(
            (result.bpm - target_bpm).abs() < 5.0,
            "Expected ~60 BPM, got {}",
            result.bpm
        );
        assert!(result.snr > 0.0, "SNR should be positive");
        assert!(
            result.optimal_alpha >= 0.0 && result.optimal_alpha <= 1.0,
            "Alpha should be in valid range [0,1], got {}",
            result.optimal_alpha
        );
    }

    #[test]
    fn test_prism_90bpm() {
        let mut processor = PrismProcessor::new();
        let fs = 30.0;
        let duration = 3.0;
        let n = (fs * duration) as usize;
        let target_bpm = 90.0; // 1.5 Hz

        let mut r = Array1::zeros(n);
        let mut g = Array1::zeros(n);
        let mut b = Array1::zeros(n);

        for i in 0..n {
            let t = i as f32 / fs;
            let pulse = (2.0 * PI * (target_bpm / 60.0) * t).sin();
            g[i] = 128.0 + pulse * 3.0;
            r[i] = 128.0 + pulse * 1.0;
            b[i] = 128.0 + pulse * 0.5;
        }

        let result = processor.process(&r, &g, &b).expect("Should process");

        assert!(
            (result.bpm - target_bpm).abs() < 5.0,
            "Expected ~90 BPM, got {}",
            result.bpm
        );
    }

    #[test]
    fn test_normalize() {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = PrismProcessor::normalize(&arr);

        // Mean should be ~0
        assert!(normalized.mean().unwrap().abs() < 1e-6);

        // Std should be ~1
        let std = normalized.mapv(|x| x.powi(2)).mean().unwrap().sqrt();
        assert!((std - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_insufficient_data() {
        let mut processor = PrismProcessor::new();
        let r = Array1::zeros(10); // Too short
        let g = Array1::zeros(10);
        let b = Array1::zeros(10);

        assert!(processor.process(&r, &g, &b).is_none());
    }

    #[test]
    fn test_snr_to_confidence() {
        let processor = PrismProcessor::new();

        // Low SNR → low confidence
        assert!(processor.snr_to_confidence(-5.0) < 0.1);

        // High SNR → high confidence
        assert!(processor.snr_to_confidence(15.0) > 0.9);
    }
}
