//! Enhanced Respiration Rate Estimation
//!
//! Extracts respiratory rate (RR) from pulse waveforms using multiple modulation
//! analysis methods. Implements SOTA 2025 techniques:
//!
//! 1. **Amplitude Modulation (AM)**: Respiration changes stroke volume
//! 2. **Frequency Modulation (FM)**: Respiratory sinus arrhythmia (RSA)
//! 3. **Baseline Wander (BW)**: Tissue blood volume changes
//!
//! # Algorithm
//!
//! ```text
//! Pulse Signal
//!     │
//!     ├──► AM: CWT envelope in resp band (0.1-0.5 Hz)
//!     │
//!     ├──► FM: Peak interval variability → spectral analysis
//!     │
//!     └──► BW: Low-pass filter → FFT peak
//!
//!     Fusion: Weight by SNR, take consensus
//! ```
//!
//! # Reference
//!
//! - "Temporal Fusion of ECG and PPG for RR Monitoring" (2025)
//! - "Enhanced Empirical Wavelet Transform for RR Estimation" (2025)

use crate::dsp::{DspProcessor, FilterConfig};
use crate::wavelet::FastCWT;
use ndarray::Array1;

/// Respiration estimation configuration.
#[derive(Debug, Clone)]
pub struct RespirationConfig {
    pub sample_rate: f32,
    /// Resp band (Hz), default 0.1..0.5 (6..30 brpm)
    pub min_freq: f32,
    pub max_freq: f32,
    /// Enable CWT-based envelope analysis
    pub use_cwt: bool,
    /// Enable frequency modulation (RSA) analysis
    pub use_fm: bool,
    /// Minimum peaks required for FM analysis
    pub min_peaks_for_fm: usize,
}

impl Default for RespirationConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 0.10,
            max_freq: 0.50,
            use_cwt: true,
            use_fm: true,
            min_peaks_for_fm: 10,
        }
    }
}

/// Respiration estimation result
#[derive(Debug, Clone)]
pub struct RespirationResult {
    /// Breaths per minute (fused estimate)
    pub brpm: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Individual method results (for debugging)
    pub method_results: MethodResults,
}

/// Results from individual estimation methods
#[derive(Debug, Clone, Default)]
pub struct MethodResults {
    /// Amplitude modulation result
    pub am: Option<(f32, f32)>,  // (brpm, snr)
    /// Frequency modulation result
    pub fm: Option<(f32, f32)>,  // (brpm, snr)
    /// Baseline wander result
    pub bw: Option<(f32, f32)>,  // (brpm, snr)
    /// CWT envelope result
    pub cwt: Option<(f32, f32)>, // (brpm, snr)
}

/// Enhanced Respiration Rate Estimator
pub struct RespirationEstimator {
    cfg: RespirationConfig,
    fcwt: FastCWT,
}

impl RespirationEstimator {
    pub fn new() -> Self {
        Self::with_config(RespirationConfig::default())
    }

    pub fn with_config(cfg: RespirationConfig) -> Self {
        let mut fcwt_cfg = crate::wavelet::FastCwtConfig::default();
        fcwt_cfg.sample_rate = cfg.sample_rate;
        
        Self {
            cfg,
            fcwt: FastCWT::with_config(fcwt_cfg),
        }
    }

    /// Estimate respiration rate using multi-method fusion
    ///
    /// Combines AM, FM, BW, and CWT methods with SNR-weighted voting.
    pub fn estimate(&mut self, pulse: &Array1<f32>) -> Option<RespirationResult> {
        let n = pulse.len();
        if n < 128 {
            return None;
        }

        let mut results = MethodResults::default();
        let mut estimates: Vec<(f32, f32)> = Vec::with_capacity(4); // (brpm, snr)

        // Method 1: Baseline Wander (BW) - simple lowpass + FFT
        if let Some((brpm, snr)) = self.estimate_bw(pulse) {
            results.bw = Some((brpm, snr));
            estimates.push((brpm, snr));
        }

        // Method 2: Amplitude Modulation (AM) - envelope in resp band
        if let Some((brpm, snr)) = self.estimate_am(pulse) {
            results.am = Some((brpm, snr));
            estimates.push((brpm, snr));
        }

        // Method 3: CWT Envelope (if enabled)
        if self.cfg.use_cwt {
            if let Some((brpm, snr)) = self.estimate_cwt_envelope(pulse.as_slice().unwrap()) {
                results.cwt = Some((brpm, snr));
                estimates.push((brpm, snr));
            }
        }

        // Method 4: Frequency Modulation (FM/RSA) - peak interval analysis
        if self.cfg.use_fm {
            if let Some((brpm, snr)) = self.estimate_fm(pulse) {
                results.fm = Some((brpm, snr));
                estimates.push((brpm, snr));
            }
        }

        if estimates.is_empty() {
            return None;
        }

        // Fuse estimates using SNR-weighted voting
        let (fused_brpm, fused_snr) = self.fuse_estimates(&estimates);
        let confidence = self.snr_to_confidence(fused_snr, estimates.len());

        Some(RespirationResult {
            brpm: fused_brpm,
            snr_db: fused_snr,
            confidence,
            method_results: results,
        })
    }

    /// Baseline Wander (BW) method - lowpass filter + FFT
    fn estimate_bw(&self, pulse: &Array1<f32>) -> Option<(f32, f32)> {
        let cfg = FilterConfig {
            sample_rate: self.cfg.sample_rate,
            min_freq: self.cfg.min_freq,
            max_freq: self.cfg.max_freq,
        };

        let resp_band = DspProcessor::bandpass_filter(pulse, &cfg);
        let (hz, snr_db) = DspProcessor::compute_peak_frequency(
            &resp_band,
            self.cfg.sample_rate,
            self.cfg.min_freq,
            self.cfg.max_freq,
        );

        if hz > 0.0 && snr_db > -10.0 {
            Some((hz * 60.0, snr_db))
        } else {
            None
        }
    }

    /// Amplitude Modulation (AM) method - envelope extraction
    fn estimate_am(&self, pulse: &Array1<f32>) -> Option<(f32, f32)> {
        // Hilbert-like envelope via squaring + lowpass
        let squared = pulse.mapv(|x| x * x);
        
        // Lowpass to get smooth envelope
        let cfg = FilterConfig {
            sample_rate: self.cfg.sample_rate,
            min_freq: 0.0,
            max_freq: 1.0, // Low pass envelope smoothing
        };
        let envelope = DspProcessor::bandpass_filter(&squared, &cfg);
        
        // Now extract resp component from envelope
        let resp_cfg = FilterConfig {
            sample_rate: self.cfg.sample_rate,
            min_freq: self.cfg.min_freq,
            max_freq: self.cfg.max_freq,
        };
        let resp_env = DspProcessor::bandpass_filter(&envelope, &resp_cfg);
        
        let (hz, snr_db) = DspProcessor::compute_peak_frequency(
            &resp_env,
            self.cfg.sample_rate,
            self.cfg.min_freq,
            self.cfg.max_freq,
        );

        if hz > 0.0 && snr_db > -10.0 {
            Some((hz * 60.0, snr_db))
        } else {
            None
        }
    }

    /// CWT Envelope method - wavelet-based envelope extraction
    fn estimate_cwt_envelope(&mut self, pulse: &[f32]) -> Option<(f32, f32)> {
        // Generate scales for respiration band
        let resp_freqs: Vec<f32> = (0..10)
            .map(|i| self.cfg.min_freq + i as f32 * 0.04)
            .collect();
        
        let scales: Vec<f32> = resp_freqs
            .iter()
            .map(|&f| self.fcwt.frequency_to_scale(f))
            .collect();
        
        // Compute CWT
        let cwt_result = self.fcwt.cwt(pulse, &scales);
        
        if cwt_result.is_empty() {
            return None;
        }
        
        // Sum power across all resp scales to get respiratory envelope
        let n = pulse.len();
        let mut resp_envelope = vec![0.0f32; n];
        
        for scale_coeffs in &cwt_result {
            for (i, c) in scale_coeffs.iter().enumerate() {
                resp_envelope[i] += c.norm();
            }
        }
        
        // Find peak frequency in envelope
        let resp_arr = Array1::from_vec(resp_envelope);
        let (hz, snr_db) = DspProcessor::compute_peak_frequency(
            &resp_arr,
            self.cfg.sample_rate,
            self.cfg.min_freq,
            self.cfg.max_freq,
        );

        if hz > 0.0 && snr_db > -10.0 {
            Some((hz * 60.0, snr_db))
        } else {
            None
        }
    }

    /// Frequency Modulation (FM/RSA) method - peak interval variability
    fn estimate_fm(&self, pulse: &Array1<f32>) -> Option<(f32, f32)> {
        // Simple peak detection
        let peaks = self.detect_peaks(pulse);
        
        if peaks.len() < self.cfg.min_peaks_for_fm {
            return None;
        }
        
        // Compute inter-peak intervals (IPI)
        let mut intervals: Vec<f32> = Vec::with_capacity(peaks.len() - 1);
        for i in 1..peaks.len() {
            let dt = (peaks[i] - peaks[i - 1]) as f32 / self.cfg.sample_rate;
            intervals.push(dt);
        }
        
        if intervals.is_empty() {
            return None;
        }
        
        // Resample intervals to uniform grid (for FFT)
        // Simple: assume intervals are at 1 Hz (one per beat)
        // HR ~ 60-100 bpm = 1-1.67 Hz sample rate for intervals
        let avg_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let interval_sample_rate = 1.0 / avg_interval;
        
        if interval_sample_rate < 0.5 || intervals.len() < 8 {
            return None;
        }
        
        let interval_arr = Array1::from_vec(intervals);
        
        let (hz, snr_db) = DspProcessor::compute_peak_frequency(
            &interval_arr,
            interval_sample_rate,
            self.cfg.min_freq,
            self.cfg.max_freq,
        );

        if hz > 0.0 && snr_db > -10.0 {
            Some((hz * 60.0, snr_db))
        } else {
            None
        }
    }

    /// Simple peak detection (threshold + refractory)
    fn detect_peaks(&self, signal: &Array1<f32>) -> Vec<usize> {
        let n = signal.len();
        if n < 3 {
            return vec![];
        }
        
        // Adaptive threshold
        let mean: f32 = signal.iter().sum::<f32>() / n as f32;
        let std: f32 = (signal.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32).sqrt();
        let threshold = mean + 0.5 * std;
        
        // Refractory period (minimum 0.3s between peaks ~ max 200 bpm)
        let refractory = (0.3 * self.cfg.sample_rate) as usize;
        
        let mut peaks = Vec::new();
        let mut last_peak = 0usize;
        
        for i in 1..n - 1 {
            if signal[i] > threshold
                && signal[i] > signal[i - 1]
                && signal[i] > signal[i + 1]
                && (last_peak == 0 || i - last_peak >= refractory)
            {
                peaks.push(i);
                last_peak = i;
            }
        }
        
        peaks
    }

    /// Fuse multiple estimates using SNR-weighted voting
    fn fuse_estimates(&self, estimates: &[(f32, f32)]) -> (f32, f32) {
        if estimates.is_empty() {
            return (0.0, 0.0);
        }
        
        if estimates.len() == 1 {
            return estimates[0];
        }
        
        // Convert SNR to weights (higher SNR = higher weight)
        let weights: Vec<f32> = estimates
            .iter()
            .map(|(_, snr)| (snr + 10.0).max(0.1)) // Shift SNR to positive range
            .collect();
        
        let total_weight: f32 = weights.iter().sum();
        
        let fused_brpm: f32 = estimates
            .iter()
            .zip(weights.iter())
            .map(|((brpm, _), w)| brpm * w / total_weight)
            .sum();
        
        let fused_snr: f32 = estimates
            .iter()
            .zip(weights.iter())
            .map(|((_, snr), w)| snr * w / total_weight)
            .sum();
        
        (fused_brpm, fused_snr)
    }

    /// Convert SNR to confidence, boosted by method agreement
    fn snr_to_confidence(&self, snr_db: f32, num_methods: usize) -> f32 {
        let base_conf = 1.0 / (1.0 + (-0.7 * (snr_db - 3.0)).exp());
        
        // Boost confidence if multiple methods agree
        let agreement_boost = match num_methods {
            1 => 1.0,
            2 => 1.05,
            3 => 1.1,
            _ => 1.15,
        };
        
        (base_conf * agreement_boost).min(1.0)
    }

    /// Get configuration
    pub fn config(&self) -> &RespirationConfig {
        &self.cfg
    }
}

impl Default for RespirationEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_pulse_with_resp_modulation(
        n: usize,
        fs: f32,
        hr_bpm: f32,
        rr_brpm: f32,
    ) -> Array1<f32> {
        let hr_hz = hr_bpm / 60.0;
        let rr_hz = rr_brpm / 60.0;
        
        (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                // Pulse wave
                let pulse = (2.0 * PI * hr_hz * t).sin();
                // Respiratory modulation (amplitude modulation)
                let resp_mod = 1.0 + 0.2 * (2.0 * PI * rr_hz * t).sin();
                pulse * resp_mod
            })
            .collect()
    }

    #[test]
    fn test_respiration_basic() {
        let mut estimator = RespirationEstimator::new();
        
        let signal = generate_pulse_with_resp_modulation(300, 30.0, 72.0, 15.0);
        let result = estimator.estimate(&signal);
        
        assert!(result.is_some(), "Should produce result");
        let result = result.unwrap();
        
        // Should detect respiration in valid range
        assert!(result.brpm > 5.0 && result.brpm < 40.0, 
            "BRPM should be in valid range, got {}", result.brpm);
    }

    #[test]
    fn test_respiration_insufficient_data() {
        let mut estimator = RespirationEstimator::new();
        let short_signal = Array1::from_vec(vec![0.0; 50]);
        
        assert!(estimator.estimate(&short_signal).is_none());
    }

    #[test]
    fn test_method_results_populated() {
        let mut estimator = RespirationEstimator::new();
        let signal = generate_pulse_with_resp_modulation(300, 30.0, 72.0, 12.0);
        
        if let Some(result) = estimator.estimate(&signal) {
            // At least one method should produce a result
            let has_result = result.method_results.bw.is_some()
                || result.method_results.am.is_some()
                || result.method_results.cwt.is_some()
                || result.method_results.fm.is_some();
            
            assert!(has_result, "At least one method should produce result");
        }
    }

    #[test]
    fn test_peak_detection() {
        let estimator = RespirationEstimator::new();
        
        // Generate simple sine wave (30 Hz sample rate, 1 Hz frequency = 30 samples per cycle)
        let signal: Array1<f32> = (0..120)
            .map(|i| (2.0 * PI * i as f32 / 30.0).sin())
            .collect();
        
        let peaks = estimator.detect_peaks(&signal);
        
        // Should detect ~4 peaks (4 cycles in 120 samples at 30 Hz)
        assert!(peaks.len() >= 2 && peaks.len() <= 6, 
            "Should detect reasonable number of peaks, got {}", peaks.len());
    }
}
