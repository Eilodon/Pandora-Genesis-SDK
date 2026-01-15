//! Digital Signal Processing (DSP) module
//!
//! Provides FFT-based heart rate computation and signal filtering.
//!
//! Reference: fft.worker.ts from TypeScript implementation

use ndarray::Array1;
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Configuration for bandpass filter
#[derive(Debug, Clone)]
pub struct FilterConfig {
    pub sample_rate: f32,
    pub min_freq: f32, // Hz (e.g., 0.67 Hz = 40 BPM)
    pub max_freq: f32, // Hz (e.g., 3.0 Hz = 180 BPM)
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 0.67, // 40 BPM
            max_freq: 3.0,  // 180 BPM
        }
    }
}

/// Digital Signal Processor
pub struct DspProcessor;

impl DspProcessor {
    /// Create Hamming window for FFT
    ///
    /// Reference: fft.worker.ts:448-452
    ///
    /// # Arguments
    /// * `size` - Window size
    ///
    /// # Returns
    /// Hamming window coefficients
    pub fn hamming_window(size: usize) -> Array1<f32> {
        let mut window = Array1::zeros(size);
        for i in 0..size {
            window[i] = 0.54 - 0.46 * ((2.0 * PI * i as f32) / ((size - 1) as f32)).cos();
        }
        window
    }

    /// Compute heart rate from signal using FFT
    ///
    /// Reference: fft.worker.ts:508-537
    ///
    /// # Arguments
    /// * `signal` - Input signal (pulse waveform)
    /// * `fs` - Sample rate (Hz)
    ///
    /// # Returns
    /// Tuple of (BPM, SNR in dB)
    pub fn compute_heart_rate(signal: &Array1<f32>, fs: f32) -> (f32, f32) {
        let n = signal.len();
        
        // Guard against too-short signals
        if n < 2 || fs <= 0.0 {
            return (0.0, 0.0);
        }

        // 1. Apply Hamming window
        let window = Self::hamming_window(n);
        let windowed: Vec<Complex32> = signal
            .iter()
            .zip(window.iter())
            .map(|(s, w)| Complex32::new(s * w, 0.0))
            .collect();

        // 2. Perform FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        // 3. Compute power spectrum (only positive frequencies)
        let half_n = n / 2;
        let power_spectrum: Vec<f32> = buffer.iter().take(half_n).map(|c| c.norm_sqr()).collect();

        // 4. Find peak in physiological range (40-180 BPM = 0.67-3.0 Hz)
        let bin_res = fs / n as f32;
        let min_bin = (0.67 / bin_res) as usize;
        let max_bin = (3.0 / bin_res).min(half_n as f32) as usize;

        let mut max_power = 0.0;
        let mut peak_bin = 0;
        let mut total_noise_power = 0.0;

        for i in min_bin..=max_bin {
            if i >= power_spectrum.len() {
                break;
            }
            let power = power_spectrum[i];
            if power > max_power {
                max_power = power;
                peak_bin = i;
            }
            total_noise_power += power;
        }

        // 5. Calculate SNR
        let noise = total_noise_power - max_power;
        let snr = if noise > 0.0 {
            10.0 * (max_power / noise).log10() // dB
        } else {
            0.0
        };

        // 6. Parabolic interpolation for sub-bin accuracy (+0.5 bpm)
        let refined_bin = if peak_bin > 0 && peak_bin + 1 < power_spectrum.len() {
            let y_m1 = power_spectrum[peak_bin - 1];
            let y_0 = power_spectrum[peak_bin];
            let y_p1 = power_spectrum[peak_bin + 1];
            let denom = y_m1 - 2.0 * y_0 + y_p1;
            if denom.abs() > 1e-12 {
                let delta = 0.5 * (y_m1 - y_p1) / denom;
                if delta.is_finite() && delta.abs() <= 1.0 {
                    peak_bin as f32 + delta
                } else {
                    peak_bin as f32
                }
            } else {
                peak_bin as f32
            }
        } else {
            peak_bin as f32
        };

        let bpm = refined_bin * bin_res * 60.0;

        (bpm, snr)
    }

    /// Compute peak frequency (Hz) and SNR (dB) in a specified band.
    ///
    /// Generalized version of `compute_heart_rate` for any frequency band,
    /// e.g., respiration (0.1-0.5 Hz).
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `fs` - Sample rate (Hz)
    /// * `min_freq` - Minimum frequency (Hz)
    /// * `max_freq` - Maximum frequency (Hz)
    ///
    /// # Returns
    /// Tuple of (peak frequency in Hz, SNR in dB)
    pub fn compute_peak_frequency(
        signal: &Array1<f32>,
        fs: f32,
        min_freq: f32,
        max_freq: f32,
    ) -> (f32, f32) {
        let n = signal.len();
        if n < 32 || fs <= 0.0 {
            return (0.0, 0.0);
        }

        // 1. Apply Hamming window
        let window = Self::hamming_window(n);
        let windowed: Vec<Complex32> = signal
            .iter()
            .zip(window.iter())
            .map(|(s, w)| Complex32::new(s * w, 0.0))
            .collect();

        // 2. FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        // 3. Power spectrum (positive freqs)
        let half_n = n / 2;
        let power_spectrum: Vec<f32> = buffer.iter().take(half_n).map(|c| c.norm_sqr()).collect();

        // 4. Peak in band
        let bin_res = fs / n as f32;
        let min_bin = (min_freq / bin_res).max(0.0) as usize;
        let max_bin = (max_freq / bin_res).min(half_n as f32) as usize;

        if min_bin >= max_bin || max_bin == 0 {
            return (0.0, 0.0);
        }

        let mut peak_bin = min_bin;
        let mut max_power = 0.0f32;
        let mut total_power = 0.0f32;

        for i in min_bin..=max_bin.min(half_n - 1) {
            let p = power_spectrum[i];
            total_power += p;
            if p > max_power {
                max_power = p;
                peak_bin = i;
            }
        }

        let noise = (total_power - max_power).max(1e-12);
        let snr = if max_power > 0.0 { 10.0 * (max_power / noise).log10() } else { 0.0 };

        // Parabolic interpolation for refined peak location
        let refined_bin = if peak_bin > 0 && peak_bin + 1 < power_spectrum.len() {
            let y_m1 = power_spectrum[peak_bin - 1];
            let y_0 = power_spectrum[peak_bin];
            let y_p1 = power_spectrum[peak_bin + 1];
            let denom = y_m1 - 2.0 * y_0 + y_p1;
            if denom.abs() > 1e-12 {
                let delta = 0.5 * (y_m1 - y_p1) / denom;
                if delta.is_finite() && delta.abs() <= 1.0 {
                    peak_bin as f32 + delta
                } else {
                    peak_bin as f32
                }
            } else {
                peak_bin as f32
            }
        } else {
            peak_bin as f32
        };

        let hz = refined_bin * bin_res;
        (hz, snr)
    }

    /// Bandpass filter using simple IIR approximation
    ///
    /// Implements a first-order high-pass followed by first-order low-pass
    /// with cutoff frequencies derived from FilterConfig.
    ///
    /// # Arguments
    /// * `signal` - Input signal
    /// * `config` - Filter configuration with min_freq (highpass) and max_freq (lowpass)
    ///
    /// # Returns
    /// Filtered signal
    pub fn bandpass_filter(signal: &Array1<f32>, config: &FilterConfig) -> Array1<f32> {
        let n = signal.len();
        if n < 2 {
            return signal.clone();
        }

        let mut filtered = signal.to_vec();
        let fs = config.sample_rate;

        // Compute IIR coefficients from cutoff frequencies
        // High-pass: alpha_hp = 1 / (1 + 2*pi*fc/fs) approximately
        // For first-order IIR: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        let hp_rc = 1.0 / (2.0 * PI * config.min_freq.max(0.01));
        let hp_dt = 1.0 / fs;
        let hp_alpha = hp_rc / (hp_rc + hp_dt);

        // Low-pass: alpha_lp = dt / (rc + dt)
        let lp_rc = 1.0 / (2.0 * PI * config.max_freq.max(0.1));
        let lp_dt = 1.0 / fs;
        let lp_alpha = lp_dt / (lp_rc + lp_dt);

        // High-pass filter (remove frequencies below min_freq)
        let mut hp_prev_in = filtered[0];
        let mut hp_prev_out = 0.0;
        for i in 1..n {
            let hp_out = hp_alpha * (hp_prev_out + filtered[i] - hp_prev_in);
            hp_prev_in = filtered[i];
            hp_prev_out = hp_out;
            filtered[i] = hp_out;
        }
        filtered[0] = 0.0; // First sample has no previous

        // Low-pass filter (remove frequencies above max_freq)
        let mut lp_prev = filtered[0];
        for i in 1..n {
            let lp_out = lp_alpha * filtered[i] + (1.0 - lp_alpha) * lp_prev;
            lp_prev = lp_out;
            filtered[i] = lp_out;
        }

        Array1::from(filtered)
    }

    /// Detrend signal (remove mean)
    ///
    /// Reference: fft.worker.ts:499-506
    pub fn detrend(signal: &Array1<f32>) -> Array1<f32> {
        let mean = signal.mean().unwrap_or(0.0);
        signal.mapv(|x| x - mean)
    }

    /// Calculate standard deviation
    pub fn std(arr: &Array1<f32>) -> f32 {
        let mean = arr.mean().unwrap_or(0.0);
        let variance = arr.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hamming_window() {
        let window = DspProcessor::hamming_window(10);
        assert_eq!(window.len(), 10);

        // First and last values should be ~0.08
        assert_relative_eq!(window[0], 0.08, epsilon = 0.01);
        assert_relative_eq!(window[9], 0.08, epsilon = 0.01);

        // Middle value should be close to 1.0 (but not exactly due to discrete window)
        assert_relative_eq!(window[4], 1.0, epsilon = 0.03);
    }

    #[test]
    fn test_detrend() {
        let signal = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let detrended = DspProcessor::detrend(&signal);

        // Mean should be ~0
        assert_relative_eq!(detrended.mean().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_std() {
        let signal = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let std = DspProcessor::std(&signal);

        // Known std for [1,2,3,4,5] is sqrt(2)
        assert_relative_eq!(std, std::f32::consts::SQRT_2, epsilon = 0.001);
    }

    #[test]
    fn test_compute_heart_rate_synthetic() {
        // Generate synthetic 60 BPM signal (1 Hz)
        let fs = 30.0;
        let duration = 3.0; // 3 seconds
        let n = (fs * duration) as usize;

        let mut signal = Array1::zeros(n);
        for i in 0..n {
            let t = i as f32 / fs;
            signal[i] = (2.0 * PI * t).sin(); // 1 Hz = 60 BPM
        }

        let (bpm, snr) = DspProcessor::compute_heart_rate(&signal, fs);

        // Should detect ~60 BPM
        assert!((bpm - 60.0).abs() < 5.0, "Expected ~60 BPM, got {}", bpm);
        assert!(snr > 0.0, "SNR should be positive");
    }
}
