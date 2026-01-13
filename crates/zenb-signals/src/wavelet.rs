//! Continuous Wavelet Transform using Morlet Wavelets
//!
//! Implements signal analysis for extracting phasor (amplitude + phase) information
//! from physiological signals. Designed for the VAJRA V5 architecture.
//!
//! # Mathematical Foundation
//!
//! The Morlet wavelet is a complex sinusoid modulated by a Gaussian:
//! ```text
//! ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)
//! ```
//!
//! The CWT projects the signal onto wavelets at different scales:
//! ```text
//! W(a,b) = (1/√a) ∫ x(t) * ψ*((t-b)/a) dt
//! ```
//!
//! # Frequency Bands
//! - Delta (0.5-4 Hz): Deep sleep, unconscious processing
//! - Theta (4-8 Hz): Meditation, drowsiness, memory
//! - Alpha (8-12 Hz): Relaxed awareness, eyes closed
//!
//! # References
//! - Morlet et al. (1982): Original wavelet formulation
//! - VAJRA V5 Specification: Sắc Uẩn (RŪPA) sensing layer

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// Configuration for Morlet wavelet
#[derive(Debug, Clone)]
pub struct MorletConfig {
    /// Central frequency parameter (omega_0). Higher values = better frequency resolution.
    /// Default: 6.0 (standard Morlet)
    pub omega0: f32,
    /// Sample rate of input signal in Hz
    pub sample_rate: f32,
}

impl Default for MorletConfig {
    fn default() -> Self {
        Self {
            omega0: 6.0,
            sample_rate: 30.0, // Typical wearable sensor rate
        }
    }
}

/// Result of band-separated wavelet transform
#[derive(Debug, Clone)]
pub struct WaveletBands {
    /// Delta band (0.5-4 Hz): Deep sleep, unconscious
    pub delta: Vec<Complex32>,
    /// Theta band (4-8 Hz): Meditation, memory
    pub theta: Vec<Complex32>,
    /// Alpha band (8-12 Hz): Relaxed awareness
    pub alpha: Vec<Complex32>,
    /// Timestamps corresponding to each sample
    pub timestamps_us: Vec<i64>,
}

impl WaveletBands {
    /// Get the dominant band based on average power
    pub fn dominant_band(&self) -> BandType {
        let delta_power: f32 = self.delta.iter().map(|c| c.norm_sqr()).sum();
        let theta_power: f32 = self.theta.iter().map(|c| c.norm_sqr()).sum();
        let alpha_power: f32 = self.alpha.iter().map(|c| c.norm_sqr()).sum();

        if delta_power >= theta_power && delta_power >= alpha_power {
            BandType::Delta
        } else if theta_power >= alpha_power {
            BandType::Theta
        } else {
            BandType::Alpha
        }
    }

    /// Get the mean phase difference across samples for a band
    /// Returns the coherence (0-1) indicating phase consistency
    pub fn phase_coherence(&self, band: BandType) -> f32 {
        let samples = match band {
            BandType::Delta => &self.delta,
            BandType::Theta => &self.theta,
            BandType::Alpha => &self.alpha,
        };

        if samples.len() < 2 {
            return 1.0; // Perfect coherence for single sample
        }

        // Compute mean resultant length (PLV - phase locking value)
        let sum: Complex32 = samples
            .iter()
            .filter(|c| c.norm() > 1e-10) // Skip near-zero values
            .map(|c| c / c.norm()) // Normalize to unit circle
            .sum();

        let n = samples.len() as f32;
        (sum.norm() / n).clamp(0.0, 1.0)
    }
}

/// Frequency band type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandType {
    Delta,
    Theta,
    Alpha,
}

/// Morlet Wavelet Transform processor
pub struct MorletWavelet {
    config: MorletConfig,
    fft_planner: FftPlanner<f32>,
    cached_fft: Option<Arc<dyn Fft<f32>>>,
    cached_ifft: Option<Arc<dyn Fft<f32>>>,
    cached_size: usize,
}

impl std::fmt::Debug for MorletWavelet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorletWavelet")
            .field("config", &self.config)
            .field("cached_size", &self.cached_size)
            .finish_non_exhaustive()
    }
}

impl MorletWavelet {
    /// Create a new Morlet wavelet processor
    pub fn new(config: MorletConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
            cached_fft: None,
            cached_ifft: None,
            cached_size: 0,
        }
    }

    /// Create with default configuration
    pub fn default_for_zenb() -> Self {
        Self::new(MorletConfig::default())
    }

    /// Compute frequency for a given scale
    fn scale_to_frequency(&self, scale: f32) -> f32 {
        self.config.omega0 * self.config.sample_rate / (2.0 * PI * scale)
    }

    /// Compute scale for a given frequency
    fn frequency_to_scale(&self, freq: f32) -> f32 {
        self.config.omega0 * self.config.sample_rate / (2.0 * PI * freq)
    }

    /// Generate Morlet wavelet in frequency domain for a given scale
    ///
    /// The Morlet wavelet in frequency domain is:
    /// Ψ̂(ω) = π^(-1/4) * exp(-0.5 * (s*ω - ω₀)²)
    ///
    /// where s is the scale and ω₀ is the central frequency parameter.
    fn morlet_fft(&self, n: usize, scale: f32) -> Vec<Complex32> {
        let omega0 = self.config.omega0;
        let fs = self.config.sample_rate;

        let mut wavelet = vec![Complex32::new(0.0, 0.0); n];

        // Nyquist frequency
        let nyquist = fs / 2.0;

        for k in 0..n {
            // Compute normalized angular frequency for bin k
            // k=0 is DC, k=n/2 is Nyquist, k>n/2 are negative frequencies
            let freq_hz = if k <= n / 2 {
                (k as f32) * fs / (n as f32)
            } else {
                -((n - k) as f32) * fs / (n as f32)
            };

            // Skip DC and negative frequencies for analytic wavelet
            if freq_hz <= 0.0 {
                continue;
            }

            // Normalized angular frequency (0 to π maps to 0 to Nyquist)
            let omega_norm = PI * freq_hz / nyquist;

            // Scaled angular frequency
            let omega_scaled = scale * omega_norm;

            // Morlet wavelet in frequency domain
            // Ψ̂(ω) = π^(-1/4) * exp(-0.5 * (s*ω - ω₀)²)
            let exponent = -0.5 * (omega_scaled - omega0).powi(2);
            let amplitude = PI.powf(-0.25) * exponent.exp();

            // Scale normalization for energy preservation
            wavelet[k] = Complex32::new(amplitude * scale.sqrt(), 0.0);
        }

        wavelet
    }

    /// Perform CWT at a single scale, returning complex coefficients
    ///
    /// Uses FFT convolution for O(N log N) complexity
    pub fn cwt_at_scale(&mut self, signal: &[f32], scale: f32) -> Vec<Complex32> {
        let n = signal.len();
        if n == 0 {
            return vec![];
        }

        // Get or create FFT plans
        if self.cached_size != n {
            self.cached_fft = Some(self.fft_planner.plan_fft_forward(n));
            self.cached_ifft = Some(self.fft_planner.plan_fft_inverse(n));
            self.cached_size = n;
        }

        let fft = self.cached_fft.as_ref().expect("FFT plan should exist");
        let ifft = self.cached_ifft.as_ref().expect("IFFT plan should exist");

        // Convert signal to complex
        let mut signal_fft: Vec<Complex32> = signal
            .iter()
            .map(|&x| Complex32::new(x, 0.0))
            .collect();

        // Forward FFT of signal
        fft.process(&mut signal_fft);

        // Get wavelet in frequency domain
        let wavelet_fft = self.morlet_fft(n, scale);

        // Multiply in frequency domain (convolution)
        for i in 0..n {
            signal_fft[i] *= wavelet_fft[i];
        }

        // Inverse FFT
        ifft.process(&mut signal_fft);

        // Normalize by n (rustfft doesn't normalize)
        let norm = 1.0 / (n as f32);
        for c in &mut signal_fft {
            *c *= norm;
        }

        signal_fft
    }

    /// Perform CWT across multiple scales
    ///
    /// Returns a 2D array: [scale_index][time_index]
    pub fn cwt(&mut self, signal: &[f32], scales: &[f32]) -> Vec<Vec<Complex32>> {
        scales
            .iter()
            .map(|&scale| self.cwt_at_scale(signal, scale))
            .collect()
    }

    /// Extract frequency bands from a signal
    ///
    /// Returns averaged complex values for Delta, Theta, and Alpha bands
    pub fn extract_bands(&mut self, signal: &[f32], start_ts_us: i64) -> WaveletBands {
        let n = signal.len();
        if n == 0 {
            return WaveletBands {
                delta: vec![],
                theta: vec![],
                alpha: vec![],
                timestamps_us: vec![],
            };
        }

        // Define frequency ranges for each band
        let delta_freqs = vec![0.5, 1.0, 2.0, 4.0];
        let theta_freqs = vec![4.0, 5.0, 6.0, 7.0, 8.0];
        let alpha_freqs = vec![8.0, 9.0, 10.0, 11.0, 12.0];

        // Convert frequencies to scales
        let delta_scales: Vec<f32> = delta_freqs
            .iter()
            .map(|&f| self.frequency_to_scale(f))
            .collect();
        let theta_scales: Vec<f32> = theta_freqs
            .iter()
            .map(|&f| self.frequency_to_scale(f))
            .collect();
        let alpha_scales: Vec<f32> = alpha_freqs
            .iter()
            .map(|&f| self.frequency_to_scale(f))
            .collect();

        // Compute CWT for each band
        let delta_cwt = self.cwt(signal, &delta_scales);
        let theta_cwt = self.cwt(signal, &theta_scales);
        let alpha_cwt = self.cwt(signal, &alpha_scales);

        // Average across scales for each time point
        let average_band = |cwt: Vec<Vec<Complex32>>| -> Vec<Complex32> {
            if cwt.is_empty() || cwt[0].is_empty() {
                return vec![];
            }
            let n_scales = cwt.len();
            let n_samples = cwt[0].len();
            let mut result = vec![Complex32::new(0.0, 0.0); n_samples];

            for t in 0..n_samples {
                let sum: Complex32 = cwt.iter().map(|scale| scale[t]).sum();
                result[t] = sum / (n_scales as f32);
            }
            result
        };

        let delta = average_band(delta_cwt);
        let theta = average_band(theta_cwt);
        let alpha = average_band(alpha_cwt);

        // Generate timestamps
        let dt_us = (1_000_000.0 / self.config.sample_rate) as i64;
        let timestamps_us: Vec<i64> = (0..n)
            .map(|i| start_ts_us + (i as i64) * dt_us)
            .collect();

        WaveletBands {
            delta,
            theta,
            alpha,
            timestamps_us,
        }
    }

    /// Get amplitude and phase from a complex value
    #[inline]
    pub fn to_phasor(c: Complex32) -> (f32, f32) {
        (c.norm(), c.arg())
    }

    /// Get instantaneous frequency from phase derivative
    ///
    /// Returns frequency in Hz for consecutive samples
    pub fn instantaneous_frequency(phases: &[f32], sample_rate: f32) -> Vec<f32> {
        if phases.len() < 2 {
            return vec![];
        }

        let mut freqs = Vec::with_capacity(phases.len() - 1);
        let dt = 1.0 / sample_rate;

        for i in 1..phases.len() {
            // Phase unwrapping
            let mut dphi = phases[i] - phases[i - 1];
            while dphi > PI {
                dphi -= 2.0 * PI;
            }
            while dphi < -PI {
                dphi += 2.0 * PI;
            }

            // Frequency = dφ/dt / (2π)
            let freq = dphi / (2.0 * PI * dt);
            freqs.push(freq.abs());
        }

        freqs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morlet_config_default() {
        let config = MorletConfig::default();
        assert_eq!(config.omega0, 6.0);
        assert_eq!(config.sample_rate, 30.0);
    }

    #[test]
    fn test_scale_frequency_conversion() {
        let wavelet = MorletWavelet::default_for_zenb();

        // Round-trip should preserve frequency
        let freq = 5.0;
        let scale = wavelet.frequency_to_scale(freq);
        let freq_back = wavelet.scale_to_frequency(scale);

        assert!(
            (freq - freq_back).abs() < 0.01,
            "Frequency round-trip failed: {} vs {}",
            freq,
            freq_back
        );
    }

    #[test]
    fn test_cwt_synthetic_sine() {
        let mut wavelet = MorletWavelet::new(MorletConfig {
            omega0: 6.0,
            sample_rate: 100.0, // Higher sample rate for this test
        });

        // Generate a 5 Hz sine wave
        let duration = 2.0; // 2 seconds
        let fs = 100.0;
        let n = (duration * fs) as usize;
        let target_freq = 5.0;

        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                (2.0 * PI * target_freq * t).sin()
            })
            .collect();

        // Test at the correct scale
        let scale = wavelet.frequency_to_scale(target_freq);
        let coeffs = wavelet.cwt_at_scale(&signal, scale);

        // The middle of the signal should have high amplitude
        let mid = n / 2;
        let amplitude = coeffs[mid].norm();

        assert!(
            amplitude > 0.1,
            "CWT amplitude at target frequency should be significant: {}",
            amplitude
        );
    }

    #[test]
    fn test_extract_bands() {
        let mut wavelet = MorletWavelet::default_for_zenb();

        // Generate a simple signal with theta-band oscillation (6 Hz)
        let n = 60; // 2 seconds at 30 Hz
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 30.0;
                (2.0 * PI * 6.0 * t).sin()
            })
            .collect();

        let bands = wavelet.extract_bands(&signal, 0);

        assert_eq!(bands.delta.len(), n);
        assert_eq!(bands.theta.len(), n);
        assert_eq!(bands.alpha.len(), n);
        assert_eq!(bands.timestamps_us.len(), n);

        // Theta should dominate for 6 Hz signal
        let dominant = bands.dominant_band();
        assert_eq!(
            dominant,
            BandType::Theta,
            "6 Hz signal should be in theta band"
        );
    }

    #[test]
    fn test_phase_coherence() {
        // Perfectly coherent signal (constant phase)
        let coherent: Vec<Complex32> = (0..10)
            .map(|_| Complex32::from_polar(1.0, PI / 4.0))
            .collect();

        let bands = WaveletBands {
            delta: coherent.clone(),
            theta: vec![],
            alpha: vec![],
            timestamps_us: vec![],
        };

        let coherence = bands.phase_coherence(BandType::Delta);
        assert!(
            coherence > 0.99,
            "Constant phase should have high coherence: {}",
            coherence
        );

        // Random phases should have low coherence
        let incoherent: Vec<Complex32> = (0..100)
            .map(|i| Complex32::from_polar(1.0, (i as f32) * 0.7))
            .collect();

        let bands_incoherent = WaveletBands {
            delta: incoherent,
            theta: vec![],
            alpha: vec![],
            timestamps_us: vec![],
        };

        let low_coherence = bands_incoherent.phase_coherence(BandType::Delta);
        assert!(
            low_coherence < 0.5,
            "Random phases should have low coherence: {}",
            low_coherence
        );
    }

    #[test]
    fn test_phasor_conversion() {
        let c = Complex32::from_polar(2.0, PI / 3.0);
        let (amp, phase) = MorletWavelet::to_phasor(c);

        assert!((amp - 2.0).abs() < 0.001);
        assert!((phase - PI / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_instantaneous_frequency() {
        // Constant frequency should give constant IF
        let fs = 100.0;
        let freq = 5.0;
        let n = 100;

        let phases: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                2.0 * PI * freq * t
            })
            .collect();

        let if_values = MorletWavelet::instantaneous_frequency(&phases, fs);

        // All values should be close to 5 Hz
        for &f in &if_values {
            assert!(
                (f - freq).abs() < 0.1,
                "Instantaneous frequency should be ~5 Hz: {}",
                f
            );
        }
    }
}
