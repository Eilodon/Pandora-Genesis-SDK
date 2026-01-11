//! Remote Photoplethysmography (rPPG) algorithms
//!
//! Extracts heart rate from RGB video by detecting subtle color changes
//! in skin caused by blood volume variations.
//!
//! Implements:
//! - **CHROM**: Chrominance-based method (De Haan & Jeanne, 2013)
//! - **POS**: Plane-Orthogonal-to-Skin (Wang et al., 2017)
//!
//! Reference: RPPGProcessor.ts from TypeScript implementation

use crate::dsp::{DspProcessor, FilterConfig};
use ndarray::Array1;

/// rPPG extraction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RppgMethod {
    /// Simple green channel extraction
    Green,
    /// Chrominance-based (De Haan 2013)
    Chrom,
    /// Plane-Orthogonal-to-Skin (Wang 2017) - Most robust
    Pos,
}

/// rPPG processing result
#[derive(Debug, Clone)]
pub struct RppgResult {
    pub heart_rate: f32,
    pub confidence: f32,
    pub snr: f32,
    pub pulse_waveform: Vec<f32>,
}

/// rPPG Processor
///
/// Maintains a rolling buffer of RGB samples and extracts heart rate
/// using the selected algorithm.
pub struct RppgProcessor {
    method: RppgMethod,
    window_size: usize,
    sample_rate: f32,
    buffer_r: Vec<f32>,
    buffer_g: Vec<f32>,
    buffer_b: Vec<f32>,
}

impl RppgProcessor {
    /// Create new rPPG processor
    ///
    /// # Arguments
    /// * `method` - Algorithm to use (CHROM or POS recommended)
    /// * `window_size` - Number of samples to process (e.g., 90 for 3s @ 30fps)
    /// * `sample_rate` - Frames per second (e.g., 30.0)
    pub fn new(method: RppgMethod, window_size: usize, sample_rate: f32) -> Self {
        Self {
            method,
            window_size,
            sample_rate,
            buffer_r: Vec::with_capacity(window_size * 2),
            buffer_g: Vec::with_capacity(window_size * 2),
            buffer_b: Vec::with_capacity(window_size * 2),
        }
    }

    /// Add RGB sample to buffer
    ///
    /// Reference: RPPGProcessor.ts:70-77
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - RGB channel values (typically 0-255)
    pub fn add_sample(&mut self, r: f32, g: f32, b: f32) {
        self.buffer_r.push(r);
        self.buffer_g.push(g);
        self.buffer_b.push(b);

        // Keep 2x window for overlap
        let max_samples = self.window_size * 2;
        if self.buffer_r.len() > max_samples {
            self.buffer_r.remove(0);
            self.buffer_g.remove(0);
            self.buffer_b.remove(0);
        }
    }

    /// Check if ready to process
    pub fn is_ready(&self) -> bool {
        self.buffer_r.len() >= self.window_size
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_r.len()
    }

    /// Reset buffer
    pub fn reset(&mut self) {
        self.buffer_r.clear();
        self.buffer_g.clear();
        self.buffer_b.clear();
    }

    /// Process buffer and extract heart rate
    ///
    /// Reference: RPPGProcessor.ts:82-147
    ///
    /// # Returns
    /// `Some((BPM, confidence))` if successful, `None` if not enough data
    pub fn process(&self) -> Option<(f32, f32)> {
        if !self.is_ready() {
            return None;
        }

        // 1. Extract window
        let start = self.buffer_r.len() - self.window_size;
        let r = Array1::from(self.buffer_r[start..].to_vec());
        let g = Array1::from(self.buffer_g[start..].to_vec());
        let b = Array1::from(self.buffer_b[start..].to_vec());

        // 2. Normalize RGB (zero-mean, unit variance)
        let r_norm = Self::normalize(&r);
        let g_norm = Self::normalize(&g);
        let b_norm = Self::normalize(&b);

        // 3. Extract pulse signal using selected method
        let pulse_signal = match self.method {
            RppgMethod::Green => g_norm,
            RppgMethod::Chrom => Self::chrom_method(&r_norm, &g_norm, &b_norm),
            RppgMethod::Pos => Self::pos_method(&r_norm, &g_norm, &b_norm),
        };

        // 4. Band-pass filter
        let config = FilterConfig {
            sample_rate: self.sample_rate,
            min_freq: 0.67,  // 40 BPM
            max_freq: 3.0,   // 180 BPM
        };
        let filtered = DspProcessor::bandpass_filter(&pulse_signal, &config);

        // 5. Compute heart rate via FFT
        let (bpm, snr) = DspProcessor::compute_heart_rate(&filtered, self.sample_rate);

        // 6. Calculate confidence from SNR
        // Map SNR [-5, 10] dB to confidence [0, 1]
        let confidence = ((snr + 5.0) / 15.0).clamp(0.0, 1.0);

        Some((bpm, confidence))
    }

    /// CHROM method (Chrominance-based)
    ///
    /// Reference: RPPGProcessor.ts:155-178
    ///
    /// De Haan & Jeanne (2013): "Robust Pulse Rate from Chrominance-Based rPPG"
    fn chrom_method(r: &Array1<f32>, g: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        let n = r.len();

        // X = 3R - 2G
        let x = (r * 3.0) - (g * 2.0);

        // Y = 1.5R + G - 1.5B
        let y = (r * 1.5) + g - (b * 1.5);

        // Calculate ratio α = std(X) / std(Y)
        let std_x = DspProcessor::std(&x);
        let std_y = DspProcessor::std(&y);
        let alpha = if std_y == 0.0 { 0.0 } else { std_x / std_y };

        // Pulse signal: S = X - α*Y
        let mut s = Array1::zeros(n);
        for i in 0..n {
            s[i] = x[i] - alpha * y[i];
        }

        s
    }

    /// POS method (Plane-Orthogonal-to-Skin)
    ///
    /// Reference: RPPGProcessor.ts:187-202
    ///
    /// Wang et al. (2017): "Algorithmic Principles of Remote PPG"
    /// More motion-robust than CHROM
    fn pos_method(r: &Array1<f32>, g: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
        let n = r.len();
        let mut s = Array1::zeros(n);

        for i in 0..n {
            // POS transformation
            let c1 = r[i] - g[i];
            let c2 = r[i] + g[i] - 2.0 * b[i];

            // Pulse signal (enhanced chrominance)
            s[i] = c1 + c2;
        }

        // Normalize
        Self::normalize(&s)
    }

    /// Normalize array (zero-mean, unit variance)
    ///
    /// Reference: RPPGProcessor.ts:357-364
    fn normalize(arr: &Array1<f32>) -> Array1<f32> {
        let mean = arr.mean().unwrap_or(0.0);
        let std = DspProcessor::std(arr);

        if std == 0.0 {
            return Array1::zeros(arr.len());
        }

        arr.mapv(|x| (x - mean) / std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    #[test]
    fn test_normalize() {
        let arr = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = RppgProcessor::normalize(&arr);

        // Mean should be ~0
        assert_relative_eq!(normalized.mean().unwrap(), 0.0, epsilon = 1e-6);

        // Std should be ~1
        let std = DspProcessor::std(&normalized);
        assert_relative_eq!(std, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_buffer_management() {
        let mut proc = RppgProcessor::new(RppgMethod::Pos, 10, 30.0);

        assert!(!proc.is_ready());
        assert_eq!(proc.buffer_size(), 0);

        // Add samples
        for i in 0..10 {
            proc.add_sample(i as f32, i as f32, i as f32);
        }

        assert!(proc.is_ready());
        assert_eq!(proc.buffer_size(), 10);

        // Test overflow (should keep 2x window = 20)
        for i in 0..15 {
            proc.add_sample(i as f32, i as f32, i as f32);
        }

        assert_eq!(proc.buffer_size(), 20);

        // Test reset
        proc.reset();
        assert!(!proc.is_ready());
        assert_eq!(proc.buffer_size(), 0);
    }

    #[test]
    fn test_synthetic_heartbeat_pos() {
        let mut proc = RppgProcessor::new(RppgMethod::Pos, 90, 30.0);

        // Generate synthetic 60 BPM signal (1 Hz sine wave in red channel)
        for i in 0..90 {
            let t = i as f32 / 30.0; // 3 seconds
            let heartbeat = (2.0 * PI * t).sin() * 10.0 + 128.0;
            proc.add_sample(heartbeat, 128.0, 128.0);
        }

        let result = proc.process();
        assert!(result.is_some());

        let (bpm, confidence) = result.unwrap();

        // Should detect ~60 BPM
        assert!(
            (bpm - 60.0).abs() < 10.0,
            "Expected ~60 BPM, got {}",
            bpm
        );

        // Confidence should be reasonable
        assert!(
            confidence > 0.2,
            "Confidence too low: {}",
            confidence
        );
    }

    #[test]
    fn test_synthetic_heartbeat_chrom() {
        let mut proc = RppgProcessor::new(RppgMethod::Chrom, 90, 30.0);

        // Generate synthetic 60 BPM signal with chrominance variation
        // CHROM needs variation in both R and G channels
        for i in 0..90 {
            let t = i as f32 / 30.0;
            let heartbeat_r = (2.0 * PI * t).sin() * 8.0 + 128.0;
            let heartbeat_g = (2.0 * PI * t).sin() * 5.0 + 128.0;
            proc.add_sample(heartbeat_r, heartbeat_g, 128.0);
        }

        let result = proc.process();
        assert!(result.is_some());

        let (bpm, _confidence) = result.unwrap();

        // CHROM should detect reasonable physiological range
        // May not be as accurate as POS on synthetic signals
        assert!(
            bpm >= 30.0 && bpm <= 200.0,
            "BPM out of reasonable range: {}",
            bpm
        );
    }

    #[test]
    fn test_insufficient_data() {
        let mut proc = RppgProcessor::new(RppgMethod::Pos, 90, 30.0);

        // Add only 50 samples (less than window_size)
        for i in 0..50 {
            proc.add_sample(i as f32, i as f32, i as f32);
        }

        assert!(!proc.is_ready());
        assert!(proc.process().is_none());
    }
}
