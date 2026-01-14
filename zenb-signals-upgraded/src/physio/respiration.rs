
use crate::dsp::{DspProcessor, FilterConfig};
use ndarray::Array1;

/// Respiration estimation configuration.
#[derive(Debug, Clone)]
pub struct RespirationConfig {
    pub sample_rate: f32,
    /// Resp band (Hz), default 0.1..0.5 (6..30 brpm)
    pub min_freq: f32,
    pub max_freq: f32,
}

impl Default for RespirationConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 0.10,
            max_freq: 0.50,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RespirationResult {
    pub brpm: f32,
    pub snr_db: f32,
    pub confidence: f32,
}

pub struct RespirationEstimator {
    cfg: RespirationConfig,
}

impl RespirationEstimator {
    pub fn new() -> Self {
        Self::with_config(RespirationConfig::default())
    }

    pub fn with_config(cfg: RespirationConfig) -> Self {
        Self { cfg }
    }

    /// Estimate respiration rate from a pulse waveform by extracting low-frequency modulation.
    ///
    /// Pipeline (simple & deployable):
    /// - Take absolute value (energy proxy) then lowpass/bandpass into resp band.
    /// - FFT peak detection.
    pub fn estimate(&self, pulse: &Array1<f32>) -> Option<RespirationResult> {
        let n = pulse.len();
        if n < 128 {
            return None;
        }

        // Energy envelope proxy (cheap alternative to Hilbert)
        let env = pulse.mapv(|x| x.abs());

        let cfg = FilterConfig {
            sample_rate: self.cfg.sample_rate,
            min_freq: self.cfg.min_freq,
            max_freq: self.cfg.max_freq,
        };

        let resp_band = DspProcessor::bandpass_filter(&env, &cfg);
        let (hz, snr_db) = DspProcessor::compute_peak_frequency(&resp_band, self.cfg.sample_rate, self.cfg.min_freq, self.cfg.max_freq);

        let brpm = hz * 60.0;
        let confidence = 1.0 / (1.0 + (-0.7 * (snr_db - 3.0)).exp());

        Some(RespirationResult { brpm, snr_db, confidence })
    }
}
