
//! Multi-metric quality scoring for rPPG / BVP signals.
//!
//! The crate already provides `SignalQualityAnalyzer` (SNR + motion-heuristic).
//! This module adds a *composite* confidence that can incorporate external
//! motion / illumination metrics from upstream (e.g., landmark jitter, exposure drift).

use super::signal_quality::{SignalQuality, SignalQualityAnalyzer, SignalQualityConfig};
use ndarray::Array1;
use num_complex::Complex32;
use rustfft::FftPlanner;
use std::f32::consts::PI;

#[derive(Debug, Clone)]
pub struct QualityScorerConfig {
    pub sample_rate: f32,
    pub min_freq: f32,
    pub max_freq: f32,

    /// SNR midpoint (dB) where confidence ~0.5.
    pub snr_mid_db: f32,
    /// SNR steepness for logistic curve.
    pub snr_k: f32,

    /// Harmonic ratio target (power(2f0)/power(f0)) for "healthy-looking" pulse.
    pub harmonic_target: f32,
    /// Spread for harmonic log-distance penalty.
    pub harmonic_log_spread: f32,

    /// If upstream provides motion_score in [0..1] where 1=worst, scale penalty.
    pub motion_penalty: f32,
    /// If upstream provides illumination_score in [0..1] where 1=worst, scale penalty.
    pub illum_penalty: f32,

    /// Minimum final confidence to consider valid.
    pub min_confidence: f32,
    /// Minimum SNR to consider valid (dB).
    pub min_snr: f32,
}

impl Default for QualityScorerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_freq: 0.67,
            max_freq: 3.0,
            snr_mid_db: 5.0,
            snr_k: 0.6,
            harmonic_target: 0.20,
            harmonic_log_spread: 1.2,
            motion_penalty: 1.0,
            illum_penalty: 1.0,
            min_confidence: 0.15,
            min_snr: 2.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExternalQuality {
    /// 0..1 where 0 = no motion, 1 = extreme motion.
    pub motion_score: Option<f32>,
    /// 0..1 where 0 = stable lighting, 1 = extreme lighting drift/flicker.
    pub illumination_score: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct QualityScore {
    pub snr_db: f32,
    pub has_motion_artifact: bool,
    pub harmonic_ratio: f32,
    pub confidence: f32,
    pub is_valid: bool,
}

pub struct QualityScorer {
    cfg: QualityScorerConfig,
    snr_analyzer: SignalQualityAnalyzer,
    fft_planner: FftPlanner<f32>,
}

impl QualityScorer {
    pub fn new() -> Self {
        Self::with_config(QualityScorerConfig::default())
    }

    pub fn with_config(cfg: QualityScorerConfig) -> Self {
        let snr_cfg = SignalQualityConfig {
            sample_rate: cfg.sample_rate,
            min_freq: cfg.min_freq,
            max_freq: cfg.max_freq,
            motion_threshold: 3.0,
            min_snr: cfg.min_snr,
        };
        Self {
            cfg,
            snr_analyzer: SignalQualityAnalyzer::with_config(snr_cfg),
            fft_planner: FftPlanner::new(),
        }
    }

    pub fn analyze(&mut self, signal: &Array1<f32>, ext: ExternalQuality) -> QualityScore {
        let base: SignalQuality = self.snr_analyzer.analyze(signal);
        let harmonic_ratio = self.harmonic_ratio(signal);

        let snr_conf = 1.0 / (1.0 + (-self.cfg.snr_k * (base.snr - self.cfg.snr_mid_db)).exp());

        let motion_conf = ext
            .motion_score
            .map(|m| (1.0 - (m.clamp(0.0, 1.0) * self.cfg.motion_penalty).clamp(0.0, 1.0)))
            .unwrap_or(1.0);

        let illum_conf = ext
            .illumination_score
            .map(|l| (1.0 - (l.clamp(0.0, 1.0) * self.cfg.illum_penalty).clamp(0.0, 1.0)))
            .unwrap_or(1.0);

        let harm_conf = {
            let ratio = harmonic_ratio.max(1e-8);
            let target = self.cfg.harmonic_target.max(1e-8);
            let log_dist = (ratio.ln() - target.ln()).abs();
            (-log_dist / self.cfg.harmonic_log_spread.max(1e-6)).exp()
        };

        // Combine multiplicatively: any severe issue collapses confidence.
        let confidence = (snr_conf * motion_conf * illum_conf * harm_conf).clamp(0.0, 1.0);

        let is_valid = confidence >= self.cfg.min_confidence
            && base.snr >= self.cfg.min_snr
            && !base.has_motion_artifact;

        QualityScore {
            snr_db: base.snr,
            has_motion_artifact: base.has_motion_artifact,
            harmonic_ratio,
            confidence,
            is_valid,
        }
    }

    /// Compute power(2*f0) / power(f0) in the HR band as a simple harmonic consistency metric.
    fn harmonic_ratio(&mut self, signal: &Array1<f32>) -> f32 {
        let n = signal.len();
        if n < 64 {
            return 0.0;
        }

        let fs = self.cfg.sample_rate;

        // Hamming window
        let windowed: Vec<Complex32> = signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / (n - 1) as f32).cos();
                Complex32::new(s * w, 0.0)
            })
            .collect();

        let fft = self.fft_planner.plan_fft_forward(n);
        let mut buffer = windowed;
        fft.process(&mut buffer);

        let half_n = n / 2;
        let bin_res = fs / n as f32;
        let min_bin = (self.cfg.min_freq / bin_res) as usize;
        let max_bin = (self.cfg.max_freq / bin_res).min(half_n as f32) as usize;

        if min_bin >= max_bin || max_bin >= half_n {
            return 0.0;
        }

        // Find fundamental peak bin
        let mut peak_bin = min_bin;
        let mut peak_power = 0.0f32;
        for i in min_bin..=max_bin.min(half_n - 1) {
            let p = buffer[i].norm_sqr();
            if p > peak_power {
                peak_power = p;
                peak_bin = i;
            }
        }

        if peak_power <= 0.0 {
            return 0.0;
        }

        // 2nd harmonic bin near 2*f0
        let harm_bin = (2.0 * peak_bin as f32).round() as usize;
        if harm_bin >= half_n || harm_bin < min_bin {
            return 0.0;
        }

        let harm_power = buffer[harm_bin].norm_sqr();
        harm_power / peak_power
    }
}
