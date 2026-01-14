
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct HrvConfig {
    /// Sample rate (Hz)
    pub sample_rate: f32,
    /// Minimum plausible IBI (ms)
    pub min_ibi_ms: f32,
    /// Maximum plausible IBI (ms)
    pub max_ibi_ms: f32,
    /// Minimum peak distance in seconds (refractory)
    pub min_peak_distance_sec: f32,
    /// Peak detection sensitivity (amplitude threshold = mean + k*std)
    pub peak_k_std: f32,
    /// Minimum valid beat ratio required to output metrics
    pub min_valid_ratio: f32,
}

impl Default for HrvConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            min_ibi_ms: 300.0,   // 200 BPM
            max_ibi_ms: 2000.0,  // 30 BPM
            min_peak_distance_sec: 0.30,
            peak_k_std: 0.5,
            min_valid_ratio: 0.70,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HrvMetrics {
    pub mean_hr_bpm: f32,
    pub mean_ibi_ms: f32,
    pub sdnn_ms: f32,
    pub rmssd_ms: f32,
}

#[derive(Debug, Clone)]
pub struct HrvResult {
    pub metrics: Option<HrvMetrics>,
    pub ibi_ms: Vec<f32>,
    pub valid_ratio: f32,
    pub beat_count: usize,
}

pub struct HrvEstimator {
    cfg: HrvConfig,
}

impl HrvEstimator {
    pub fn new() -> Self {
        Self::with_config(HrvConfig::default())
    }

    pub fn with_config(cfg: HrvConfig) -> Self {
        Self { cfg }
    }

    /// Estimate HRV from a pulse waveform (BVP) assumed to already be band-limited around HR.
    pub fn estimate(&self, pulse: &Array1<f32>) -> HrvResult {
        let fs = self.cfg.sample_rate.max(1e-3);
        let n = pulse.len();
        if n < (fs as usize * 5).max(64) {
            return HrvResult { metrics: None, ibi_ms: vec![], valid_ratio: 0.0, beat_count: 0 };
        }

        let mean = pulse.mean().unwrap_or(0.0);
        let var = pulse.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        let std = var.sqrt().max(1e-6);
        let thr = mean + self.cfg.peak_k_std * std;

        let min_dist = (self.cfg.min_peak_distance_sec * fs).round().max(1.0) as usize;

        // Simple peak picking (local maxima above threshold with refractory)
        let mut peaks: Vec<usize> = Vec::new();
        let mut last_peak: Option<usize> = None;

        for i in 1..(n.saturating_sub(1)) {
            if pulse[i] > thr && pulse[i] >= pulse[i-1] && pulse[i] >= pulse[i+1] {
                if let Some(lp) = last_peak {
                    if i - lp < min_dist { continue; }
                }
                peaks.push(i);
                last_peak = Some(i);
            }
        }

        if peaks.len() < 3 {
            return HrvResult { metrics: None, ibi_ms: vec![], valid_ratio: 0.0, beat_count: peaks.len() };
        }

        let mut ibi: Vec<f32> = Vec::with_capacity(peaks.len()-1);
        let mut valid = 0usize;

        for w in peaks.windows(2) {
            let dt = (w[1] - w[0]) as f32 / fs * 1000.0;
            ibi.push(dt);
            if dt >= self.cfg.min_ibi_ms && dt <= self.cfg.max_ibi_ms {
                valid += 1;
            }
        }

        let valid_ratio = valid as f32 / ibi.len().max(1) as f32;

        let metrics = if valid_ratio >= self.cfg.min_valid_ratio {
            let ibi_valid: Vec<f32> = ibi.iter().cloned().filter(|x| *x >= self.cfg.min_ibi_ms && *x <= self.cfg.max_ibi_ms).collect();
            if ibi_valid.len() >= 2 {
                let mean_ibi = ibi_valid.iter().sum::<f32>() / ibi_valid.len() as f32;
                let mean_hr = 60000.0 / mean_ibi.max(1e-3);

                // SDNN
                let var = ibi_valid.iter().map(|x| (x - mean_ibi).powi(2)).sum::<f32>() / (ibi_valid.len() as f32);
                let sdnn = var.sqrt();

                // RMSSD
                let mut diffsq_sum = 0.0f32;
                let mut cnt = 0usize;
                for w in ibi_valid.windows(2) {
                    let d = w[1] - w[0];
                    diffsq_sum += d * d;
                    cnt += 1;
                }
                let rmssd = if cnt > 0 { (diffsq_sum / cnt as f32).sqrt() } else { 0.0 };

                Some(HrvMetrics {
                    mean_hr_bpm: mean_hr,
                    mean_ibi_ms: mean_ibi,
                    sdnn_ms: sdnn,
                    rmssd_ms: rmssd,
                })
            } else {
                None
            }
        } else {
            None
        };

        HrvResult { metrics, ibi_ms: ibi, valid_ratio, beat_count: peaks.len() }
    }
}
