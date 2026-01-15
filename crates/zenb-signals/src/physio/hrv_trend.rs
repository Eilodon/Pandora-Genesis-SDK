//! HRV Trend Tracker with Per-User Baseline
//!
//! Tracks HRV metrics over time to detect trends and deviations from baseline.
//! Supports personalized analysis by maintaining per-user baseline statistics.

use super::hrv::{HrvMetrics, HrvResult};

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
    Unknown,
}

/// HRV baseline statistics (per-user)
#[derive(Debug, Clone)]
pub struct HrvBaseline {
    /// Mean RMSSD (ms)
    pub rmssd_mean: f32,
    /// Standard deviation of RMSSD
    pub rmssd_std: f32,
    /// Mean SDNN (ms)
    pub sdnn_mean: f32,
    /// Standard deviation of SDNN
    pub sdnn_std: f32,
    /// Mean HR (bpm)
    pub hr_mean: f32,
    /// Number of samples used to compute baseline
    pub sample_count: usize,
    /// Minimum samples needed for valid baseline
    pub min_samples: usize,
}

impl Default for HrvBaseline {
    fn default() -> Self {
        Self {
            rmssd_mean: 35.0, // Population average
            rmssd_std: 15.0,
            sdnn_mean: 50.0,
            sdnn_std: 20.0,
            hr_mean: 70.0,
            sample_count: 0,
            min_samples: 10, // Need 10 measurements for reliable baseline
        }
    }
}

impl HrvBaseline {
    /// Check if baseline has enough data
    pub fn is_valid(&self) -> bool {
        self.sample_count >= self.min_samples
    }
    
    /// Compute z-score for RMSSD relative to baseline
    pub fn rmssd_z_score(&self, rmssd: f32) -> f32 {
        if self.rmssd_std > 0.0 {
            (rmssd - self.rmssd_mean) / self.rmssd_std
        } else {
            0.0
        }
    }
    
    /// Compute z-score for SDNN relative to baseline
    pub fn sdnn_z_score(&self, sdnn: f32) -> f32 {
        if self.sdnn_std > 0.0 {
            (sdnn - self.sdnn_mean) / self.sdnn_std
        } else {
            0.0
        }
    }
}

/// Configuration for HRV trend tracking
#[derive(Debug, Clone)]
pub struct HrvTrendConfig {
    /// Size of sliding window for trend detection
    pub window_size: usize,
    /// Minimum change to consider significant (as % of baseline)
    pub significance_threshold: f32,
    /// EMA alpha for baseline updates
    pub baseline_alpha: f32,
    /// Z-score threshold for anomaly detection
    pub anomaly_z_threshold: f32,
}

impl Default for HrvTrendConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            significance_threshold: 0.15, // 15% change
            baseline_alpha: 0.1, // Slow baseline adaptation
            anomaly_z_threshold: 2.0, // 2 std deviations
        }
    }
}

/// Result of trend analysis
#[derive(Debug, Clone)]
pub struct HrvTrendResult {
    /// Current RMSSD value
    pub rmssd: f32,
    /// Current SDNN value
    pub sdnn: f32,
    /// RMSSD trend direction
    pub rmssd_trend: TrendDirection,
    /// SDNN trend direction
    pub sdnn_trend: TrendDirection,
    /// RMSSD z-score relative to baseline
    pub rmssd_z_score: f32,
    /// SDNN z-score relative to baseline
    pub sdnn_z_score: f32,
    /// Whether current values are anomalous
    pub is_anomaly: bool,
    /// Recovery score (0-1, based on RMSSD relative to baseline)
    pub recovery_score: f32,
    /// Stress indicator (0-1, inverse of recovery)
    pub stress_indicator: f32,
}

/// HRV Trend Tracker with per-user baseline
#[derive(Debug, Clone)]
pub struct HrvTrendTracker {
    config: HrvTrendConfig,
    baseline: HrvBaseline,
    /// Rolling window of RMSSD values
    rmssd_window: Vec<f32>,
    /// Rolling window of SDNN values
    sdnn_window: Vec<f32>,
    /// Sum for running mean (RMSSD)
    rmssd_sum: f32,
    /// Sum of squares for running std (RMSSD)
    rmssd_sq_sum: f32,
    /// Sum for running mean (SDNN)
    sdnn_sum: f32,
    /// Sum of squares for running std (SDNN)
    sdnn_sq_sum: f32,
    /// Sum for running mean (HR)
    hr_sum: f32,
    /// Total samples seen for baseline computation
    total_samples: usize,
}

impl HrvTrendTracker {
    /// Create new tracker with default config
    pub fn new() -> Self {
        Self::with_config(HrvTrendConfig::default())
    }
    
    /// Create tracker with custom config
    pub fn with_config(config: HrvTrendConfig) -> Self {
        Self {
            config,
            baseline: HrvBaseline::default(),
            rmssd_window: Vec::new(),
            sdnn_window: Vec::new(),
            rmssd_sum: 0.0,
            rmssd_sq_sum: 0.0,
            sdnn_sum: 0.0,
            sdnn_sq_sum: 0.0,
            hr_sum: 0.0,
            total_samples: 0,
        }
    }
    
    /// Create tracker with pre-loaded baseline (e.g., from storage)
    pub fn with_baseline(baseline: HrvBaseline) -> Self {
        let mut tracker = Self::new();
        let sample_count = baseline.sample_count;
        tracker.baseline = baseline;
        tracker.total_samples = sample_count;
        tracker
    }
    
    /// Update tracker with new HRV metrics
    pub fn update(&mut self, metrics: &HrvMetrics) -> HrvTrendResult {
        let rmssd = metrics.rmssd_ms;
        let sdnn = metrics.sdnn_ms;
        let hr = metrics.mean_hr_bpm;
        
        // Update running statistics for baseline
        self.total_samples += 1;
        self.rmssd_sum += rmssd;
        self.rmssd_sq_sum += rmssd * rmssd;
        self.sdnn_sum += sdnn;
        self.sdnn_sq_sum += sdnn * sdnn;
        self.hr_sum += hr;
        
        // Update baseline
        self.update_baseline();
        
        // Update sliding windows
        self.rmssd_window.push(rmssd);
        self.sdnn_window.push(sdnn);
        
        if self.rmssd_window.len() > self.config.window_size {
            self.rmssd_window.remove(0);
        }
        if self.sdnn_window.len() > self.config.window_size {
            self.sdnn_window.remove(0);
        }
        
        // Compute trends
        let rmssd_trend = self.compute_trend(&self.rmssd_window, self.baseline.rmssd_mean);
        let sdnn_trend = self.compute_trend(&self.sdnn_window, self.baseline.sdnn_mean);
        
        // Compute z-scores
        let rmssd_z = self.baseline.rmssd_z_score(rmssd);
        let sdnn_z = self.baseline.sdnn_z_score(sdnn);
        
        // Anomaly detection
        let is_anomaly = rmssd_z.abs() > self.config.anomaly_z_threshold
            || sdnn_z.abs() > self.config.anomaly_z_threshold;
        
        // Recovery score (higher RMSSD = better recovery)
        let recovery_score = self.compute_recovery_score(rmssd);
        
        HrvTrendResult {
            rmssd,
            sdnn,
            rmssd_trend,
            sdnn_trend,
            rmssd_z_score: rmssd_z,
            sdnn_z_score: sdnn_z,
            is_anomaly,
            recovery_score,
            stress_indicator: 1.0 - recovery_score,
        }
    }
    
    /// Update with HrvResult (convenience)
    pub fn update_from_result(&mut self, result: &HrvResult) -> Option<HrvTrendResult> {
        result.metrics.as_ref().map(|m| self.update(m))
    }
    
    /// Get current baseline
    pub fn baseline(&self) -> &HrvBaseline {
        &self.baseline
    }
    
    /// Export baseline for persistence
    pub fn export_baseline(&self) -> HrvBaseline {
        self.baseline.clone()
    }
    
    /// Reset tracker state (keeps config)
    pub fn reset(&mut self) {
        self.baseline = HrvBaseline::default();
        self.rmssd_window.clear();
        self.sdnn_window.clear();
        self.rmssd_sum = 0.0;
        self.rmssd_sq_sum = 0.0;
        self.sdnn_sum = 0.0;
        self.sdnn_sq_sum = 0.0;
        self.hr_sum = 0.0;
        self.total_samples = 0;
    }
    
    // --- Private ---
    
    fn update_baseline(&mut self) {
        if self.total_samples >= self.baseline.min_samples {
            let n = self.total_samples as f32;
            
            // Compute mean
            let rmssd_mean = self.rmssd_sum / n;
            let sdnn_mean = self.sdnn_sum / n;
            let hr_mean = self.hr_sum / n;
            
            // Compute std (using variance = E[X^2] - E[X]^2)
            let rmssd_var = (self.rmssd_sq_sum / n) - (rmssd_mean * rmssd_mean);
            let sdnn_var = (self.sdnn_sq_sum / n) - (sdnn_mean * sdnn_mean);
            
            // EMA update for baseline
            let alpha = self.config.baseline_alpha;
            self.baseline.rmssd_mean = self.baseline.rmssd_mean * (1.0 - alpha) + rmssd_mean * alpha;
            self.baseline.rmssd_std = self.baseline.rmssd_std * (1.0 - alpha) + rmssd_var.sqrt().max(1.0) * alpha;
            self.baseline.sdnn_mean = self.baseline.sdnn_mean * (1.0 - alpha) + sdnn_mean * alpha;
            self.baseline.sdnn_std = self.baseline.sdnn_std * (1.0 - alpha) + sdnn_var.sqrt().max(1.0) * alpha;
            self.baseline.hr_mean = self.baseline.hr_mean * (1.0 - alpha) + hr_mean * alpha;
            self.baseline.sample_count = self.total_samples;
        }
    }
    
    fn compute_trend(&self, window: &[f32], baseline_mean: f32) -> TrendDirection {
        if window.len() < 3 {
            return TrendDirection::Unknown;
        }
        
        let n = window.len();
        let first_half: f32 = window[..n/2].iter().sum::<f32>() / (n/2) as f32;
        let second_half: f32 = window[n/2..].iter().sum::<f32>() / (n - n/2) as f32;
        
        let change_pct = (second_half - first_half) / baseline_mean.max(1.0);
        
        if change_pct > self.config.significance_threshold {
            TrendDirection::Increasing
        } else if change_pct < -self.config.significance_threshold {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
    
    fn compute_recovery_score(&self, rmssd: f32) -> f32 {
        // Map RMSSD to 0-1 recovery score
        // Below baseline mean = lower recovery
        // Above baseline mean = higher recovery
        let z = self.baseline.rmssd_z_score(rmssd);
        
        // Sigmoid-like mapping: z=0 -> 0.5, z=2 -> ~0.88, z=-2 -> ~0.12
        let raw = 1.0 / (1.0 + (-z).exp());
        raw.clamp(0.0, 1.0)
    }
}

impl Default for HrvTrendTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_baseline_validity() {
        let baseline = HrvBaseline::default();
        assert!(!baseline.is_valid()); // No samples yet
    }
    
    #[test]
    fn test_z_score() {
        let mut baseline = HrvBaseline::default();
        baseline.rmssd_mean = 40.0;
        baseline.rmssd_std = 10.0;
        
        assert!((baseline.rmssd_z_score(50.0) - 1.0).abs() < 0.01);
        assert!((baseline.rmssd_z_score(30.0) - (-1.0)).abs() < 0.01);
    }
    
    #[test]
    fn test_trend_tracking() {
        let mut tracker = HrvTrendTracker::new();
        
        // Simulate 15 samples to build baseline
        for i in 0..15 {
            let metrics = HrvMetrics {
                mean_hr_bpm: 70.0,
                mean_ibi_ms: 857.0,
                sdnn_ms: 50.0,
                rmssd_ms: 35.0 + (i as f32), // Slowly increasing
            };
            let result = tracker.update(&metrics);
            
            if i > 10 {
                // After enough samples, trend should be detected
                assert!(matches!(result.rmssd_trend, TrendDirection::Increasing | TrendDirection::Stable));
            }
        }
    }
    
    #[test]
    fn test_recovery_score() {
        let mut tracker = HrvTrendTracker::new();
        tracker.baseline.rmssd_mean = 40.0;
        tracker.baseline.rmssd_std = 10.0;
        
        // High RMSSD = good recovery
        let high_recovery = tracker.compute_recovery_score(60.0);
        let low_recovery = tracker.compute_recovery_score(20.0);
        
        assert!(high_recovery > 0.7);
        assert!(low_recovery < 0.3);
    }
}
