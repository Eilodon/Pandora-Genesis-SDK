
//! Heart-rate tracking (Kalman-style) for stabilizing BPM over time.
//!
//! This is intentionally lightweight and allocation-free.
//! Upstream can provide BPM + confidence from any rPPG algorithm (POS/CHROM/PRISM/Ensemble).

#[derive(Debug, Clone)]
pub struct HrTrackerConfig {
    /// Base measurement variance in (BPM^2) at confidence=1.0.
    pub meas_var_base: f32,
    /// Process variance for HR (BPM^2 per second).
    pub process_var_hr: f32,
    /// Process variance for dHR (BPM^2 per second).
    pub process_var_dhr: f32,
    /// Hard gate on implausible jumps (BPM).
    pub max_jump_bpm: f32,
    /// Innovation gate in "sigma" units (e.g., 3.0).
    pub gate_sigma: f32,
    /// Minimum confidence required to accept a measurement update.
    pub min_meas_confidence: f32,
}

impl Default for HrTrackerConfig {
    fn default() -> Self {
        Self {
            meas_var_base: 4.0,      // ~2 BPM std at conf=1
            process_var_hr: 1.0,     // moderate drift
            process_var_dhr: 0.5,    // dHR changes slowly
            max_jump_bpm: 25.0,
            gate_sigma: 3.5,
            min_meas_confidence: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HrTracker {
    cfg: HrTrackerConfig,
    // State: [hr, dhr]
    x_hr: f32,
    x_dhr: f32,
    // Covariance P (2x2)
    p00: f32,
    p01: f32,
    p10: f32,
    p11: f32,
    initialized: bool,
}

#[derive(Debug, Clone)]
pub struct HrTrackedValue {
    pub bpm: f32,
    /// Tracker's internal stability confidence (0..1).
    pub stability_confidence: f32,
    /// Whether the last measurement was accepted.
    pub accepted_measurement: bool,
}

impl HrTracker {
    pub fn new() -> Self {
        Self::with_config(HrTrackerConfig::default())
    }

    pub fn with_config(cfg: HrTrackerConfig) -> Self {
        Self {
            cfg,
            x_hr: 0.0,
            x_dhr: 0.0,
            // Start with high uncertainty (will collapse quickly after a few updates)
            p00: 100.0,
            p01: 0.0,
            p10: 0.0,
            p11: 25.0,
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::with_config(self.cfg.clone());
    }

    /// Update tracker with a new BPM measurement.
    ///
    /// - `bpm_meas`: measured HR (BPM)
    /// - `confidence`: 0..1 (higher = more reliable)
    /// - `dt_sec`: time since last update in seconds
    pub fn update(&mut self, bpm_meas: f32, confidence: f32, dt_sec: f32) -> HrTrackedValue {
        let dt = dt_sec.max(1e-3).min(5.0);

        // Initialize quickly on first valid measurement
        if !self.initialized && bpm_meas.is_finite() && confidence >= self.cfg.min_meas_confidence {
            self.x_hr = bpm_meas;
            self.x_dhr = 0.0;
            self.p00 = 25.0;
            self.p01 = 0.0;
            self.p10 = 0.0;
            self.p11 = 9.0;
            self.initialized = true;

            return HrTrackedValue {
                bpm: self.x_hr,
                stability_confidence: 0.5,
                accepted_measurement: true,
            };
        }

        // Predict: x = F x, P = F P F^T + Q
        // F = [[1, dt], [0, 1]]
        let x_hr_pred = self.x_hr + self.x_dhr * dt;
        let x_dhr_pred = self.x_dhr;

        let p00_pred = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11
            + self.cfg.process_var_hr * dt;
        let p01_pred = self.p01 + dt * self.p11;
        let p10_pred = self.p10 + dt * self.p11;
        let p11_pred = self.p11 + self.cfg.process_var_dhr * dt;

        // If measurement unusable, just accept prediction.
        if !bpm_meas.is_finite() || confidence < self.cfg.min_meas_confidence {
            self.x_hr = x_hr_pred;
            self.x_dhr = x_dhr_pred;
            self.p00 = p00_pred;
            self.p01 = p01_pred;
            self.p10 = p10_pred;
            self.p11 = p11_pred;

            let stability = Self::stability_from_sigma(p00_pred.sqrt());
            return HrTrackedValue {
                bpm: self.x_hr,
                stability_confidence: stability,
                accepted_measurement: false,
            };
        }

        // Hard jump gate
        let jump = (bpm_meas - x_hr_pred).abs();
        if jump > self.cfg.max_jump_bpm {
            self.x_hr = x_hr_pred;
            self.x_dhr = x_dhr_pred;
            self.p00 = p00_pred;
            self.p01 = p01_pred;
            self.p10 = p10_pred;
            self.p11 = p11_pred;

            let stability = Self::stability_from_sigma(p00_pred.sqrt());
            return HrTrackedValue {
                bpm: self.x_hr,
                stability_confidence: stability,
                accepted_measurement: false,
            };
        }

        // Measurement update (H = [1, 0])
        let r = self.cfg.meas_var_base / (confidence.max(1e-3) * confidence.max(1e-3));
        let y = bpm_meas - x_hr_pred; // innovation
        let s = p00_pred + r; // innovation covariance

        // Innovation gate (sigma)
        let sigma = s.sqrt().max(1e-6);
        if (y / sigma).abs() > self.cfg.gate_sigma {
            self.x_hr = x_hr_pred;
            self.x_dhr = x_dhr_pred;
            self.p00 = p00_pred;
            self.p01 = p01_pred;
            self.p10 = p10_pred;
            self.p11 = p11_pred;

            let stability = Self::stability_from_sigma(p00_pred.sqrt());
            return HrTrackedValue {
                bpm: self.x_hr,
                stability_confidence: stability,
                accepted_measurement: false,
            };
        }

        // Kalman gain K = P H^T / S = [p00_pred, p10_pred]^T / s
        let k0 = p00_pred / s;
        let k1 = p10_pred / s;

        // Update state
        self.x_hr = x_hr_pred + k0 * y;
        self.x_dhr = x_dhr_pred + k1 * y;

        // Update covariance: P = (I - K H) P
        // (I - K H) = [[1 - k0, 0], [-k1, 1]]
        self.p00 = (1.0 - k0) * p00_pred;
        self.p01 = (1.0 - k0) * p01_pred;
        self.p10 = p10_pred - k1 * p00_pred;
        self.p11 = p11_pred - k1 * p01_pred;

        let stability = Self::stability_from_sigma(self.p00.sqrt());
        HrTrackedValue {
            bpm: self.x_hr,
            stability_confidence: stability,
            accepted_measurement: true,
        }
    }

    #[inline]
    fn stability_from_sigma(sigma_bpm: f32) -> f32 {
        // sigma ~2 -> ~0.7 ; sigma ~5 -> ~0.5 ; sigma ~10 -> ~0.33
        1.0 / (1.0 + (sigma_bpm / 4.0).max(0.0))
    }

    pub fn state(&self) -> (f32, f32) {
        (self.x_hr, self.x_dhr)
    }
}
