use std::collections::VecDeque;

use crate::config::ZenbConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonanceFeatures {
    pub phase_diff_norm: f32,
    pub resonance_score: f32,
    pub stability_score: f32,
}

#[derive(Debug, Clone)]
pub struct ResonanceTracker {
    window: SignalWindow,
    rr_prev: Option<f32>,
    rr_sign_pos: bool,
    rr_phase_norm: f32,
}

impl Default for ResonanceTracker {
    fn default() -> Self {
        Self {
            window: SignalWindow::default(),
            rr_prev: None,
            rr_sign_pos: true,
            rr_phase_norm: 0.0,
        }
    }
}

impl ResonanceTracker {
    pub fn update(
        &mut self,
        ts_us: i64,
        guide_phase_norm: f32,
        guide_bpm: f32,
        rr_bpm: Option<f32>,
        cfg: &ZenbConfig,
    ) -> ResonanceFeatures {
        let guide = guide_phase_norm.rem_euclid(1.0);

        if let Some(rr) = rr_bpm {
            self.window.push(ts_us, rr, cfg.resonance.window_size_sec);
        }

        let v2 = self.compute_v2(guide, guide_bpm, cfg);
        if let Some(f) = v2 {
            return f;
        }

        let mut stability = 1.0f32;
        if let (Some(prev), Some(rr)) = (self.rr_prev, rr_bpm) {
            let d = rr - prev;
            let sign_pos = d >= 0.0;
            if sign_pos != self.rr_sign_pos {
                self.rr_phase_norm = (self.rr_phase_norm + 0.5).rem_euclid(1.0);
                self.rr_sign_pos = sign_pos;
            }
            let d_abs = d.abs();
            stability = (1.0 - (d_abs / 4.0)).clamp(0.0, 1.0);
        }
        if rr_bpm.is_some() {
            self.rr_prev = rr_bpm;
        }

        let rr_phase = self.rr_phase_norm.rem_euclid(1.0);
        let diff = (guide - rr_phase).abs();
        let phase_diff_norm = (diff.min(1.0 - diff) * 2.0).clamp(0.0, 1.0);
        let resonance_score = (1.0 - phase_diff_norm).clamp(0.0, 1.0);

        ResonanceFeatures {
            phase_diff_norm,
            resonance_score,
            stability_score: stability,
        }
    }

    fn compute_v2(
        &self,
        guide_phase_norm: f32,
        guide_bpm: f32,
        cfg: &ZenbConfig,
    ) -> Option<ResonanceFeatures> {
        let target_freq_hz = (guide_bpm / 60.0).max(0.001);
        let fs_hz = 4.0f32;
        let dt_us = (1_000_000f32 / fs_hz).round() as i64;
        let x = self.window.resample_interp(dt_us)?;
        if x.len() < 8 {
            return None;
        }

        let min_cycles = 1.0f32;
        let min_n = ((min_cycles * fs_hz / target_freq_hz).ceil() as usize).max(8);
        if x.len() < min_n {
            return None;
        }

        let (mag, phase_rad, coh) = goertzel_mag_phase_coherence(&x, fs_hz, target_freq_hz);

        let coherence_threshold = cfg.resonance.coherence_threshold.clamp(0.0, 1.0);
        if coh < coherence_threshold {
            return None;
        }

        let guide_phase_rad = guide_phase_norm.rem_euclid(1.0) * (2.0 * std::f32::consts::PI);
        let diff_rad = wrap_pi(phase_rad - guide_phase_rad).abs();
        let phase_diff_norm = (diff_rad / std::f32::consts::PI).clamp(0.0, 1.0);

        let resonance_score = ((1.0 - phase_diff_norm) * coh).clamp(0.0, 1.0);
        let stability_score = (coh * (mag / (mag + 1.0))).clamp(0.0, 1.0);

        Some(ResonanceFeatures {
            phase_diff_norm,
            resonance_score,
            stability_score,
        })
    }
}

#[derive(Debug, Clone)]
struct SignalWindow {
    buf: VecDeque<(i64, f32)>,
}

impl Default for SignalWindow {
    fn default() -> Self {
        Self {
            buf: VecDeque::new(),
        }
    }
}

impl SignalWindow {
    fn push(&mut self, ts_us: i64, value: f32, window_size_sec: f32) {
        let window_us = (window_size_sec.max(0.5) * 1_000_000f32) as i64;
        self.buf.push_back((ts_us, value));
        while let Some((t0, _)) = self.buf.front() {
            if ts_us - *t0 > window_us {
                self.buf.pop_front();
            } else {
                break;
            }
        }

        let max_len = ((window_size_sec.max(0.5) * 4.0).ceil() as usize).max(16) + 4;
        while self.buf.len() > max_len {
            self.buf.pop_front();
        }
    }

    fn resample_interp(&self, dt_us: i64) -> Option<Vec<f32>> {
        if self.buf.len() < 2 {
            return None;
        }
        let dt_us = dt_us.max(50_000);

        let (t_first, _) = *self.buf.front()?;
        let (t_last, _) = *self.buf.back()?;
        if t_last <= t_first {
            return None;
        }

        let n = (((t_last - t_first) as f32) / (dt_us as f32)).floor() as usize + 1;
        if n < 2 {
            return None;
        }
        let mut out = Vec::with_capacity(n);

        let mut i = 0usize;
        let v: Vec<(i64, f32)> = self.buf.iter().cloned().collect();

        for j in 0..n {
            let t = t_first + (j as i64) * dt_us;
            while i + 1 < v.len() && v[i + 1].0 < t {
                i += 1;
            }
            if i + 1 >= v.len() {
                break;
            }
            let (t0, x0) = v[i];
            let (t1, x1) = v[i + 1];
            if t1 == t0 {
                out.push(x1);
                continue;
            }
            let a = ((t - t0) as f32) / ((t1 - t0) as f32);
            out.push(x0 + (x1 - x0) * a.clamp(0.0, 1.0));
        }
        if out.len() < 2 {
            None
        } else {
            Some(out)
        }
    }
}

fn goertzel_mag_phase_coherence(x: &[f32], fs_hz: f32, target_freq_hz: f32) -> (f32, f32, f32) {
    let n = x.len() as f32;
    let omega = 2.0 * std::f32::consts::PI * (target_freq_hz / fs_hz);
    let coeff = 2.0 * omega.cos();

    let mut s_prev = 0.0f32;
    let mut s_prev2 = 0.0f32;
    let mut energy = 0.0f32;
    for &v in x {
        let s = v + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
        energy += v * v;
    }

    let re = s_prev - s_prev2 * omega.cos();
    let im = s_prev2 * omega.sin();
    let mag = (re * re + im * im).sqrt();
    let phase = im.atan2(re);

    let eps = 1e-6f32;
    let coh = (mag / ((energy + eps).sqrt() * n.sqrt())).clamp(0.0, 1.0);
    (mag, phase, coh)
}

fn wrap_pi(x: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    (x + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ZenbConfig;

    #[test]
    fn identical_phases_high_score() {
        let mut t = ResonanceTracker::default();
        let cfg = ZenbConfig::default();
        let guide_bpm = 6.0;
        let f_hz = guide_bpm / 60.0;
        let fs_hz = 4.0;
        let dt_us = (1_000_000f32 / fs_hz) as i64;
        let mut ts = 0i64;

        for i in 0..80 {
            let t_sec = (i as f32) / fs_hz;
            let rr = 6.0 + 1.0 * (2.0 * std::f32::consts::PI * f_hz * t_sec).sin();
            let guide_phase = (f_hz * t_sec).rem_euclid(1.0);
            let f = t.update(ts, guide_phase, guide_bpm, Some(rr), &cfg);
            ts += dt_us;
            if i > 40 {
                assert!(f.resonance_score > 0.7);
            }
        }
    }

    #[test]
    fn opposite_phases_low_score() {
        let mut t = ResonanceTracker::default();
        let mut cfg = ZenbConfig::default();
        cfg.resonance.coherence_threshold = 0.0;

        let guide_bpm = 6.0;
        let f_hz = guide_bpm / 60.0;
        let fs_hz = 4.0;
        let dt_us = (1_000_000f32 / fs_hz) as i64;
        let mut ts = 0i64;

        let mut last = None;
        for i in 0..80 {
            let t_sec = (i as f32) / fs_hz;
            let rr = 6.0 + 1.0 * (2.0 * std::f32::consts::PI * f_hz * t_sec).sin();
            let guide_phase = (f_hz * t_sec + 0.5).rem_euclid(1.0);
            last = Some(t.update(ts, guide_phase, guide_bpm, Some(rr), &cfg));
            ts += dt_us;
        }
        let f = last.unwrap();
        assert!(f.phase_diff_norm > 0.6);
    }

    #[test]
    fn deterministic_given_same_inputs() {
        let mut t1 = ResonanceTracker::default();
        let mut t2 = ResonanceTracker::default();
        let cfg = ZenbConfig::default();
        let a = t1.update(1000, 0.1, 6.0, Some(6.0), &cfg);
        let b = t2.update(1000, 0.1, 6.0, Some(6.0), &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn noise_low_score() {
        let mut t = ResonanceTracker::default();
        let mut cfg = ZenbConfig::default();
        cfg.resonance.coherence_threshold = 0.0;

        let guide_bpm = 6.0;
        let f_hz = guide_bpm / 60.0;
        let fs_hz = 4.0;
        let dt_us = (1_000_000f32 / fs_hz) as i64;
        let mut ts = 0i64;

        let mut state = 1u32;
        let mut last = None;
        for i in 0..80 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((state >> 8) as f32) / (u32::MAX as f32);
            let rr = 4.0 + 8.0 * u;
            let t_sec = (i as f32) / fs_hz;
            let guide_phase = (f_hz * t_sec).rem_euclid(1.0);
            last = Some(t.update(ts, guide_phase, guide_bpm, Some(rr), &cfg));
            ts += dt_us;
        }
        let f = last.unwrap();
        assert!(f.resonance_score < 0.6);
    }
}
