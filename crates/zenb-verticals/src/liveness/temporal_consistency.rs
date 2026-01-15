//! Temporal Consistency Analysis
//!
//! Detects video replay attacks by analyzing:
//! - Landmark motion patterns
//! - Frame-to-frame consistency
//! - Periodic artifacts (video loops)

/// Consistency check result
#[derive(Debug, Clone, Default)]
pub struct ConsistencyResult {
    /// Overall consistency score (0-1)
    pub score: f32,
    /// Motion is natural (not looped)
    pub natural_motion: bool,
    /// No periodic patterns detected
    pub no_loops: bool,
    /// Frame transitions are smooth
    pub smooth_transitions: bool,
}

/// Temporal Consistency Checker
pub struct TemporalConsistencyChecker {
    motion_history: Vec<f32>,
    window_size: usize,
}

impl TemporalConsistencyChecker {
    pub fn new() -> Self {
        Self {
            motion_history: Vec::with_capacity(300),
            window_size: 90,
        }
    }

    /// Check temporal consistency from landmark history
    pub fn check(&mut self, landmark_history: &[Vec<[f32; 2]>]) -> ConsistencyResult {
        if landmark_history.len() < 10 {
            return ConsistencyResult::default();
        }

        let mut motions = Vec::new();
        for i in 1..landmark_history.len() {
            let motion = self.compute_motion(&landmark_history[i - 1], &landmark_history[i]);
            motions.push(motion);
        }

        self.motion_history.extend(motions.iter());
        if self.motion_history.len() > 300 {
            self.motion_history
                .drain(0..self.motion_history.len() - 300);
        }

        let natural_motion = self.check_natural_motion(&motions);
        let no_loops = self.check_no_loops();
        let smooth_transitions = self.check_smooth_transitions(&motions);

        let score = (natural_motion as u8 as f32 * 0.4
            + no_loops as u8 as f32 * 0.3
            + smooth_transitions as u8 as f32 * 0.3)
            .clamp(0.0, 1.0);

        ConsistencyResult {
            score,
            natural_motion,
            no_loops,
            smooth_transitions,
        }
    }

    fn compute_motion(&self, prev: &[[f32; 2]], curr: &[[f32; 2]]) -> f32 {
        if prev.len() != curr.len() || prev.is_empty() {
            return 0.0;
        }

        let key_indices = [4, 133, 362, 152];
        let mut total = 0.0;
        let mut count = 0;

        for &idx in &key_indices {
            if idx < prev.len() && idx < curr.len() {
                let dx = curr[idx][0] - prev[idx][0];
                let dy = curr[idx][1] - prev[idx][1];
                total += (dx * dx + dy * dy).sqrt();
                count += 1;
            }
        }

        if count > 0 { total / count as f32 } else { 0.0 }
    }

    fn check_natural_motion(&self, motions: &[f32]) -> bool {
        if motions.len() < 5 {
            return true;
        }

        let mean: f32 = motions.iter().sum::<f32>() / motions.len() as f32;
        let variance: f32 = motions
            .iter()
            .map(|&m| (m - mean).powi(2))
            .sum::<f32>()
            / motions.len() as f32;

        variance > 0.00001
    }

    fn check_no_loops(&self) -> bool {
        if self.motion_history.len() < self.window_size {
            return true;
        }

        let recent = &self.motion_history[self.motion_history.len() - self.window_size..];
        let mean: f32 = recent.iter().sum::<f32>() / recent.len() as f32;

        let mut max_corr = 0.0f32;
        for lag in 10..recent.len() / 2 {
            let mut corr = 0.0;
            for i in 0..recent.len() - lag {
                corr += (recent[i] - mean) * (recent[i + lag] - mean);
            }
            corr /= (recent.len() - lag) as f32;
            max_corr = max_corr.max(corr.abs());
        }

        max_corr < 0.5
    }

    fn check_smooth_transitions(&self, motions: &[f32]) -> bool {
        if motions.len() < 3 {
            return true;
        }

        for i in 1..motions.len() {
            let diff = (motions[i] - motions[i - 1]).abs();
            if diff > 0.1 {
                return false;
            }
        }

        true
    }

    pub fn reset(&mut self) {
        self.motion_history.clear();
    }
}

impl Default for TemporalConsistencyChecker {
    fn default() -> Self {
        Self::new()
    }
}
