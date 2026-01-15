//! APON (Adaptive Plane-Orthogonal-to-Noise) Algorithm
//!
//! Estimates the dominant noise direction in RGB color space using PCA,
//! then provides a warm-start suggestion for PRISM's α parameter search.
//!
//! # Algorithm
//!
//! 1. Compute 3x3 covariance matrix of RGB signals
//! 2. Find eigenvector with largest eigenvalue (dominant noise direction)
//! 3. Project signal onto plane orthogonal to noise
//! 4. Suggest α range to warm-start PRISM grid search
//!
//! # Reference
//!
//! APON: Non-contact Pulse Rate Detection via Adaptive Projection Plane

use ndarray::Array1;

/// APON algorithm configuration
#[derive(Debug, Clone)]
pub struct AponConfig {
    /// Minimum variance ratio to consider noise direction valid
    pub min_variance_ratio: f32,
    /// Width of α search window around warm-start suggestion
    pub alpha_search_width: f32,
}

impl Default for AponConfig {
    fn default() -> Self {
        Self {
            min_variance_ratio: 0.5,   // Noise must explain >50% variance
            alpha_search_width: 0.2,    // ±0.1 around suggested α
        }
    }
}

/// APON noise estimation result
#[derive(Debug, Clone)]
pub struct AponResult {
    /// Estimated noise direction unit vector [r, g, b]
    pub noise_direction: [f32; 3],
    /// Variance explained by noise direction (0-1)
    pub variance_ratio: f32,
    /// Suggested α for PRISM (Signal = α·G + (1-α)·(R-B))
    pub suggested_alpha: f32,
    /// Suggested search range (α_min, α_max)
    pub alpha_range: (f32, f32),
    /// Whether the noise estimate is reliable
    pub is_reliable: bool,
}

/// APON Noise Estimator for warm-starting PRISM
pub struct AponNoiseEstimator {
    config: AponConfig,
}

impl AponNoiseEstimator {
    /// Create new estimator with default config
    pub fn new() -> Self {
        Self::with_config(AponConfig::default())
    }

    /// Create estimator with custom config
    pub fn with_config(config: AponConfig) -> Self {
        Self { config }
    }

    /// Estimate noise direction from RGB signals
    ///
    /// # Arguments
    /// * `r`, `g`, `b` - Normalized RGB channels (same length)
    ///
    /// # Returns
    /// AponResult with noise direction and suggested α
    pub fn estimate(
        &self,
        r: &Array1<f32>,
        g: &Array1<f32>,
        b: &Array1<f32>,
    ) -> AponResult {
        let n = r.len();
        if n < 10 || g.len() != n || b.len() != n {
            return self.fallback_result();
        }

        // 1. Compute means
        let r_mean = r.mean().unwrap_or(0.0);
        let g_mean = g.mean().unwrap_or(0.0);
        let b_mean = b.mean().unwrap_or(0.0);

        // 2. Compute 3x3 covariance matrix
        let mut cov = [[0.0f32; 3]; 3];
        for i in 0..n {
            let dr = r[i] - r_mean;
            let dg = g[i] - g_mean;
            let db = b[i] - b_mean;

            cov[0][0] += dr * dr;
            cov[0][1] += dr * dg;
            cov[0][2] += dr * db;
            cov[1][1] += dg * dg;
            cov[1][2] += dg * db;
            cov[2][2] += db * db;
        }

        // Symmetric
        cov[1][0] = cov[0][1];
        cov[2][0] = cov[0][2];
        cov[2][1] = cov[1][2];

        // Normalize
        let nf = n as f32;
        for row in &mut cov {
            for val in row {
                *val /= nf;
            }
        }

        // 3. Find dominant eigenvector using power iteration
        let (eigenvalue, eigenvector) = self.power_iteration(&cov, 20);

        // 4. Compute variance ratio
        let total_variance = cov[0][0] + cov[1][1] + cov[2][2];
        let variance_ratio = if total_variance > 1e-10 {
            (eigenvalue / total_variance).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let is_reliable = variance_ratio >= self.config.min_variance_ratio;

        // 5. Compute suggested α from noise direction
        // The PRISM signal is S = α·G + (1-α)·(R-B)
        // We want the signal orthogonal to noise, so:
        // α should minimize |noise · [-(1-α), α, (1-α)]|
        let suggested_alpha = self.compute_suggested_alpha(&eigenvector);

        // 6. Compute search range
        let half_width = self.config.alpha_search_width / 2.0;
        let alpha_min = (suggested_alpha - half_width).clamp(0.0, 1.0);
        let alpha_max = (suggested_alpha + half_width).clamp(0.0, 1.0);

        AponResult {
            noise_direction: eigenvector,
            variance_ratio,
            suggested_alpha,
            alpha_range: (alpha_min, alpha_max),
            is_reliable,
        }
    }

    /// Power iteration to find dominant eigenvector of 3x3 matrix
    fn power_iteration(&self, matrix: &[[f32; 3]; 3], iterations: usize) -> (f32, [f32; 3]) {
        // Start with initial guess
        let mut v = [1.0f32, 1.0, 1.0];
        let mut eigenvalue = 0.0f32;

        for _ in 0..iterations {
            // w = A * v
            let w = [
                matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2],
                matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2],
                matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2],
            ];

            // Compute norm
            let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
            if norm < 1e-10 {
                break;
            }

            eigenvalue = norm;

            // Normalize
            v = [w[0] / norm, w[1] / norm, w[2] / norm];
        }

        (eigenvalue, v)
    }

    /// Compute suggested α that makes PRISM signal orthogonal to noise
    fn compute_suggested_alpha(&self, noise: &[f32; 3]) -> f32 {
        // Signal direction as function of α:
        // s(α) = [-(1-α), α, (1-α)] = [-1+α, α, 1-α]
        //
        // Orthogonality: noise · s(α) = 0
        // noise[0]·(-1+α) + noise[1]·α + noise[2]·(1-α) = 0
        // -noise[0] + α·noise[0] + α·noise[1] + noise[2] - α·noise[2] = 0
        // α·(noise[0] + noise[1] - noise[2]) = noise[0] - noise[2]
        //
        // α = (noise[0] - noise[2]) / (noise[0] + noise[1] - noise[2])

        let numerator = noise[0] - noise[2];
        let denominator = noise[0] + noise[1] - noise[2];

        if denominator.abs() < 1e-6 {
            // Fall back to green-channel dominant
            0.7
        } else {
            (numerator / denominator).clamp(0.0, 1.0)
        }
    }

    /// Fallback result when estimation fails
    fn fallback_result(&self) -> AponResult {
        AponResult {
            noise_direction: [0.0, 0.0, 0.0],
            variance_ratio: 0.0,
            suggested_alpha: 0.5,
            alpha_range: (0.0, 1.0),
            is_reliable: false,
        }
    }
}

impl Default for AponNoiseEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apon_clean_signal() {
        let estimator = AponNoiseEstimator::new();

        // Clean pulse signal (green dominant)
        let n = 90;
        let r: Array1<f32> = (0..n)
            .map(|i| 128.0 + (i as f32 * 0.1).sin() * 5.0)
            .collect();
        let g: Array1<f32> = (0..n)
            .map(|i| 128.0 + (i as f32 * 0.1).sin() * 10.0)
            .collect();
        let b: Array1<f32> = (0..n)
            .map(|i| 128.0 + (i as f32 * 0.1).sin() * 3.0)
            .collect();

        let result = estimator.estimate(&r, &g, &b);

        assert!(result.variance_ratio > 0.5, "Should detect dominant direction");
        assert!(
            result.suggested_alpha >= 0.0 && result.suggested_alpha <= 1.0,
            "Alpha should be in valid range"
        );
    }

    #[test]
    fn test_apon_with_motion_noise() {
        let estimator = AponNoiseEstimator::new();

        // Signal with strong motion artifact (correlated across all channels)
        let n = 90;
        let motion: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.3).sin() * 20.0))
            .collect();

        let r: Array1<f32> = motion.iter().map(|m| 128.0 + m).collect();
        let g: Array1<f32> = motion.iter().map(|m| 128.0 + m).collect();
        let b: Array1<f32> = motion.iter().map(|m| 128.0 + m).collect();

        let result = estimator.estimate(&r, &g, &b);

        // Noise should be detected along [1,1,1] direction
        assert!(result.variance_ratio > 0.9, "Motion should dominate variance");
    }

    #[test]
    fn test_apon_insufficient_data() {
        let estimator = AponNoiseEstimator::new();

        let r = Array1::from(vec![1.0, 2.0, 3.0]);
        let g = Array1::from(vec![1.0, 2.0, 3.0]);
        let b = Array1::from(vec![1.0, 2.0, 3.0]);

        let result = estimator.estimate(&r, &g, &b);

        assert!(!result.is_reliable, "Short signals should not be reliable");
        assert_eq!(result.alpha_range, (0.0, 1.0), "Should use full search range");
    }

    #[test]
    fn test_alpha_range_bounds() {
        let estimator = AponNoiseEstimator::new();

        // Create signal where suggested α would be near boundary
        let n = 90;
        let r: Array1<f32> = (0..n).map(|_| 128.0).collect();
        let g: Array1<f32> = (0..n)
            .map(|i| 128.0 + (i as f32 * 0.1).sin() * 20.0)
            .collect();
        let b: Array1<f32> = (0..n).map(|_| 128.0).collect();

        let result = estimator.estimate(&r, &g, &b);

        assert!(result.alpha_range.0 >= 0.0, "Min should be >= 0");
        assert!(result.alpha_range.1 <= 1.0, "Max should be <= 1");
        assert!(
            result.alpha_range.0 <= result.alpha_range.1,
            "Min should be <= Max"
        );
    }
}
