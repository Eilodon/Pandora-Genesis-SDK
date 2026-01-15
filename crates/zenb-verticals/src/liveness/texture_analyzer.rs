//! Texture Analysis for 3D vs 2D Detection
//!
//! Detects photo/screen attacks by analyzing:
//! - Color distribution
//! - Frequency content
//! - Specular highlights

/// Texture analysis result
#[derive(Debug, Clone, Default)]
pub struct TextureResult {
    pub is_3d: bool,
    pub confidence: f32,
    pub color_gamut: f32,
    pub frequency_score: f32,
    pub specular_score: f32,
}

/// Texture Analyzer
pub struct TextureAnalyzer {
    min_samples: usize,
}

impl TextureAnalyzer {
    pub fn new() -> Self {
        Self { min_samples: 30 }
    }

    /// Analyze texture from RGB history
    pub fn analyze(&self, rgb_history: &[[f32; 3]]) -> TextureResult {
        if rgb_history.len() < self.min_samples {
            return TextureResult::default();
        }

        let color_gamut = self.analyze_color_gamut(rgb_history);
        let frequency_score = self.analyze_frequency(rgb_history);
        let specular_score = self.analyze_specular(rgb_history);

        let combined = color_gamut * 0.3 + frequency_score * 0.4 + specular_score * 0.3;
        let is_3d = combined > 0.6;

        TextureResult {
            is_3d,
            confidence: combined,
            color_gamut,
            frequency_score,
            specular_score,
        }
    }

    fn analyze_color_gamut(&self, rgb_history: &[[f32; 3]]) -> f32 {
        let mut r_min = 255.0f32;
        let mut r_max = 0.0f32;
        let mut g_min = 255.0f32;
        let mut g_max = 0.0f32;
        let mut b_min = 255.0f32;
        let mut b_max = 0.0f32;

        for rgb in rgb_history {
            r_min = r_min.min(rgb[0]);
            r_max = r_max.max(rgb[0]);
            g_min = g_min.min(rgb[1]);
            g_max = g_max.max(rgb[1]);
            b_min = b_min.min(rgb[2]);
            b_max = b_max.max(rgb[2]);
        }

        let r_range = r_max - r_min;
        let g_range = g_max - g_min;
        let b_range = b_max - b_min;

        let expected_range = 20.0;
        let avg_range = (r_range + g_range + b_range) / 3.0;

        let score = 1.0 - ((avg_range - expected_range).abs() / expected_range).min(1.0);
        score.clamp(0.0, 1.0)
    }

    fn analyze_frequency(&self, rgb_history: &[[f32; 3]]) -> f32 {
        if rgb_history.len() < 10 {
            return 0.5;
        }

        let g: Vec<f32> = rgb_history.iter().map(|rgb| rgb[1]).collect();
        let mean: f32 = g.iter().sum::<f32>() / g.len() as f32;
        let centered: Vec<f32> = g.iter().map(|&v| v - mean).collect();

        let mut max_corr = 0.0f32;
        for lag in 2..g.len().min(20) {
            let mut corr = 0.0;
            for i in 0..g.len() - lag {
                corr += centered[i] * centered[i + lag];
            }
            corr /= (g.len() - lag) as f32;
            max_corr = max_corr.max(corr.abs());
        }

        let score = 1.0 - (max_corr / 100.0).min(1.0);
        score.clamp(0.0, 1.0)
    }

    fn analyze_specular(&self, rgb_history: &[[f32; 3]]) -> f32 {
        let brightness: Vec<f32> = rgb_history
            .iter()
            .map(|rgb| (rgb[0] + rgb[1] + rgb[2]) / 3.0)
            .collect();

        let mean: f32 = brightness.iter().sum::<f32>() / brightness.len() as f32;
        let std: f32 = (brightness
            .iter()
            .map(|&b| (b - mean).powi(2))
            .sum::<f32>()
            / brightness.len() as f32)
            .sqrt();

        let expected_std = 10.0;
        let score = 1.0 - ((std - expected_std).abs() / expected_std).min(1.0);
        score.clamp(0.0, 1.0)
    }
}

impl Default for TextureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
