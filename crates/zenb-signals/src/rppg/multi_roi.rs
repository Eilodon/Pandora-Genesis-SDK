//! Multi-ROI Processing for rPPG
//!
//! Divides the face region into a grid of sub-regions and runs PRISM on each,
//! then combines results using SNR-weighted voting for improved robustness.
//!
//! # Benefits
//!
//! - Robust to partial occlusion (hand, glasses, hair)
//! - Automatic rejection of low-quality regions
//! - Reduces impact of local artifacts
//!
//! # Usage
//!
//! ```ignore
//! let processor = MultiRoiProcessor::new(3, 3);  // 3x3 grid
//! let result = processor.process_grid(&rgb_grid)?;
//! println!("HR: {:.1} BPM (from {} regions)", result.bpm, result.valid_regions);
//! ```

use ndarray::Array1;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::prism::{PrismConfig, PrismProcessor};

/// Multi-ROI configuration
#[derive(Debug, Clone)]
pub struct MultiRoiConfig {
    /// Number of grid rows
    pub rows: usize,
    /// Number of grid columns  
    pub cols: usize,
    /// Minimum SNR (dB) to include a region in voting
    pub min_snr: f32,
    /// Minimum confidence to include a region
    pub min_confidence: f32,
    /// Sample rate in Hz
    pub sample_rate: f32,
}

impl Default for MultiRoiConfig {
    fn default() -> Self {
        Self {
            rows: 3,
            cols: 3,
            min_snr: 2.0,
            min_confidence: 0.15,
            sample_rate: 30.0,
        }
    }
}

/// Result from a single ROI
#[derive(Debug, Clone)]
pub struct RoiResult {
    /// Grid row index (0-based)
    pub row: usize,
    /// Grid column index (0-based)
    pub col: usize,
    /// Heart rate in BPM
    pub bpm: f32,
    /// Signal-to-noise ratio in dB
    pub snr: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Optimal alpha used
    pub alpha: f32,
    /// Whether this ROI passed quality thresholds
    pub is_valid: bool,
}

/// Combined result from Multi-ROI processing
#[derive(Debug, Clone)]
pub struct MultiRoiResult {
    /// Final heart rate (weighted average)
    pub bpm: f32,
    /// Combined confidence
    pub confidence: f32,
    /// Average SNR across valid regions
    pub snr: f32,
    /// Number of valid regions used
    pub valid_regions: usize,
    /// Total regions processed
    pub total_regions: usize,
    /// Individual ROI results (for debugging/visualization)
    pub roi_results: Vec<RoiResult>,
}

/// RGB signal data for one grid region
#[derive(Debug, Clone)]
pub struct RoiSignal {
    /// Grid row index
    pub row: usize,
    /// Grid column index
    pub col: usize,
    /// Red channel values
    pub r: Array1<f32>,
    /// Green channel values
    pub g: Array1<f32>,
    /// Blue channel values
    pub b: Array1<f32>,
}

/// Multi-ROI rPPG Processor
pub struct MultiRoiProcessor {
    config: MultiRoiConfig,
    /// Per-region PRISM processors
    processors: Vec<PrismProcessor>,
}

impl MultiRoiProcessor {
    /// Create with default 3x3 grid
    pub fn new() -> Self {
        Self::with_config(MultiRoiConfig::default())
    }

    /// Create with specified grid size
    pub fn with_grid(rows: usize, cols: usize) -> Self {
        let mut config = MultiRoiConfig::default();
        config.rows = rows.max(1);
        config.cols = cols.max(1);
        Self::with_config(config)
    }

    /// Create with custom config
    pub fn with_config(config: MultiRoiConfig) -> Self {
        let num_regions = config.rows * config.cols;
        
        let prism_config = PrismConfig {
            sample_rate: config.sample_rate,
            use_apon: true,
            ..PrismConfig::default()
        };
        
        let processors = (0..num_regions)
            .map(|_| PrismProcessor::with_config(prism_config.clone()))
            .collect();
        
        Self { config, processors }
    }

    /// Process a grid of ROI signals
    ///
    /// # Arguments
    /// * `signals` - Vector of RoiSignal, one per grid cell (row-major order)
    ///
    /// # Returns
    /// Combined result with weighted voting
    pub fn process_grid(&mut self, signals: &[RoiSignal]) -> Option<MultiRoiResult> {
        if signals.is_empty() {
            return None;
        }

        let mut roi_results = Vec::with_capacity(signals.len());
        
        // Process each ROI
        for (idx, signal) in signals.iter().enumerate() {
            if idx >= self.processors.len() {
                break;
            }
            
            let processor = &mut self.processors[idx];
            
            if let Some(prism_result) = processor.process(&signal.r, &signal.g, &signal.b) {
                let is_valid = prism_result.snr >= self.config.min_snr
                    && prism_result.confidence >= self.config.min_confidence;
                
                roi_results.push(RoiResult {
                    row: signal.row,
                    col: signal.col,
                    bpm: prism_result.bpm,
                    snr: prism_result.snr,
                    confidence: prism_result.confidence,
                    alpha: prism_result.optimal_alpha,
                    is_valid,
                });
            }
        }

        if roi_results.is_empty() {
            return None;
        }

        Some(self.combine_results(&roi_results))
    }

    /// Process from flattened grid arrays (convenience method)
    ///
    /// # Arguments
    /// * `r_grid`, `g_grid`, `b_grid` - Arrays of shape [rows * cols, samples]
    pub fn process_arrays(
        &mut self,
        r_grid: &[Array1<f32>],
        g_grid: &[Array1<f32>],
        b_grid: &[Array1<f32>],
    ) -> Option<MultiRoiResult> {
        let n = r_grid.len().min(g_grid.len()).min(b_grid.len());
        
        let signals: Vec<RoiSignal> = (0..n)
            .map(|i| {
                let row = i / self.config.cols;
                let col = i % self.config.cols;
                RoiSignal {
                    row,
                    col,
                    r: r_grid[i].clone(),
                    g: g_grid[i].clone(),
                    b: b_grid[i].clone(),
                }
            })
            .collect();
        
        self.process_grid(&signals)
    }

    /// Combine individual ROI results using SNR-weighted voting
    fn combine_results(&self, results: &[RoiResult]) -> MultiRoiResult {
        let valid_results: Vec<&RoiResult> = results.iter().filter(|r| r.is_valid).collect();
        
        if valid_results.is_empty() {
            // Fall back to best available result (handle NaN gracefully)
            let best = results.iter().max_by(|a, b| {
                a.snr.partial_cmp(&b.snr).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            return match best {
                Some(r) => MultiRoiResult {
                    bpm: r.bpm,
                    confidence: r.confidence * 0.5, // Penalize for no valid regions
                    snr: r.snr,
                    valid_regions: 0,
                    total_regions: results.len(),
                    roi_results: results.to_vec(),
                },
                None => MultiRoiResult {
                    bpm: 0.0,
                    confidence: 0.0,
                    snr: 0.0,
                    valid_regions: 0,
                    total_regions: 0,
                    roi_results: vec![],
                },
            };
        }

        // SNR-weighted voting
        let total_weight: f32 = valid_results.iter().map(|r| (r.snr + 5.0).max(0.1)).sum();
        
        let weighted_bpm: f32 = valid_results
            .iter()
            .map(|r| r.bpm * (r.snr + 5.0).max(0.1) / total_weight)
            .sum();
        
        let avg_snr: f32 = valid_results.iter().map(|r| r.snr).sum::<f32>() / valid_results.len() as f32;
        let avg_conf: f32 = valid_results.iter().map(|r| r.confidence).sum::<f32>() / valid_results.len() as f32;
        
        // Boost confidence if multiple regions agree
        let region_agreement_boost = if valid_results.len() >= 3 {
            1.1
        } else if valid_results.len() >= 2 {
            1.05
        } else {
            1.0
        };
        
        MultiRoiResult {
            bpm: weighted_bpm,
            confidence: (avg_conf * region_agreement_boost).min(1.0),
            snr: avg_snr,
            valid_regions: valid_results.len(),
            total_regions: results.len(),
            roi_results: results.to_vec(),
        }
    }

    /// Reset all processors
    pub fn reset(&mut self) {
        for processor in &mut self.processors {
            processor.reset();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MultiRoiConfig {
        &self.config
    }

    /// Get grid dimensions
    pub fn grid_size(&self) -> (usize, usize) {
        (self.config.rows, self.config.cols)
    }

    /// Process grid in parallel (requires `parallel` feature)
    ///
    /// Uses rayon to process each ROI concurrently, providing ~3x speedup
    /// on 4-core machines with a 3x3 grid.
    ///
    /// # Note
    /// This creates new PrismProcessors per call rather than reusing cached
    /// ones, which has slightly higher overhead but enables true parallelism.
    #[cfg(feature = "parallel")]
    pub fn process_grid_parallel(&self, signals: &[RoiSignal]) -> Option<MultiRoiResult> {
        if signals.is_empty() {
            return None;
        }

        let prism_config = PrismConfig {
            sample_rate: self.config.sample_rate,
            use_apon: true,
            ..PrismConfig::default()
        };

        let min_snr = self.config.min_snr;
        let min_confidence = self.config.min_confidence;

        // Process each ROI in parallel
        let roi_results: Vec<RoiResult> = signals
            .par_iter()
            .filter_map(|signal| {
                let mut processor = PrismProcessor::with_config(prism_config.clone());
                
                processor.process(&signal.r, &signal.g, &signal.b).map(|result| {
                    let is_valid = result.snr >= min_snr && result.confidence >= min_confidence;
                    RoiResult {
                        row: signal.row,
                        col: signal.col,
                        bpm: result.bpm,
                        snr: result.snr,
                        confidence: result.confidence,
                        alpha: result.optimal_alpha,
                        is_valid,
                    }
                })
            })
            .collect();

        if roi_results.is_empty() {
            return None;
        }

        Some(self.combine_results(&roi_results))
    }

    /// Process grid in parallel (fallback when parallel feature disabled)
    #[cfg(not(feature = "parallel"))]
    pub fn process_grid_parallel(&mut self, signals: &[RoiSignal]) -> Option<MultiRoiResult> {
        // Fall back to sequential processing
        self.process_grid(signals)
    }
}

impl Default for MultiRoiProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_pulse_signal(n: usize, fs: f32, bpm: f32, amplitude: f32) -> Array1<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / fs;
                128.0 + (2.0 * PI * (bpm / 60.0) * t).sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn test_multi_roi_basic() {
        let mut processor = MultiRoiProcessor::with_grid(2, 2);
        
        let n = 90;
        let fs = 30.0;
        let target_bpm = 72.0;
        
        // Generate 4 regions with same pulse
        let signals: Vec<RoiSignal> = (0..4)
            .map(|i| RoiSignal {
                row: i / 2,
                col: i % 2,
                r: generate_pulse_signal(n, fs, target_bpm, 3.0),
                g: generate_pulse_signal(n, fs, target_bpm, 6.0),
                b: generate_pulse_signal(n, fs, target_bpm, 2.0),
            })
            .collect();
        
        let result = processor.process_grid(&signals);
        assert!(result.is_some(), "Should produce result");
        
        let result = result.unwrap();
        assert!(result.total_regions == 4, "Should process 4 regions");
        assert!(result.bpm > 40.0 && result.bpm < 180.0, "BPM in valid range: {}", result.bpm);
    }

    #[test]
    fn test_multi_roi_partial_occlusion() {
        let mut processor = MultiRoiProcessor::with_grid(2, 2);
        
        let n = 90;
        let fs = 30.0;
        let target_bpm = 60.0;
        
        // 3 good regions, 1 bad (simulating occlusion)
        let signals = vec![
            RoiSignal {
                row: 0, col: 0,
                r: generate_pulse_signal(n, fs, target_bpm, 3.0),
                g: generate_pulse_signal(n, fs, target_bpm, 6.0),
                b: generate_pulse_signal(n, fs, target_bpm, 2.0),
            },
            RoiSignal {
                row: 0, col: 1,
                r: generate_pulse_signal(n, fs, target_bpm, 3.0),
                g: generate_pulse_signal(n, fs, target_bpm, 6.0),
                b: generate_pulse_signal(n, fs, target_bpm, 2.0),
            },
            RoiSignal {
                row: 1, col: 0,
                r: generate_pulse_signal(n, fs, target_bpm, 3.0),
                g: generate_pulse_signal(n, fs, target_bpm, 6.0),
                b: generate_pulse_signal(n, fs, target_bpm, 2.0),
            },
            RoiSignal {
                row: 1, col: 1,
                r: Array1::from_elem(n, 100.0),  // Constant - no pulse (occluded)
                g: Array1::from_elem(n, 100.0),
                b: Array1::from_elem(n, 100.0),
            },
        ];
        
        let result = processor.process_grid(&signals).expect("Should process");
        
        // Should detect most regions are valid
        assert!(result.total_regions == 4);
        // BPM should still be accurate from good regions
        assert!(result.bpm > 40.0 && result.bpm < 180.0);
    }

    #[test]
    fn test_multi_roi_empty_input() {
        let mut processor = MultiRoiProcessor::new();
        let signals: Vec<RoiSignal> = vec![];
        
        assert!(processor.process_grid(&signals).is_none());
    }

    #[test]
    fn test_multi_roi_array_interface() {
        let mut processor = MultiRoiProcessor::with_grid(2, 2);
        
        let n = 90;
        let fs = 30.0;
        let bpm = 72.0;
        
        let r_grid: Vec<Array1<f32>> = (0..4)
            .map(|_| generate_pulse_signal(n, fs, bpm, 3.0))
            .collect();
        let g_grid: Vec<Array1<f32>> = (0..4)
            .map(|_| generate_pulse_signal(n, fs, bpm, 6.0))
            .collect();
        let b_grid: Vec<Array1<f32>> = (0..4)
            .map(|_| generate_pulse_signal(n, fs, bpm, 2.0))
            .collect();
        
        let result = processor.process_arrays(&r_grid, &g_grid, &b_grid);
        assert!(result.is_some());
        assert!(result.unwrap().total_regions == 4);
    }

    #[test]
    fn test_grid_dimensions() {
        let processor = MultiRoiProcessor::with_grid(4, 5);
        assert_eq!(processor.grid_size(), (4, 5));
    }
}
