//! Video Pipeline
//!
//! Real-time video processing pipeline that orchestrates:
//! - Frame buffering with timestamps
//! - Face detection integration
//! - rPPG signal extraction
//! - Multi-ROI processing

use super::face_roi::{FaceDetection, FaceDetector, extract_roi_mean_rgb};
use super::image_ops::Frame;
use crate::rppg::{EnsembleProcessor, MultiRoiProcessor};

use std::collections::VecDeque;

/// Video pipeline configuration
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Expected frame rate (Hz)
    pub fps: f32,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Buffer size in frames
    pub buffer_size: usize,
    /// Enable multi-ROI processing
    pub use_multi_roi: bool,
    /// Grid size for multi-ROI (rows, cols)
    pub roi_grid: (usize, usize),
    /// Minimum detection confidence
    pub min_detection_confidence: f32,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            fps: 30.0,
            width: 640,
            height: 480,
            buffer_size: 150, // 5 seconds at 30 fps
            use_multi_roi: true,
            roi_grid: (3, 3),
            min_detection_confidence: 0.5,
        }
    }
}

/// Pipeline processing result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Heart rate in BPM (if available)
    pub heart_rate_bpm: Option<f32>,
    /// Signal quality / confidence (0-1)
    pub confidence: f32,
    /// Raw RGB signal from face ROI
    pub rgb_signal: [f32; 3],
    /// Face detected in this frame
    pub face_detected: bool,
    /// Face detection result
    pub face_detection: Option<FaceDetection>,
    /// Processing timestamp
    pub timestamp_us: i64,
    /// Current buffer fill ratio (0-1)
    pub buffer_fill: f32,
}

impl Default for PipelineResult {
    fn default() -> Self {
        Self {
            heart_rate_bpm: None,
            confidence: 0.0,
            rgb_signal: [0.0, 0.0, 0.0],
            face_detected: false,
            face_detection: None,
            timestamp_us: 0,
            buffer_fill: 0.0,
        }
    }
}

/// Frame with metadata in buffer
#[derive(Debug, Clone)]
#[allow(dead_code)] // timestamp_us and face_detected reserved for temporal analysis
struct BufferedFrame {
    rgb_mean: [f32; 3],
    timestamp_us: i64,
    face_detected: bool,
}

/// Video Processing Pipeline
///
/// Orchestrates frame processing from raw camera input to heart rate output.
pub struct VideoPipeline {
    config: VideoConfig,
    
    /// Ring buffer of processed frames
    frame_buffer: VecDeque<BufferedFrame>,
    
    /// Face detector (pluggable)
    face_detector: Option<Box<dyn FaceDetector>>,
    
    /// Last known face detection (for tracking)
    last_detection: Option<FaceDetection>,
    
    /// rPPG processor
    rppg_processor: EnsembleProcessor,
    
    /// Multi-ROI processor (optional)
    multi_roi_processor: Option<MultiRoiProcessor>,
    
    /// Frame counter
    frame_count: usize,
    
    /// Session start timestamp
    session_start_us: i64,
}

impl VideoPipeline {
    /// Create new pipeline with default config
    pub fn new() -> Self {
        Self::with_config(VideoConfig::default())
    }

    /// Create pipeline with custom config
    pub fn with_config(config: VideoConfig) -> Self {
        let multi_roi = if config.use_multi_roi {
            Some(MultiRoiProcessor::new())
        } else {
            None
        };

        Self {
            frame_buffer: VecDeque::with_capacity(config.buffer_size),
            face_detector: None,
            last_detection: None,
            rppg_processor: EnsembleProcessor::new(),
            multi_roi_processor: multi_roi,
            config,
            frame_count: 0,
            session_start_us: 0,
        }
    }

    /// Set custom face detector
    pub fn set_face_detector(&mut self, detector: Box<dyn FaceDetector>) {
        self.face_detector = Some(detector);
    }

    /// Process raw frame bytes (RGB888)
    ///
    /// # Arguments
    /// * `frame_data` - Raw RGB8 pixel data
    /// * `width` - Frame width (can differ from config for dynamic resize)
    /// * `height` - Frame height
    /// * `timestamp_us` - Frame capture timestamp in microseconds
    ///
    /// # Returns
    /// Processing result with heart rate if enough data collected
    pub fn process_raw(
        &mut self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        timestamp_us: i64,
    ) -> PipelineResult {
        if self.frame_count == 0 {
            self.session_start_us = timestamp_us;
        }
        self.frame_count += 1;

        // 1. Face detection (if detector available)
        let detection = if let Some(ref mut detector) = self.face_detector {
            detector.detect(frame_data, width, height, 3)
        } else {
            // Use last known detection or full-frame fallback
            self.last_detection.clone()
        };

        let face_detected = detection.as_ref()
            .map(|d| d.confidence >= self.config.min_detection_confidence)
            .unwrap_or(false);

        // Update last detection for tracking
        if face_detected {
            self.last_detection = detection.clone();
        }

        // 2. Extract ROI RGB signal
        let rgb_signal = if let Some(ref det) = detection {
            let [x, y, w, h] = det.bbox;
            extract_roi_mean_rgb(
                frame_data, width, height,
                x as u32, y as u32, w as u32, h as u32,
            )
        } else {
            // Fallback: center region
            let roi_w = width / 3;
            let roi_h = height / 3;
            let roi_x = width / 3;
            let roi_y = height / 3;
            extract_roi_mean_rgb(frame_data, width, height, roi_x, roi_y, roi_w, roi_h)
        };

        // 3. Add to buffer
        self.frame_buffer.push_back(BufferedFrame {
            rgb_mean: rgb_signal,
            timestamp_us,
            face_detected,
        });

        // Maintain buffer size
        while self.frame_buffer.len() > self.config.buffer_size {
            self.frame_buffer.pop_front();
        }

        // 4. Process rPPG if enough data
        let (heart_rate, confidence) = self.process_rppg();

        let buffer_fill = self.frame_buffer.len() as f32 / self.config.buffer_size as f32;

        PipelineResult {
            heart_rate_bpm: heart_rate,
            confidence,
            rgb_signal,
            face_detected,
            face_detection: detection,
            timestamp_us,
            buffer_fill,
        }
    }

    /// Process Frame struct
    pub fn process_frame(&mut self, frame: &Frame) -> PipelineResult {
        self.process_raw(&frame.data, frame.width, frame.height, frame.timestamp_us)
    }

    /// Inject external face detection (from mobile SDK)
    ///
    /// Use this when face detection is handled externally (MediaPipe, ARKit)
    pub fn inject_detection(&mut self, detection: FaceDetection) {
        self.last_detection = Some(detection);
    }

    /// Inject external landmarks and ROI colors
    ///
    /// Complete bypass of internal detection - client provides everything
    pub fn inject_signal(
        &mut self,
        rgb_mean: [f32; 3],
        timestamp_us: i64,
        face_detected: bool,
    ) -> PipelineResult {
        if self.frame_count == 0 {
            self.session_start_us = timestamp_us;
        }
        self.frame_count += 1;

        self.frame_buffer.push_back(BufferedFrame {
            rgb_mean,
            timestamp_us,
            face_detected,
        });

        while self.frame_buffer.len() > self.config.buffer_size {
            self.frame_buffer.pop_front();
        }

        let (heart_rate, confidence) = self.process_rppg();
        let buffer_fill = self.frame_buffer.len() as f32 / self.config.buffer_size as f32;

        PipelineResult {
            heart_rate_bpm: heart_rate,
            confidence,
            rgb_signal: rgb_mean,
            face_detected,
            face_detection: self.last_detection.clone(),
            timestamp_us,
            buffer_fill,
        }
    }

    fn process_rppg(&mut self) -> (Option<f32>, f32) {
        let min_samples = 90; // 3 seconds at 30fps
        if self.frame_buffer.len() < min_samples {
            return (None, 0.0);
        }

        // Extract RGB arrays
        let r: Vec<f32> = self.frame_buffer.iter().map(|f| f.rgb_mean[0]).collect();
        let g: Vec<f32> = self.frame_buffer.iter().map(|f| f.rgb_mean[1]).collect();
        let b: Vec<f32> = self.frame_buffer.iter().map(|f| f.rgb_mean[2]).collect();

        let r_arr = ndarray::Array1::from_vec(r);
        let g_arr = ndarray::Array1::from_vec(g);
        let b_arr = ndarray::Array1::from_vec(b);

        if let Some(result) = self.rppg_processor.process_arrays(&r_arr, &g_arr, &b_arr) {
            (Some(result.bpm), result.confidence)
        } else {
            (None, 0.0)
        }
    }

    /// Get current session duration in seconds
    pub fn session_duration_sec(&self, current_us: i64) -> f32 {
        (current_us - self.session_start_us) as f32 / 1_000_000.0
    }

    /// Get buffer fill percentage (0-100)
    pub fn buffer_fill_percent(&self) -> f32 {
        (self.frame_buffer.len() as f32 / self.config.buffer_size as f32) * 100.0
    }

    /// Get frames processed count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.last_detection = None;
        self.rppg_processor.reset();
        if let Some(ref mut multi) = self.multi_roi_processor {
            multi.reset();
        }
        self.frame_count = 0;
        self.session_start_us = 0;
    }
}

impl Default for VideoPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = VideoPipeline::new();
        assert_eq!(pipeline.frame_count, 0);
        assert!(pipeline.frame_buffer.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = VideoConfig::default();
        assert!((config.fps - 30.0).abs() < 0.01);
        assert_eq!(config.buffer_size, 150);
    }

    #[test]
    fn test_inject_signal() {
        let mut pipeline = VideoPipeline::new();
        
        let result = pipeline.inject_signal([128.0, 128.0, 128.0], 0, true);
        assert!(result.face_detected);
        assert_eq!(pipeline.frame_count, 1);
    }

    #[test]
    fn test_buffer_management() {
        let mut config = VideoConfig::default();
        config.buffer_size = 10;
        let mut pipeline = VideoPipeline::with_config(config);

        for i in 0..20 {
            pipeline.inject_signal([100.0, 100.0, 100.0], i * 33333, true);
        }

        assert_eq!(pipeline.frame_buffer.len(), 10);
    }

    #[test]
    fn test_reset() {
        let mut pipeline = VideoPipeline::new();
        pipeline.inject_signal([128.0, 128.0, 128.0], 0, true);
        
        pipeline.reset();
        
        assert_eq!(pipeline.frame_count, 0);
        assert!(pipeline.frame_buffer.is_empty());
    }
}
