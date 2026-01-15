//! Face Detection Trait and ROI Extraction
//!
//! Provides pluggable face detection interface and grid-based ROI extraction.

use crate::rppg::RoiSignal;

/// Face detection result
#[derive(Debug, Clone)]
pub struct FaceDetection {
    /// Bounding box [x, y, width, height] in pixels
    pub bbox: [f32; 4],
    /// Optional 468 MediaPipe-style landmarks [(x, y), ...]
    pub landmarks: Option<Vec<[f32; 2]>>,
    /// Detection confidence (0-1)
    pub confidence: f32,
}

impl Default for FaceDetection {
    fn default() -> Self {
        Self {
            bbox: [0.0, 0.0, 100.0, 100.0],
            landmarks: None,
            confidence: 0.0,
        }
    }
}

/// Pluggable face detection trait
///
/// Implement this trait to integrate different face detection backends:
/// - MediaPipe Face Mesh (via FFI)
/// - ARKit Face Tracking (on iOS)
/// - Custom ML models
pub trait FaceDetector: Send + Sync {
    /// Detect face in frame
    ///
    /// # Arguments
    /// * `frame` - Raw frame bytes (RGB888 or RGBA8888)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    /// * `channels` - Number of color channels (3 or 4)
    ///
    /// # Returns
    /// Face detection result if face found, None otherwise
    fn detect(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
        channels: u8,
    ) -> Option<FaceDetection>;
}

/// No-op detector for when client provides landmarks externally
///
/// Use this when face detection is handled by the client application
/// (e.g., MediaPipe on mobile, ARKit on iOS) and you only need to
/// use the ROI extraction utilities.
#[derive(Debug, Clone, Default)]
pub struct ExternalLandmarkDetector;

impl FaceDetector for ExternalLandmarkDetector {
    fn detect(
        &mut self,
        _frame: &[u8],
        _width: u32,
        _height: u32,
        _channels: u8,
    ) -> Option<FaceDetection> {
        // No-op: client must inject landmarks separately
        None
    }
}

/// Extract mean RGB from a rectangular ROI
///
/// # Arguments
/// * `frame` - Raw frame bytes (RGB888)
/// * `width` - Frame width in pixels
/// * `x`, `y` - Top-left corner of ROI
/// * `roi_w`, `roi_h` - ROI dimensions
///
/// # Returns
/// Mean [R, G, B] values in the ROI
pub fn extract_roi_mean_rgb(
    frame: &[u8],
    width: u32,
    x: u32,
    y: u32,
    roi_w: u32,
    roi_h: u32,
) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut count = 0u32;
    
    let frame_stride = (width * 3) as usize;
    
    for dy in 0..roi_h {
        let row_y = y + dy;
        if row_y >= width { continue; }  // bounds check
        
        for dx in 0..roi_w {
            let row_x = x + dx;
            if row_x >= width { continue; }
            
            let idx = (row_y as usize * frame_stride) + (row_x as usize * 3);
            if idx + 2 < frame.len() {
                sum[0] += frame[idx] as f32;
                sum[1] += frame[idx + 1] as f32;
                sum[2] += frame[idx + 2] as f32;
                count += 1;
            }
        }
    }
    
    if count > 0 {
        let inv = 1.0 / count as f32;
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

/// Extract grid of ROIs from face bounding box
///
/// Divides the face bounding box into a grid and extracts mean RGB
/// for each cell. Used with Multi-ROI processor for occlusion robustness.
///
/// # Arguments
/// * `frame` - Raw frame bytes (RGB888)
/// * `frame_width` - Frame width in pixels
/// * `frame_height` - Frame height in pixels
/// * `detection` - Face detection with bounding box
/// * `grid_rows` - Number of grid rows
/// * `grid_cols` - Number of grid columns
///
/// # Returns
/// Vector of RoiSignal for Multi-ROI processing
pub fn extract_roi_grid(
    frame: &[u8],
    frame_width: u32,
    frame_height: u32,
    detection: &FaceDetection,
    grid_rows: usize,
    grid_cols: usize,
) -> Vec<RoiSignal> {
    let [bbox_x, bbox_y, bbox_w, bbox_h] = detection.bbox;
    
    let cell_w = bbox_w / grid_cols as f32;
    let cell_h = bbox_h / grid_rows as f32;
    
    let mut signals = Vec::with_capacity(grid_rows * grid_cols);
    
    for row in 0..grid_rows {
        for col in 0..grid_cols {
            let x = (bbox_x + col as f32 * cell_w) as u32;
            let y = (bbox_y + row as f32 * cell_h) as u32;
            let w = cell_w as u32;
            let h = cell_h as u32;
            
            // Bounds check
            if x + w <= frame_width && y + h <= frame_height {
                let rgb = extract_roi_mean_rgb(frame, frame_width, x, y, w, h);
                signals.push(RoiSignal {
                    r: ndarray::Array1::from(vec![rgb[0]]),
                    g: ndarray::Array1::from(vec![rgb[1]]),
                    b: ndarray::Array1::from(vec![rgb[2]]),
                    row,
                    col,
                });
            }
        }
    }
    
    signals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_detector_returns_none() {
        let mut detector = ExternalLandmarkDetector;
        let result = detector.detect(&[], 100, 100, 3);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_roi_mean_rgb() {
        // Create 4x4 frame with known values
        let frame: Vec<u8> = vec![
            128, 64, 32,  128, 64, 32,  128, 64, 32,  128, 64, 32,
            128, 64, 32,  128, 64, 32,  128, 64, 32,  128, 64, 32,
            128, 64, 32,  128, 64, 32,  128, 64, 32,  128, 64, 32,
            128, 64, 32,  128, 64, 32,  128, 64, 32,  128, 64, 32,
        ];
        
        let rgb = extract_roi_mean_rgb(&frame, 4, 0, 0, 2, 2);
        assert!((rgb[0] - 128.0).abs() < 0.01);
        assert!((rgb[1] - 64.0).abs() < 0.01);
        assert!((rgb[2] - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_roi_grid() {
        // Create 100x100 frame with uniform color
        let frame: Vec<u8> = vec![100u8; 100 * 100 * 3];
        
        let detection = FaceDetection {
            bbox: [10.0, 10.0, 30.0, 30.0],
            landmarks: None,
            confidence: 0.9,
        };
        
        let grid = extract_roi_grid(&frame, 100, 100, &detection, 3, 3);
        assert_eq!(grid.len(), 9); // 3x3 grid
    }
}
