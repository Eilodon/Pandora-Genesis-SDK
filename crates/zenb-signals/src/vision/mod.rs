//! Vision module for face detection and ROI extraction
//!
//! This module provides:
//! - `FaceDetector` trait for pluggable face detection backends
//! - `FaceDetection` struct with bounding box and optional landmarks
//! - ROI extraction utilities for rPPG processing
//!
//! # Design
//!
//! The module is designed as a pure-Rust interface that allows clients
//! to inject face detection results from external sources (MediaPipe,
//! ARKit, etc.) without adding native dependencies.
//!
//! # Example
//!
//! ```ignore
//! use zenb_signals::vision::{FaceDetection, extract_roi_grid};
//!
//! let detection = FaceDetection {
//!     bbox: [100.0, 50.0, 200.0, 250.0],
//!     landmarks: None,
//!     confidence: 0.95,
//! };
//!
//! let grid = extract_roi_grid(&frame, width, height, &detection, 3, 3);
//! ```

mod face_roi;
mod landmark_roi;

pub use face_roi::{
    extract_roi_grid, extract_roi_mean_rgb,
    FaceDetection, FaceDetector, ExternalLandmarkDetector,
};
pub use landmark_roi::{
    forehead_roi, left_cheek_roi, right_cheek_roi,
    Polygon, compute_polygon_mean_rgb, FOREHEAD_LANDMARKS, CHEEK_LEFT_LANDMARKS, CHEEK_RIGHT_LANDMARKS,
};
