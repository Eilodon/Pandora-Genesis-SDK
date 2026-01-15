//! Vision module for face detection and ROI extraction
//!
//! This module provides:
//! - `FaceDetector` trait for pluggable face detection backends
//! - `FaceDetection` struct with bounding box and optional landmarks
//! - ROI extraction utilities for rPPG processing
//! - `Frame` struct for image processing (basic ops available always)
//! - `VideoPipeline` for real-time video processing
//! - `BlazeFaceDetector` for pure-Rust face detection (with Candle)
//!
//! # Features
//!
//! - `image-processing`: Enables `image` crate integration for file I/O
//! - `candle-detection`: Enables BlazeFace with Candle ML framework (GPU support)
//!
//! # Design
//!
//! The module is designed as a pure-Rust interface that allows clients
//! to inject face detection results from external sources (MediaPipe,
//! ARKit, etc.) without adding native dependencies. With `candle-detection`,
//! full inference can run locally.
//!
//! # Example
//!
//! ```ignore
//! use zenb_signals::vision::{VideoPipeline, BlazeFaceDetector};
//!
//! let mut pipeline = VideoPipeline::new();
//!
//! // Set up BlazeFace detector
//! let detector = BlazeFaceDetector::new();
//! pipeline.set_face_detector(Box::new(detector));
//!
//! // Process frames
//! let result = pipeline.process_raw(&frame_bytes, 640, 480, timestamp_us);
//! ```

mod blazeface;
mod face_roi;
mod image_ops;
mod landmark_roi;
mod video_pipeline;

// Face detection and ROI extraction
pub use face_roi::{
    extract_roi_grid, extract_roi_mean_rgb,
    FaceDetection, FaceDetector, ExternalLandmarkDetector,
};
pub use landmark_roi::{
    forehead_roi, left_cheek_roi, right_cheek_roi,
    Polygon, compute_polygon_mean_rgb, FOREHEAD_LANDMARKS, CHEEK_LEFT_LANDMARKS, CHEEK_RIGHT_LANDMARKS,
};

// Image processing
pub use image_ops::{Frame, rgba_to_rgb, nv21_to_rgb};

// Video pipeline
pub use video_pipeline::{VideoPipeline, VideoConfig, PipelineResult};

// BlazeFace detector (works without candle feature, but uses heuristics only)
pub use blazeface::{BlazeFaceDetector, BlazeFaceConfig, BlazeFaceDetection};
