//! Vision module for face detection and ROI extraction
//!
//! This module provides:
//! - `FaceDetector` trait for pluggable face detection backends
//! - `FaceDetection` struct with bounding box and optional landmarks
//! - ROI extraction utilities for rPPG processing
//! - `Frame` struct for image processing (basic ops available always)
//! - `VideoPipeline` for real-time video processing
//! - `BlazeFaceDetector` for pure-Rust face detection (with Candle)
//! - `GpuDevice` for GPU acceleration (CUDA/Metal/CPU)
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
//! full inference can run locally with GPU acceleration.
//!
//! # Example
//!
//! ```ignore
//! use zenb_signals::vision::{VideoPipeline, BlazeFaceDetector, get_best_device};
//!
//! // Auto-detect best GPU
//! let device = get_best_device();
//! println!("Using: {:?}", device.backend());
//!
//! let mut pipeline = VideoPipeline::new();
//! let detector = BlazeFaceDetector::new();
//! pipeline.set_face_detector(Box::new(detector));
//!
//! let result = pipeline.process_raw(&frame_bytes, 640, 480, timestamp_us);
//! ```

mod blazeface;
mod face_roi;
pub mod gpu;
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

// GPU acceleration
pub use gpu::{GpuDevice, GpuBackend, GpuInfo, GpuMetrics, get_best_device, list_devices};

// GPU-accelerated ops (candle feature only)
#[cfg(feature = "candle-detection")]
pub use gpu::{GpuImageOps, BatchProcessor};
