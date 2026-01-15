//! Vertical Market Modules for AGOLOS Platform
//!
//! # Modules
//! - `shared`: Common components reused across verticals
//! - `liveness`: Liveness detection for authentication
//! - `automotive`: Driver monitoring system
//! - `retail`: Emotion analytics for retail
//! - `fintech`: Fraud detection
//! - `education`: Exam proctoring

pub mod shared;

#[cfg(feature = "liveness")]
pub mod liveness;

#[cfg(feature = "automotive")]
pub mod automotive;

#[cfg(feature = "retail")]
pub mod retail;

#[cfg(feature = "fintech")]
pub mod fintech;

#[cfg(feature = "education")]
pub mod education;

pub use shared::*;
