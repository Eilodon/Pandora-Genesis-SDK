//! Perception Module - Sheaf-based Sensor Fusion
//!
//! Implements Sheaf Laplacian diffusion to achieve consensus among sensors,
//! automatically filtering contradictory or malicious inputs.
//!
//! # Mathematical Foundation (Sắc Uẩn - Rūpa-skandha)
//! In Buddhist philosophy, Sắc Uẩn represents the aggregate of form/matter.
//! This module embodies coherent perception: seeing reality as it is,
//! free from the distortions of contradictory sensor data.
//!
//! # Sheaf Theory Application
//! A sheaf is a structure that tracks how local data can be consistently
//! "glued" into global data. The Sheaf Laplacian measures how much local
//! observations disagree, and diffusion naturally filters inconsistencies.

pub mod sheaf;

pub use sheaf::{PhysiologicalContext, SheafPerception};
