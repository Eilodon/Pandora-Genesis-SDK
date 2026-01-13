//! Memory Module
//!
//! Provides multiple associative memory implementations:
//! - **Holographic Memory**: FFT-based, O(N log N), requires floating point
//! - **Krylov Projector**: Subspace acceleration for iterative methods
//! - **HDC Memory**: Binary hyperdimensional, integer-only, NPU-friendly
//!
//! # Mathematical Foundation
//! Holographic Associative Memory (HAM) stores key-value pairs as interference patterns
//! in frequency space. The key insight is that convolution in time domain equals
//! multiplication in frequency domain, giving us O(N log N) instead of O(N²).
//!
//! Binary HDC uses high-dimensional binary vectors with XOR binding and majority
//! bundling, achieving similar content-addressable properties with integer-only ops.
//!
//! # Invariants
//! - Memory trace dimension is fixed at construction
//! - FFT/IFFT operations preserve energy (Parseval's theorem)
//! - Retrieval complexity is O(dim log dim), independent of stored items

pub mod hdc; // Binary Hyperdimensional Computing (NPU-accelerated)
pub mod hologram;
pub mod krylov; // TIER 4b: Krylov Subspace Acceleration

pub use hdc::{HdcMemory, HdcVector, HdcConfig};
pub use hologram::HolographicMemory;
pub use krylov::KrylovProjector;
