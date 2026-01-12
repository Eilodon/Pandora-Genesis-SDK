//! Holographic Memory Module
//!
//! Implements FFT-based associative memory following the principle:
//! "Information is not located at an address, but superposed across the entire memory trace"
//!
//! # Mathematical Foundation
//! Holographic Associative Memory (HAM) stores key-value pairs as interference patterns
//! in frequency space. The key insight is that convolution in time domain equals
//! multiplication in frequency domain, giving us O(N log N) instead of O(N²).
//!
//! # Invariants
//! - Memory trace dimension is fixed at construction
//! - FFT/IFFT operations preserve energy (Parseval's theorem)
//! - Retrieval complexity is O(dim log dim), independent of stored items

pub mod hologram;
pub mod modern_hopfield; // Modern Hopfield Networks (2020-2025) - exponential capacity

pub use hologram::HolographicMemory;
pub use modern_hopfield::ModernHopfieldNetwork;
