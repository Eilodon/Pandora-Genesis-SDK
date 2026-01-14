//! Memory Module
//!
//! Provides multiple associative memory implementations:
//! - **Holographic Memory**: FFT-based, O(N log N), requires floating point
//! - **Krylov Projector**: Subspace acceleration for iterative methods
//! - **HDC Memory**: Binary hyperdimensional, integer-only, NPU-friendly
//! - **Tiered HDC Memory**: Two-tier HDC with working + long-term storage (LifeHD pattern)
//! - **Saccade Linker**: LTC-based predictive memory addressing
//!
//! # Mathematical Foundation
//! Holographic Associative Memory (HAM) stores key-value pairs as interference patterns
//! in frequency space. The key insight is that convolution in time domain equals
//! multiplication in frequency domain, giving us O(N log N) instead of O(N²).
//!
//! Binary HDC uses high-dimensional binary vectors with XOR binding and majority
//! bundling, achieving similar content-addressable properties with integer-only ops.
//!
//! Tiered HDC implements the LifeHD pattern: frequently accessed patterns are
//! consolidated to long-term memory for anti-forgetting properties.
//!
//! Saccade Linker uses LTC neurons to predict memory locations for O(1) lookup.
//!
//! # Invariants
//! - Memory trace dimension is fixed at construction
//! - FFT/IFFT operations preserve energy (Parseval's theorem)
//! - Retrieval complexity is O(dim log dim), independent of stored items

pub mod hdc; // Binary Hyperdimensional Computing (NPU-accelerated)
pub mod hologram;
pub mod krylov; // TIER 4b: Krylov Subspace Acceleration
pub mod router; // ZENITH Tier 1: Adaptive memory routing
pub mod saccade; // VAJRA V5: Predictive memory addressing
pub mod tiered_hdc; // LifeHD pattern: Two-tier memory with consolidation
pub mod uncertainty; // ZENITH Tier 4: Uncertainty-aware retrieval
pub mod zenith; // ZENITH Unified API

pub use hdc::{HdcConfig, HdcMemory, HdcVector, SparseHdcVector, SparsityController};
pub use hologram::HolographicMemory;
pub use krylov::KrylovProjector;
pub use router::{AdaptiveMemoryRouter, MemoryBackend, RouterConfig, RouterStats, TaskType};
pub use saccade::{SaccadeConfig, SaccadeLinker, SaccadeStats};
pub use tiered_hdc::{MemoryTier, TieredHdcConfig, TieredHdcMemory, TieredHdcStats};
pub use uncertainty::{UncertaintyAwareRetrieval, UncertaintyRetrievalConfig, UncertainRetrievalResult};
pub use zenith::{ZenithConfig, ZenithMemory, ZenithRetrievalResult, ZenithStats};


