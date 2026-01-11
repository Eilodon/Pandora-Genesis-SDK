//! Basic (stateless) skandha implementations.
//!
//! These are the original V1 implementations - fast, simple, deterministic.
//!
//! # Performance Characteristics
//!
//! - Latency: ~30-40µs per full cycle
//! - Memory: ~2KB per cycle (with recycling)
//! - Throughput: >30,000 cycles/sec (single thread)
//!
//! # Use Cases
//!
//! - High-throughput event processing
//! - Deterministic testing
//! - Baseline performance benchmarking
//! - Production systems requiring predictable latency

pub mod implementations;

// Re-export for convenience
pub use implementations::{
    BasicRupaSkandha,
    BasicVedanaSkandha,
    BasicSannaSkandha,
    BasicSankharaSkandha,
    BasicVinnanaSkandha,
};
