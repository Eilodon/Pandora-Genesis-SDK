//! Skandha pipeline processors.
//!
//! # Processor Tiers
//!
//! - **Linear**: Fast, stateless, deterministic (V1)
//! - **Recurrent**: Stateful, reflective, adaptive (V2 - Phase 1)
//! - **Async**: Non-blocking, I/O efficient (V3 - Phase 2, future)
//! - **Adaptive**: Self-tuning, meta-cognitive (V4 - Phase 3, future)

pub mod linear;
pub mod recurrent;

// Future processors
// pub mod async_linear; // Phase 2
// pub mod adaptive;     // Phase 3

pub use linear::LinearProcessor;
pub use recurrent::RecurrentProcessor;
