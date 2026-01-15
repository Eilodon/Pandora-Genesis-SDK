//! Domain modules: Pluggable application domains for the AGOLOS engine.
//!
//! Each domain provides:
//! - Configuration structure implementing `OscillatorConfig`
//! - Signal variables implementing `SignalVariable`
//! - Belief modes implementing `BeliefMode`
//! - Action types implementing `ActionKind`
//! - A `Domain` implementation tying them together
//!
//! # Available Domains
//!
//! - [`biofeedback`]: Reference implementation for breath guidance, HRV tracking,
//!   and physiological signal processing.
//! - [`trading`]: Example domain for market analysis and algorithmic trading.
//! - [`text`]: Natural language processing with LLM integration.

pub mod biofeedback;
pub mod trading;
pub mod text;

pub use biofeedback::BiofeedbackDomain;
pub use trading::TradingDomain;
pub use text::TextDomain;

