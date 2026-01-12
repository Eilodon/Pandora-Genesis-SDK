//! Domain modules: Pluggable application domains for the AGOLOS engine.
//!
//! Each domain provides:
//! - Configuration structure implementing `OscillatorConfig`
//! - Signal variables implementing `SignalVariable`
//! - Action types implementing `ActionKind`
//! - A `Domain` implementation tying them together
//!
//! # Available Domains
//!
//! - [`biofeedback`]: Reference implementation for breath guidance, HRV tracking,
//!   and physiological signal processing.

pub mod biofeedback;

pub use biofeedback::BiofeedbackDomain;
