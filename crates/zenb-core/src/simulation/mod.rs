//! Simulation module for testing Active Inference agents.
//!
//! # Components
//! - `GridWorld`: 2D grid environment with partial observability

mod grid_world;

pub use grid_world::{
    Action, ActionResult, Cell, Direction, GridWorld, ObservabilityMode, Viewshed,
};
