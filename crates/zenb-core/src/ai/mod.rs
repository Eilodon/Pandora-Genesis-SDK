//! AI Tools module
//!
//! Safe AI function calling with validation and rate limiting

pub mod tools;

pub use tools::{AiTool, AiToolRegistry, ToolContext, ToolResult};
