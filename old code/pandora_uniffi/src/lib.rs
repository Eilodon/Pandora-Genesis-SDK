#![allow(clippy::empty_line_after_doc_comments)]
#![allow(unpredictable_function_pointer_comparisons)]

// Simple Python bindings for Pandora SDK
pub fn hello(name: String) -> String {
    format!("Hello from Pandora SDK, {}!", name)
}

pub fn get_version() -> String {
    "Pandora Genesis SDK v1.0.0".to_string()
}

pub fn run_gridworld_simulation() -> String {
    "GridWorld simulation completed successfully!".to_string()
}

uniffi::include_scaffolding!("pandora");
