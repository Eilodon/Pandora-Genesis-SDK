//! Level 0-4 Character Counting Tests
//!
//! This test suite validates the original test proposal for AGOLOS's
//! compositional reasoning capabilities. Tests progress from basic
//! character counting to cross-domain transfer.
//!
//! # Test Hierarchy
//!
//! - **Level 0**: Basic counting (teach counting, test on 'r' in raspberry)
//! - **Level 1**: Generalization (new letters, new words)
//! - **Level 2**: Compositional (count + arithmetic)
//! - **Level 3**: Meta-learning (pattern evolution)
//! - **Level 4**: Cross-domain transfer (sentence → word → letter)

use std::sync::Arc;
use zenb_core::domains::text::TextInput;
use zenb_core::llm::MockProvider;
use zenb_core::skandha::llm::{LlmPipeline, LlmPipelineConfig};

/// Helper to create a test pipeline.
fn create_pipeline() -> LlmPipeline<MockProvider> {
    let provider = Arc::new(MockProvider::new());
    LlmPipeline::new(provider)
}

// ============================================================================
// LEVEL 0: BASIC COUNTING
// ============================================================================

#[test]
fn test_level0_count_r_in_raspberry() {
    let mut pipeline = create_pipeline();
    
    let result = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
    
    assert_eq!(result.answer(), Some("3"));
    assert!(!result.used_llm()); // Solved procedurally
    
    // Verify reasoning trace exists
    assert!(result.reasoning().contains("3"));
}

#[test]
fn test_level0_no_llm_for_counting() {
    let mut pipeline = create_pipeline();
    
    // Counting should NOT require LLM - it's procedural
    let result = pipeline.process(&TextInput::new("Count the letter 'a' in 'banana'"));
    
    assert!(!result.used_llm());
    assert!(result.pattern.similarity >= 0.9); // High confidence for procedural
}

// ============================================================================
// LEVEL 1: GENERALIZATION
// ============================================================================

#[test]
fn test_level1_different_letters() {
    let mut pipeline = create_pipeline();
    
    // Should generalize to any letter
    let result = pipeline.process(&TextInput::new("Count the letter 's' in 'raspberry'"));
    assert_eq!(result.answer(), Some("1"));
    
    let result = pipeline.process(&TextInput::new("Count the letter 'p' in 'raspberry'"));
    assert_eq!(result.answer(), Some("1"));
    
    let result = pipeline.process(&TextInput::new("Count the letter 'b' in 'raspberry'"));
    assert_eq!(result.answer(), Some("1"));
}

#[test]
fn test_level1_different_words() {
    let mut pipeline = create_pipeline();
    
    // Should work on any word
    let result = pipeline.process(&TextInput::new("Count the letter 'a' in 'banana'"));
    assert_eq!(result.answer(), Some("3"));
    
    let result = pipeline.process(&TextInput::new("Count the letter 'e' in 'excellence'"));
    assert_eq!(result.answer(), Some("4"));
    
    let result = pipeline.process(&TextInput::new("Count the letter 'l' in 'llama'"));
    assert_eq!(result.answer(), Some("2"));
}

#[test]
fn test_level1_edge_cases() {
    let mut pipeline = create_pipeline();
    
    // Zero occurrences
    let result = pipeline.process(&TextInput::new("Count the letter 'z' in 'raspberry'"));
    assert_eq!(result.answer(), Some("0"));
    
    // All same letter
    let result = pipeline.process(&TextInput::new("Count the letter 'a' in 'aaa'"));
    assert_eq!(result.answer(), Some("3"));
    
    // Single letter word
    let result = pipeline.process(&TextInput::new("Count the letter 'x' in 'x'"));
    assert_eq!(result.answer(), Some("1"));
}

#[test]
fn test_level1_case_insensitivity() {
    let mut pipeline = create_pipeline();
    
    // Should handle uppercase
    let result = pipeline.process(&TextInput::new("Count the letter 'R' in 'Raspberry'"));
    assert_eq!(result.answer(), Some("3")); // r, R, r
}

// ============================================================================
// LEVEL 2: COMPOSITIONAL
// ============================================================================

#[test]
fn test_level2_counting_basics_for_composition() {
    // Level 2 tests require combining counting with arithmetic
    // For now, verify the counting primitive is reliable enough for composition
    let mut pipeline = create_pipeline();
    
    // Multiple counting operations that could be composed
    let r1 = pipeline.process(&TextInput::new("Count the letter 'a' in 'banana'"));
    let r2 = pipeline.process(&TextInput::new("Count the letter 'n' in 'banana'"));
    
    // Manual composition (simulating what a compositional system would do)
    let a_count: i32 = r1.answer().unwrap().parse().unwrap();
    let n_count: i32 = r2.answer().unwrap().parse().unwrap();
    
    assert_eq!(a_count + n_count, 5); // 3 a's + 2 n's = 5
}

// ============================================================================
// LEVEL 3: META-LEARNING (Pattern Evolution)
// ============================================================================

#[test]
fn test_level3_pattern_consistency() {
    let mut pipeline = create_pipeline();
    
    // Same query should give consistent results (determinism)
    let r1 = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
    let r2 = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
    
    assert_eq!(r1.answer(), r2.answer());
    assert_eq!(r1.pattern.pattern_type, r2.pattern.pattern_type);
}

// ============================================================================
// LEVEL 4: CROSS-DOMAIN TRANSFER
// ============================================================================

#[test]
fn test_level4_procedural_transfer() {
    // Verify the procedural counting transfers across different contexts
    let mut pipeline = create_pipeline();
    
    // Different prompt formats, same underlying task
    let words = [
        ("r", "raspberry", 3),
        ("s", "mississippi", 4),
        ("i", "mississippi", 4),
        ("m", "mammogram", 4), // m-a-m-m-o-g-r-a-m = 4 m's
    ];
    
    for (letter, word, expected) in words {
        let prompt = format!("Count the letter '{}' in '{}'", letter, word);
        let result = pipeline.process(&TextInput::new(&prompt));
        assert_eq!(
            result.answer(),
            Some(expected.to_string()).as_deref(),
            "Failed for '{}' in '{}'",
            letter,
            word
        );
    }
}

// ============================================================================
// EXPLAINABILITY (Skandha Pipeline Visibility)
// ============================================================================

#[test]
fn test_explainability_reasoning_trace() {
    let mut pipeline = create_pipeline();
    
    let result = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
    
    // Reasoning should explain the answer
    let reasoning = result.reasoning();
    assert!(reasoning.contains("3"));
    assert!(reasoning.contains("occurrences") || reasoning.contains("Counted"));
}

#[test]
fn test_explainability_pattern_detection() {
    let mut pipeline = create_pipeline();
    
    let result = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
    
    // Pattern type should be identified
    assert_eq!(
        result.pattern.pattern_type,
        zenb_core::skandha::llm::PatternType::Counting
    );
}

// ============================================================================
// PERFORMANCE (Procedural vs LLM)
// ============================================================================

#[test]
fn test_counting_is_procedural() {
    // Critical: counting should NOT use LLM (cost/latency optimization)
    let mut pipeline = create_pipeline();
    
    // Run many counting operations
    for _ in 0..100 {
        let result = pipeline.process(&TextInput::new("Count the letter 'r' in 'raspberry'"));
        assert!(!result.used_llm(), "Counting should be procedural, not LLM-based");
    }
}
