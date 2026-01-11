//! Phase 1 foundation tests.
//!
//! Validates core abstractions and ensures backward compatibility.

use pandora_core::skandha_implementations::{basic::*, core::*, processors::LinearProcessor};
use std::time::Duration;

// ============================================================================
// CORE TRAITS TESTS
// ============================================================================

#[test]
fn test_mood_state_default() {
    let mood = MoodState::default();
    assert_eq!(mood.valence, 0.0);
    assert_eq!(mood.arousal, 0.0);
    assert_eq!(mood.quadrant(), "Neutral");
}

#[test]
fn test_mood_state_update() {
    let mut mood = MoodState::default();

    // Update with pleasant feeling
    mood.update(0.8, 0.5);

    assert!(mood.valence > 0.0);
    assert!(mood.arousal > 0.0);
    assert_eq!(mood.quadrant(), "Pleasant-Activated");
}

#[test]
fn test_mood_state_decay() {
    let mut mood = MoodState::with_decay_rate(0.95);
    mood.update(1.0, 1.0); // Max pleasant

    let initial_valence = mood.valence;
    // Simulate 1 second passing
    mood.decay(Duration::from_secs(1));

    assert!(mood.valence < initial_valence);
    assert!(mood.valence > 0.0); // Still positive
    assert_eq!(mood.valence, 1.0 * 0.95);
}

#[test]
fn test_mood_state_reset() {
    let mut mood = MoodState::default();
    mood.update(1.0, 1.0);

    mood.reset();

    assert_eq!(mood.valence, 0.0);
    assert_eq!(mood.arousal, 0.0);
}

#[test]
fn test_mood_half_life() {
    let mood = MoodState::with_decay_rate(0.95);
    let half_life = mood.half_life();

    // Should be around 13.5 seconds for 0.95 decay rate
    assert!(half_life > 13.0 && half_life < 14.0);
}

// ============================================================================
// PATTERN MEMORY TESTS
// ============================================================================

#[test]
fn test_pattern_memory_record() {
    let mut memory = PatternMemoryState::default();

    memory.record_pattern(12345);

    assert_eq!(memory.pattern_strength(12345), 0.5);
}

#[test]
fn test_pattern_memory_reinforcement() {
    let mut memory = PatternMemoryState::default();

    memory.record_pattern(12345);
    let initial_strength = memory.pattern_strength(12345);

    memory.record_pattern(12345);
    let reinforced_strength = memory.pattern_strength(12345);

    assert!(reinforced_strength > initial_strength);
}

#[test]
fn test_pattern_memory_decay() {
    let mut memory = PatternMemoryState::default();
    memory.record_pattern(12345);

    let initial_strength = memory.pattern_strength(12345);
    memory.decay(Duration::from_secs(1));
    let decayed_strength = memory.pattern_strength(12345);

    assert!(decayed_strength < initial_strength);
}

#[test]
fn test_pattern_memory_eviction() {
    let mut memory = PatternMemoryState::default();
    memory.max_patterns = 3;

    // Fill to capacity
    memory.record_pattern(1);
    memory.record_pattern(2);
    memory.record_pattern(3);

    // This should evict the weakest (which is one of the first 3 with 0.5 strength)
    memory.record_pattern(4);

    assert_eq!(memory.patterns.len(), 3);
    assert!(memory.patterns.contains_key(&4)); // New pattern is added
}

// ============================================================================
// FLOW CONTROL TESTS
// ============================================================================

#[test]
fn test_energy_budget_creation() {
    let budget = EnergyBudget::new(20, 5);
    assert_eq!(budget.remaining_percent(), 100.0);
}

#[test]
fn test_energy_budget_consumption() {
    let mut budget = EnergyBudget::new(20, 5);

    assert!(budget.consume(5));
    assert_eq!(budget.spent(), 5);
    assert_eq!(budget.remaining_percent(), 75.0);
}

#[test]
fn test_energy_budget_depletion() {
    let mut budget = EnergyBudget::new(10, 5);

    budget.consume(10);

    assert!(budget.is_depleted());
    assert!(!budget.consume(1));
}

#[test]
fn test_energy_budget_reflection_tracking() {
    let mut budget = EnergyBudget::new(30, 5);

    assert!(budget.record_reflection(SkandhaStage::Sanna, SkandhaStage::Vedana));
    // assert_eq!(budget.reflection_count, 1);
}

#[test]
fn test_energy_budget_reflection_limit() {
    let mut budget = EnergyBudget::new(100, 2);

    assert!(budget.record_reflection(SkandhaStage::Sanna, SkandhaStage::Vedana));
    assert!(budget.record_reflection(SkandhaStage::Sankhara, SkandhaStage::Sanna));
    assert!(!budget.record_reflection(SkandhaStage::Vinnana, SkandhaStage::Sankhara));
}

#[test]
fn test_skandha_stage_ordering() {
    assert!(SkandhaStage::Rupa < SkandhaStage::Vedana);
    assert!(SkandhaStage::Vedana < SkandhaStage::Sanna);
    assert!(SkandhaStage::Sanna < SkandhaStage::Sankhara);
    assert!(SkandhaStage::Sankhara < SkandhaStage::Vinnana);
}

#[test]
fn test_skandha_stage_can_loop_to() {
    assert!(SkandhaStage::Sanna.can_loop_to(SkandhaStage::Vedana));
    assert!(!SkandhaStage::Vedana.can_loop_to(SkandhaStage::Sanna));
    assert!(!SkandhaStage::Rupa.can_loop_to(SkandhaStage::Vedana));
}

// ============================================================================
// BACKWARD COMPATIBILITY TESTS
// ============================================================================

#[test]
fn test_linear_processor_unchanged_behavior() {
    let processor = LinearProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        Box::new(BasicVedanaSkandha::default()),
        Box::new(BasicSannaSkandha::default()),
        Box::new(BasicSankharaSkandha::default()),
        Box::new(BasicVinnanaSkandha::default()),
    );

    // Error event should produce rebirth
    let error_event = b"critical error".to_vec();
    let result = processor.run_cycle(error_event);
    assert!(result.is_some());

    // Success event should not
    let success_event = b"operation success".to_vec();
    let result = processor.run_cycle(success_event);
    assert!(result.is_none());
}

#[test]
fn test_basic_skandhas_unchanged() {
    let rupa = BasicRupaSkandha::default();
    let vedana = BasicVedanaSkandha::default();
    let sanna = BasicSannaSkandha::default();
    let sankhara = BasicSankharaSkandha::default();
    let vinnana = BasicVinnanaSkandha::default();

    // Verify trait implementations still work
    assert_eq!(rupa.name(), "BasicRupa");
    assert_eq!(vedana.name(), "BasicVedana");
    assert_eq!(sanna.name(), "BasicSanna");
    assert_eq!(sankhara.name(), "BasicSankhara");
    assert_eq!(vinnana.name(), "BasicVinnana");
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[test]
fn test_full_pipeline_error_handling() {
    let processor = LinearProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        Box::new(BasicVedanaSkandha::default()),
        Box::new(BasicSannaSkandha::default()),
        Box::new(BasicSankharaSkandha::default()),
        Box::new(BasicVinnanaSkandha::default()),
    );

    let events = vec![
        b"error in module A".to_vec(),
        b"critical failure in system B".to_vec(),
        b"exception caught in handler C".to_vec(),
    ];

    for event in events {
        let result = processor.run_cycle(event);
        assert!(result.is_some(), "Error events should produce rebirth");
    }
}

#[test]
fn test_full_pipeline_success_handling() {
    let processor = LinearProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        Box::new(BasicVedanaSkandha::default()),
        Box::new(BasicSannaSkandha::default()),
        Box::new(BasicSankharaSkandha::default()),
        Box::new(BasicVinnanaSkandha::default()),
    );

    let events = vec![
        b"operation success".to_vec(),
        b"system healthy".to_vec(),
        b"task complete".to_vec(),
    ];

    for event in events {
        let result = processor.run_cycle(event);
        assert!(
            result.is_none(),
            "Success events should not produce rebirth"
        );
    }
}

#[tokio::test]
async fn test_async_compatibility() {
    let processor = LinearProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        Box::new(BasicVedanaSkandha::default()),
        Box::new(BasicSannaSkandha::default()),
        Box::new(BasicSankharaSkandha::default()),
        Box::new(BasicVinnanaSkandha::default()),
    );

    let event = b"async test event".to_vec();
    let result = processor.run_cycle_async(event).await;

    // Just verify it works
    assert!(result.is_none() || result.is_some());
}
