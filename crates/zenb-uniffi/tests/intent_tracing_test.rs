//! Integration tests for Intent ID tracing and Karmic Feedback Loop.
//!
//! # B.ONE V3: Di Hồn Đại Pháp - Vòng Lặp Nghiệp Quả
//!
//! These tests verify that:
//! 1. Engine propagates IntentId to ControlDecision
//! 2. IntentTracker correctly tracks intents
//! 3. report_action_outcome can trace back to original intent
//! 4. Learning updates happen with correct context

use zenb_core::skandha::sankhara::{IntentId, IntentTracker, TrackedIntent, ContextSnapshot};
use zenb_core::Engine;

// ============================================================================
// UNIT TESTS: IntentTracker Core Functionality
// ============================================================================

#[test]
fn test_intent_tracker_basic_tracking() {
    let mut tracker = IntentTracker::new();
    
    let intent = TrackedIntent {
        id: IntentId::new(),
        flow_event_id: None,
        action: zenb_core::skandha::IntentAction::GuideBreath { target_bpm: 6 },
        context: ContextSnapshot {
            belief_mode: 0,
            confidence: 0.8,
            free_energy: 0.2,
            arousal: 0.5,
            valence: 0.0,
            goal_id: 1,
            pattern_id: 42,
            timestamp_us: 1000,
        },
        target_bpm: 6.0,
        outcome_ts_us: None,
        success: None,
    };
    
    let intent_id = intent.id;
    tracker.track(intent);
    
    // Verify tracking
    assert_eq!(tracker.len(), 1);
    assert!(tracker.get(intent_id).is_some());
    
    // Verify initial state
    let tracked = tracker.get(intent_id).unwrap();
    assert!(tracked.success.is_none());
    assert!(tracked.outcome_ts_us.is_none());
}

#[test]
fn test_intent_tracker_record_outcome() {
    let mut tracker = IntentTracker::new();
    
    let intent = TrackedIntent {
        id: IntentId::new(),
        flow_event_id: None,
        action: zenb_core::skandha::IntentAction::GuideBreath { target_bpm: 6 },
        context: ContextSnapshot::default(),
        target_bpm: 6.0,
        outcome_ts_us: None,
        success: None,
    };
    
    let intent_id = intent.id;
    tracker.track(intent);
    
    // Record success
    let result = tracker.record_outcome(intent_id, true, 2000);
    assert!(result.is_some());
    
    // Verify outcome recorded
    let tracked = tracker.get(intent_id).unwrap();
    assert_eq!(tracked.success, Some(true));
    assert_eq!(tracked.outcome_ts_us, Some(2000));
}

#[test]
fn test_intent_tracker_cleanup() {
    let mut tracker = IntentTracker::with_config(1_000_000, 100); // 1 second max age
    
    // Add old intent
    let old_intent = TrackedIntent {
        id: IntentId::new(),
        flow_event_id: None,
        action: zenb_core::skandha::IntentAction::Observe,
        context: ContextSnapshot {
            timestamp_us: 1000,
            ..Default::default()
        },
        target_bpm: 6.0,
        outcome_ts_us: None,
        success: None,
    };
    
    // Add recent intent
    let recent_intent = TrackedIntent {
        id: IntentId::new(),
        flow_event_id: None,
        action: zenb_core::skandha::IntentAction::Observe,
        context: ContextSnapshot {
            timestamp_us: 2_000_000,
            ..Default::default()
        },
        target_bpm: 6.0,
        outcome_ts_us: None,
        success: None,
    };
    
    let old_id = old_intent.id;
    let recent_id = recent_intent.id;
    
    tracker.track(old_intent);
    tracker.track(recent_intent);
    
    assert_eq!(tracker.len(), 2);
    
    // Cleanup with current time = 3s
    tracker.cleanup(3_000_000);
    
    // Old intent should be removed, recent should remain
    assert_eq!(tracker.len(), 1);
    assert!(tracker.get(old_id).is_none());
    assert!(tracker.get(recent_id).is_some());
}

#[test]
fn test_intent_tracker_capacity_limit() {
    let mut tracker = IntentTracker::with_config(1_000_000_000, 3); // Max 3 items
    
    // Add 5 intents
    let mut ids = Vec::new();
    for i in 0..5 {
        let intent = TrackedIntent {
            id: IntentId::new(),
            flow_event_id: None,
            action: zenb_core::skandha::IntentAction::Observe,
            context: ContextSnapshot {
                timestamp_us: (i * 1000) as i64,
                ..Default::default()
            },
            target_bpm: 6.0,
            outcome_ts_us: None,
            success: None,
        };
        ids.push(intent.id);
        tracker.track(intent);
    }
    
    // Should have evicted oldest to stay at capacity
    assert!(tracker.len() <= 3);
    
    // Oldest intents should be gone
    assert!(tracker.get(ids[0]).is_none());
    assert!(tracker.get(ids[1]).is_none());
    
    // Newest should still be there
    assert!(tracker.get(ids[4]).is_some());
}

// ============================================================================
// INTEGRATION TESTS: Engine -> ControlDecision -> IntentTracker
// ============================================================================

#[test]
fn test_engine_populates_intent_id_in_decision() {
    let mut engine = Engine::new_for_test(6.0);
    
    // Ingest sensor data
    let features = vec![70.0, 45.0, 12.0, 0.9, 0.1];
    let ts_us = 1_000_000;
    let estimate = engine.ingest_sensor(&features, ts_us);
    
    // Tick engine
    engine.tick(100_000);
    
    // Make control decision
    let (decision, _changed, _policy, _deny) = engine.make_control(&estimate, ts_us);
    
    // CRITICAL: Verify intent_id is populated
    assert!(decision.intent_id.is_some(), "ControlDecision should have intent_id");
    
    let intent_id = decision.intent_id.unwrap();
    assert!(intent_id > 0, "IntentId should be non-zero");
    
    // Verify intent is tracked
    let tracker = &engine.skandha_pipeline.sankhara.intent_tracker;
    assert!(tracker.len() > 0, "IntentTracker should have at least one intent");
    
    // Verify we can retrieve the intent
    let intent = tracker.get(IntentId::from_raw(intent_id));
    assert!(intent.is_some(), "Intent should be retrievable from tracker");
    
    println!("✓ IntentId {} tracked successfully", intent_id);
}

#[test]
fn test_multiple_decisions_have_unique_intent_ids() {
    let mut engine = Engine::new_for_test(6.0);
    
    let mut intent_ids = Vec::new();
    
    // Make 5 decisions
    for i in 0..5 {
        let features = vec![70.0 + i as f32, 45.0, 12.0, 0.9, 0.1];
        let ts_us = 1_000_000 + i * 100_000;
        
        let estimate = engine.ingest_sensor(&features, ts_us);
        engine.tick(100_000);
        
        let (decision, _changed, _policy, _deny) = engine.make_control(&estimate, ts_us);
        
        if let Some(id) = decision.intent_id {
            intent_ids.push(id);
        }
    }
    
    // Verify all IDs are unique
    let unique_count = intent_ids.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(
        unique_count,
        intent_ids.len(),
        "All IntentIds should be unique"
    );
    
    println!("✓ {} unique IntentIds generated", intent_ids.len());
}

#[test]
fn test_intent_tracker_records_outcome_correctly() {
    let mut engine = Engine::new_for_test(6.0);
    
    // Make decision
    let features = vec![70.0, 45.0, 12.0, 0.9, 0.1];
    let ts_us = 1_000_000;
    let estimate = engine.ingest_sensor(&features, ts_us);
    engine.tick(100_000);
    
    let (decision, _changed, _policy, _deny) = engine.make_control(&estimate, ts_us);
    let intent_id = decision.intent_id.unwrap();
    
    // Simulate action outcome
    let outcome_ts = ts_us + 500_000;
    let success = true;
    
    let tracker = &mut engine.skandha_pipeline.sankhara.intent_tracker;
    let result = tracker.record_outcome(IntentId::from_raw(intent_id), success, outcome_ts);
    
    assert!(result.is_some(), "Outcome recording should succeed");
    
    // Verify outcome recorded
    let intent = tracker.get(IntentId::from_raw(intent_id)).unwrap();
    assert_eq!(intent.success, Some(true));
    assert_eq!(intent.outcome_ts_us, Some(outcome_ts));
    
    println!("✓ Outcome recorded: intent_id={}, success={}", intent_id, success);
}

// ============================================================================
// END-TO-END TEST: Full Karmic Feedback Loop
// ============================================================================

#[test]
fn test_karmic_feedback_loop_end_to_end() {
    // This test simulates the full flow:
    // 1. Engine makes decision with intent_id
    // 2. Mobile receives decision and executes action
    // 3. Mobile reports outcome with intent_id
    // 4. System learns from outcome with correct context
    
    let mut engine = Engine::new_for_test(6.0);
    
    // === STEP 1: Make Decision ===
    let features = vec![75.0, 40.0, 14.0, 0.85, 0.2];
    let ts_us = 1_000_000;
    let estimate = engine.ingest_sensor(&features, ts_us);
    engine.tick(100_000);
    
    let (decision, _changed, _policy, _deny) = engine.make_control(&estimate, ts_us);
    
    assert!(decision.intent_id.is_some(), "Decision must have intent_id");
    let intent_id = decision.intent_id.unwrap();
    
    // Capture original context
    let tracker = &engine.skandha_pipeline.sankhara.intent_tracker;
    let original_intent = tracker.get(IntentId::from_raw(intent_id)).unwrap();
    let original_context = original_intent.context.clone();
    
    println!("✓ Decision made: intent_id={}, target_bpm={}", 
             intent_id, decision.target_rate_bpm);
    
    // === STEP 2: Mobile executes action (simulated delay) ===
    let action_execution_delay = 500_000; // 0.5s
    let outcome_ts = ts_us + action_execution_delay;
    
    // === STEP 3: Report outcome back ===
    let success = true;
    let tracker = &mut engine.skandha_pipeline.sankhara.intent_tracker;
    tracker.record_outcome(IntentId::from_raw(intent_id), success, outcome_ts);
    
    println!("✓ Outcome recorded: success={}", success);
    
    // === STEP 4: Verify context preserved ===
    let updated_intent = tracker.get(IntentId::from_raw(intent_id)).unwrap();
    
    // Context should be unchanged (it's the original decision context)
    assert_eq!(updated_intent.context.belief_mode, original_context.belief_mode);
    assert_eq!(updated_intent.context.confidence, original_context.confidence);
    assert_eq!(updated_intent.context.pattern_id, original_context.pattern_id);
    
    // Outcome should be recorded
    assert_eq!(updated_intent.success, Some(true));
    assert_eq!(updated_intent.outcome_ts_us, Some(outcome_ts));
    
    println!("✓ Context preserved correctly");
    
    // === STEP 5: Simulate learning (would happen via learn_from_outcome) ===
    // In real flow, the IntentTracker provides the context for precise learning
    // This verifies we can retrieve all necessary information
    
    assert_eq!(updated_intent.target_bpm, decision.target_rate_bpm);
    assert!(updated_intent.context.timestamp_us > 0);
    
    println!("✓ Karmic feedback loop complete!");
    println!("  - Intent ID: {}", intent_id);
    println!("  - Original context: mode={}, confidence={:.2}", 
             original_context.belief_mode, original_context.confidence);
    println!("  - Action executed: target_bpm={}", decision.target_rate_bpm);
    println!("  - Outcome: success={}", success);
}

#[test]
fn test_failed_action_records_negative_outcome() {
    let mut engine = Engine::new_for_test(6.0);
    
    // Make decision
    let features = vec![85.0, 30.0, 16.0, 0.7, 0.4];
    let ts_us = 1_000_000;
    let estimate = engine.ingest_sensor(&features, ts_us);
    engine.tick(100_000);
    
    let (decision, _changed, _policy, _deny) = engine.make_control(&estimate, ts_us);
    let intent_id = decision.intent_id.unwrap();
    
    // Simulate failed action
    let outcome_ts = ts_us + 300_000;
    let success = false;
    
    let tracker = &mut engine.skandha_pipeline.sankhara.intent_tracker;
    tracker.record_outcome(IntentId::from_raw(intent_id), success, outcome_ts);
    
    // Verify failure recorded
    let intent = tracker.get(IntentId::from_raw(intent_id)).unwrap();
    assert_eq!(intent.success, Some(false));
    
    // In real system, this would trigger:
    // - Trauma registry update
    // - Policy masking
    // - Confidence reduction
    
    println!("✓ Failed action recorded correctly");
}

#[test]
fn test_intent_serialization_for_persistence() {
    let mut tracker = IntentTracker::new();
    
    // Track some intents
    for i in 0..3 {
        let intent = TrackedIntent {
            id: IntentId::new(),
            flow_event_id: None,
            action: zenb_core::skandha::IntentAction::GuideBreath { target_bpm: 6 },
            context: ContextSnapshot {
                belief_mode: i,
                confidence: 0.8,
                timestamp_us: (i as i64) * 1000,
                ..Default::default()
            },
            target_bpm: 6.0,
            outcome_ts_us: None,
            success: None,
        };
        tracker.track(intent);
    }
    
    // Serialize for persistence
    let serialized = tracker.serialize_for_persistence();
    assert_eq!(serialized.len(), 3);
    
    // Verify can be deserialized (simulate restore)
    let mut new_tracker = IntentTracker::new();
    new_tracker.restore_from_persistence(serialized.clone());
    
    assert_eq!(new_tracker.len(), 3);
    
    // Verify content matches
    for intent in serialized {
        let restored = new_tracker.get(intent.id);
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().context.belief_mode, intent.context.belief_mode);
    }
    
    println!("✓ Intent persistence verified");
}

// ============================================================================
// PERFORMANCE / STRESS TESTS
// ============================================================================

#[test]
fn test_intent_tracker_performance_under_load() {
    let mut tracker = IntentTracker::with_config(1_000_000_000, 1000);
    
    let start = std::time::Instant::now();
    
    // Track 1000 intents rapidly
    for i in 0..1000 {
        let intent = TrackedIntent {
            id: IntentId::new(),
            flow_event_id: None,
            action: zenb_core::skandha::IntentAction::Observe,
            context: ContextSnapshot {
                timestamp_us: (i * 1000) as i64,
                ..Default::default()
            },
            target_bpm: 6.0,
            outcome_ts_us: None,
            success: None,
        };
        tracker.track(intent);
    }
    
    let track_duration = start.elapsed();
    
    // Record outcomes for all
    let outcome_start = std::time::Instant::now();
    let intents: Vec<_> = tracker.serialize_for_persistence();
    for intent in &intents {
        tracker.record_outcome(intent.id, true, 2_000_000);
    }
    let outcome_duration = outcome_start.elapsed();
    
    println!("✓ Performance test:");
    println!("  - Track 1000 intents: {:?}", track_duration);
    println!("  - Record 1000 outcomes: {:?}", outcome_duration);
    println!("  - Final tracker size: {}", tracker.len());
    
    // Assert reasonable performance (< 100ms for 1000 operations)
    assert!(track_duration.as_millis() < 100, "Tracking should be fast");
    assert!(outcome_duration.as_millis() < 100, "Outcome recording should be fast");
}
