//! Tests for P0.1: Floating Point Determinism

use crate::domain::{BreathState, ControlDecision, Envelope, Event, SessionId};

#[test]
fn test_f32_to_canonical_normal_values() {
    // Test that normal values produce consistent fixed-point representation
    let state = BreathState {
        session_active: true,
        total_cycles: 100,
        last_decision: Some(ControlDecision {
            target_rate_bpm: 6.5,
            confidence: 0.85,
            recommended_poll_interval_ms: 1000,
        }),
        current_mode: Some(2),
        belief_conf: Some(0.75),
        belief_p: Some([0.1, 0.2, 0.5, 0.15, 0.05]),
        config_hash: None,
    };

    let hash1 = state.hash();
    let hash2 = state.hash();

    assert_eq!(
        hash1, hash2,
        "Hash must be deterministic for identical state"
    );
}

#[test]
fn test_f32_to_canonical_edge_cases() {
    // Test NaN handling
    let mut state1 = BreathState::default();
    state1.belief_conf = Some(f32::NAN);
    let hash1 = state1.hash();

    let mut state2 = BreathState::default();
    state2.belief_conf = Some(f32::NAN);
    let hash2 = state2.hash();

    assert_eq!(hash1, hash2, "NaN values must hash consistently");

    // Test infinity handling
    let mut state3 = BreathState::default();
    state3.belief_conf = Some(f32::INFINITY);
    let hash3 = state3.hash();

    let mut state4 = BreathState::default();
    state4.belief_conf = Some(f32::INFINITY);
    let hash4 = state4.hash();

    assert_eq!(hash3, hash4, "Infinity values must hash consistently");
}

#[test]
fn test_cross_platform_determinism() {
    // Simulate values that might differ across platforms
    let mut state = BreathState::default();

    // Very small differences that might occur due to floating point precision
    state.belief_conf = Some(0.123456789);
    state.belief_p = Some([0.2, 0.2, 0.2, 0.2, 0.2]);

    let hash1 = state.hash();

    // Simulate slight floating point drift (within 1 millionth)
    state.belief_conf = Some(0.123456788);
    let hash2 = state.hash();

    // Should be different since we're using fixed-point with 6 decimal precision
    assert_ne!(
        hash1, hash2,
        "Sub-microsecond differences should be detected"
    );

    // But identical values should always hash the same
    state.belief_conf = Some(0.123456789);
    let hash3 = state.hash();
    assert_eq!(
        hash1, hash3,
        "Identical values must produce identical hashes"
    );
}

#[test]
fn test_hash_changes_with_state_changes() {
    let mut state = BreathState::default();
    let hash_initial = state.hash();

    state.session_active = true;
    let hash_after_active = state.hash();
    assert_ne!(hash_initial, hash_after_active);

    state.total_cycles = 42;
    let hash_after_cycles = state.hash();
    assert_ne!(hash_after_active, hash_after_cycles);

    state.current_mode = Some(3);
    let hash_after_mode = state.hash();
    assert_ne!(hash_after_cycles, hash_after_mode);
}

#[test]
fn test_option_none_vs_some_hashing() {
    let mut state1 = BreathState::default();
    state1.belief_conf = None;
    let hash1 = state1.hash();

    let mut state2 = BreathState::default();
    state2.belief_conf = Some(0.0);
    let hash2 = state2.hash();

    assert_ne!(hash1, hash2, "None and Some(0.0) must hash differently");
}

#[test]
fn test_array_order_matters() {
    let mut state1 = BreathState::default();
    state1.belief_p = Some([0.1, 0.2, 0.3, 0.2, 0.2]);
    let hash1 = state1.hash();

    let mut state2 = BreathState::default();
    state2.belief_p = Some([0.2, 0.1, 0.3, 0.2, 0.2]);
    let hash2 = state2.hash();

    assert_ne!(hash1, hash2, "Array order must affect hash");
}

#[test]
fn test_replay_determinism_with_floats() {
    use crate::replay::replay_envelopes;

    let sid = SessionId::new();
    let envelopes = vec![
        Envelope {
            session_id: sid.clone(),
            seq: 1,
            ts_us: 1000,
            event: Event::SessionStarted {
                mode: "test".into(),
            },
            meta: serde_json::json!({}),
        },
        Envelope {
            session_id: sid.clone(),
            seq: 2,
            ts_us: 2000,
            event: Event::ControlDecisionMade {
                decision: ControlDecision {
                    target_rate_bpm: 6.123456,
                    confidence: 0.876543,
                    recommended_poll_interval_ms: 1000,
                },
            },
            meta: serde_json::json!({}),
        },
        Envelope {
            session_id: sid.clone(),
            seq: 3,
            ts_us: 3000,
            event: Event::BeliefUpdatedV2 {
                p: [0.15, 0.25, 0.35, 0.15, 0.10],
                conf: 0.82,
                mode: 2,
                free_energy_ema: 1.234,
                lr: 0.567,
                resonance_score: 0.89,
            },
            meta: serde_json::json!({}),
        },
    ];

    let state1 = replay_envelopes(&envelopes).unwrap();
    let state2 = replay_envelopes(&envelopes).unwrap();

    assert_eq!(state1.hash(), state2.hash(), "Replay must be deterministic");
}
