use zenb_core::domain::{ControlDecision, Envelope, Event, SessionId};

fn main() {
    // minimal in-memory demo (no persistence) showcasing deterministic behavior
    let sid = SessionId::new();
    let mut seq = 1u64;
    let mut state = zenb_core::domain::BreathState::default();

    for tick in 0..10u64 {
        let ts_us = (tick * 100_000) as i64;
        let env = Envelope {
            session_id: sid.clone(),
            seq,
            ts_us,
            event: Event::ControlDecisionMade {
                decision: ControlDecision {
                    target_rate_bpm: 6.0,
                    confidence: 0.9,
                },
            },
            meta: serde_json::json!({}),
        };
        state.apply(&env);
        seq += 1;
    }

    let h = state.hash();
    println!("WASM demo state hash: {}", hex::encode(h));
}
