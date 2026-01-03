use zenb_core::breath_engine::{BreathEngine, BreathMode};
use zenb_core::domain::BreathState;
use zenb_core::engine::Engine;
use zenb_core::phase_machine::PhaseDurations;

#[test]
fn long_session_fixed_to_dynamic_hash() {
    // create engine with fixed 4-7-8 (in seconds -> microseconds)
    let d = PhaseDurations {
        inhale_us: 4_000_000,
        hold_in_us: 7_000_000,
        exhale_us: 8_000_000,
        hold_out_us: 0,
    };
    let mut eng = Engine::new(6.0);
    eng.breath.set_fixed_pattern(d);
    eng.update_context(crate::belief::Context {
        local_hour: 12,
        is_charging: true,
        recent_sessions: 0,
    });

    let mut ts_us: i64 = 0;
    // simulate 2 minutes of fixed pattern (advance in 1s steps)
    for _ in 0..(2 * 60) {
        eng.tick(1_000_000);
        // noisy sensor occasionally but generally good
        let features = vec![60.0, 40.0, 6.0, 0.9, 0.1];
        eng.ingest_sensor(&features, ts_us);
        ts_us += 1_000_000;
    }

    // switch to dynamic (biofeedback)
    eng.breath.set_target_bpm(6.0);

    // simulate remaining 8 minutes with occasional sensor noise and a battery unplug event
    for i in 0..(8 * 60) {
        if i == 10 {
            // simulate unplugging at 2m10s into dynamic phase
            eng.update_context(crate::belief::Context {
                local_hour: 12,
                is_charging: false,
                recent_sessions: 1,
            });
        }
        // sprinkle noisy readings to sometimes reduce confidence
        if i % 13 == 0 {
            eng.ingest_sensor(&[60.0, 1.0, 6.0, 0.0, 1.0], ts_us);
        } else {
            eng.ingest_sensor(&[60.0, 40.0, 6.0, 0.9, 0.0], ts_us);
        }
        eng.tick(1_000_000);
        ts_us += 1_000_000;
    }

    // deterministic hash check
    let st = &eng.belief_state; // use belief_state as representative
    let mut bstate1 = BreathState::default();
    bstate1.apply(&zenb_core::domain::Envelope {
        session_id: zenb_core::domain::SessionId::new(),
        seq: 0,
        ts_us: ts_us,
        event: zenb_core::domain::Event::BeliefUpdatedV2 {
            p: st.p,
            conf: st.conf,
            mode: st.mode as u8,
            free_energy_ema: eng.fep_state.free_energy_ema,
            lr: eng.fep_state.lr,
            resonance_score: eng.resonance_score_ema,
        },
        meta: serde_json::json!({}),
    });
    let h1 = bstate1.hash();

    // Repeat the same deterministic simulation and expect same hash
    let mut eng2 = Engine::new(6.0);
    eng2.breath.set_fixed_pattern(PhaseDurations {
        inhale_us: 4_000_000,
        hold_in_us: 7_000_000,
        exhale_us: 8_000_000,
        hold_out_us: 0,
    });
    eng2.update_context(crate::belief::Context {
        local_hour: 12,
        is_charging: true,
        recent_sessions: 0,
    });
    let mut ts2: i64 = 0;
    for _ in 0..(2 * 60) {
        eng2.tick(1_000_000);
        eng2.ingest_sensor(&vec![60.0, 40.0, 6.0, 0.9, 0.1], ts2);
        ts2 += 1_000_000;
    }
    eng2.breath.set_target_bpm(6.0);
    for i in 0..(8 * 60) {
        if i == 10 {
            eng2.update_context(crate::belief::Context {
                local_hour: 12,
                is_charging: false,
                recent_sessions: 1,
            });
        }
        if i % 13 == 0 {
            eng2.ingest_sensor(&[60.0, 1.0, 6.0, 0.0, 1.0], ts2);
        } else {
            eng2.ingest_sensor(&[60.0, 40.0, 6.0, 0.9, 0.0], ts2);
        }
        eng2.tick(1_000_000);
        ts2 += 1_000_000;
    }
    let mut bstate2 = BreathState::default();
    let st2 = &eng2.belief_state;
    bstate2.apply(&zenb_core::domain::Envelope {
        session_id: zenb_core::domain::SessionId::new(),
        seq: 0,
        ts_us: ts2,
        event: zenb_core::domain::Event::BeliefUpdatedV2 {
            p: st2.p,
            conf: st2.conf,
            mode: st2.mode as u8,
            free_energy_ema: eng2.fep_state.free_energy_ema,
            lr: eng2.fep_state.lr,
            resonance_score: eng2.resonance_score_ema,
        },
        meta: serde_json::json!({}),
    });
    let h2 = bstate2.hash();
    assert_eq!(h1, h2, "Hashes must be deterministic across identical runs");
}
