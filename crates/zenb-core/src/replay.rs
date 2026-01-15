use crate::domain::{BreathState, DomainError, Envelope};

pub fn replay_envelopes(envelopes: &[Envelope]) -> Result<BreathState, DomainError> {
    let mut state = BreathState::default();
    for (i, e) in envelopes.iter().enumerate() {
        // basic sequence validation
        if e.seq != (i as u64 + 1) {
            return Err(DomainError::InvalidSequence {
                expected: i as u64 + 1,
                got: e.seq,
            });
        }
        state.apply(e);
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Envelope, Event, SessionId};

    #[test]
    fn replay_is_deterministic() {
        let sid = SessionId::new();
        let e1 = Envelope {
            session_id: sid.clone(),
            seq: 1,
            ts_us: 1000,
            event: Event::SessionStarted {
                mode: "gentle".into(),
            },
            meta: serde_json::json!({}),
        };
        let e2 = Envelope {
            session_id: sid.clone(),
            seq: 2,
            ts_us: 2000,
            event: Event::ControlDecisionMade {
                decision: crate::domain::ControlDecision {
                    target_rate_bpm: 6.0,
                    confidence: 0.9,
                    recommended_poll_interval_ms: 1000,
                    intent_id: None,
                },
            },
            meta: serde_json::json!({}),
        };
        let envs = vec![e1.clone(), e2.clone()];
        let s1 = replay_envelopes(&envs).unwrap();
        let s2 = replay_envelopes(&envs).unwrap();
        assert_eq!(s1.hash(), s2.hash());
    }
}
