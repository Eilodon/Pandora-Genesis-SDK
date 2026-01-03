use std::fs;
use tempfile::NamedTempFile;
use zenb_core::domain::{ControlDecision, Envelope, Event, SessionId};
use zenb_store::EventStore;

fn master_key() -> [u8; 32] {
    // for tests, deterministic master key
    [42u8; 32]
}

#[test]
fn append_and_read_roundtrip() {
    let tf = NamedTempFile::new().unwrap();
    let path = tf.path().to_path_buf();
    let store = EventStore::open(&path, master_key()).unwrap();

    let sid = SessionId::new();
    store.create_session_key(&sid).unwrap();
    let e1 = Envelope {
        session_id: sid.clone(),
        seq: 1,
        ts_us: 1000,
        event: Event::SessionStarted {
            mode: "sleep".into(),
        },
        meta: serde_json::json!({}),
    };
    let e2 = Envelope {
        session_id: sid.clone(),
        seq: 2,
        ts_us: 2000,
        event: Event::ControlDecisionMade {
            decision: ControlDecision {
                target_rate_bpm: 5.0,
                confidence: 0.95,
            },
        },
        meta: serde_json::json!({"src":"sensor"}),
    };
    store.append_batch(&sid, &[e1.clone(), e2.clone()]).unwrap();

    let rows = store.read_events(&sid).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].seq, 1);
    assert_eq!(rows[1].seq, 2);
}

#[test]
fn tamper_detection() {
    let tf = NamedTempFile::new().unwrap();
    let path = tf.path().to_path_buf();
    let store = EventStore::open(&path, master_key()).unwrap();

    let sid = SessionId::new();
    store.create_session_key(&sid).unwrap();
    let e1 = Envelope {
        session_id: sid.clone(),
        seq: 1,
        ts_us: 1000,
        event: Event::SessionStarted {
            mode: "sleep".into(),
        },
        meta: serde_json::json!({}),
    };
    store.append_batch(&sid, &[e1.clone()]).unwrap();

    // tamper with payload bytes directly
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute("UPDATE events SET payload = x'00' WHERE seq = 1", ())
        .unwrap();

    let res = store.read_events(&sid);
    assert!(res.is_err());
}
