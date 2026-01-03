use tempfile::NamedTempFile;
use zenb_store::EventStore;

fn master_key() -> [u8; 32] {
    [42u8; 32]
}

#[test]
fn severity_decays_with_time() {
    let tf = NamedTempFile::new().unwrap();
    let store = EventStore::open(tf.path(), master_key()).unwrap();

    let sig = [7u8; 32];
    let now = 0i64;
    let decay_per_day = 1.0f32;

    store
        .record_trauma(&sig, 1, 10, 5, 1.0, now, decay_per_day)
        .unwrap();

    let hit0 = store.query_trauma(&sig, now).unwrap().unwrap();
    assert!(hit0.sev_eff > 0.9);

    // 1 day later => exp(-1)
    let one_day_us: i64 = 86_400_000_000;
    let hit1 = store.query_trauma(&sig, now + one_day_us).unwrap().unwrap();
    assert!(hit1.sev_eff < hit0.sev_eff);
    assert!((hit1.sev_eff - (1.0f32 * (-1.0f32).exp())).abs() < 1e-3);
}

#[test]
fn inhibit_until_blocks_even_if_sev_low() {
    let tf = NamedTempFile::new().unwrap();
    let store = EventStore::open(tf.path(), master_key()).unwrap();

    let sig = [9u8; 32];

    // Insert row by recording once, then set inhibit_until_ts_us directly.
    store.record_trauma(&sig, 1, 10, 5, 0.01, 0, 10.0).unwrap();

    let db = rusqlite::Connection::open(tf.path()).unwrap();
    db.execute(
        "UPDATE trauma_registry SET inhibit_until_ts_us = ?1 WHERE sig_hash = ?2",
        rusqlite::params![1000i64, &sig as &[u8]],
    )
    .unwrap();

    let hit = store.query_trauma(&sig, 10).unwrap().unwrap();
    assert_eq!(hit.inhibit_until_ts_us, 1000);
    assert!(hit.sev_eff < 0.02);
}
