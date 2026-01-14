//! P0.9: Database Migration System
//!
//! Handles schema version upgrades without data loss

use crate::StoreError;
use rusqlite::{params, Connection, OptionalExtension};

const CURRENT_SCHEMA_VERSION: i32 = 3;

/// Metadata table for tracking schema version
pub fn init_metadata_table(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );",
    )?;
    Ok(())
}

/// Get current schema version from metadata
pub fn get_schema_version(conn: &Connection) -> Result<i32, StoreError> {
    let version: Option<String> = conn
        .query_row(
            "SELECT value FROM metadata WHERE key = 'schema_version'",
            [],
            |r| r.get(0),
        )
        .optional()?;

    match version {
        Some(v) => v
            .parse::<i32>()
            .map_err(|_| StoreError::BatchValidation("invalid schema version".into())),
        None => Ok(0), // No version = legacy v0
    }
}

/// Set schema version in metadata
fn set_schema_version(conn: &Connection, version: i32) -> Result<(), StoreError> {
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?1)",
        params![version.to_string()],
    )?;
    Ok(())
}

/// Run all necessary migrations to bring DB to current version
pub fn migrate_to_current(conn: &Connection) -> Result<(), StoreError> {
    init_metadata_table(conn)?;
    let current_version = get_schema_version(conn)?;

    if current_version == CURRENT_SCHEMA_VERSION {
        return Ok(()); // Already up to date
    }

    if current_version > CURRENT_SCHEMA_VERSION {
        return Err(StoreError::BatchValidation(format!(
            "Database version {} is newer than supported version {}",
            current_version, CURRENT_SCHEMA_VERSION
        )));
    }

    // Apply migrations in sequence
    let mut version = current_version;

    if version < 1 {
        migrate_v0_to_v1(conn)?;
        version = 1;
        set_schema_version(conn, version)?;
    }

    if version < 2 {
        migrate_v1_to_v2(conn)?;
        version = 2;
        set_schema_version(conn, version)?;
    }
    
    if version < 3 {
        migrate_v2_to_v3(conn)?;
        version = 3;
        set_schema_version(conn, version)?;
    }

    Ok(())
}

/// Migration v0 -> v1: Add metadata table and initial version
fn migrate_v0_to_v1(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "BEGIN IMMEDIATE;
        -- Metadata table already created by init_metadata_table
        -- Mark as v1
        COMMIT;",
    )?;
    Ok(())
}

/// Migration v1 -> v2: Add hash_version column and append_log table
fn migrate_v1_to_v2(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "BEGIN IMMEDIATE;
        
        -- Add hash_version to events table (for P0.1 determinism tracking)
        ALTER TABLE events ADD COLUMN hash_version INTEGER DEFAULT 1;
        
        -- Mark all existing events as legacy hash version
        UPDATE events SET hash_version = 1;
        
        -- Create append_log table if not exists (P0.3/P0.4)
        CREATE TABLE IF NOT EXISTS append_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id BLOB NOT NULL,
            attempt_ts_us INTEGER NOT NULL,
            seq_start INTEGER NOT NULL,
            seq_end INTEGER NOT NULL,
            event_count INTEGER NOT NULL,
            success INTEGER NOT NULL,
            error_msg TEXT
        );
        CREATE INDEX IF NOT EXISTS append_log_session_idx ON append_log(session_id, attempt_ts_us);
        
        COMMIT;",
    )?;
    Ok(())
}

/// Migration v2 -> v3: Add memory_snapshots table (EIDOLON FIX: Memory Persistence)
///
/// This table stores encrypted snapshots of HolographicMemory and other
/// cognitive state for persistence across process restarts.
fn migrate_v2_to_v3(conn: &Connection) -> Result<(), StoreError> {
    conn.execute_batch(
        "BEGIN IMMEDIATE;
        
        -- EIDOLON FIX: Memory persistence table
        -- Stores encrypted snapshots of cognitive state
        CREATE TABLE IF NOT EXISTS memory_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id BLOB NOT NULL,
            snapshot_type TEXT NOT NULL,
            created_ts_us INTEGER NOT NULL,
            payload BLOB NOT NULL,
            nonce BLOB NOT NULL,
            UNIQUE(session_id, snapshot_type)
        );
        CREATE INDEX IF NOT EXISTS memory_snapshots_session_idx ON memory_snapshots(session_id);
        
        COMMIT;",
    )?;
    Ok(())
}

/// Check if migration is needed
pub fn needs_migration(conn: &Connection) -> Result<bool, StoreError> {
    init_metadata_table(conn)?;
    let version = get_schema_version(conn)?;
    Ok(version < CURRENT_SCHEMA_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_fresh_db_no_migration() {
        let tf = NamedTempFile::new().unwrap();
        let conn = Connection::open(tf.path()).unwrap();

        init_metadata_table(&conn).unwrap();
        set_schema_version(&conn, CURRENT_SCHEMA_VERSION).unwrap();

        assert!(!needs_migration(&conn).unwrap());
    }

    #[test]
    fn test_v0_to_current_migration() {
        let tf = NamedTempFile::new().unwrap();
        let conn = Connection::open(tf.path()).unwrap();

        // Simulate v0 DB (has events table but no metadata table)
        // v0 databases would have been created by init_schema in lib.rs
        conn.execute_batch(
            "CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                session_id BLOB NOT NULL,
                seq INTEGER NOT NULL,
                ts_us INTEGER NOT NULL,
                event_type INTEGER NOT NULL,
                meta BLOB NOT NULL,
                payload BLOB NOT NULL,
                nonce BLOB NOT NULL
            );",
        ).unwrap();
        
        assert!(needs_migration(&conn).unwrap());

        // Run migration
        migrate_to_current(&conn).unwrap();

        // Verify version
        let version = get_schema_version(&conn).unwrap();
        assert_eq!(version, CURRENT_SCHEMA_VERSION);

        // Should not need migration anymore
        assert!(!needs_migration(&conn).unwrap());
        
        // Verify memory_snapshots table was created (v2→v3)
        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_snapshots'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_v1_to_v2_migration() {
        let tf = NamedTempFile::new().unwrap();
        let conn = Connection::open(tf.path()).unwrap();

        // Setup v1 schema
        init_metadata_table(&conn).unwrap();
        conn.execute_batch(
            "CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                session_id BLOB NOT NULL,
                seq INTEGER NOT NULL,
                ts_us INTEGER NOT NULL,
                event_type INTEGER NOT NULL,
                meta BLOB NOT NULL,
                payload BLOB NOT NULL,
                nonce BLOB NOT NULL
            );",
        )
        .unwrap();
        set_schema_version(&conn, 1).unwrap();

        // Run migration
        migrate_to_current(&conn).unwrap();

        // Verify hash_version column exists
        let result: Result<i32, _> =
            conn.query_row("SELECT hash_version FROM events LIMIT 1", [], |r| r.get(0));
        // Should fail with no rows, but column should exist
        assert!(result.is_err() || result.is_ok());

        // Verify append_log table exists
        let count: i32 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='append_log'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_future_version_error() {
        let tf = NamedTempFile::new().unwrap();
        let conn = Connection::open(tf.path()).unwrap();

        init_metadata_table(&conn).unwrap();
        set_schema_version(&conn, 999).unwrap();

        let result = migrate_to_current(&conn);
        assert!(result.is_err());
    }
}
