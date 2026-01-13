//! Encrypted SQLite event store with session keys and batch append.

use blake3::Hasher;
use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{Key, KeyInit, XChaCha20Poly1305, XNonce};
use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json;
use sha2::Sha256;
use std::path::Path;
use thiserror::Error;
use zeroize::Zeroize;

use zenb_core::domain::{Envelope, SessionId};
use zenb_core::safety_swarm;

pub mod migration;

#[derive(Error, Debug)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("crypto error: {0}")]
    CryptoError(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid sequence: expected {expected}, got {got}, session={session}")]
    InvalidSequence {
        expected: u64,
        got: u64,
        session: String,
    },
    #[error("sequence conflict: {inserted} of {total} events inserted, possible race condition")]
    SequenceConflict { inserted: usize, total: usize },
    #[error("batch validation failed: {0}")]
    BatchValidation(String),
}

impl safety_swarm::TraumaSource for EventStore {
    fn query_trauma(
        &self,
        sig_hash: &[u8],
        now_ts_us: i64,
    ) -> Result<Option<safety_swarm::TraumaHit>, String> {
        match EventStore::query_trauma(self, sig_hash, now_ts_us) {
            Ok(Some(hit)) => Ok(Some(safety_swarm::TraumaHit {
                sev_eff: hit.sev_eff,
                count: hit.count,
                inhibit_until_ts_us: hit.inhibit_until_ts_us,
                last_ts_us: hit.last_ts_us,
            })),
            Ok(None) => Ok(None),
            Err(e) => Err(format!("{:?}", e)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraumaHit {
    pub sev_eff: f32,
    pub count: u32,
    pub inhibit_until_ts_us: i64,
    pub last_ts_us: i64,
}

pub struct SessionKey([u8; 32]);

impl SessionKey {
    pub fn random() -> Self {
        let mut k = [0u8; 32];
        // SECURITY: Use OsRng for cryptographic key generation
        // OsRng is guaranteed to be cryptographically secure on all platforms
        OsRng.fill_bytes(&mut k);
        SessionKey(k)
    }
}
impl Drop for SessionKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

pub struct EventStore {
    conn: Connection,
    master_key: [u8; 32],
}

impl EventStore {
    pub fn open<P: AsRef<Path>>(path: P, master_key: [u8; 32]) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;
        // performance pragmas
        conn.pragma_update(None, "journal_mode", &"WAL")?;
        conn.pragma_update(None, "synchronous", &"NORMAL")?;
        let s = EventStore { conn, master_key };
        s.init_schema()?;

        // Run database migration to current version
        migration::migrate_to_current(&s.conn)?;

        Ok(s)
    }

    pub fn init_schema(&self) -> Result<(), StoreError> {
        self.conn.execute_batch(
            "BEGIN;
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id BLOB NOT NULL,
                seq INTEGER NOT NULL,
                ts_us INTEGER NOT NULL,
                event_type INTEGER NOT NULL,
                meta BLOB NOT NULL,
                payload BLOB NOT NULL,
                nonce BLOB NOT NULL,
                UNIQUE(session_id, seq)
            );
            CREATE INDEX IF NOT EXISTS events_session_idx ON events(session_id, seq);
            CREATE TABLE IF NOT EXISTS session_keys (
                session_id BLOB PRIMARY KEY,
                wrapped_key BLOB NOT NULL,
                wrap_nonce BLOB NOT NULL,
                created_ts_us INTEGER NOT NULL,
                kdf_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trauma_registry (
                sig_hash BLOB PRIMARY KEY,
                mode INTEGER NOT NULL,
                pattern_id INTEGER NOT NULL,
                goal INTEGER NOT NULL,
                severity_ema REAL NOT NULL,
                count INTEGER NOT NULL,
                last_ts_us INTEGER NOT NULL,
                decay_rate REAL NOT NULL,
                inhibit_until_ts_us INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS trauma_registry_mode_pattern_idx ON trauma_registry(mode, pattern_id);
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
            COMMIT;"
        )?;
        Ok(())
    }

    pub fn record_trauma(
        &self,
        sig_hash: &[u8],
        mode: i64,
        pattern_id: i64,
        goal: i64,
        severity: f32,
        now_ts_us: i64,
        decay_rate_default: f32,
    ) -> Result<(), StoreError> {
        const BETA: f32 = 0.2;

        let row: Option<(f32, i64, i64, f32)> = self
            .conn
            .query_row(
                "SELECT severity_ema, count, inhibit_until_ts_us, decay_rate FROM trauma_registry WHERE sig_hash = ?1",
                params![sig_hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;

        let (new_sev, new_count, inhibit_until, decay_rate) = match row {
            Some((old_sev, old_count, old_inhibit, old_decay)) => {
                let sev = ema(old_sev, severity, BETA);
                let cnt = old_count.saturating_add(1);
                let inhibit = old_inhibit;
                let dr = if old_decay > 0.0 {
                    old_decay
                } else {
                    decay_rate_default
                };
                (sev, cnt, inhibit, dr)
            }
            None => (severity, 1, 0, decay_rate_default),
        };

        self.conn.execute(
            "INSERT INTO trauma_registry (sig_hash, mode, pattern_id, goal, severity_ema, count, last_ts_us, decay_rate, inhibit_until_ts_us)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(sig_hash) DO UPDATE SET
               mode = excluded.mode,
               pattern_id = excluded.pattern_id,
               goal = excluded.goal,
               severity_ema = excluded.severity_ema,
               count = excluded.count,
               last_ts_us = excluded.last_ts_us,
               decay_rate = excluded.decay_rate,
               inhibit_until_ts_us = excluded.inhibit_until_ts_us",
            params![
                sig_hash,
                mode,
                pattern_id,
                goal,
                new_sev,
                new_count,
                now_ts_us,
                decay_rate,
                inhibit_until,
            ],
        )?;

        Ok(())
    }

    /// Record trauma with explicit inhibit_until timestamp.
    /// This variant is used by the learning system to implement exponential backoff.
    pub fn record_trauma_with_inhibit(
        &self,
        sig_hash: &[u8],
        mode: i64,
        pattern_id: i64,
        goal: i64,
        severity: f32,
        now_ts_us: i64,
        inhibit_until_ts_us: i64,
        decay_rate_default: f32,
    ) -> Result<(), StoreError> {
        const BETA: f32 = 0.2;

        let row: Option<(f32, i64, f32)> = self
            .conn
            .query_row(
                "SELECT severity_ema, count, decay_rate FROM trauma_registry WHERE sig_hash = ?1",
                params![sig_hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;

        let (new_sev, new_count, decay_rate) = match row {
            Some((old_sev, old_count, old_decay)) => {
                let sev = ema(old_sev, severity, BETA);
                let cnt = old_count.saturating_add(1);
                let dr = if old_decay > 0.0 {
                    old_decay
                } else {
                    decay_rate_default
                };
                (sev, cnt, dr)
            }
            None => (severity, 1, decay_rate_default),
        };

        self.conn.execute(
            "INSERT INTO trauma_registry (sig_hash, mode, pattern_id, goal, severity_ema, count, last_ts_us, decay_rate, inhibit_until_ts_us)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(sig_hash) DO UPDATE SET
               mode = excluded.mode,
               pattern_id = excluded.pattern_id,
               goal = excluded.goal,
               severity_ema = excluded.severity_ema,
               count = excluded.count,
               last_ts_us = excluded.last_ts_us,
               decay_rate = excluded.decay_rate,
               inhibit_until_ts_us = excluded.inhibit_until_ts_us",
            params![
                sig_hash,
                mode,
                pattern_id,
                goal,
                new_sev,
                new_count,
                now_ts_us,
                decay_rate,
                inhibit_until_ts_us,
            ],
        )?;

        Ok(())
    }

    pub fn query_trauma(
        &self,
        sig_hash: &[u8],
        now_ts_us: i64,
    ) -> Result<Option<TraumaHit>, StoreError> {
        let row: Option<(f32, i64, i64, f32, i64)> = self
            .conn
            .query_row(
                "SELECT severity_ema, count, inhibit_until_ts_us, decay_rate, last_ts_us FROM trauma_registry WHERE sig_hash = ?1",
                params![sig_hash],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .optional()?;

        let Some((severity_ema, count, inhibit_until_ts_us, decay_rate, last_ts_us)) = row else {
            return Ok(None);
        };

        let dt_us = (now_ts_us - last_ts_us).max(0) as f64;
        let dt_days = dt_us / 86_400_000_000f64;
        let sev_eff = (severity_ema as f64) * (-(decay_rate as f64) * dt_days).exp();

        Ok(Some(TraumaHit {
            sev_eff: (sev_eff as f32).max(0.0),
            count: count.max(0) as u32,
            inhibit_until_ts_us,
            last_ts_us,
        }))
    }

    /// Load active trauma entries for cache hydration on startup.
    ///
    /// Selects trauma entries that are either:
    /// - Still inhibited (inhibit_until_ts_us > now)
    /// - High severity (severity_ema > threshold)
    ///
    /// Orders by most recent first and limits the result set.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of trauma entries to load (e.g., 1000)
    ///
    /// # Returns
    /// Vector of (sig_hash, TraumaHit) tuples for cache population
    pub fn load_active_trauma(
        &self,
        limit: usize,
    ) -> Result<Vec<([u8; 32], TraumaHit)>, StoreError> {
        let now_ts_us = chrono::Utc::now().timestamp_micros();
        const SEVERITY_THRESHOLD: f32 = 0.1; // Load traumas with severity > 0.1

        let mut stmt = self.conn.prepare(
            "SELECT sig_hash, severity_ema, count, inhibit_until_ts_us, decay_rate, last_ts_us 
             FROM trauma_registry 
             WHERE inhibit_until_ts_us > ?1 OR severity_ema > ?2
             ORDER BY last_ts_us DESC 
             LIMIT ?3",
        )?;

        let mut rows = stmt.query(params![now_ts_us, SEVERITY_THRESHOLD, limit as i64])?;
        let mut results = Vec::new();

        while let Some(row) = rows.next()? {
            let sig_hash_blob: Vec<u8> = row.get(0)?;
            let severity_ema: f32 = row.get(1)?;
            let count: i64 = row.get(2)?;
            let inhibit_until_ts_us: i64 = row.get(3)?;
            let decay_rate: f32 = row.get(4)?;
            let last_ts_us: i64 = row.get(5)?;

            // Convert sig_hash blob to [u8; 32]
            if sig_hash_blob.len() != 32 {
                continue; // Skip invalid entries
            }
            let mut sig_hash = [0u8; 32];
            sig_hash.copy_from_slice(&sig_hash_blob);

            // Apply decay to get effective severity
            let dt_us = (now_ts_us - last_ts_us).max(0) as f64;
            let dt_days = dt_us / 86_400_000_000f64;
            let sev_eff = (severity_ema as f64) * (-(decay_rate as f64) * dt_days).exp();

            let trauma_hit = TraumaHit {
                sev_eff: (sev_eff as f32).max(0.0),
                count: count.max(0) as u32,
                inhibit_until_ts_us,
                last_ts_us,
            };

            results.push((sig_hash, trauma_hit));
        }

        Ok(results)
    }

    fn xchacha(&self, key: &[u8; 32]) -> XChaCha20Poly1305 {
        XChaCha20Poly1305::new(Key::from_slice(key))
    }

    /// Derive a wrapping key from the master key using HKDF-SHA256.
    fn derive_wrapping_key(&self) -> Result<[u8; 32], StoreError> {
        const INFO: &[u8] = b"zenb-session-key-wrap-v1";
        let hk = Hkdf::<Sha256>::new(None, &self.master_key);
        let mut okm = [0u8; 32];
        hk.expand(INFO, &mut okm)
            .map_err(|e| StoreError::CryptoError(format!("hkdf expand failed: {:?}", e)))?;
        Ok(okm)
    }

    /// AEAD for wrapping/unwrapping session keys.
    fn wrapping_aead(&self) -> Result<XChaCha20Poly1305, StoreError> {
        let wrap_key = self.derive_wrapping_key()?;
        Ok(self.xchacha(&wrap_key))
    }

    pub fn create_session_key(&self, session_id: &SessionId) -> Result<(), StoreError> {
        // generate session key and wrap with master_key
        let sk = SessionKey::random();
        let mut nonce = [0u8; 24];
        OsRng.fill_bytes(&mut nonce);
        let aead = self.wrapping_aead()?;
        let ciphertext = aead
            .encrypt(XNonce::from_slice(&nonce), &sk.0[..])
            .map_err(|e| StoreError::CryptoError(format!("{:?}", e)))?;
        let ts = chrono::Utc::now().timestamp_micros();
        self.conn.execute(
            "INSERT OR REPLACE INTO session_keys (session_id, wrapped_key, wrap_nonce, created_ts_us, kdf_version) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id.as_bytes() as &[u8], ciphertext, &nonce as &[u8], ts, 1u32],
        )?;
        Ok(())
    }

    pub fn load_session_key(&self, session_id: &SessionId) -> Result<SessionKey, StoreError> {
        let row: Option<(Vec<u8>, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT wrapped_key, wrap_nonce FROM session_keys WHERE session_id = ?1",
                params![session_id.as_bytes() as &[u8]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (wrapped, nonce) = row.ok_or_else(|| {
            StoreError::NotFound(format!(
                "session key not found for session {:?}",
                session_id.as_bytes()
            ))
        })?;
        let aead = self.wrapping_aead()?;
        let pk = aead
            .decrypt(XNonce::from_slice(&nonce), wrapped.as_slice())
            .map_err(|e| {
                StoreError::CryptoError(format!("failed to decrypt session key: {:?}", e))
            })?;
        if pk.len() != 32 {
            return Err(StoreError::CryptoError(format!(
                "invalid session key length: {}",
                pk.len()
            )));
        }
        let mut k = [0u8; 32];
        k.copy_from_slice(&pk);
        Ok(SessionKey(k))
    }

    pub fn delete_session_keys(&self, session_id: &SessionId) -> Result<(), StoreError> {
        self.conn.execute(
            "DELETE FROM session_keys WHERE session_id = ?1",
            params![session_id.as_bytes() as &[u8]],
        )?;
        self.conn.execute(
            "DELETE FROM events WHERE session_id = ?1",
            params![session_id.as_bytes() as &[u8]],
        )?;
        Ok(())
    }

    pub fn get_last_seq(&self, session_id: &SessionId) -> Result<u64, StoreError> {
        let r: Option<i64> = self
            .conn
            .query_row(
                "SELECT MAX(seq) FROM events WHERE session_id = ?1",
                params![session_id.as_bytes() as &[u8]],
                |r| r.get(0),
            )
            .optional()?;
        Ok(r.unwrap_or(0) as u64)
    }

    pub fn append_batch(
        &mut self,
        session_id: &SessionId,
        envelopes: &[Envelope],
    ) -> Result<(), StoreError> {
        if envelopes.is_empty() {
            return Ok(());
        }

        let attempt_ts_us = chrono::Utc::now().timestamp_micros();
        let seq_start = envelopes[0].seq;
        let seq_end = envelopes.last().unwrap().seq;
        let event_count = envelopes.len();

        // Validate batch sequence continuity
        for (i, env) in envelopes.iter().enumerate() {
            let expected_seq = seq_start + i as u64;
            if env.seq != expected_seq {
                let err_msg = format!(
                    "batch sequence gap at index {}: expected {}, got {}",
                    i, expected_seq, env.seq
                );
                self.log_append_attempt(
                    session_id,
                    attempt_ts_us,
                    seq_start,
                    seq_end,
                    event_count,
                    false,
                    Some(&err_msg),
                )?;
                return Err(StoreError::BatchValidation(err_msg));
            }
        }

        // Load session key before transaction
        let sk = self.load_session_key(session_id)?;
        let aead = self.xchacha(&sk.0);

        // Start IMMEDIATE transaction to lock database and prevent TOCTOU
        let mut tx = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Validate sequence against current DB state (inside transaction)
        let db_max_seq: u64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?1",
            params![session_id.as_bytes() as &[u8]],
            |r| r.get(0),
        )?;

        if seq_start != db_max_seq + 1 {
            let err_msg = format!(
                "sequence mismatch: expected {}, got {}",
                db_max_seq + 1,
                seq_start
            );
            // Log before rollback
            let _ = tx.execute(
                "INSERT INTO append_log (session_id, attempt_ts_us, seq_start, seq_end, event_count, success, error_msg) VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
                params![session_id.as_bytes() as &[u8], attempt_ts_us, seq_start as i64, seq_end as i64, event_count as i64, &err_msg]
            );
            let _ = tx.rollback();
            return Err(StoreError::InvalidSequence {
                expected: db_max_seq + 1,
                got: seq_start,
                session: hex::encode(session_id.as_bytes()),
            });
        }

        // Use INSERT OR IGNORE for idempotency (protects against duplicate seq)
        let mut inserted = 0usize;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO events (session_id, seq, ts_us, event_type, meta, payload, nonce) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
            )?;

            for env in envelopes {
                // Encrypt the event payload
                let payload_plain = serde_json::to_vec(&env.event).map_err(|e| {
                    StoreError::CryptoError(format!("event serialization failed: {}", e))
                })?;

                // Build AAD
                let meta_bytes = serde_json::to_vec(&env.meta).map_err(|e| {
                    StoreError::CryptoError(format!("meta serialization failed: {}", e))
                })?;
                let mut hasher = Hasher::new();
                hasher.update(&meta_bytes);
                let meta_hash = hasher.finalize();

                let mut aad = Vec::new();
                aad.extend_from_slice(env.session_id.as_bytes());
                aad.extend_from_slice(&env.seq.to_le_bytes());
                aad.extend_from_slice(&env.event_type_code().to_le_bytes());
                aad.extend_from_slice(&env.ts_us.to_le_bytes());
                aad.extend_from_slice(meta_hash.as_bytes());

                let mut nonce = [0u8; 24];
                OsRng.fill_bytes(&mut nonce);
                let ct = aead
                    .encrypt(
                        XNonce::from_slice(&nonce),
                        Payload {
                            msg: &payload_plain,
                            aad: &aad,
                        },
                    )
                    .map_err(|e| StoreError::CryptoError(format!("encryption failed: {:?}", e)))?;
                let changes = stmt.execute(params![
                    env.session_id.as_bytes() as &[u8],
                    env.seq as i64,
                    env.ts_us as i64,
                    env.event_type_code() as i64,
                    env.meta_as_bytes()
                        .map_err(|e| StoreError::CryptoError(format!(
                            "meta serialization failed: {}",
                            e
                        )))?,
                    ct,
                    nonce,
                ])?;
                inserted += changes;
            }
        }

        if inserted != envelopes.len() {
            let err_msg = format!("only {} of {} events inserted", inserted, envelopes.len());
            let _ = tx.execute(
                "INSERT INTO append_log (session_id, attempt_ts_us, seq_start, seq_end, event_count, success, error_msg) VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
                params![session_id.as_bytes() as &[u8], attempt_ts_us, seq_start as i64, seq_end as i64, event_count as i64, &err_msg]
            );
            let _ = tx.rollback();
            return Err(StoreError::SequenceConflict {
                inserted,
                total: envelopes.len(),
            });
        }

        // Log successful append
        tx.execute(
            "INSERT INTO append_log (session_id, attempt_ts_us, seq_start, seq_end, event_count, success, error_msg) VALUES (?1, ?2, ?3, ?4, ?5, 1, NULL)",
            params![session_id.as_bytes() as &[u8], attempt_ts_us, seq_start as i64, seq_end as i64, event_count as i64]
        )?;

        tx.commit()?;
        Ok(())
    }

    fn log_append_attempt(
        &self,
        session_id: &SessionId,
        attempt_ts_us: i64,
        seq_start: u64,
        seq_end: u64,
        event_count: usize,
        success: bool,
        error_msg: Option<&str>,
    ) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT INTO append_log (session_id, attempt_ts_us, seq_start, seq_end, event_count, success, error_msg) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![session_id.as_bytes() as &[u8], attempt_ts_us, seq_start as i64, seq_end as i64, event_count as i64, if success { 1 } else { 0 }, error_msg]
        )?;
        Ok(())
    }

    pub fn read_events(&self, session_id: &SessionId) -> Result<Vec<Envelope>, StoreError> {
        let sk = self.load_session_key(session_id)?;
        let aead = self.xchacha(&sk.0);
        let mut stmt = self.conn.prepare("SELECT seq, ts_us, event_type, meta, payload, nonce FROM events WHERE session_id = ?1 ORDER BY seq ASC")?;
        let mut rows = stmt.query(params![session_id.as_bytes() as &[u8]])?;
        let mut out = Vec::new();
        while let Some(r) = rows.next()? {
            let seq: i64 = r.get(0)?;
            let ts_us: i64 = r.get(1)?;
            let _etype: i64 = r.get(2)?;
            let meta_bytes: Vec<u8> = r.get(3)?;
            let payload: Vec<u8> = r.get(4)?;
            let nonce: Vec<u8> = r.get(5)?;
            // compute aad
            let mut hasher = Hasher::new();
            hasher.update(&meta_bytes);
            let meta_hash = hasher.finalize();
            let mut aad = Vec::new();
            aad.extend_from_slice(session_id.as_bytes());
            aad.extend_from_slice(&(seq as u64).to_le_bytes());
            aad.extend_from_slice(&(_etype as u16).to_le_bytes());
            aad.extend_from_slice(&ts_us.to_le_bytes());
            aad.extend_from_slice(meta_hash.as_bytes());
            let plain = aead
                .decrypt(
                    XNonce::from_slice(&nonce),
                    Payload {
                        msg: &payload,
                        aad: &aad,
                    },
                )
                .map_err(|_| StoreError::CryptoError("decryption failed".into()))?;
            let event: zenb_core::domain::Event = serde_json::from_slice(&plain)
                .map_err(|_| StoreError::CryptoError("event deserialization failed".into()))?;
            let meta = serde_json::from_slice(&meta_bytes)
                .map_err(|_| StoreError::CryptoError("meta deserialization failed".into()))?;
            out.push(Envelope {
                session_id: session_id.clone(),
                seq: seq as u64,
                ts_us,
                event,
                meta,
            });
        }
        Ok(out)
    }

    /// Perform full WAL checkpoint (for flush operations)
    pub fn checkpoint_full(&mut self) -> Result<(), StoreError> {
        // PRAGMA wal_checkpoint(FULL) forces all WAL data to main DB file
        self.conn.execute("PRAGMA wal_checkpoint(FULL)", [])?;
        Ok(())
    }
}

fn ema(a: f32, b: f32, beta: f32) -> f32 {
    a * (1.0 - beta) + b * beta
}
