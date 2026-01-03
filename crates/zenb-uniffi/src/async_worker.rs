//! P0.2 & P0.8: Async Worker with Retry Queue and Backpressure
//!
//! Architecture:
//! - Bounded channel (50 capacity) for backpressure
//! - Retry queue with exponential backoff
//! - Emergency dump to disk for unrecoverable failures
//! - Atomic metrics for observability

use crossbeam_channel::{Receiver, Sender, bounded, TryRecvError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use zenb_core::domain::{Envelope, SessionId};
use zenb_store::{EventStore, StoreError};

const CHANNEL_CAPACITY: usize = 50;
const MAX_RETRIES: u8 = 3;
const RETRY_BACKOFF_MS: u64 = 100;

/// Worker metrics tracked atomically
#[derive(Debug, Default)]
pub struct WorkerMetrics {
    pub appends_success: AtomicU64,
    pub appends_failed: AtomicU64,
    pub retries: AtomicU64,
    pub emergency_dumps: AtomicU64,
    pub channel_full_drops: AtomicU64,
    /// PR2: Track dropped HighFreq events separately for visibility
    pub highfreq_drops: AtomicU64,
    /// PR2: Track coalesced HighFreq events
    pub highfreq_coalesced: AtomicU64,
}

impl WorkerMetrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            appends_success: self.appends_success.load(Ordering::Relaxed),
            appends_failed: self.appends_failed.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
            emergency_dumps: self.emergency_dumps.load(Ordering::Relaxed),
            channel_full_drops: self.channel_full_drops.load(Ordering::Relaxed),
            highfreq_drops: self.highfreq_drops.load(Ordering::Relaxed),
            highfreq_coalesced: self.highfreq_coalesced.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub appends_success: u64,
    pub appends_failed: u64,
    pub retries: u64,
    pub emergency_dumps: u64,
    pub channel_full_drops: u64,
    pub highfreq_drops: u64,
    pub highfreq_coalesced: u64,
}

/// Commands sent to worker thread
pub enum WorkerCmd {
    Append {
        session_id: SessionId,
        envelopes: Vec<Envelope>,
    },
    FlushSync {
        response_tx: crossbeam_channel::Sender<Result<(), StoreError>>,
    },
    Shutdown,
}

/// Retry queue entry
struct RetryEntry {
    session_id: SessionId,
    envelopes: Vec<Envelope>,
    retry_count: u8,
    last_error: String,
}

/// Async worker handle
pub struct AsyncWorker {
    tx: Sender<WorkerCmd>,
    metrics: Arc<WorkerMetrics>,
    worker_thread: Option<thread::JoinHandle<()>>,
}

impl AsyncWorker {
    /// Create and start async worker
    pub fn start(store: EventStore) -> Self {
        let (tx, rx) = bounded(CHANNEL_CAPACITY);
        let metrics = Arc::new(WorkerMetrics::default());
        let metrics_clone = Arc::clone(&metrics);
        
        let worker_thread = thread::spawn(move || {
            Self::loop_forever(store, rx, metrics_clone);
        });
        
        AsyncWorker {
            tx,
            metrics,
            worker_thread: Some(worker_thread),
        }
    }
    
    /// PR2: Submit append command with priority-aware delivery.
    /// Critical events use blocking send (GUARANTEED delivery).
    /// HighFreq events use try_send with drop visibility.
    pub fn submit_append(&self, session_id: SessionId, envelopes: Vec<Envelope>) -> Result<(), &'static str> {
        use zenb_core::domain::EventPriority;
        
        // Classify batch priority: Critical if ANY event is Critical
        let has_critical = envelopes.iter().any(|e| e.event.priority() == EventPriority::Critical);
        
        let cmd = WorkerCmd::Append { session_id, envelopes: envelopes.clone() };
        
        if has_critical {
            // CRITICAL PATH: Blocking send - MUST NOT DROP
            // This will block the caller until space is available in the channel.
            // Acceptable because Critical events are rare (session lifecycle, decisions).
            match self.tx.send(cmd) {
                Ok(_) => Ok(()),
                Err(_) => {
                    // Channel closed - worker shutdown
                    // Last resort: emergency dump
                    self.metrics.emergency_dumps.fetch_add(1, Ordering::Relaxed);
                    if let Err(e) = Self::emergency_dump(&session_id, &envelopes, "worker_shutdown") {
                        eprintln!("CRITICAL: Emergency dump failed during shutdown: {:?}", e);
                    }
                    Err("worker_shutdown")
                }
            }
        } else {
            // HIGH-FREQ PATH: Non-blocking send - can drop with visibility
            match self.tx.try_send(cmd) {
                Ok(_) => Ok(()),
                Err(_) => {
                    // Channel full - backpressure
                    // Count drops for visibility (PR2 requirement)
                    self.metrics.channel_full_drops.fetch_add(1, Ordering::Relaxed);
                    self.metrics.highfreq_drops.fetch_add(envelopes.len() as u64, Ordering::Relaxed);
                    
                    eprintln!("WARN: Dropped {} HighFreq events due to backpressure", envelopes.len());
                    Err("channel_full")
                }
            }
        }
    }
    
    /// Flush and wait for completion (blocking)
    pub fn flush_sync(&self) -> Result<(), StoreError> {
        let (response_tx, response_rx) = crossbeam_channel::bounded(1);
        
        self.tx.send(WorkerCmd::FlushSync { response_tx })
            .map_err(|_| StoreError::CryptoError("worker channel closed".into()))?;
        
        response_rx.recv()
            .map_err(|_| StoreError::CryptoError("flush response channel closed".into()))?
    }
    
    /// Get current metrics snapshot
    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }
    
    /// Shutdown worker gracefully
    pub fn shutdown(mut self) {
        let _ = self.tx.send(WorkerCmd::Shutdown);
        
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
    
    /// Main worker loop - processes retry queue and channel
    fn loop_forever(store: EventStore, rx: Receiver<WorkerCmd>, metrics: Arc<WorkerMetrics>) {
        let mut retry_queue: Vec<RetryEntry> = Vec::new();
        
        loop {
            // 1. Process Retry Queue (Priority: clear backlog first)
            if !retry_queue.is_empty() {
                let entry = retry_queue.remove(0);
                
                match Self::try_append(&store, &entry.session_id, &entry.envelopes) {
                    Ok(_) => {
                        metrics.appends_success.fetch_add(1, Ordering::Relaxed);
                        // Success - continue to next
                    }
                    Err(e) => {
                        if entry.retry_count < MAX_RETRIES {
                            // Re-queue with incremented retry count
                            metrics.retries.fetch_add(1, Ordering::Relaxed);
                            retry_queue.push(RetryEntry {
                                session_id: entry.session_id,
                                envelopes: entry.envelopes,
                                retry_count: entry.retry_count + 1,
                                last_error: format!("{:?}", e),
                            });
                            
                            // Backoff before next retry
                            thread::sleep(Duration::from_millis(RETRY_BACKOFF_MS * (entry.retry_count as u64)));
                        } else {
                            // Max retries exceeded - emergency dump
                            metrics.emergency_dumps.fetch_add(1, Ordering::Relaxed);
                            metrics.appends_failed.fetch_add(1, Ordering::Relaxed);
                            
                            if let Err(dump_err) = Self::emergency_dump(&entry.session_id, &entry.envelopes, &entry.last_error) {
                                eprintln!("CRITICAL: Emergency dump failed: {:?}", dump_err);
                            }
                        }
                    }
                }
            }
            
            // 2. Process Channel (use try_recv if retrying to avoid blocking)
            let msg = if retry_queue.is_empty() {
                // No retries pending - block on channel
                match rx.recv() {
                    Ok(cmd) => Some(cmd),
                    Err(_) => break, // Channel closed
                }
            } else {
                // Retries pending - non-blocking receive
                match rx.try_recv() {
                    Ok(cmd) => Some(cmd),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => break,
                }
            };
            
            match msg {
                Some(WorkerCmd::Append { session_id, envelopes }) => {
                    match Self::try_append(&store, &session_id, &envelopes) {
                        Ok(_) => {
                            metrics.appends_success.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            // Failed - push to retry queue
                            retry_queue.push(RetryEntry {
                                session_id,
                                envelopes,
                                retry_count: 1,
                                last_error: format!("{:?}", e),
                            });
                        }
                    }
                }
                
                Some(WorkerCmd::FlushSync { response_tx }) => {
                    // Process all pending retries first
                    while !retry_queue.is_empty() {
                        let entry = retry_queue.remove(0);
                        let _ = Self::try_append(&store, &entry.session_id, &entry.envelopes);
                    }
                    
                    // WAL checkpoint
                    let result = Self::checkpoint_wal(&store);
                    let _ = response_tx.send(result);
                }
                
                Some(WorkerCmd::Shutdown) => {
                    // Process remaining retries before shutdown
                    while !retry_queue.is_empty() {
                        let entry = retry_queue.remove(0);
                        let _ = Self::try_append(&store, &entry.session_id, &entry.envelopes);
                    }
                    break;
                }
                
                None => {
                    // No message, continue retry loop
                    continue;
                }
            }
        }
    }
    
    /// Attempt to append batch to store
    fn try_append(store: &EventStore, session_id: &SessionId, envelopes: &[Envelope]) -> Result<(), StoreError> {
        store.append_batch(session_id, envelopes)
    }
    
    /// WAL checkpoint for flush
    fn checkpoint_wal(store: &EventStore) -> Result<(), StoreError> {
        // SQLite WAL checkpoint - forces write to main DB file
        store.checkpoint_full()
    }
    
    /// Emergency dump to disk when all retries fail
    fn emergency_dump(session_id: &SessionId, envelopes: &[Envelope], last_error: &str) -> Result<(), std::io::Error> {
        use std::fs::{create_dir_all, OpenOptions};
        use std::io::Write;
        
        // Create emergency dump directory
        let dump_dir = std::path::Path::new("emergency_dumps");
        create_dir_all(dump_dir)?;
        
        // Generate filename with timestamp
        let timestamp = chrono::Utc::now().timestamp_micros();
        let session_hex = hex::encode(&session_id.0);
        let filename = format!("dump_{}_{}.json", session_hex, timestamp);
        let filepath = dump_dir.join(filename);
        
        // Serialize envelopes with metadata
        let dump_data = serde_json::json!({
            "session_id": session_hex,
            "timestamp_us": timestamp,
            "last_error": last_error,
            "envelope_count": envelopes.len(),
            "envelopes": envelopes,
        });
        
        let json = serde_json::to_string_pretty(&dump_data)?;
        
        // Write to file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&filepath)?;
        
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        
        eprintln!("EMERGENCY DUMP: Saved {} envelopes to {:?}", envelopes.len(), filepath);
        
        Ok(())
    }
}

impl Drop for AsyncWorker {
    fn drop(&mut self) {
        // Send shutdown signal
        let _ = self.tx.send(WorkerCmd::Shutdown);
        
        // Wait for worker thread to finish
        if let Some(handle) = self.worker_thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use zenb_core::domain::{Event, ControlDecision};
    
    fn mk_key() -> [u8; 32] {
        [99u8; 32]
    }
    
    fn mk_envelope(sid: &SessionId, seq: u64) -> Envelope {
        Envelope {
            session_id: sid.clone(),
            seq,
            ts_us: seq as i64 * 1000,
            event: Event::ControlDecisionMade {
                decision: ControlDecision {
                    target_rate_bpm: 6.0,
                    confidence: 0.9,
                },
            },
            meta: serde_json::json!({}),
        }
    }
    
    #[test]
    fn test_worker_basic_append() {
        let tf = NamedTempFile::new().unwrap();
        let store = EventStore::open(tf.path(), mk_key()).unwrap();
        let sid = SessionId::new();
        store.create_session_key(&sid).unwrap();
        
        let worker = AsyncWorker::start(store);
        
        let envelopes = vec![mk_envelope(&sid, 1), mk_envelope(&sid, 2)];
        worker.submit_append(sid.clone(), envelopes).unwrap();
        
        // Flush and verify
        worker.flush_sync().unwrap();
        
        let metrics = worker.metrics();
        assert!(metrics.appends_success > 0);
        
        worker.shutdown();
    }
    
    #[test]
    fn test_worker_retry_on_failure() {
        let tf = NamedTempFile::new().unwrap();
        let store = EventStore::open(tf.path(), mk_key()).unwrap();
        let sid = SessionId::new();
        store.create_session_key(&sid).unwrap();
        
        let worker = AsyncWorker::start(store);
        
        // Submit valid batch
        worker.submit_append(sid.clone(), vec![mk_envelope(&sid, 1)]).unwrap();
        worker.flush_sync().unwrap();
        
        // Submit invalid batch (wrong sequence) - should retry
        worker.submit_append(sid.clone(), vec![mk_envelope(&sid, 10)]).unwrap();
        
        std::thread::sleep(Duration::from_millis(500));
        
        let metrics = worker.metrics();
        assert!(metrics.retries > 0 || metrics.emergency_dumps > 0);
        
        worker.shutdown();
    }
    
    #[test]
    fn test_backpressure() {
        let tf = NamedTempFile::new().unwrap();
        let store = EventStore::open(tf.path(), mk_key()).unwrap();
        let sid = SessionId::new();
        store.create_session_key(&sid).unwrap();
        
        let worker = AsyncWorker::start(store);
        
        // Flood channel beyond capacity
        let mut dropped = 0;
        for i in 0..100 {
            let envelopes = vec![mk_envelope(&sid, i + 1)];
            if worker.submit_append(sid.clone(), envelopes).is_err() {
                dropped += 1;
            }
        }
        
        // Should have some drops due to backpressure
        assert!(dropped > 0 || worker.metrics().channel_full_drops.load(Ordering::Relaxed) > 0);
        
        worker.shutdown();
    }
}
