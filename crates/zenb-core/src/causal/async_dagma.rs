//! Async DAGMA: Background Causal Learning
//!
//! Runs DAGMA causal structure learning in a background thread,
//! enabling non-blocking causal discovery without impacting main loop latency.
//!
//! # Architecture
//! ```text
//! Main Thread          Background Thread
//!     |                      |
//!     | submit_data() ----> |
//!     |                      | [DAGMA::fit_warm_start]
//!     | <----- notify ----- |
//!     | get_result()         |
//! ```
//!
//! # Usage
//! ```ignore
//! let async_dagma = AsyncDagma::spawn(5);
//!
//! // Submit data (non-blocking)
//! async_dagma.submit(data_matrix, Some(&prev_weights));
//!
//! // Check for results
//! if let Some(result) = async_dagma.try_get_result() {
//!     // Use new causal graph
//! }
//! ```

use nalgebra::DMatrix;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use super::dagma::{Dagma, DagmaConfig};

/// Request sent to background DAGMA thread.
pub struct DagmaRequest {
    /// Data matrix (n_samples × n_vars)
    pub data: DMatrix<f32>,
    /// Optional warm start weights for faster convergence
    pub warm_start: Option<DMatrix<f32>>,
    /// Request ID for tracking
    pub request_id: u64,
}

/// Result from background DAGMA computation.
#[derive(Debug, Clone)]
pub struct DagmaResult {
    /// Learned weighted adjacency matrix
    pub weights: DMatrix<f32>,
    /// Request ID this result corresponds to
    pub request_id: u64,
    /// Time taken in milliseconds
    pub duration_ms: u64,
}

/// Error when submitting to AsyncDagma
#[derive(Debug, Clone)]
pub enum SubmitError {
    /// Channel is full (backpressure active)
    BackpressureFull,
    /// Worker thread has disconnected
    Disconnected,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackpressureFull => write!(f, "AsyncDagma channel full (backpressure)"),
            Self::Disconnected => write!(f, "AsyncDagma worker disconnected"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Metrics for AsyncDagma performance monitoring
#[derive(Debug, Clone, Default)]
pub struct AsyncDagmaMetrics {
    /// Total requests submitted
    pub total_submitted: u64,
    /// Requests dropped due to backpressure
    pub dropped_backpressure: u64,
    /// Completed requests
    pub completed: u64,
    /// Average duration in milliseconds
    pub avg_duration_ms: f64,
    /// Maximum duration in milliseconds
    pub max_duration_ms: u64,
}

/// Async DAGMA: Background causal structure learning.
///
/// Spawns a dedicated thread for DAGMA computation, allowing the main
/// Engine loop to continue without blocking on causal discovery.
///
/// # Backpressure
/// Uses bounded channel (default capacity: 2) to prevent unbounded memory growth.
/// When channel is full, `try_submit` returns `Err(SubmitError::BackpressureFull)`.
pub struct AsyncDagma {
    /// Channel to send requests to background thread (bounded)
    tx: SyncSender<DagmaRequest>,
    /// Channel to receive results from background thread
    rx: Receiver<DagmaResult>,
    /// Handle to background thread (for cleanup)
    _handle: JoinHandle<()>,
    /// Latest result (cached for multiple reads)
    latest_result: Arc<Mutex<Option<DagmaResult>>>,
    /// Next request ID
    next_request_id: u64,
    /// Most recent request ID (for ordering)
    pending_request_id: Option<u64>,
    /// Metrics (shared with caller)
    metrics: Arc<Mutex<AsyncDagmaMetrics>>,
    /// Channel capacity for backpressure
    capacity: usize,
}

impl AsyncDagma {
    /// Default channel capacity for backpressure
    pub const DEFAULT_CAPACITY: usize = 2;

    /// Spawn a new async DAGMA worker.
    ///
    /// # Arguments
    /// * `n_vars` - Number of variables in causal graph
    /// * `config` - Optional DAGMA configuration
    pub fn spawn(n_vars: usize, config: Option<DagmaConfig>) -> Self {
        Self::spawn_with_capacity(n_vars, config, Self::DEFAULT_CAPACITY)
    }

    /// Spawn a new async DAGMA worker with custom capacity.
    ///
    /// # Arguments
    /// * `n_vars` - Number of variables in causal graph
    /// * `config` - Optional DAGMA configuration
    /// * `capacity` - Channel capacity for backpressure (0 = rendezvous, 1+ = bounded)
    pub fn spawn_with_capacity(n_vars: usize, config: Option<DagmaConfig>, capacity: usize) -> Self {
        let (request_tx, request_rx) = sync_channel::<DagmaRequest>(capacity);
        let (result_tx, result_rx) = sync_channel::<DagmaResult>(capacity.max(1));

        let dagma = Dagma::new(n_vars, config);
        let metrics = Arc::new(Mutex::new(AsyncDagmaMetrics::default()));
        let metrics_clone = Arc::clone(&metrics);

        let handle = thread::Builder::new()
            .name("async-dagma".to_string())
            .spawn(move || {
                Self::worker_loop(dagma, request_rx, result_tx, metrics_clone);
            })
            .expect("Failed to spawn DAGMA thread");

        Self {
            tx: request_tx,
            rx: result_rx,
            _handle: handle,
            latest_result: Arc::new(Mutex::new(None)),
            next_request_id: 0,
            pending_request_id: None,
            metrics,
            capacity,
        }
    }

    /// Worker loop for background thread.
    fn worker_loop(
        dagma: Dagma,
        rx: Receiver<DagmaRequest>,
        tx: SyncSender<DagmaResult>,
        metrics: Arc<Mutex<AsyncDagmaMetrics>>,
    ) {
        log::info!("AsyncDagma worker started");

        while let Ok(request) = rx.recv() {
            let start = std::time::Instant::now();

            // Run DAGMA with optional warm start
            let weights = dagma.fit_warm_start(
                &request.data,
                request.warm_start.as_ref(),
            );

            let duration_ms = start.elapsed().as_millis() as u64;

            let result = DagmaResult {
                weights,
                request_id: request.request_id,
                duration_ms,
            };

            // Update metrics
            if let Ok(mut m) = metrics.lock() {
                m.completed += 1;
                m.max_duration_ms = m.max_duration_ms.max(duration_ms);
                // Rolling average
                let n = m.completed as f64;
                m.avg_duration_ms = m.avg_duration_ms * (n - 1.0) / n + duration_ms as f64 / n;
            }

            log::debug!(
                "AsyncDagma completed request {} in {}ms",
                request.request_id,
                duration_ms
            );

            // Send result (ignore error if receiver dropped)
            let _ = tx.try_send(result);
        }

        log::info!("AsyncDagma worker stopped");
    }

    /// Submit a new learning request (non-blocking).
    ///
    /// If the channel is full, this will still succeed by waiting briefly.
    /// For strict non-blocking, use `try_submit`.
    ///
    /// # Arguments
    /// * `data` - Data matrix (n_samples × n_vars)
    /// * `warm_start` - Optional previous weights for faster convergence
    ///
    /// # Returns
    /// Request ID for tracking
    pub fn submit(&mut self, data: DMatrix<f32>, warm_start: Option<&DMatrix<f32>>) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request = DagmaRequest {
            data,
            warm_start: warm_start.cloned(),
            request_id,
        };

        // Update metrics
        if let Ok(mut m) = self.metrics.lock() {
            m.total_submitted += 1;
        }

        // Blocking send (will wait if channel full)
        if let Err(e) = self.tx.send(request) {
            log::error!("AsyncDagma: Worker disconnected: {}", e);
        } else {
            self.pending_request_id = Some(request_id);
        }

        request_id
    }

    /// Try to submit a new learning request (non-blocking with backpressure).
    ///
    /// # Arguments
    /// * `data` - Data matrix (n_samples × n_vars)
    /// * `warm_start` - Optional previous weights for faster convergence
    ///
    /// # Returns
    /// - `Ok(request_id)` if submitted successfully
    /// - `Err(SubmitError::BackpressureFull)` if channel is full
    /// - `Err(SubmitError::Disconnected)` if worker thread died
    pub fn try_submit(&mut self, data: DMatrix<f32>, warm_start: Option<&DMatrix<f32>>) -> Result<u64, SubmitError> {
        let request_id = self.next_request_id;

        let request = DagmaRequest {
            data,
            warm_start: warm_start.cloned(),
            request_id,
        };

        match self.tx.try_send(request) {
            Ok(()) => {
                self.next_request_id += 1;
                self.pending_request_id = Some(request_id);
                
                if let Ok(mut m) = self.metrics.lock() {
                    m.total_submitted += 1;
                }
                
                Ok(request_id)
            }
            Err(TrySendError::Full(_)) => {
                if let Ok(mut m) = self.metrics.lock() {
                    m.dropped_backpressure += 1;
                }
                Err(SubmitError::BackpressureFull)
            }
            Err(TrySendError::Disconnected(_)) => {
                Err(SubmitError::Disconnected)
            }
        }
    }

    /// Get current metrics.
    pub fn metrics(&self) -> AsyncDagmaMetrics {
        self.metrics.lock().ok().map(|m| m.clone()).unwrap_or_default()
    }

    /// Get channel capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Try to get the latest result (non-blocking).
    ///
    /// # Returns
    /// - `Some(result)` if a new result is available
    /// - `None` if no new result (still computing or no pending request)
    pub fn try_get_result(&mut self) -> Option<DagmaResult> {
        // Check for new result from channel
        match self.rx.try_recv() {
            Ok(result) => {
                // Cache the result
                if let Ok(mut cached) = self.latest_result.lock() {
                    *cached = Some(result.clone());
                }
                
                // Clear pending if this matches
                if Some(result.request_id) == self.pending_request_id {
                    self.pending_request_id = None;
                }
                
                Some(result)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                log::error!("AsyncDagma: Worker thread disconnected");
                None
            }
        }
    }

    /// Check if there's a pending request.
    pub fn is_pending(&self) -> bool {
        self.pending_request_id.is_some()
    }

    /// Get the latest cached result (even if already retrieved).
    pub fn latest(&self) -> Option<DagmaResult> {
        self.latest_result.lock().ok().and_then(|g| g.clone())
    }
}

impl std::fmt::Debug for AsyncDagma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncDagma")
            .field("pending_request_id", &self.pending_request_id)
            .field("next_request_id", &self.next_request_id)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_async_dagma_basic() {
        let mut async_dagma = AsyncDagma::spawn(3, None);

        // Create simple test data
        let data = DMatrix::from_row_slice(10, 3, &[
            1.0, 0.5, 0.2,
            1.1, 0.6, 0.3,
            0.9, 0.4, 0.1,
            1.2, 0.7, 0.4,
            0.8, 0.3, 0.0,
            1.0, 0.5, 0.2,
            1.1, 0.6, 0.3,
            0.9, 0.4, 0.1,
            1.2, 0.7, 0.4,
            0.8, 0.3, 0.0,
        ]);

        // Submit request
        let req_id = async_dagma.submit(data, None);
        assert_eq!(req_id, 0);
        assert!(async_dagma.is_pending());

        // Wait for result (with timeout)
        let mut result = None;
        for _ in 0..100 {
            if let Some(r) = async_dagma.try_get_result() {
                result = Some(r);
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.request_id, 0);
        assert_eq!(result.weights.nrows(), 3);
        assert_eq!(result.weights.ncols(), 3);
    }

    #[test]
    fn test_async_dagma_warm_start() {
        let mut async_dagma = AsyncDagma::spawn(3, None);

        let data = DMatrix::from_row_slice(10, 3, &[
            1.0, 0.5, 0.2,
            1.1, 0.6, 0.3,
            0.9, 0.4, 0.1,
            1.2, 0.7, 0.4,
            0.8, 0.3, 0.0,
            1.0, 0.5, 0.2,
            1.1, 0.6, 0.3,
            0.9, 0.4, 0.1,
            1.2, 0.7, 0.4,
            0.8, 0.3, 0.0,
        ]);

        // First request (cold start)
        async_dagma.submit(data.clone(), None);
        
        let mut first_result = None;
        for _ in 0..100 {
            if let Some(r) = async_dagma.try_get_result() {
                first_result = Some(r);
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(first_result.is_some());

        // Second request (warm start)
        let warm = first_result.unwrap().weights;
        let req_id = async_dagma.submit(data, Some(&warm));
        assert_eq!(req_id, 1);

        let mut second_result = None;
        for _ in 0..100 {
            if let Some(r) = async_dagma.try_get_result() {
                second_result = Some(r);
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        assert!(second_result.is_some());

        // Warm start should be faster (usually)
        // Note: This is non-deterministic, just verify it completed
        assert_eq!(second_result.unwrap().request_id, 1);
    }
}
