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
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
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

/// Async DAGMA: Background causal structure learning.
///
/// Spawns a dedicated thread for DAGMA computation, allowing the main
/// Engine loop to continue without blocking on causal discovery.
pub struct AsyncDagma {
    /// Channel to send requests to background thread
    tx: Sender<DagmaRequest>,
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
}

impl AsyncDagma {
    /// Spawn a new async DAGMA worker.
    ///
    /// # Arguments
    /// * `n_vars` - Number of variables in causal graph
    /// * `config` - Optional DAGMA configuration
    pub fn spawn(n_vars: usize, config: Option<DagmaConfig>) -> Self {
        let (request_tx, request_rx) = channel::<DagmaRequest>();
        let (result_tx, result_rx) = channel::<DagmaResult>();

        let dagma = Dagma::new(n_vars, config);

        let handle = thread::Builder::new()
            .name("async-dagma".to_string())
            .spawn(move || {
                Self::worker_loop(dagma, request_rx, result_tx);
            })
            .expect("Failed to spawn DAGMA thread");

        Self {
            tx: request_tx,
            rx: result_rx,
            _handle: handle,
            latest_result: Arc::new(Mutex::new(None)),
            next_request_id: 0,
            pending_request_id: None,
        }
    }

    /// Worker loop for background thread.
    fn worker_loop(
        dagma: Dagma,
        rx: Receiver<DagmaRequest>,
        tx: Sender<DagmaResult>,
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

            log::debug!(
                "AsyncDagma completed request {} in {}ms",
                request.request_id,
                duration_ms
            );

            // Send result (ignore error if receiver dropped)
            let _ = tx.send(result);
        }

        log::info!("AsyncDagma worker stopped");
    }

    /// Submit a new learning request (non-blocking).
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

        // Non-blocking send (buffer in channel)
        if let Err(e) = self.tx.send(request) {
            log::error!("AsyncDagma: Failed to submit request: {}", e);
        } else {
            self.pending_request_id = Some(request_id);
        }

        request_id
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
