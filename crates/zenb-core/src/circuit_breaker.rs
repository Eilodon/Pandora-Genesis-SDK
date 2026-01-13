//! Sharded Circuit Breaker for resilient operation execution.
//!
//! Ported from Pandora's `pandora_orchestrator::circuit_breaker` module.
//! Uses 16 shards to reduce lock contention in concurrent scenarios.
//!
//! # Performance Requirements
//! - < 1µs per is_open() check (measured via criterion benchmark)
//! - No contention between different operation names
//!
//! # Example
//! ```rust
//! use zenb_core::circuit_breaker::{CircuitBreakerManager, CircuitBreakerConfig};
//!
//! let config = CircuitBreakerConfig::default();
//! let manager = CircuitBreakerManager::new(config);
//!
//!
//! // Mock function for doctest
//! fn do_sensor_ingest() -> Result<(), ()> { Ok(()) }
//!
//! // Check before calling external service
//! if !manager.is_open("sensor_ingest") {
//!     match do_sensor_ingest() {
//!         Ok(_) => manager.record_success("sensor_ingest"),
//!         Err(_) => manager.record_failure("sensor_ingest"),
//!     }
//! }
//! ```

use lru::LruCache;
use parking_lot::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

// ============================================================================
// Circuit State
// ============================================================================

/// Circuit breaker state with timestamp for TTL-based expiry.
#[derive(Debug, Clone)]
pub enum CircuitState {
    /// Circuit is closed (healthy). Requests flow through normally.
    Closed {
        /// Number of consecutive failures
        failures: u32,
        /// Last time the state was updated
        last_updated: Instant,
    },
    /// Circuit is open (tripped). All requests are rejected.
    Open {
        /// When the circuit was opened
        opened_at: Instant,
    },
    /// Circuit is half-open (testing). Limited requests allowed.
    HalfOpen {
        /// Number of trial permits remaining
        trial_permits: u32,
        /// Last time the state was updated
        last_updated: Instant,
    },
}

impl CircuitState {
    /// Check if state has expired based on TTL.
    #[inline]
    pub fn is_expired(&self, ttl: Duration) -> bool {
        let last_touch = match self {
            CircuitState::Closed { last_updated, .. } => *last_updated,
            CircuitState::Open { opened_at } => *opened_at,
            CircuitState::HalfOpen { last_updated, .. } => *last_updated,
        };
        last_touch.elapsed() > ttl
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for circuit breaker behavior.
#[derive(Clone, Debug)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit (default: 3)
    pub failure_threshold: u32,
    /// Duration to wait before trying half-open (default: 5000ms)
    pub open_cooldown_ms: u64,
    /// Number of trial requests in half-open state (default: 1)
    pub half_open_trial: u32,
    /// Maximum number of circuits to track (default: 1000)
    pub max_circuits: usize,
    /// TTL for idle circuit states (default: 3600s = 1 hour)
    pub state_ttl_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_cooldown_ms: 5_000,
            half_open_trial: 1,
            max_circuits: 1000,
            state_ttl_secs: 3600, // 1 hour
        }
    }
}

// ============================================================================
// Sharded Circuit Breaker Manager
// ============================================================================

/// Number of shards for the circuit breaker state.
/// Power of 2 for efficient modulo via bitwise AND.
const SHARD_COUNT: usize = 16;
const SHARD_MASK: usize = SHARD_COUNT - 1;

/// Sharded circuit breaker manager to reduce lock contention.
///
/// Each shard maintains its own LRU cache of circuit states,
/// allowing concurrent access to different operations without contention.
pub struct ShardedCircuitBreakerManager {
    shards: [Mutex<LruCache<String, CircuitState>>; SHARD_COUNT],
    config: CircuitBreakerConfig,
}

impl std::fmt::Debug for ShardedCircuitBreakerManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ShardedCircuitBreakerManager")
            .field("shard_count", &SHARD_COUNT)
            .field("total_circuits", &stats.total_circuits)
            .field("closed", &stats.closed)
            .field("open", &stats.open)
            .field("half_open", &stats.half_open)
            .field("config", &self.config)
            .finish()
    }
}

impl ShardedCircuitBreakerManager {
    /// Creates a new sharded circuit breaker manager.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let capacity_per_shard = config.max_circuits / SHARD_COUNT;
        let capacity = NonZeroUsize::new(capacity_per_shard.max(1))
            .unwrap_or_else(|| NonZeroUsize::new(1).expect("1 is non-zero"));

        // Initialize array of shards using const generics
        let shards = std::array::from_fn(|_| Mutex::new(LruCache::new(capacity)));

        Self { shards, config }
    }

    /// Calculate which shard an operation belongs to.
    /// Uses fast hash with bitwise AND instead of modulo.
    #[inline]
    fn shard_index(&self, operation_name: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        operation_name.hash(&mut hasher);
        (hasher.finish() as usize) & SHARD_MASK
    }

    /// Get mutable access to the shard for an operation.
    #[inline]
    fn get_shard(&self, operation_name: &str) -> &Mutex<LruCache<String, CircuitState>> {
        &self.shards[self.shard_index(operation_name)]
    }

    /// Check if circuit is open for an operation.
    ///
    /// This method only locks ONE shard, not the entire state.
    /// Returns `true` if the circuit is open (requests should be rejected).
    pub fn is_open(&self, operation_name: &str) -> bool {
        let shard = self.get_shard(operation_name);
        let mut states = shard.lock();

        // Ensure entry exists
        if states.get(operation_name).is_none() {
            states.put(
                operation_name.to_string(),
                CircuitState::Closed {
                    failures: 0,
                    last_updated: Instant::now(),
                },
            );
        }

        // Get mutable reference for in-place transitions
        let state = states
            .get_mut(operation_name)
            .expect("state exists after insert");

        // Check if expired and reset if needed
        let ttl = Duration::from_secs(self.config.state_ttl_secs);
        if state.is_expired(ttl) {
            log::debug!("Circuit state expired for '{}'", operation_name);
            states.pop(operation_name);
            return false;
        }

        // Check state and potentially transition
        match state {
            CircuitState::Closed { .. } => false,
            CircuitState::Open { opened_at } => {
                let elapsed = opened_at.elapsed();
                let cooldown = Duration::from_millis(self.config.open_cooldown_ms);

                if elapsed >= cooldown {
                    // Transition to half-open
                    log::debug!(
                        "Circuit cooldown elapsed for '{}', entering half-open",
                        operation_name
                    );
                    let consumed = self.config.half_open_trial.saturating_sub(1);
                    *state = CircuitState::HalfOpen {
                        trial_permits: consumed,
                        last_updated: Instant::now(),
                    };
                    false
                } else {
                    true
                }
            }
            CircuitState::HalfOpen { trial_permits, .. } => {
                if *trial_permits > 0 {
                    let remaining = trial_permits.saturating_sub(1);
                    *state = CircuitState::HalfOpen {
                        trial_permits: remaining,
                        last_updated: Instant::now(),
                    };
                    false
                } else {
                    true
                }
            }
        }
    }

    /// Record successful execution.
    pub fn record_success(&self, operation_name: &str) {
        let shard = self.get_shard(operation_name);
        let mut states = shard.lock();

        states.put(
            operation_name.to_string(),
            CircuitState::Closed {
                failures: 0,
                last_updated: Instant::now(),
            },
        );
        log::debug!("Circuit reset to closed for '{}'", operation_name);
    }

    /// Record failed execution.
    pub fn record_failure(&self, operation_name: &str) {
        let shard = self.get_shard(operation_name);
        let mut states = shard.lock();

        let new_state = match states.get(operation_name) {
            Some(CircuitState::Closed { failures, .. }) => {
                let new_failures = failures + 1;
                if new_failures >= self.config.failure_threshold {
                    log::warn!(
                        "Circuit opened for '{}' after {} failures",
                        operation_name,
                        new_failures
                    );
                    CircuitState::Open {
                        opened_at: Instant::now(),
                    }
                } else {
                    log::debug!(
                        "Circuit failure {}/{} for '{}'",
                        new_failures,
                        self.config.failure_threshold,
                        operation_name
                    );
                    CircuitState::Closed {
                        failures: new_failures,
                        last_updated: Instant::now(),
                    }
                }
            }
            Some(CircuitState::HalfOpen { .. }) => {
                log::warn!(
                    "Circuit re-opened for '{}' after half-open failure",
                    operation_name
                );
                CircuitState::Open {
                    opened_at: Instant::now(),
                }
            }
            _ => {
                if self.config.failure_threshold <= 1 {
                    CircuitState::Open {
                        opened_at: Instant::now(),
                    }
                } else {
                    CircuitState::Closed {
                        failures: 1,
                        last_updated: Instant::now(),
                    }
                }
            }
        };

        states.put(operation_name.to_string(), new_state);
    }

    /// Get aggregated statistics across all shards.
    pub fn stats(&self) -> CircuitStats {
        let mut total_circuits = 0;
        let mut closed = 0;
        let mut open = 0;
        let mut half_open = 0;

        // Lock each shard sequentially (not all at once)
        for shard in &self.shards {
            let states = shard.lock();
            total_circuits += states.len();

            for (_, state) in states.iter() {
                match state {
                    CircuitState::Closed { .. } => closed += 1,
                    CircuitState::Open { .. } => open += 1,
                    CircuitState::HalfOpen { .. } => half_open += 1,
                }
            }
        }

        CircuitStats {
            total_circuits,
            closed,
            open,
            half_open,
            capacity: self.config.max_circuits,
        }
    }

    /// Manually cleanup expired states across all shards.
    pub fn cleanup_expired(&self) {
        let ttl = Duration::from_secs(self.config.state_ttl_secs);

        for (shard_idx, shard) in self.shards.iter().enumerate() {
            let mut states = shard.lock();

            let to_remove: Vec<String> = states
                .iter()
                .filter_map(|(name, state)| {
                    if state.is_expired(ttl) {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect();

            for name in to_remove {
                states.pop(&name);
                log::debug!(
                    "Cleaned up expired circuit state for '{}' in shard {}",
                    name,
                    shard_idx
                );
            }
        }
    }

    /// Get per-shard statistics for monitoring distribution.
    pub fn shard_stats(&self) -> Vec<usize> {
        self.shards.iter().map(|shard| shard.lock().len()).collect()
    }
}

// ============================================================================
// Type Alias
// ============================================================================

/// Default circuit breaker manager (sharded for better concurrency).
pub type CircuitBreakerManager = ShardedCircuitBreakerManager;

// ============================================================================
// Statistics
// ============================================================================

/// Statistics about circuit breaker state.
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub total_circuits: usize,
    pub closed: usize,
    pub open: usize,
    pub half_open: usize,
    pub capacity: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_circuit_opens_after_threshold() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let manager = CircuitBreakerManager::new(config);

        // Record failures
        assert!(!manager.is_open("test_op"));
        manager.record_failure("test_op");
        assert!(!manager.is_open("test_op"));
        manager.record_failure("test_op");
        assert!(!manager.is_open("test_op"));
        manager.record_failure("test_op");

        // Should be open now
        assert!(manager.is_open("test_op"));
    }

    #[test]
    fn test_circuit_cooldown() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            open_cooldown_ms: 100,
            ..Default::default()
        };
        let manager = CircuitBreakerManager::new(config);

        // Open circuit
        manager.record_failure("test_op");
        assert!(manager.is_open("test_op"));

        // Wait for cooldown
        sleep(Duration::from_millis(150));

        // Should transition to half-open
        assert!(!manager.is_open("test_op")); // First trial
        assert!(manager.is_open("test_op")); // No more trials
    }

    #[test]
    fn test_circuit_success_resets() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let manager = CircuitBreakerManager::new(config);

        // Build up failures
        manager.record_failure("test_op");
        manager.record_failure("test_op");

        // Success resets
        manager.record_success("test_op");

        // Need 3 more failures to open
        manager.record_failure("test_op");
        manager.record_failure("test_op");
        assert!(!manager.is_open("test_op"));
        manager.record_failure("test_op");
        assert!(manager.is_open("test_op"));
    }

    #[test]
    fn test_shard_distribution() {
        let config = CircuitBreakerConfig::default();
        let manager = ShardedCircuitBreakerManager::new(config);

        // Add many operations to see distribution
        for i in 0..100 {
            let op_name = format!("operation_{}", i);
            manager.record_failure(&op_name);
        }

        let shard_stats = manager.shard_stats();
        assert_eq!(shard_stats.len(), SHARD_COUNT);

        // All shards should have some circuits
        let total: usize = shard_stats.iter().sum();
        assert_eq!(total, 100);

        // At least some shards should be used
        let used_shards = shard_stats.iter().filter(|&&count| count > 0).count();
        assert!(
            used_shards > 1,
            "Operations should be distributed across multiple shards"
        );
    }
}
