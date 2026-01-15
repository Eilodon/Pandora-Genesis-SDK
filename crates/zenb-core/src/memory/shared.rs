//! Thread-Safe Holographic Memory Wrapper
//!
//! Provides `Arc<RwLock<HolographicMemory>>` wrapper with convenient methods
//! for concurrent read/write access.
//!
//! # Usage
//! ```ignore
//! use zenb_core::memory::{HolographicMemory, SharedMemory};
//!
//! let memory = SharedMemory::new(512);
//!
//! // Clone handle for another thread
//! let handle = memory.clone();
//! std::thread::spawn(move || {
//!     handle.entangle(&key, &value);
//! });
//!
//! // Read from main thread
//! let recalled = memory.recall(&key);
//! ```

use num_complex::Complex32;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use super::HolographicMemory;

/// Thread-safe wrapper for HolographicMemory.
///
/// Internally uses `Arc<RwLock<HolographicMemory>>` to enable:
/// - Multiple concurrent readers (recall, energy, diagnostics)
/// - Exclusive writer (entangle, decay, clear)
///
/// # Performance Notes
/// - Read operations acquire shared lock (non-blocking with other reads)
/// - Write operations acquire exclusive lock (blocks all other access)
/// - For hot paths, consider batching writes to reduce lock contention
#[derive(Clone)]
pub struct SharedMemory {
    inner: Arc<RwLock<HolographicMemory>>,
}

impl std::fmt::Debug for SharedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner.try_read() {
            Ok(guard) => write!(f, "SharedMemory({:?})", &*guard),
            Err(_) => write!(f, "SharedMemory(<locked>)"),
        }
    }
}

impl SharedMemory {
    /// Create new shared memory with given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HolographicMemory::new(dim))),
        }
    }

    /// Create with capacity limit.
    pub fn with_capacity(dim: usize, max_items: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HolographicMemory::with_capacity(dim, max_items))),
        }
    }

    /// Create with default ZenB settings (dim=512, max_items=10000).
    pub fn default_for_zenb() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HolographicMemory::default_for_zenb())),
        }
    }

    /// Wrap an existing HolographicMemory.
    pub fn from_memory(memory: HolographicMemory) -> Self {
        Self {
            inner: Arc::new(RwLock::new(memory)),
        }
    }

    // =========================================================================
    // Read Operations (shared lock)
    // =========================================================================

    /// Recall a value by key (acquires read lock).
    pub fn recall(&self, key: &[Complex32]) -> Vec<Complex32> {
        let guard = self.inner.read().expect("RwLock poisoned");
        guard.recall(key)
    }

    /// Get current energy (acquires read lock).
    pub fn energy(&self) -> f32 {
        let guard = self.inner.read().expect("RwLock poisoned");
        guard.energy()
    }

    /// Get item count (acquires read lock).
    pub fn item_count(&self) -> usize {
        let guard = self.inner.read().expect("RwLock poisoned");
        guard.item_count()
    }

    /// Get dimension (acquires read lock).
    pub fn dim(&self) -> usize {
        let guard = self.inner.read().expect("RwLock poisoned");
        guard.dim()
    }

    // =========================================================================
    // Write Operations (exclusive lock)
    // =========================================================================

    /// Entangle key-value pair (acquires write lock).
    pub fn entangle(&self, key: &[Complex32], value: &[Complex32]) {
        let mut guard = self.inner.write().expect("RwLock poisoned");
        guard.entangle(key, value);
    }

    /// Apply decay (acquires write lock).
    pub fn decay(&self, factor: f32) {
        let mut guard = self.inner.write().expect("RwLock poisoned");
        guard.decay(factor);
    }

    /// Clear memory (acquires write lock).
    pub fn clear(&self) {
        let mut guard = self.inner.write().expect("RwLock poisoned");
        guard.clear();
    }

    // =========================================================================
    // Direct Access (for advanced use cases)
    // =========================================================================

    /// Acquire read lock for batch operations.
    ///
    /// # Warning
    /// Hold this lock for as short as possible to avoid blocking writers.
    pub fn read(&self) -> RwLockReadGuard<'_, HolographicMemory> {
        self.inner.read().expect("RwLock poisoned")
    }

    /// Acquire write lock for batch operations.
    ///
    /// # Warning
    /// Hold this lock for as short as possible to avoid blocking all access.
    pub fn write(&self) -> RwLockWriteGuard<'_, HolographicMemory> {
        self.inner.write().expect("RwLock poisoned")
    }

    /// Try to acquire read lock without blocking.
    pub fn try_read(&self) -> Option<RwLockReadGuard<'_, HolographicMemory>> {
        self.inner.try_read().ok()
    }

    /// Try to acquire write lock without blocking.
    pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, HolographicMemory>> {
        self.inner.try_write().ok()
    }

    /// Get number of strong references to this memory.
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl Default for SharedMemory {
    fn default() -> Self {
        Self::default_for_zenb()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_shared_memory_basic() {
        let memory = SharedMemory::new(64);
        
        let key: Vec<Complex32> = (0..64).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let value = key.clone();
        
        memory.entangle(&key, &value);
        assert_eq!(memory.item_count(), 1);
        
        let recalled = memory.recall(&key);
        assert_eq!(recalled.len(), 64);
    }

    #[test]
    fn test_shared_memory_concurrent_reads() {
        let memory = SharedMemory::new(64);
        let key: Vec<Complex32> = (0..64).map(|i| Complex32::new(i as f32, 0.0)).collect();
        memory.entangle(&key, &key);

        // Spawn multiple readers
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let mem = memory.clone();
                let k = key.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = mem.recall(&k);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_shared_memory_concurrent_write() {
        let memory = SharedMemory::new(64);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let mem = memory.clone();
                thread::spawn(move || {
                    for j in 0..10 {
                        let key: Vec<Complex32> = (0..64)
                            .map(|k| Complex32::new((i * 10 + j + k) as f32, 0.0))
                            .collect();
                        mem.entangle(&key, &key);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All writes should have succeeded
        assert_eq!(memory.item_count(), 40);
    }
}
