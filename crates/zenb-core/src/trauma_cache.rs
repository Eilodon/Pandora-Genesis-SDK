use crate::safety_swarm::{TraumaHit, TraumaSource};
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct TraumaCache {
    cache: LruCache<[u8; 32], TraumaHit>,
}

impl TraumaCache {
    pub fn new() -> Self {
        // SAFETY: 1000 is guaranteed non-zero at compile time
        let capacity = unsafe { NonZeroUsize::new_unchecked(1000) };
        Self {
            cache: LruCache::new(capacity),
        }
    }

    pub fn update(&mut self, sig: [u8; 32], hit: TraumaHit) {
        self.cache.put(sig, hit);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for TraumaCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TraumaSource for TraumaCache {
    fn query_trauma(&self, sig_hash: &[u8], _now_ts_us: i64) -> Result<Option<TraumaHit>, String> {
        if sig_hash.len() != 32 {
            return Ok(None);
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(sig_hash);
        Ok(self.cache.peek(&key).copied())
    }
}
