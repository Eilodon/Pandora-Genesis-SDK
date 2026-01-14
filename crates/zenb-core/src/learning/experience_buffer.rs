//! Priority Experience Buffer for Active Inference.
//!
//! Implements prioritized experience replay where samples with higher prediction error
//! are sampled more frequently, leading to faster convergence.
//!
//! # Key Insight
//! `priority = prediction_error` — experiences where the model was most wrong
//! get replayed more often, maximizing learning signal.

use std::marker::PhantomData;

/// A single experience sample stored in the buffer.
#[derive(Debug, Clone)]
pub struct ExperienceSample<T: Clone> {
    /// The state/observation at this timestep
    pub state: T,
    /// Reward or outcome signal
    pub reward: f64,
    /// Timestamp in microseconds
    pub ts_us: i64,
}

/// Priority-based experience replay buffer.
///
/// Uses O(n) weighted sampling — acceptable for typical buffer sizes (< 10k).
/// For larger buffers, consider SumTree implementation.
#[derive(Debug)]
pub struct PriorityExperienceBuffer<T: Clone> {
    items: Vec<(ExperienceSample<T>, f64)>, // (sample, priority)
    capacity: usize,
    _marker: PhantomData<T>,
}

impl<T: Clone> PriorityExperienceBuffer<T> {
    /// Create a new buffer with fixed capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            capacity,
            _marker: PhantomData,
        }
    }

    /// Push a new experience with given priority (typically prediction error).
    ///
    /// When buffer is full, oldest experience is evicted (FIFO).
    /// Priority is clamped to minimum 1e-6 to avoid zero-probability samples.
    pub fn push(&mut self, sample: ExperienceSample<T>, priority: f64) {
        if self.items.len() >= self.capacity {
            self.items.remove(0);
        }
        self.items.push((sample, priority.max(1e-6)));
    }

    /// Returns the number of experiences in the buffer.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Sample an index proportionally to priority using deterministic RNG.
    ///
    /// Higher priority items are more likely to be selected.
    /// Returns None if buffer is empty.
    ///
    /// Time complexity: O(n) — acceptable for buffer sizes < 10k.
    pub fn sample_index(&self, seed: u64) -> Option<usize> {
        if self.items.is_empty() {
            return None;
        }

        let total: f64 = self.items.iter().map(|(_, p)| *p).sum();
        if total <= 0.0 {
            return Some(0);
        }

        // Use SplitMix64 for better mixing of sequential seeds
        // This ensures different seeds produce very different random values
        let mut x = seed;
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        
        let r = (x as f64) / (u64::MAX as f64);
        let target = r * total;

        let mut acc = 0.0;
        for (i, (_, p)) in self.items.iter().enumerate() {
            acc += *p;
            if acc >= target {
                return Some(i);
            }
        }

        Some(self.items.len() - 1)
    }

    /// Sample multiple indices without replacement (for batch training).
    pub fn sample_batch(&self, batch_size: usize, base_seed: u64) -> Vec<usize> {
        let mut indices = Vec::with_capacity(batch_size);
        let mut used = std::collections::HashSet::new();
        let mut attempts = 0;
        let max_attempts = batch_size * 10;

        while indices.len() < batch_size && attempts < max_attempts {
            if let Some(idx) = self.sample_index(base_seed.wrapping_add(attempts as u64)) {
                if !used.contains(&idx) {
                    used.insert(idx);
                    indices.push(idx);
                }
            }
            attempts += 1;
        }

        indices
    }

    /// Get reference to experience at given index.
    pub fn get(&self, index: usize) -> Option<&ExperienceSample<T>> {
        self.items.get(index).map(|(s, _)| s)
    }

    /// Get mutable reference to experience at given index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut ExperienceSample<T>> {
        self.items.get_mut(index).map(|(s, _)| s)
    }

    /// Update priority for a specific index (e.g., after TD-error update).
    pub fn update_priority(&mut self, index: usize, new_priority: f64) {
        if let Some((_, p)) = self.items.get_mut(index) {
            *p = new_priority.max(1e-6);
        }
    }

    /// Get current priority for an index.
    pub fn priority(&self, index: usize) -> Option<f64> {
        self.items.get(index).map(|(_, p)| *p)
    }

    /// Clear all experiences from the buffer.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Iterate over all experiences.
    pub fn iter(&self) -> impl Iterator<Item = &ExperienceSample<T>> {
        self.items.iter().map(|(s, _)| s)
    }

    /// Get the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Clone> Default for PriorityExperienceBuffer<T> {
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_respects_capacity() {
        let mut buf: PriorityExperienceBuffer<f32> = PriorityExperienceBuffer::with_capacity(2);

        buf.push(ExperienceSample { state: 1.0, reward: 1.0, ts_us: 1000 }, 1.0);
        buf.push(ExperienceSample { state: 2.0, reward: 2.0, ts_us: 2000 }, 1.0);
        buf.push(ExperienceSample { state: 3.0, reward: 3.0, ts_us: 3000 }, 1.0);

        assert_eq!(buf.len(), 2);
        // Oldest (state=1.0) should be evicted
        assert!((buf.get(0).unwrap().state - 2.0).abs() < 1e-6);
        assert!((buf.get(1).unwrap().state - 3.0).abs() < 1e-6);
    }

    #[test]
    fn priority_sampling_prefers_higher_priority() {
        let mut buf: PriorityExperienceBuffer<&str> = PriorityExperienceBuffer::with_capacity(3);

        buf.push(ExperienceSample { state: "low", reward: 0.0, ts_us: 1000 }, 1.0);
        buf.push(ExperienceSample { state: "high", reward: 0.0, ts_us: 2000 }, 10.0);
        buf.push(ExperienceSample { state: "low2", reward: 0.0, ts_us: 3000 }, 1.0);

        // Sample 100 times, expect index 1 (priority=10) to appear often
        let mut hits_high = 0;
        for k in 0..100 {
            if let Some(i) = buf.sample_index(k) {
                if i == 1 {
                    hits_high += 1;
                }
            }
        }

        // With priority 10 vs (1+1) = 12 total, expect ~83% hits on high
        assert!(hits_high > 50, "Expected high-priority to be sampled often, got {} hits", hits_high);
    }

    #[test]
    fn sample_batch_returns_unique_indices() {
        let mut buf: PriorityExperienceBuffer<i32> = PriorityExperienceBuffer::with_capacity(10);

        for i in 0..10 {
            buf.push(ExperienceSample { state: i, reward: 0.0, ts_us: i as i64 * 1000 }, 1.0);
        }

        let batch = buf.sample_batch(5, 42);
        assert_eq!(batch.len(), 5);

        // Check uniqueness
        let mut unique: Vec<_> = batch.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 5, "Batch should contain unique indices");
    }

    #[test]
    fn update_priority_changes_sampling_distribution() {
        let mut buf: PriorityExperienceBuffer<i32> = PriorityExperienceBuffer::with_capacity(2);

        buf.push(ExperienceSample { state: 0, reward: 0.0, ts_us: 1000 }, 1.0);
        buf.push(ExperienceSample { state: 1, reward: 0.0, ts_us: 2000 }, 1.0);

        // Initially equal priority - count how often index 0 is sampled
        let mut hits_0 = 0;
        // Use larger sample for statistical stability  
        for k in 0..500 {
            if buf.sample_index(k) == Some(0) {
                hits_0 += 1;
            }
        }
        // With equal priorities, expect roughly 50%, allow wider range [20%, 80%]
        assert!(hits_0 > 100 && hits_0 < 400, 
            "With equal priorities, should be roughly balanced, got {}/500 = {}%", 
            hits_0, hits_0 * 100 / 500);

        // Now boost priority of index 1 to 100x
        buf.update_priority(1, 100.0);

        let mut hits_1 = 0;
        for k in 0..500 {
            if buf.sample_index(k) == Some(1) {
                hits_1 += 1;
            }
        }
        // With 100:1 ratio, index 1 should be >90% = 450 hits
        assert!(hits_1 > 400, 
            "Index 1 should dominate with 100:1 priority, got {}/500 = {}%", 
            hits_1, hits_1 * 100 / 500);
    }

    #[test]
    fn empty_buffer_returns_none() {
        let buf: PriorityExperienceBuffer<()> = PriorityExperienceBuffer::with_capacity(10);
        assert!(buf.sample_index(42).is_none());
    }
}
