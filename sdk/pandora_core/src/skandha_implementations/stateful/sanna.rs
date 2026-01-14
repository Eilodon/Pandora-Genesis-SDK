//! Stateful Saññā implementation with pattern memory reinforcement.

#[allow(unused_imports)]
use crate::skandha_implementations::core::{
    traits::{Skandha, StatefulSkandha, SannaSkandha, StatefulSannaSkandha, DecayableState},
    state_management::PatternMemoryState,
};
use crate::ontology::EpistemologicalFlow;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tracing::debug;

/// Stateful Saññā implementation with pattern memory.
///
/// This skandha wraps a stateless `SannaSkandha` and adds a layer of
/// statefulness using `PatternMemoryState`.
///
/// # Architecture
///
/// - **Inner Skandha**: A stateless `SannaSkandha` (e.g., `BasicSannaSkandha`)
///   provides the initial pattern recognition.
/// - **State**: A `Mutex`-protected `PatternMemoryState` tracks the frequency
///   and strength of perceived patterns over time.
/// - **Reinforcement**: Each time a pattern is perceived, its entry in the
///   state is reinforced, increasing its strength.
pub struct StatefulSanna {
    #[allow(dead_code)]
    name: String,
    inner_sanna: Arc<dyn SannaSkandha>,
    state: Arc<Mutex<PatternMemoryState>>,
}

impl std::fmt::Debug for StatefulSanna {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatefulSanna")
            .field("name", &self.name)
            .field("state", &self.state.lock())
            .finish()
    }
}

impl StatefulSanna {
    /// Create a new `StatefulSanna` with default pattern memory.
    pub fn new(name: &'static str, inner_sanna: Arc<dyn SannaSkandha>) -> Self {
        Self {
            name: name.to_string(),
            inner_sanna,
            state: Arc::new(Mutex::new(PatternMemoryState::default())),
        }
    }
}

impl Skandha for StatefulSanna {
    fn name(&self) -> &'static str {
        "StatefulSanna"
    }
}

impl SannaSkandha for StatefulSanna {
    /// Performs perception and then reinforces the detected pattern.
    fn perceive(&self, flow: &mut EpistemologicalFlow) {
        // Since `perceive` is immutable, we can't update state here.
        // We defer the state update logic to `perceive_with_state`.
        // This method will just perform the basic perception.
        self.inner_sanna.perceive(flow);
    }
}

impl StatefulSkandha for StatefulSanna {
    type State = PatternMemoryState;

    fn state(&self) -> &Self::State {
        unimplemented!("Direct access to state under mutex is not provided for safety.");
    }

    fn state_mut(&mut self) -> &mut Self::State {
        unimplemented!("Direct access to state under mutex is not provided for safety.");
    }
}

impl StatefulSannaSkandha for StatefulSanna {
    /// Stateful perception with pattern reinforcement.
    ///
    /// 1. Runs the inner stateless `SannaSkandha` to get a base perception.
    /// 2. Hashes the raw event data to get a stable pattern ID.
    /// 3. Locks the pattern memory state.
    /// 4. Records the pattern, reinforcing it if already present.
    /// 5. (Optional) Adjusts the confidence of the perception in the flow
    ///    based on the pattern's new strength.
    fn perceive_with_state(&mut self, flow: &mut EpistemologicalFlow) {
        // 1. Get base perception
        self.inner_sanna.perceive(flow);

        if let Some(rupa_bytes) = &flow.rupa {
            // 2. Hash the raw data to get a consistent pattern ID
            let mut hasher = DefaultHasher::new();
            rupa_bytes.hash(&mut hasher);
            let pattern_hash = hasher.finish();

            // 3. Lock state and record the pattern
            let mut memory = self.state.lock();
            memory.record_pattern(pattern_hash);

            let strength = memory.pattern_strength(pattern_hash);
            debug!(
                "Pattern {} reinforced. New strength: {}",
                pattern_hash, strength
            );

            // 4. (Optional) Adjust flow based on strength.
            // For now, we just log. A more advanced implementation could
            // modify `flow.sanna` or add metadata.
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skandha_implementations::basic::BasicSannaSkandha;
    use bytes::Bytes;

    fn create_flow(content: &'static str) -> EpistemologicalFlow {
        EpistemologicalFlow::from_bytes(Bytes::from(content))
    }

    fn get_hash(content: &'static str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.as_bytes().hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_stateful_sanna_initial_state() {
        let sanna = StatefulSanna::new("TestSanna", Arc::new(BasicSannaSkandha));
        let memory = sanna.state.lock();
        assert_eq!(memory.patterns.len(), 0);
    }

    #[test]
    fn test_stateful_sanna_records_new_pattern() {
        let mut sanna = StatefulSanna::new("TestSanna", Arc::new(BasicSannaSkandha));
        let mut flow = create_flow("new pattern here");
        let hash = get_hash("new pattern here");

        sanna.perceive_with_state(&mut flow);

        let memory = sanna.state.lock();
        assert_eq!(memory.patterns.len(), 1);
        assert!(memory.patterns.contains_key(&hash));
        assert_eq!(memory.pattern_strength(hash), 0.5); // Initial strength
    }

    #[test]
    fn test_stateful_sanna_reinforces_existing_pattern() {
        let mut sanna = StatefulSanna::new("TestSanna", Arc::new(BasicSannaSkandha));
        let mut flow1 = create_flow("repeated pattern");
        let hash = get_hash("repeated pattern");

        // First perception
        sanna.perceive_with_state(&mut flow1);
        let strength1 = sanna.state.lock().pattern_strength(hash);
        assert_eq!(strength1, 0.5);

        // Second perception
        let mut flow2 = create_flow("repeated pattern");
        sanna.perceive_with_state(&mut flow2);
        let strength2 = sanna.state.lock().pattern_strength(hash);

        assert!(strength2 > strength1); // Strength should increase
    }

    #[test]
    fn test_stateful_sanna_decay_works() {
        let mut sanna = StatefulSanna::new("TestSanna", Arc::new(BasicSannaSkandha));
        let mut flow = create_flow("pattern to decay");
        let hash = get_hash("pattern to decay");

        sanna.perceive_with_state(&mut flow);
        let strength1 = sanna.state.lock().pattern_strength(hash);

        // Manually decay the state
        {
            let mut state = sanna.state.lock();
            state.decay(std::time::Duration::from_secs(1));
        }

        let strength2 = sanna.state.lock().pattern_strength(hash);
        assert!(strength2 < strength1);
    }
}
