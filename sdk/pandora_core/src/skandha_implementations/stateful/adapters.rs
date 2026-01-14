//! Adapter to make a stateless skandha compatible with stateful processors.

use crate::skandha_implementations::core::traits::{
    Skandha, StatefulSkandha, RupaSkandha, VedanaSkandha, SannaSkandha, SankharaSkandha, VinnanaSkandha,
    StatefulVedanaSkandha, StatefulSannaSkandha,
};
use crate::ontology::EpistemologicalFlow;

/// Wraps a stateless `Skandha` to make it implement `StatefulSkandha`.
///
/// This is a crucial compatibility layer that allows us to mix and match
/// stateless and stateful components within a stateful processor (like the
/// upcoming RecurrentProcessor).
///
/// It implements `StatefulSkandha` by providing a unit type `()` as the state,
/// which has zero runtime cost. All stateful method calls are simply forwarded
/// to their stateless counterparts.
#[derive(Debug, Clone, Copy)]
pub struct StatelessAdapter<T: Skandha> {
    inner: T,
}

impl<T: Skandha> StatelessAdapter<T> {
    /// Create a new adapter for a stateless skandha.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

// ============================================================================
// BASE TRAIT IMPLEMENTATIONS
// ============================================================================

impl<T: Skandha> Skandha for StatelessAdapter<T> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

impl<T: Skandha> StatefulSkandha for StatelessAdapter<T> {
    type State = (); // The "state" is a zero-sized unit type.

    fn state(&self) -> &Self::State {
        &()
    }

    fn state_mut(&mut self) -> &mut Self::State {
        // We just return a mutable reference to the unit type.
        // This is a no-op, but it satisfies the trait bounds.
        static mut UNIT: () = ();
        unsafe { &mut *(&raw mut UNIT) }
    }
}

// ============================================================================
// RUPA ADAPTER
// ============================================================================

impl<T: RupaSkandha> RupaSkandha for StatelessAdapter<T> {
    #[inline]
    fn process_event(&self, event: Vec<u8>) -> EpistemologicalFlow {
        self.inner.process_event(event)
    }
}

// ============================================================================
// VEDANA ADAPTER
// ============================================================================

impl<T: VedanaSkandha> VedanaSkandha for StatelessAdapter<T> {
    #[inline]
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        self.inner.feel(flow)
    }
}

/// For a stateless Vedana, `feel_with_state` is identical to `feel`.
#[async_trait::async_trait]
impl<T: VedanaSkandha> StatefulVedanaSkandha for StatelessAdapter<T> {
    async fn feel_with_state(&mut self, flow: &mut EpistemologicalFlow) {
        self.inner.feel(flow);
    }
}

// ============================================================================
// SANNA ADAPTER
// ============================================================================

impl<T: SannaSkandha> SannaSkandha for StatelessAdapter<T> {
    #[inline]
    fn perceive(&self, flow: &mut EpistemologicalFlow) {
        self.inner.perceive(flow)
    }
}

/// For a stateless Sanna, `perceive_with_state` is identical to `perceive`.
impl<T: SannaSkandha> StatefulSannaSkandha for StatelessAdapter<T> {
    fn perceive_with_state(&mut self, flow: &mut EpistemologicalFlow) {
        self.inner.perceive(flow);
    }
}

// ============================================================================
// SANKHARA ADAPTER
// ============================================================================

impl<T: SankharaSkandha> SankharaSkandha for StatelessAdapter<T> {
    #[inline]
    fn form_intent(&self, flow: &mut EpistemologicalFlow) {
        self.inner.form_intent(flow)
    }
}

// ============================================================================
// VINNANA ADAPTER
// ============================================================================

impl<T: VinnanaSkandha> VinnanaSkandha for StatelessAdapter<T> {
    #[inline]
    fn synthesize(&self, flow: &EpistemologicalFlow) -> Option<Vec<u8>> {
        self.inner.synthesize(flow)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skandha_implementations::basic::*;

    #[test]
    fn test_adapter_wraps_basic_rupa() {
        let adapter = StatelessAdapter::new(BasicRupaSkandha::default());
        let event = b"test".to_vec();
        let flow = adapter.process_event(event);
        
        assert!(flow.rupa.is_some());
    }

    #[test]
    fn test_adapter_preserves_name() {
        let adapter = StatelessAdapter::new(BasicVedanaSkandha::default());
        assert_eq!(adapter.name(), "BasicVedana");
    }

    #[test]
    fn test_adapter_stateful_trait() {
        let mut adapter = StatelessAdapter::new(BasicSankharaSkandha::default());
        
        // Should implement StatefulSkandha with unit state
        let _state = adapter.state();
        let _state_mut = adapter.state_mut();
        // This test just checks that the methods exist and can be called without panicking.
    }
}
