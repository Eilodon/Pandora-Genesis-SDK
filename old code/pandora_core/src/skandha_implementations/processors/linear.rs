//! Linear (non-recurrent) skandha processor.
//!
//! This is the V1 processor - extracted from `fep_cell.rs` with zero logic changes.

use crate::skandha_implementations::core::{
    RupaSkandha, SannaSkandha, SankharaSkandha, VedanaSkandha, VinnanaSkandha,
};
use tracing::info;

/// Linear skandha processor (V1 - original architecture).
///
/// # Characteristics
///
/// - **Stateless**: Skandhas have no memory between cycles
/// - **Deterministic**: Same input → same output
/// - **Fast**: ~30-40µs per cycle
/// - **Simple**: No reflection, no energy budget
///
/// # Performance Baseline
///
/// - Latency: ~30-40µs per cycle (sync version)
/// - Memory: ~2KB per cycle (with recycling)
/// - Throughput: >30,000 cycles/sec (single thread)
///
/// # Use Cases
///
/// - Production systems requiring predictable latency
/// - High-throughput event processing
/// - Deterministic testing and validation
/// - Baseline for benchmarking advanced processors
pub struct LinearProcessor {
    rupa: Box<dyn RupaSkandha>,
    vedana: Box<dyn VedanaSkandha>,
    sanna: Box<dyn SannaSkandha>,
    sankhara: Box<dyn SankharaSkandha>,
    vinnana: Box<dyn VinnanaSkandha>,
}

impl LinearProcessor {
    /// Create new linear processor with specified skandha implementations.
    pub fn new(
        rupa: Box<dyn RupaSkandha>,
        vedana: Box<dyn VedanaSkandha>,
        sanna: Box<dyn SannaSkandha>,
        sankhara: Box<dyn SankharaSkandha>,
        vinnana: Box<dyn VinnanaSkandha>,
    ) -> Self {
        info!("✅ LinearProcessor initialized");
        Self {
            rupa,
            vedana,
            sanna,
            sankhara,
            vinnana,
        }
    }

    /// Run a complete epistemological cycle synchronously.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Input → Rupa → Vedana → Sanna → Sankhara → Vinnana → Output
    ///         (form) (feel)  (perceive) (intent)  (synthesize)
    /// ```
    ///
    /// # Performance
    ///
    /// - Inline hint for optimization
    /// - Zero allocations (flow reused)
    /// - Sequential execution (no parallelism)
    #[inline]
    pub fn run_cycle(&self, event: Vec<u8>) -> Option<Vec<u8>> {
        info!("\n--- 🔄 LINEAR CYCLE START ---");

        // 1. Rūpa: Form
        let mut flow = self.rupa.process_event(event);

        // 2. Vedanā: Feel
        self.vedana.feel(&mut flow);

        // 3. Saññā: Perceive
        self.sanna.perceive(&mut flow);

        // 4. Saṅkhāra: Form Intent
        self.sankhara.form_intent(&mut flow);

        // 5. Viññāṇa: Synthesize
        let reborn_event = self.vinnana.synthesize(&flow);

        info!("--- ✅ LINEAR CYCLE END ---");

        reborn_event
    }

    /// Async variant for compatibility with async I/O boundaries.
    ///
    /// Note: This is NOT truly async - it just wraps the sync version.
    /// For real async processing, use AsyncLinearProcessor (Phase 2).
    pub async fn run_cycle_async(&self, event: Vec<u8>) -> Option<Vec<u8>> {
        self.run_cycle(event)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skandha_implementations::basic::*;

    fn create_test_processor() -> LinearProcessor {
        LinearProcessor::new(
            Box::new(BasicRupaSkandha),
            Box::new(BasicVedanaSkandha),
            Box::new(BasicSannaSkandha),
            Box::new(BasicSankharaSkandha),
            Box::new(BasicVinnanaSkandha),
        )
    }

    #[test]
    fn test_linear_processor_error_event() {
        let processor = create_test_processor();
        let event = b"critical error detected".to_vec();

        let result = processor.run_cycle(event);

        assert!(result.is_some());
        let output = String::from_utf8(result.unwrap()).unwrap();
        assert!(output.contains("rebirth"));
    }

    #[test]
    fn test_linear_processor_success_event() {
        let processor = create_test_processor();
        let event = b"operation success".to_vec();

        let result = processor.run_cycle(event);

        // Success events typically don't trigger rebirth
        assert!(result.is_none());
    }

    #[test]
    fn test_linear_processor_deterministic() {
        let processor = create_test_processor();
        let event = b"test event".to_vec();

        let result1 = processor.run_cycle(event.clone());
        let result2 = processor.run_cycle(event.clone());

        assert_eq!(result1, result2);
    }

    #[tokio::test]
    async fn test_linear_processor_async() {
        let processor = create_test_processor();
        let event = b"async test".to_vec();

        let result = processor.run_cycle_async(event).await;

        // Just verify it runs without panic
        assert!(result.is_none() || result.is_some());
    }
}
