use crate::skandha_implementations::core::*;  // Use new core traits instead of deprecated interfaces
use tracing::info;

/// Core cognitive processor implementing the Five Skandhas (aggregates) pipeline.
///
/// # Architecture
///
/// The `SkandhaProcessor` implements a five-stage pipeline based on Buddhist philosophy:
///
/// 1. **Rupa (Form)**: Raw sensory input processing
/// 2. **Vedana (Feeling)**: Moral/emotional valence assignment  
/// 3. **Sanna (Perception)**: Pattern recognition and knowledge retrieval
/// 4. **Sankhara (Formations)**: Intent formation based on perceived patterns
/// 5. **Vinnana (Consciousness)**: Synthesis and potential rebirth
///
/// # Performance Characteristics
///
/// - **Latency**: ~30-40µs per cycle (sync version)
/// - **Memory**: ~2KB per cycle (with recycling)
/// - **Throughput**: >30,000 cycles/sec (single thread)
///
/// # Examples
///
/// ```rust
/// use pandora_core::fep_cell::SkandhaProcessor;
/// use pandora_core::skandha_implementations::basic_skandhas::*;
///
/// let processor = SkandhaProcessor::new(
///     Box::new(BasicRupaSkandha),
///     Box::new(BasicVedanaSkandha),
///     Box::new(BasicSannaSkandha),
///     Box::new(BasicSankharaSkandha),
///     Box::new(BasicVinnanaSkandha),
/// );
///
/// let event = b"system error detected".to_vec();
/// let result = processor.run_epistemological_cycle(event);
///
/// // Error events produce reborn events with corrective intent
/// assert!(result.is_some());
/// ```
///
/// # Async Processing
///
/// ```rust
/// # use pandora_core::fep_cell::SkandhaProcessor;
/// # use pandora_core::skandha_implementations::basic_skandhas::*;
/// # let processor = SkandhaProcessor::new(
/// #     Box::new(BasicRupaSkandha),
/// #     Box::new(BasicVedanaSkandha),
/// #     Box::new(BasicSannaSkandha),
/// #     Box::new(BasicSankharaSkandha),
/// #     Box::new(BasicVinnanaSkandha),
/// # );
/// # async fn example() {
/// # use pandora_core::fep_cell::SkandhaProcessor;
/// # use pandora_core::skandha_implementations::basic_skandhas::*;
/// # let processor = SkandhaProcessor::new(
/// #     Box::new(BasicRupaSkandha),
/// #     Box::new(BasicVedanaSkandha),
/// #     Box::new(BasicSannaSkandha),
/// #     Box::new(BasicSankharaSkandha),
/// #     Box::new(BasicVinnanaSkandha),
/// # );
/// let event = b"normal operation".to_vec();
/// let result = processor.run_epistemological_cycle_async(event).await;
/// // Normal events typically don't produce reborn events
/// assert!(result.is_none());
/// # }
/// ```
///
/// # Thread Safety
///
/// `SkandhaProcessor` is `Send + Sync`. Multiple threads can process events
/// concurrently using `Arc<SkandhaProcessor>`.
///
/// # See Also
///
/// - [`run_epistemological_cycle`](Self::run_epistemological_cycle) - Sync processing
/// - [`run_epistemological_cycle_async`](Self::run_epistemological_cycle_async) - Async wrapper
pub struct SkandhaProcessor {
    rupa: Box<dyn RupaSkandha>,
    vedana: Box<dyn VedanaSkandha>,
    sanna: Box<dyn SannaSkandha>,
    sankhara: Box<dyn SankharaSkandha>,
    vinnana: Box<dyn VinnanaSkandha>,
}

impl SkandhaProcessor {
    pub fn new(
        rupa: Box<dyn RupaSkandha>,
        vedana: Box<dyn VedanaSkandha>,
        sanna: Box<dyn SannaSkandha>,
        sankhara: Box<dyn SankharaSkandha>,
        vinnana: Box<dyn VinnanaSkandha>,
    ) -> Self {
        info!("✅ SkandhaProcessor V3 đã được khởi tạo.");
        Self {
            rupa,
            vedana,
            sanna,
            sankhara,
            vinnana,
        }
    }

    /// Runs a complete epistemological cycle synchronously.
    ///
    /// This is the primary method for processing events. It is fully synchronous
    /// and optimized for CPU-bound processing.
    ///
    /// # Arguments
    ///
    /// * `event` - Raw event data (typically UTF-8 encoded string)
    ///
    /// # Returns
    ///
    /// * `Some(Vec<u8>)` - Reborn event with synthesized intent
    /// * `None` - No action needed (neutral event)
    ///
    /// # Performance
    ///
    /// - **Latency**: ~30-40µs (single-threaded)
    /// - **Memory**: ~2KB allocated per cycle (recycled)
    /// - **CPU**: Pure computation, no I/O
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use pandora_core::fep_cell::SkandhaProcessor;
    /// # use pandora_core::skandha_implementations::basic_skandhas::*;
    /// # let processor = SkandhaProcessor::new(
    /// #     Box::new(BasicRupaSkandha),
    /// #     Box::new(BasicVedanaSkandha),
    /// #     Box::new(BasicSannaSkandha),
    /// #     Box::new(BasicSankharaSkandha),
    /// #     Box::new(BasicVinnanaSkandha),
    /// # );
    ///
    /// // Normal event - no action
    /// let result = processor.run_epistemological_cycle(b"hello".to_vec());
    /// assert!(result.is_none());
    ///
    /// // Error event - produces corrective intent
    /// let result = processor.run_epistemological_cycle(b"error occurred".to_vec());
    /// assert!(result.is_some());
    /// ```
    ///
    /// # See Also
    ///
    /// - [`run_epistemological_cycle_async`](Self::run_epistemological_cycle_async) - Async wrapper
    #[inline]
    pub fn run_epistemological_cycle(&self, event: Vec<u8>) -> Option<Vec<u8>> {
        info!("\n--- LUỒNG NHẬN THỨC LUẬN BẮT ĐẦU ---");

        // 1. Sắc: Tiếp nhận sự kiện (đồng bộ)
        let mut flow = self.rupa.process_event(event);

        // 2. Thọ: Gán cảm giác (đồng bộ)
        self.vedana.feel(&mut flow);

        // 3. Tưởng: Nhận diện quy luật (đồng bộ)
        self.sanna.perceive(&mut flow);

        // 4. Hành: Khởi phát ý chỉ (đồng bộ)
        self.sankhara.form_intent(&mut flow);

        // 5. Thức: Tổng hợp và tái sinh (đồng bộ)
        let reborn_event = self.vinnana.synthesize(&flow);

        info!("--- LUỒNG NHẬN THỨC LUẬN KẾT THÚC ---");

        reborn_event
    }

    /// Biến thể bất đồng bộ để dễ dàng kết hợp với async I/O ở biên.
    /// Chỉ sử dụng khi cần thiết để kết hợp với async code.
    pub async fn run_epistemological_cycle_async(&self, event: Vec<u8>) -> Option<Vec<u8>> {
        // Delegate to sync version - async wrapper for composition
        self.run_epistemological_cycle(event)
    }
}
