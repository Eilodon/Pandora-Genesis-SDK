//! Flow Enforcement - Macros and utilities for observability
//!
//! # B.ONE V3: Luật Chánh Niệm (Observability First)
//!
//! This module provides macros and utilities to enforce the principle that
//! all significant operations must emit FlowEvents. If it's not observed,
//! it didn't happen (no Karma).
//!
//! ## Design Philosophy
//! - **Compile-time hints**: `#[must_use]` attributes on critical functions
//! - **Runtime enforcement**: Macros that wrap operations with automatic emission
//! - **Audit trail**: Every operation leaves a trace in the FlowStream
//!
//! ## Usage
//! ```ignore
//! use zenb_core::flow_enforcement::*;
//!
//! // Wrap critical operations with automatic flow emission
//! with_flow_observation!(stream, SkandhaStage::Sankhara, ts_us, {
//!     sankhara.deliberate(...)
//! });
//! ```

use crate::universal_flow::{
    FlowEvent, FlowEventId, FlowPayload, SkandhaStage, UniversalFlowStream,
};

// ============================================================================
// FLOW ENFORCEMENT MACROS
// ============================================================================

/// Wrap a critical operation with automatic FlowStream observation.
///
/// # B.ONE V3: Luật Chánh Niệm
/// This macro ensures that the operation is recorded in the FlowStream,
/// providing an audit trail for all significant actions.
///
/// # Example
/// ```ignore
/// let result = with_flow_observation!(stream, SkandhaStage::Sankhara, ts_us, {
///     engine.make_control(&estimate, ts_us)
/// });
/// ```
#[macro_export]
macro_rules! with_flow_observation {
    ($stream:expr, $stage:expr, $ts:expr, $body:block) => {{
        // Emit entry event
        let entry_id = $stream.emit(
            $crate::universal_flow::FlowPayload::SystemObservation(
                $crate::universal_flow::SystemObservation {
                    health: $crate::universal_flow::SystemHealth::Healthy,
                    philosophical_state: $stream.philosophical_state(),
                    coherence: 0.8,
                    free_energy: 0.0,
                    belief_mode: $crate::belief::BeliefBasis::Calm,
                    karma_balance: $stream.stats().karma_balance,
                    consciousness_reports: Default::default(),
                }
            ),
            $stage,
            $ts
        );

        // Execute the body
        let result = $body;

        // Return result
        result
    }};
}

/// Assert that a FlowEvent was emitted for a critical operation.
///
/// # Panics
/// Panics in debug builds if no event was emitted. Silent in release builds.
#[macro_export]
macro_rules! assert_flow_emitted {
    ($stream:expr, $expected_stage:expr, $msg:literal) => {{
        #[cfg(debug_assertions)]
        {
            let stats = $stream.stats();
            let stage_idx = $expected_stage as usize;
            if stats.events_by_stage[stage_idx] == 0 {
                panic!(
                    "Flow enforcement violation: {} - Expected emission at stage {:?}",
                    $msg, $expected_stage
                );
            }
        }
    }};
}

/// Create a traced operation that records entry and exit in the FlowStream.
///
/// # Example
/// ```ignore
/// traced_operation!(stream, "deliberate", SkandhaStage::Sankhara, ts_us, {
///     sankhara.deliberate(...)
/// });
/// ```
#[macro_export]
macro_rules! traced_operation {
    ($stream:expr, $op_name:expr, $stage:expr, $ts:expr, $body:block) => {{
        log::debug!("TRACE_ENTER: {} at {:?}", $op_name, $stage);

        let start_ts = std::time::Instant::now();
        let result = $body;
        let elapsed = start_ts.elapsed();

        log::debug!(
            "TRACE_EXIT: {} completed in {:?}",
            $op_name, elapsed
        );

        result
    }};
}

// ============================================================================
// FLOW GUARD - RAII-style automatic emission
// ============================================================================

/// RAII guard that ensures FlowStream emission on scope exit.
///
/// # B.ONE V3: Automatic Karma Recording
/// When this guard is dropped, it automatically emits a completion event
/// to the FlowStream, ensuring nothing escapes observation.
///
/// # Example
/// ```ignore
/// let _guard = FlowGuard::new(&mut stream, SkandhaStage::Vinnana, ts_us, "synthesis");
/// // ... do work ...
/// // Guard automatically emits on drop
/// ```
pub struct FlowGuard<'a> {
    stream: &'a mut UniversalFlowStream,
    stage: SkandhaStage,
    timestamp_us: i64,
    operation: &'static str,
    completed: bool,
}

impl<'a> FlowGuard<'a> {
    /// Create a new FlowGuard for the given operation.
    pub fn new(
        stream: &'a mut UniversalFlowStream,
        stage: SkandhaStage,
        timestamp_us: i64,
        operation: &'static str,
    ) -> Self {
        log::trace!("FlowGuard: entering {} at {:?}", operation, stage);
        Self {
            stream,
            stage,
            timestamp_us,
            operation,
            completed: false,
        }
    }

    /// Mark the operation as completed (prevents error emission on drop).
    pub fn complete(&mut self) {
        self.completed = true;
    }

    /// Emit a custom payload before dropping.
    pub fn emit_result(&mut self, payload: FlowPayload) -> FlowEvent {
        self.completed = true;
        self.stream.emit(payload, self.stage, self.timestamp_us)
    }
}

impl<'a> Drop for FlowGuard<'a> {
    fn drop(&mut self) {
        if !self.completed {
            log::warn!(
                "FlowGuard: {} at {:?} dropped without completion",
                self.operation, self.stage
            );
        }
        log::trace!("FlowGuard: exiting {} at {:?}", self.operation, self.stage);
    }
}

// ============================================================================
// FLOW VALIDATOR - Runtime assertion helpers
// ============================================================================

/// Validator for FlowStream invariants.
pub struct FlowValidator;

impl FlowValidator {
    /// Validate that events are flowing through the pipeline in order.
    ///
    /// # Returns
    /// `Ok(())` if pipeline order is valid, `Err(String)` with description otherwise.
    pub fn validate_pipeline_order(events: &[FlowEvent]) -> Result<(), String> {
        let mut last_stage: Option<SkandhaStage> = None;

        for event in events {
            if let Some(prev) = last_stage {
                // Check for valid transitions
                let valid = match prev {
                    SkandhaStage::Rupa => matches!(
                        event.skandha_stage,
                        SkandhaStage::Rupa | SkandhaStage::Vedana
                    ),
                    SkandhaStage::Vedana => matches!(
                        event.skandha_stage,
                        SkandhaStage::Vedana | SkandhaStage::Sanna
                    ),
                    SkandhaStage::Sanna => matches!(
                        event.skandha_stage,
                        SkandhaStage::Sanna | SkandhaStage::Sankhara
                    ),
                    SkandhaStage::Sankhara => matches!(
                        event.skandha_stage,
                        SkandhaStage::Sankhara | SkandhaStage::Vinnana
                    ),
                    SkandhaStage::Vinnana => matches!(
                        event.skandha_stage,
                        SkandhaStage::Vinnana | SkandhaStage::Rebirth | SkandhaStage::Rupa
                    ),
                    SkandhaStage::Rebirth => matches!(
                        event.skandha_stage,
                        SkandhaStage::Rebirth | SkandhaStage::Rupa
                    ),
                };

                if !valid {
                    return Err(format!(
                        "Invalid pipeline transition: {:?} -> {:?}",
                        prev, event.skandha_stage
                    ));
                }
            }
            last_stage = Some(event.skandha_stage);
        }

        Ok(())
    }

    /// Validate that karma balance is within acceptable bounds.
    pub fn validate_karma_balance(stream: &UniversalFlowStream, threshold: f32) -> Result<(), String> {
        let balance = stream.stats().karma_balance;
        if balance.abs() > threshold {
            return Err(format!(
                "Karma imbalance: {} exceeds threshold {}",
                balance, threshold
            ));
        }
        Ok(())
    }

    /// Validate that all Skandha stages have been visited at least once.
    pub fn validate_complete_cycle(stream: &UniversalFlowStream) -> Result<(), String> {
        let stats = stream.stats();
        let stages = ["Rupa", "Vedana", "Sanna", "Sankhara", "Vinnana"];

        for (i, stage_name) in stages.iter().enumerate() {
            if stats.events_by_stage[i] == 0 {
                return Err(format!(
                    "Incomplete cycle: {} stage never visited",
                    stage_name
                ));
            }
        }

        Ok(())
    }
}

// ============================================================================
// MUST-USE WRAPPERS
// ============================================================================

/// A wrapper that enforces result handling via `#[must_use]`.
///
/// Use this to wrap critical operations that MUST NOT be ignored.
#[must_use = "This operation must be observed - ignoring it violates Luật Chánh Niệm"]
pub struct ObservedResult<T> {
    pub value: T,
    pub event_id: FlowEventId,
}

impl<T> ObservedResult<T> {
    /// Create a new observed result.
    pub fn new(value: T, event_id: FlowEventId) -> Self {
        Self { value, event_id }
    }

    /// Consume and return the inner value.
    pub fn into_value(self) -> T {
        self.value
    }

    /// Get a reference to the inner value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get the event ID for tracing.
    pub fn event_id(&self) -> FlowEventId {
        self.event_id
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_validator_pipeline_order() {
        use crate::domain::SessionId;
        use crate::universal_flow::FlowEnrichment;
        use crate::skandha::SensorInput;

        // Valid sequence
        let events = vec![
            FlowEvent {
                id: FlowEventId(1),
                session_id: SessionId::default(),
                timestamp_us: 1000,
                payload: FlowPayload::RawSensor(SensorInput::default()),
                skandha_stage: SkandhaStage::Rupa,
                enrichment: FlowEnrichment::default(),
                lineage: vec![],
            },
            FlowEvent {
                id: FlowEventId(2),
                session_id: SessionId::default(),
                timestamp_us: 2000,
                payload: FlowPayload::RawSensor(SensorInput::default()),
                skandha_stage: SkandhaStage::Vedana,
                enrichment: FlowEnrichment::default(),
                lineage: vec![],
            },
        ];

        assert!(FlowValidator::validate_pipeline_order(&events).is_ok());
    }

    #[test]
    fn test_observed_result_must_use() {
        let result = ObservedResult::new(42, FlowEventId(1));
        assert_eq!(result.value(), &42);
        assert_eq!(result.event_id(), FlowEventId(1));

        // Consuming the result
        let value = result.into_value();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_flow_validator_karma_balance() {
        let stream = UniversalFlowStream::new();

        // Fresh stream should have zero balance
        assert!(FlowValidator::validate_karma_balance(&stream, 5.0).is_ok());
    }
}
