//! Flow control primitives for recurrent processing.
//!
//! Enables reflection, looping, and energy-bounded computation.

use std::fmt;

// ============================================================================
// COGNITIVE FLOW STATUS
// ============================================================================

/// Control flow status returned by stateful skandhas during recurrent processing.
///
/// This enum allows skandhas to:
/// - Continue to next stage (normal flow)
/// - Request re-processing of a previous stage (reflection)
/// - Yield final result (early termination)
/// - Signal energy depletion (forced termination)
///
/// # State Machine
///
/// ```text
/// Input Event
///    ↓
/// [Rupa] → Continue
///    ↓
/// [Vedana] → Continue
///    ↓
/// [Sanna] → LoopTo(Vedana)  ← Reflection!
///    ↓
/// [Vedana] → Continue
///    ↓
/// [Sanna] → Continue
///    ↓
/// [Sankhara] → Continue
///    ↓
/// [Vinnana] → Yield(output)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitiveFlowStatus {
    /// Continue to next skandha in pipeline.
    Continue,

    /// Loop back to specified skandha for re-processing.
    ///
    /// # Use Cases
    ///
    /// - **Vedana → Vedana**: Mood stabilization (rare)
    /// - **Sanna → Vedana**: Pattern triggers emotional re-evaluation
    /// - **Sankhara → Sanna**: Intent formation needs better perception
    /// - **Vinnana → Any**: Synthesis detects need for deeper processing
    ///
    /// # Constraints
    ///
    /// - Cannot loop forward (only backward reflection)
    /// - Energy budget prevents infinite loops
    /// - Maximum reflection depth configurable per processor
    LoopTo(SkandhaStage),

    /// Yield final output and terminate cycle.
    ///
    /// Used when:
    /// - Vinnana produces rebirth event
    /// - Early termination due to high confidence
    /// - Error conditions requiring immediate response
    Yield(Vec<u8>),

    /// Energy budget depleted, forced termination.
    ///
    /// Processor will:
    /// - Synthesize best-effort output from current flow
    /// - Log energy depletion event
    /// - Return partial result
    EnergyDepleted,
}

/// Stages in the skandha pipeline.
///
/// Order matters: numerical value used for loop validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SkandhaStage {
    Rupa = 0,
    Vedana = 1,
    Sanna = 2,
    Sankhara = 3,
    Vinnana = 4,
}

impl SkandhaStage {
    /// Get next stage in pipeline (None if at end).
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Rupa => Some(Self::Vedana),
            Self::Vedana => Some(Self::Sanna),
            Self::Sanna => Some(Self::Sankhara),
            Self::Sankhara => Some(Self::Vinnana),
            Self::Vinnana => None,
        }
    }

    /// Check if looping to target is valid (backward only).
    pub fn can_loop_to(self, target: SkandhaStage) -> bool {
        target < self
    }

    /// Get stage name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rupa => "Rupa",
            Self::Vedana => "Vedana",
            Self::Sanna => "Sanna",
            Self::Sankhara => "Sankhara",
            Self::Vinnana => "Vinnana",
        }
    }
}

impl fmt::Display for SkandhaStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}


// ============================================================================
// ENERGY BUDGET
// ============================================================================

/// Energy budget for limiting computational cost of recurrent processing.
///
/// Prevents infinite loops and runaway computation by:
/// - Tracking total "energy" spent on processing
/// - Enforcing maximum reflection depth
/// - Providing early warnings before depletion
///
/// # Energy Model
///
/// - Each skandha execution costs 1 energy unit
/// - Initial budget typically 15-25 units (3-5 reflections)
/// - Budget depletes on each processing step
/// - No "recharge" within a cycle (monotonic decrease)
#[derive(Debug, Clone)]
pub struct EnergyBudget {
    /// Initial energy allocation
    initial: u32,

    /// Remaining energy
    remaining: u32,

    /// Number of reflection loops taken
    reflection_count: u32,

    /// Maximum reflections allowed
    max_reflections: u32,

    /// Per-stage reflection counts (for loop detection)
    stage_reflections: [u32; 5],
}

impl EnergyBudget {
    /// Create new energy budget with specified capacity.
    ///
    /// # Recommendations
    ///
    /// - **Low complexity (3-5 reflections)**: 15-25 units
    /// - **Medium complexity (5-10 reflections)**: 25-50 units
    /// - **High complexity (10+ reflections)**: 50-100 units
    ///
    /// Note: Each reflection involves multiple skandha re-executions,
    /// so reflection count ≠ energy units.
    pub fn new(capacity: u32, max_reflections: u32) -> Self {
        Self {
            initial: capacity,
            remaining: capacity,
            reflection_count: 0,
            max_reflections,
            stage_reflections: [0; 5],
        }
    }

    /// Default budget: 20 units, max 5 reflections.
    pub fn default_budget() -> Self {
        Self::new(20, 5)
    }

    /// Conservative budget: 15 units, max 3 reflections.
    pub fn conservative() -> Self {
        Self::new(15, 3)
    }

    /// Generous budget: 30 units, max 7 reflections.
    pub fn generous() -> Self {
        Self::new(30, 7)
    }

    /// Consume energy for processing a skandha.
    ///
    /// Returns `true` if energy was consumed, `false` if depleted.
    pub fn consume(&mut self, amount: u32) -> bool {
        if self.remaining >= amount {
            self.remaining -= amount;
            true
        } else {
            self.remaining = 0;
            false
        }
    }

    /// Record a reflection loop.
    ///
    /// Returns `true` if reflection is allowed, `false` if limit exceeded.
    pub fn record_reflection(&mut self, from_stage: SkandhaStage, to_stage: SkandhaStage) -> bool {
        // Check global reflection limit
        if self.reflection_count >= self.max_reflections {
            return false;
        }

        // Check per-stage reflection limit (prevent tight loops)
        let stage_idx = from_stage as usize;
        if self.stage_reflections[stage_idx] >= 2 {
            return false; // Max 2 reflections from same stage
        }
        
        // This check is important! A stage can only loop back to a previous stage.
        if !from_stage.can_loop_to(to_stage) {
            return false;
        }

        self.reflection_count += 1;
        self.stage_reflections[stage_idx] += 1;

        true
    }

    /// Check if energy is depleted.
    pub fn is_depleted(&self) -> bool {
        self.remaining == 0
    }

    /// Check if approaching depletion (< 20% remaining).
    pub fn is_low(&self) -> bool {
        (self.remaining as f32) < (self.initial as f32 * 0.2)
    }

    /// Get remaining energy as percentage.
    pub fn remaining_percent(&self) -> f32 {
        (self.remaining as f32 / self.initial as f32) * 100.0
    }

    /// Get total energy spent.
    pub fn spent(&self) -> u32 {
        self.initial - self.remaining
    }

    /// Get number of reflections taken.
    pub fn reflections(&self) -> u32 {
        self.reflection_count
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.remaining = self.initial;
        self.reflection_count = 0;
        self.stage_reflections = [0; 5];
    }
}

impl fmt::Display for EnergyBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Energy: {}/{} ({:.1}%), Reflections: {}/{}",
            self.remaining,
            self.initial,
            self.remaining_percent(),
            self.reflection_count,
            self.max_reflections
        )
    }
}


// ============================================================================
// CYCLE RESULT
// ============================================================================

/// Result of a complete epistemological cycle.
///
/// Contains both the output and metadata about the processing.
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// Output event (if any) from Vinnana synthesis.
    pub output: Option<Vec<u8>>,

    /// Final energy budget state.
    pub energy: EnergyBudget,

    /// Total number of skandha executions.
    pub executions: u32,

    /// Number of reflection loops taken.
    pub reflections: u32,

    /// Termination reason.
    pub termination: TerminationReason,
}

/// Why the processing cycle terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Normal completion (reached Vinnana, no loops).
    NormalCompletion,

    /// Completed with reflections.
    CompletedWithReflection,

    /// Energy budget depleted.
    EnergyDepleted,

    /// Early yield from Vinnana.
    EarlyYield,

    /// Reflection limit exceeded.
    ReflectionLimitExceeded,
}


impl CycleResult {
    /// Create result for normal completion.
    pub fn normal(output: Option<Vec<u8>>, energy: EnergyBudget, executions: u32) -> Self {
        Self {
            output,
            energy,
            executions,
            reflections: 0,
            termination: TerminationReason::NormalCompletion,
        }
    }

    /// Create result for completion with reflection.
    pub fn with_reflection(output: Option<Vec<u8>>, energy: EnergyBudget, executions: u32, reflections: u32) -> Self {
        Self {
            output,
            energy,
            executions,
            reflections,
            termination: TerminationReason::CompletedWithReflection,
        }
    }

    /// Create result for energy depletion.
    pub fn energy_depleted(output: Option<Vec<u8>>, energy: EnergyBudget, executions: u32, reflections: u32) -> Self {
        Self {
            output,
            energy,
            executions,
            reflections,
            termination: TerminationReason::EnergyDepleted,
        }
    }

    /// Check if cycle was successful.
    pub fn is_successful(&self) -> bool {
        matches!(
            self.termination,
            TerminationReason::NormalCompletion | TerminationReason::CompletedWithReflection
        )
    }
}

impl fmt::Display for CycleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cycle Result: {:?}, Executions: {}, Reflections: {}, {}",
            self.termination, self.executions, self.reflections, self.energy
        )
    }
}
