//! Recurrent skandha processor with reflection capability.
//!
//! This is the Phase 1 culmination - a processor that can "think again".

use crate::skandha_implementations::core::{
    traits::*,
    flow_control::*,
};
use tracing::{debug, info, warn};

/// Recurrent processor with stateful skandhas and reflection capability.
///
/// # Architecture
///
/// Unlike LinearProcessor which processes events in a single forward pass,
/// RecurrentProcessor can loop back to previous stages for deeper analysis.
///
/// ```text
/// Normal Flow:
/// Input → Rupa → Vedana → Sanna → Sankhara → Vinnana → Output
///
/// Reflective Flow:
/// Input → Rupa → Vedana → Sanna ─┐
///                            ↑    │ "I need to re-feel this!"
///                            └────┘ LoopTo(Vedana)
///              → Vedana → Sanna → Sankhara → Vinnana → Output
/// ```
///
/// # State Machine
///
/// The processor maintains:
/// - **Current stage**: Which skandha to execute next.
/// - **Energy budget**: Limits the total number of processing steps.
/// It terminates when it reaches the end (Vinnana), runs out of energy,
/// or a skandha explicitly yields a result.
pub struct RecurrentProcessor<V, S>
where
    V: StatefulVedanaSkandha,
    S: StatefulSannaSkandha,
{
    pub rupa: Box<dyn RupaSkandha>,
    pub vedana: V,
    pub sanna: S,
    pub sankhara: Box<dyn SankharaSkandha>,
    pub vinnana: Box<dyn VinnanaSkandha>,
}


impl<V, S> RecurrentProcessor<V, S>
where
    V: StatefulVedanaSkandha,
    S: StatefulSannaSkandha,
{
    pub fn new(
        rupa: Box<dyn RupaSkandha>,
        vedana: V,
        sanna: S,
        sankhara: Box<dyn SankharaSkandha>,
        vinnana: Box<dyn VinnanaSkandha>,
    ) -> Self {
        info!("✅ RecurrentProcessor initialized");
        Self {
            rupa,
            vedana,
            sanna,
            sankhara,
            vinnana,
        }
    }

    /// Run a complete epistemological cycle with potential for recurrent loops.
    pub fn run_cycle(&mut self, event: Vec<u8>, initial_budget: EnergyBudget) -> CycleResult {
        info!("\n--- 🌀 RECURRENT CYCLE START ---");

        let mut budget = initial_budget;
        let mut flow = self.rupa.process_event(event);
        let mut current_stage = SkandhaStage::Vedana;
        let mut executions = 1; // Rupa is always first

        loop {
            if !budget.consume(1) {
                warn!("Energy depleted. Terminating cycle.");
                return CycleResult::energy_depleted(None, budget.clone(), executions, budget.reflections());
            }

            executions += 1;
            debug!("Stage: {}, {}", current_stage, budget);

            // =================================================================
            // CORE LOGIC UPDATE - THIS IS THE CRITICAL CHANGE
            // =================================================================
            // Now, we get the status from the skandha's execution.
            let status = match current_stage {
                SkandhaStage::Vedana => {
                    // Note: feel_with_state is async but we use a blocking approach here
                    // for compatibility with the synchronous processor interface.
                    // In production, the processor itself should be async.
                    let _ = futures::executor::block_on(self.vedana.feel_with_state(&mut flow));
                    CognitiveFlowStatus::Continue
                }
                SkandhaStage::Sanna => {
                    self.sanna.perceive_with_state(&mut flow);
                    CognitiveFlowStatus::Continue
                }
                SkandhaStage::Sankhara => {
                    self.sankhara.form_intent(&mut flow);
                    CognitiveFlowStatus::Continue
                }
                SkandhaStage::Vinnana => {
                    if let Some(output) = self.vinnana.synthesize(&flow) {
                        CognitiveFlowStatus::Yield(output)
                    } else {
                        CognitiveFlowStatus::Continue
                    }
                }
                SkandhaStage::Rupa => {
                    warn!("Rupa stage reached within recurrent loop. This is a bug.");
                    CognitiveFlowStatus::Continue
                }
            };
            // =================================================================
            // END OF CRITICAL CHANGE
            // =================================================================

            match status {
                CognitiveFlowStatus::Continue => {
                    if let Some(next_stage) = current_stage.next() {
                        current_stage = next_stage;
                    } else {
                        info!("--- ✅ RECURRENT CYCLE END (Normal Completion) ---");
                        return CycleResult::with_reflection(None, budget.clone(), executions, budget.reflections());
                    }
                }
                CognitiveFlowStatus::LoopTo(target_stage) => {
                    if budget.record_reflection(current_stage, target_stage) {
                        info!("Looping from {} back to {}", current_stage, target_stage);
                        current_stage = target_stage;
                    } else {
                        warn!("Reflection limit exceeded. Terminating cycle.");
                        return CycleResult {
                            output: None,
                            energy: budget.clone(),
                            executions,
                            reflections: budget.reflections(),
                            termination: TerminationReason::ReflectionLimitExceeded,
                        };
                    }
                }
                CognitiveFlowStatus::Yield(output) => {
                    info!("--- ✅ RECURRENT CYCLE END (Early Yield) ---");
                    return CycleResult {
                        output: Some(output),
                        energy: budget.clone(),
                        executions,
                        reflections: budget.reflections(),
                        termination: TerminationReason::EarlyYield,
                    };
                }
                CognitiveFlowStatus::EnergyDepleted => {
                     warn!("Energy depleted signal received. Terminating cycle.");
                     return CycleResult::energy_depleted(None, budget.clone(), executions, budget.reflections());
                }
            }
        }
    }
}
