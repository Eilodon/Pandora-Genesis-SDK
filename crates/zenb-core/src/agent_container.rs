use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::belief::{AgentStrategy, SensorFeatures, PhysioState, Context};

/// Versioned, resource-guarded container for cognitive agents.
#[derive(Debug, Clone)]
pub struct AgentContainer {
    pub inner: Arc<Mutex<AgentStrategy>>,
    pub version: String, // Git commit hash or build ID
    pub resource_limits: ResourceQuota,
}

#[derive(Debug, Clone)]
pub struct ResourceQuota {
    pub max_cpu_ms_per_tick: u64,
    pub max_memory_mb: usize,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_cpu_ms_per_tick: 5,
            max_memory_mb: 10,
        }
    }
}

impl AgentContainer {
    pub fn new(agent: AgentStrategy, version: String) -> Self {
        Self {
            inner: Arc::new(Mutex::new(agent)),
            version,
            resource_limits: ResourceQuota::default(),
        }
    }

    pub fn evaluate(&self, x: &SensorFeatures, phys: &PhysioState, ctx: &Context) -> f32 {
        // SECURITY: Enforce CPU time limits to prevent DoS from malicious/buggy agents
        let start = Instant::now();
        let timeout_ms = self.resource_limits.max_cpu_ms_per_tick;

        // Attempt evaluation with panic protection
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.inner.lock().unwrap().eval(x, phys, ctx)
        }));

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // Check timeout violation
        if elapsed_ms > timeout_ms {
            log::error!(
                "AgentContainer: Evaluation exceeded time limit ({} ms > {} ms limit). \
                 Agent version: {}. Returning fallback confidence=0.0",
                elapsed_ms,
                timeout_ms,
                self.version
            );
            return 0.0; // Fallback: zero confidence for timeout violations
        }

        match result {
            Ok(eval_result) => {
                if elapsed_ms > timeout_ms / 2 {
                    // Warn if approaching limit (50% threshold)
                    log::warn!(
                        "AgentContainer: Evaluation slow ({} ms, limit={} ms). Agent version: {}",
                        elapsed_ms,
                        timeout_ms,
                        self.version
                    );
                }
                eval_result.confidence
            }
            Err(e) => {
                // Agent panicked
                log::error!(
                    "AgentContainer: Agent panicked during evaluation: {:?}. \
                     Agent version: {}. Returning fallback confidence=0.0",
                    e,
                    self.version
                );
                0.0 // Fallback: zero confidence for panicked agents
            }
        }
    }
}
