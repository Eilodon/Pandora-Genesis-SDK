use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub fn rng_for_tests(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

#[derive(Debug, Default)]
pub struct AgentMetrics {
    pub success_count: u32,
    pub failure_count: u32,
    pub steps_taken: u32,
    pub optimal_steps: u32,
}

impl AgentMetrics {
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }
}
