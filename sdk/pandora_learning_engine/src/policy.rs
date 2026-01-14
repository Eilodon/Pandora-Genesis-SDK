use crate::value_estimator::{NeuralQValueEstimator, QValueEstimator};
use pandora_core::ontology::EpistemologicalFlow;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyAction {
    Noop,
    Explore,
    Exploit,
    UnlockDoor,
    PickUpKey,
    MoveForward,
}

impl PolicyAction {
    /// Convert action to string for Q-value estimation
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyAction::Noop => "noop",
            PolicyAction::Explore => "explore",
            PolicyAction::Exploit => "exploit",
            PolicyAction::UnlockDoor => "unlock_door",
            PolicyAction::PickUpKey => "pick_up_key",
            PolicyAction::MoveForward => "move_forward",
        }
    }

    /// Get all available actions
    pub fn all_actions() -> Vec<PolicyAction> {
        vec![
            PolicyAction::Noop,
            PolicyAction::Explore,
            PolicyAction::Exploit,
            PolicyAction::UnlockDoor,
            PolicyAction::PickUpKey,
            PolicyAction::MoveForward,
        ]
    }
}

pub trait Policy {
    fn select_action(&self, flow: &EpistemologicalFlow, explore_rate: f64) -> PolicyAction;
    fn update(&mut self, _flow: &EpistemologicalFlow, _advantage: f64) {}
}

#[derive(Default)]
pub struct EpsilonGreedyPolicy;

impl Policy for EpsilonGreedyPolicy {
    fn select_action(&self, _flow: &EpistemologicalFlow, explore_rate: f64) -> PolicyAction {
        if explore_rate > 0.0 {
            PolicyAction::Explore
        } else {
            PolicyAction::Exploit
        }
    }
    fn update(&mut self, _flow: &EpistemologicalFlow, _advantage: f64) {
        // no-op for sample stub
    }
}

/// Value-driven policy that uses Q-value estimation for action selection
pub struct ValueDrivenPolicy {
    q_estimator: NeuralQValueEstimator,
    pub total_visits: u32,
    exploration_constant: f64,
    base_exploration_constant: f64,
    recent_rewards: Vec<f64>,
    max_recent: usize,
    temp_explore_boost_steps: u32,
    // Reward normalization (EMA)
    running_mean: f64,
    running_var: f64,
    ema_alpha: f64,
    // Phase detection
    phase_shift_mode: bool,
    phase_steps_remaining: u32,
    // Masking of recently harmful action
    masked_action: Option<String>,
    mask_steps_remaining: u32,
    // Track consecutive negative rewards and recent positive action
    neg_reward_streak: u32,
    last_positive_action: Option<String>,
    // Stagnation detection
    best_recent_avg: f64,
    stagnation_counter: u32,
}

impl ValueDrivenPolicy {
    /// Creates a new value-driven policy
    pub fn new(learning_rate: f64, discount_factor: f64, exploration_constant: f64) -> Self {
        Self {
            q_estimator: NeuralQValueEstimator::new(learning_rate, discount_factor),
            total_visits: 0,
            exploration_constant,
            base_exploration_constant: exploration_constant,
            recent_rewards: Vec::new(),
            max_recent: 20,
            temp_explore_boost_steps: 0,
            running_mean: 0.0,
            running_var: 1e-6,
            ema_alpha: 0.05,
            phase_shift_mode: false,
            phase_steps_remaining: 0,
            masked_action: None,
            mask_steps_remaining: 0,
            neg_reward_streak: 0,
            last_positive_action: None,
            best_recent_avg: 0.0,
            stagnation_counter: 0,
        }
    }

    /// Calculate UCB1 score for action selection with exploration bonus
    pub fn ucb1_score(&self, flow: &EpistemologicalFlow, action: &PolicyAction) -> f64 {
        let action_str = action.as_str();
        let q_value = self
            .q_estimator
            .get_q_values(flow)
            .unwrap_or_default()
            .iter()
            .find(|(action_name, _)| *action_name == action_str)
            .map(|(_, q)| *q)
            .unwrap_or(0.0);

        let visit_count = self.q_estimator.get_visit_count(flow, action_str);

        if visit_count == 0 {
            // If never visited, return high score to encourage exploration
            f64::INFINITY
        } else {
            // UCB1 formula: Q(s,a) + c * sqrt(ln(N) / n(s,a))
            let total = (self.total_visits.max(1)) as f64;
            let exploration_bonus =
                self.exploration_constant * ((total.ln() / visit_count as f64).sqrt());
            q_value + exploration_bonus
        }
    }

    /// Get current exploration rate (for use with select_action)
    /// This returns the dynamic exploration constant that changes based on agent's learning state
    pub fn get_exploration_rate(&self) -> f64 {
        self.exploration_constant
    }

    /// Update the policy with new experience
    pub fn update_with_experience(
        &mut self,
        flow: &EpistemologicalFlow,
        action: &PolicyAction,
        reward: f64,
        next_flow: &EpistemologicalFlow,
    ) {
        // Reward normalization via EMA (z-score like)
        let delta = reward - self.running_mean;
        self.running_mean += self.ema_alpha * delta;
        self.running_var = (1.0 - self.ema_alpha) * (self.running_var + self.ema_alpha * delta * delta);
        let norm_reward = if self.running_var > 1e-8 {
            (reward - self.running_mean) / self.running_var.sqrt()
        } else { reward };

        self.q_estimator
            .update_q_value(flow, action.as_str(), norm_reward, next_flow);
        self.total_visits += 1;

        // Maintain rolling reward window
        self.recent_rewards.push(reward);
        if self.recent_rewards.len() > self.max_recent {
            let _ = self.recent_rewards.remove(0);
        }
        let avg_recent = if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().copied().sum::<f64>() / (self.recent_rewards.len() as f64)
        };

        // Track neg/pos streaks and last good action
        if reward < 0.0 {
            self.neg_reward_streak = self.neg_reward_streak.saturating_add(1);
        } else {
            self.neg_reward_streak = 0;
        }
        if reward > 0.6 {
            self.last_positive_action = Some(action.as_str().to_string());
        }

        // IMMEDIATE CATASTROPHIC DETECTION: React instantly to negative rewards
        // This is critical for avoiding catastrophic scenarios where an action
        // that was previously good suddenly becomes harmful
        if reward < 0.0 {
            // Apply strong penalty to break the habit immediately
            self.q_estimator.update_q_value(flow, action.as_str(), reward * 5.0, next_flow);
            
            // Mask this harmful action immediately to force exploration
            self.masked_action = Some(action.as_str().to_string());
            
            // Trigger aggressive exploration mode
            self.phase_shift_mode = true;
            self.phase_steps_remaining = self.phase_steps_remaining.max(60);
            self.temp_explore_boost_steps = self.temp_explore_boost_steps.max(30);
            
            // Scale exploration boost based on negative streak severity
            if self.neg_reward_streak >= 3 {
                // Catastrophic: extremely strong exploration
                self.mask_steps_remaining = self.mask_steps_remaining.max(50);
                self.exploration_constant = (self.base_exploration_constant * 8.0).min(15.0);
            } else if self.neg_reward_streak >= 2 {
                // Severe: very strong exploration
                self.mask_steps_remaining = self.mask_steps_remaining.max(40);
                self.exploration_constant = (self.base_exploration_constant * 6.0).min(12.0);
            } else {
                // Single negative: strong exploration
                self.mask_steps_remaining = self.mask_steps_remaining.max(25);
                self.exploration_constant = (self.base_exploration_constant * 4.0).min(10.0);
            }
        }

        // STAGNATION DETECTION: Detect when performance plateaus at suboptimal level
        // This handles gradual environment changes where rewards don't go negative
        // but the agent gets stuck in a local optimum
        // MUST BE CHECKED BEFORE decay logic to prevent immediate decay
        let stagnation_boost_triggered = if self.recent_rewards.len() >= self.max_recent {
            // Check if performance has improved
            if avg_recent > self.best_recent_avg + 0.05 {
                // Improvement detected, reset stagnation counter
                self.best_recent_avg = avg_recent;
                self.stagnation_counter = 0;
                false
            } else if avg_recent < self.best_recent_avg - 0.05 {
                // Performance degraded, boost exploration
                self.stagnation_counter += 1;
                false
            } else {
                // Performance stagnant (not improving)
                self.stagnation_counter += 1;
                false
            }
        } else {
            false
        };

        // If stagnated for too long at suboptimal level, force exploration
        // Threshold 0.75 catches agents stuck at moderate rewards (~0.55-0.60)
        let stagnation_boost_triggered = if self.stagnation_counter >= 10 && avg_recent < 0.75 {
            // Trigger strong exploration boost to escape local optimum
            // Boost 5x for balanced exploration without overshooting
            self.temp_explore_boost_steps = 50;
            self.exploration_constant = (self.base_exploration_constant * 5.0).min(10.0);
            self.stagnation_counter = 0; // Reset after triggering
            true
        } else {
            stagnation_boost_triggered
        };

        // Phase detection & dynamic exploration adjustment (gradual trends)
        if avg_recent < 0.4 {
            // Trigger phase-shift mode for gradual performance degradation
            self.phase_shift_mode = true;
            self.phase_steps_remaining = self.phase_steps_remaining.max(50);
            self.temp_explore_boost_steps = self.temp_explore_boost_steps.max(20);
            // Only boost if not already boosted by catastrophic detection
            if self.exploration_constant < self.base_exploration_constant * 3.0 {
                self.exploration_constant = (self.base_exploration_constant * 3.0).min(10.0);
            }
        } else if avg_recent > 0.70 {
            // Favor exploitation when performance is good (0.70+ threshold)
            // Lowered from 0.85 to encourage exploitation sooner after finding good action
            // Reduced penalty from 0.75x to 0.85x to maintain moderate exploration
            self.exploration_constant = (self.base_exploration_constant * 0.85).max(0.01);
            if self.temp_explore_boost_steps > 0 {
                self.temp_explore_boost_steps -= 1;
            }
        } else if !stagnation_boost_triggered {
            // Gradually decay boost ONLY if stagnation didn't just trigger
            if self.temp_explore_boost_steps > 0 {
                self.temp_explore_boost_steps -= 1;
                if self.temp_explore_boost_steps == 0 {
                    self.exploration_constant = self.base_exploration_constant;
                }
            } else {
                self.exploration_constant = self.base_exploration_constant;
            }
        }

        if self.phase_shift_mode {
            if self.phase_steps_remaining > 0 {
                self.phase_steps_remaining -= 1;
            } else {
                self.phase_shift_mode = false;
                self.exploration_constant = self.base_exploration_constant;
            }
        }

        // Decay mask
        if let Some(_) = self.masked_action {
            if self.mask_steps_remaining > 0 {
                self.mask_steps_remaining -= 1;
            } else {
                self.masked_action = None;
            }
        }
    }
}

impl Policy for ValueDrivenPolicy {
    fn select_action(&self, flow: &EpistemologicalFlow, explore_rate: f64) -> PolicyAction {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Epsilon-greedy with UCB1 exploration
        if rng.gen::<f64>() < explore_rate {
            // Exploration: use UCB1 to select action
            PolicyAction::all_actions()
                .into_iter()
                .max_by(|a, b| {
                    let score_a = self.ucb1_score(flow, a);
                    let score_b = self.ucb1_score(flow, b);
                    score_a
                        .partial_cmp(&score_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(PolicyAction::Explore)
        } else {
            // Exploitation with Thompson-like sampling: add small noise inversely to visits
            let qmap = self.q_estimator.get_q_values(flow).unwrap_or_default();
            let mut best: Option<(String, f64)> = None;
            for (a, q) in qmap.into_iter() {
                if let (Some(mask), true) = (&self.masked_action, self.mask_steps_remaining > 0) {
                    if &a == mask { continue; }
                }
                let n = (self.q_estimator.get_visit_count(flow, &a) as f64).max(1.0);
                // Simple Gaussian-like noise using Box-Muller without external crates
                let u1: f64 = (rng.gen::<f64>()).clamp(1e-9, 1.0);
                let u2: f64 = rng.gen::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let sigma = (1.0 / n).sqrt();
                let noise = z * sigma;
                // Bias towards recently good action during phase shift
                let mut score = q + noise;
                if self.phase_shift_mode {
                    if let Some(ref good) = self.last_positive_action {
                        if &a == good {
                            score += 0.2;
                        }
                    }
                }
                if best.as_ref().map(|(_, bq)| score > *bq).unwrap_or(true) {
                    best = Some((a.to_string(), score));
                }
            }
            if let Some((action_str, _)) = best {
                match action_str.as_str() {
                    "noop" => PolicyAction::Noop,
                    "explore" => PolicyAction::Explore,
                    "exploit" => PolicyAction::Exploit,
                    "unlock_door" => PolicyAction::UnlockDoor,
                    "pick_up_key" => PolicyAction::PickUpKey,
                    "move_forward" => PolicyAction::MoveForward,
                    _ => PolicyAction::Noop,
                }
            } else {
                PolicyAction::Noop
            }
        }
    }

    fn update(&mut self, _flow: &EpistemologicalFlow, _advantage: f64) {
        // This is called by the learning system
        // We can use the advantage to update our Q-values
        // For now, we'll just increment the total visits
        self.total_visits += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn epsilon_greedy_switches_mode() {
        let pol = EpsilonGreedyPolicy::default();
        let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"state"));
        assert_eq!(pol.select_action(&flow, 0.1), PolicyAction::Explore);
        assert_eq!(pol.select_action(&flow, 0.0), PolicyAction::Exploit);
    }

    #[test]
    fn action_string_conversion() {
        assert_eq!(PolicyAction::Noop.as_str(), "noop");
        assert_eq!(PolicyAction::UnlockDoor.as_str(), "unlock_door");
        assert_eq!(PolicyAction::PickUpKey.as_str(), "pick_up_key");
        assert_eq!(PolicyAction::MoveForward.as_str(), "move_forward");
    }

    #[test]
    fn value_driven_policy_creation() {
        let policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);
        assert_eq!(policy.total_visits, 0);
    }

    #[test]
    fn value_driven_policy_action_selection() {
        let policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);
        let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));

        // Test exploitation (explore_rate = 0.0)
        let action = policy.select_action(&flow, 0.0);
        assert!(matches!(
            action,
            PolicyAction::Noop
                | PolicyAction::Explore
                | PolicyAction::Exploit
                | PolicyAction::UnlockDoor
                | PolicyAction::PickUpKey
                | PolicyAction::MoveForward
        ));
    }
}
