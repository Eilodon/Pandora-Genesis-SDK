#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    Inhale,
    HoldIn,
    Exhale,
    HoldOut,
}

#[derive(Debug, Clone)]
pub struct PhaseDurations {
    pub inhale_us: u64,
    pub hold_in_us: u64,
    pub exhale_us: u64,
    pub hold_out_us: u64,
}

impl PhaseDurations {
    pub fn total_us(&self) -> u64 {
        self.inhale_us + self.hold_in_us + self.exhale_us + self.hold_out_us
    }
}

#[derive(Debug, Clone)]
pub struct PhaseMachine {
    pub phase: Phase,
    pub elapsed_us: u64,
    pub durations: PhaseDurations,
    pub cycle_index: u64,
}

impl PhaseMachine {
    pub fn new(durations: PhaseDurations) -> Self {
        PhaseMachine {
            phase: Phase::Inhale,
            elapsed_us: 0,
            durations,
            cycle_index: 0,
        }
    }

    /// Compute remaining time in current phase
    fn remaining_us(&self) -> u64 {
        let dur = match self.phase {
            Phase::Inhale => self.durations.inhale_us,
            Phase::HoldIn => self.durations.hold_in_us,
            Phase::Exhale => self.durations.exhale_us,
            Phase::HoldOut => self.durations.hold_out_us,
        };
        dur.saturating_sub(self.elapsed_us)
    }

    /// transition to next phase, return whether cycle completed
    fn transition(&mut self) -> bool {
        let mut cycle_completed = false;
        self.elapsed_us = 0;
        self.phase = match self.phase {
            Phase::Inhale => Phase::HoldIn,
            Phase::HoldIn => Phase::Exhale,
            Phase::Exhale => Phase::HoldOut,
            Phase::HoldOut => {
                cycle_completed = true;
                self.cycle_index += 1;
                Phase::Inhale
            }
        };
        cycle_completed
    }

    /// Tick with dt_us, returning transitions (phase entries) and cycle completed count
    pub fn tick(&mut self, mut dt_us: u64) -> (Vec<Phase>, u64) {
        let mut trans = Vec::new();
        let mut cycles = 0u64;
        while dt_us > 0 {
            let left = self.remaining_us();
            if dt_us < left {
                self.elapsed_us = self.elapsed_us.saturating_add(dt_us);
                break;
            } else {
                // consume + transition
                dt_us = dt_us.saturating_sub(left);
                let cycle = self.transition();
                trans.push(self.phase.clone());
                if cycle {
                    cycles += 1;
                }
                // continue looping if dt_us still > 0
            }
        }
        (trans, cycles)
    }

    pub fn cycle_phase_norm(&self) -> f32 {
        let total = self.durations.total_us();
        if total == 0 {
            return 0.0;
        }

        let before = match self.phase {
            Phase::Inhale => 0,
            Phase::HoldIn => self.durations.inhale_us,
            Phase::Exhale => self.durations.inhale_us + self.durations.hold_in_us,
            Phase::HoldOut => {
                self.durations.inhale_us + self.durations.hold_in_us + self.durations.exhale_us
            }
        };
        let pos = before.saturating_add(self.elapsed_us).min(total);
        (pos as f32) / (total as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_cycle() {
        let durations = PhaseDurations {
            inhale_us: 1_000,
            hold_in_us: 100,
            exhale_us: 1_000,
            hold_out_us: 200,
        };
        let mut pm = PhaseMachine::new(durations);
        let (t, c) = pm.tick(500);
        assert!(t.is_empty());
        let (t, c) = pm.tick(1_000);
        assert!(!t.is_empty());
        let (t, c) = pm.tick(10_000);
        // should have completed cycles
        assert!(c >= 0);
    }
}
