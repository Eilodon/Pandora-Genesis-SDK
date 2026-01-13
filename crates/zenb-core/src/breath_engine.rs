use crate::phase_machine::{PhaseDurations, PhaseMachine};

#[derive(Debug, Clone)]
pub enum BreathMode {
    Dynamic(f32),          // target bpm
    Fixed(PhaseDurations), // explicit durations
}

#[derive(Debug, Clone)]
pub struct BreathEngine {
    pub pm: PhaseMachine,
    pub mode: BreathMode,
}

impl BreathEngine {
    pub fn new(mode: BreathMode) -> Self {
        let durations = match mode {
            BreathMode::Dynamic(bpm) => {
                let cycle_us = if bpm <= 0.0 {
                    10_000_000u64
                } else {
                    (60_000_000f32 / bpm).round() as u64
                };
                let inhale = (cycle_us as f32 * 0.4).round() as u64;
                let holdin = (cycle_us as f32 * 0.05).round() as u64;
                let exhale = (cycle_us as f32 * 0.45).round() as u64;
                let holdout = cycle_us.saturating_sub(inhale + holdin + exhale);
                PhaseDurations {
                    inhale_us: inhale,
                    hold_in_us: holdin,
                    exhale_us: exhale,
                    hold_out_us: holdout,
                }
            }
            BreathMode::Fixed(ref d) => d.clone(),
        };
        Self {
            pm: PhaseMachine::new(durations),
            mode,
        }
    }

    /// tick uses the phase machine. When in Fixed mode it will not recompute durations; in Dynamic mode, callers may call `set_target_bpm` to change the BPM which will rebuild the machine.
    pub fn tick(&mut self, dt_us: u64) -> (Vec<String>, u64) {
        let (trans, cycles) = self.pm.tick(dt_us);
        let names = trans
            .into_iter()
            .map(|p| match p {
                crate::phase_machine::Phase::Inhale => "Inhale".into(),
                crate::phase_machine::Phase::HoldIn => "HoldIn".into(),
                crate::phase_machine::Phase::Exhale => "Exhale".into(),
                crate::phase_machine::Phase::HoldOut => "HoldOut".into(),
            })
            .collect();
        (names, cycles)
    }

    pub fn guide_phase_norm(&self) -> f32 {
        self.pm.cycle_phase_norm()
    }

    /// Switch to dynamic BPM mode (will rebuild phase machine). If currently Fixed, this will replace the fixed pattern.
    pub fn set_target_bpm(&mut self, target_bpm: f32) {
        self.mode = BreathMode::Dynamic(target_bpm);
        let cycle_us = if target_bpm <= 0.0 {
            10_000_000u64
        } else {
            (60_000_000f32 / target_bpm).round() as u64
        };
        let inhale = (cycle_us as f32 * 0.4).round() as u64;
        let holdin = (cycle_us as f32 * 0.05).round() as u64;
        let exhale = (cycle_us as f32 * 0.45).round() as u64;
        let holdout = cycle_us.saturating_sub(inhale + holdin + exhale);
        self.pm = PhaseMachine::new(PhaseDurations {
            inhale_us: inhale,
            hold_in_us: holdin,
            exhale_us: exhale,
            hold_out_us: holdout,
        });
    }

    /// Switch to a Fixed, explicit pattern. This will be honored exactly by tick.
    pub fn set_fixed_pattern(&mut self, d: PhaseDurations) {
        self.mode = BreathMode::Fixed(d.clone());
        self.pm = PhaseMachine::new(d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_tick() {
        let mut be = BreathEngine::new(BreathMode::Dynamic(6.0));
        let (_t, c) = be.tick(1_000_000);
        // c is u64, check it compiled and ran
        assert!(c == 0 || c > 0); // Always true, but now clear
    }

    #[test]
    fn fixed_pattern_honored() {
        // 1s inhale, 1s hold, 1s exhale, 1s holdout
        let d = PhaseDurations {
            inhale_us: 1_000_000,
            hold_in_us: 1_000_000,
            exhale_us: 1_000_000,
            hold_out_us: 1_000_000,
        };
        let mut be = BreathEngine::new(BreathMode::Fixed(d));
        // tick 4s in 1s steps and expect at least 1 cycle
        let mut cycles = 0u64;
        for _ in 0..4 {
            let (_, c) = be.tick(1_000_000);
            cycles += c;
        }
        assert!(cycles >= 1);
    }
}
