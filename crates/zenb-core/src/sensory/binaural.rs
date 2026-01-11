//! Binaural beat frequency calculator
//!
//! Maps breath phases to brain wave frequencies

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainWaveState {
    Delta,  // 1-4 Hz: Deep sleep
    Theta,  // 4-8 Hz: Meditation
    Alpha,  // 8-13 Hz: Relaxed
    Beta,   // 13-30 Hz: Active
}

impl BrainWaveState {
    /// Get (carrier_hz, beat_hz) for this state
    pub fn config(&self) -> (f32, f32) {
        match self {
            Self::Delta => (200.0, 2.5),
            Self::Theta => (200.0, 6.0),
            Self::Alpha => (200.0, 10.0),
            Self::Beta => (220.0, 18.0),
        }
    }
}

pub struct BinauralEngine {
    current_state: BrainWaveState,
}

impl BinauralEngine {
    pub fn new() -> Self {
        Self {
            current_state: BrainWaveState::Alpha,
        }
    }
    
    /// Update based on breath phase and arousal
    pub fn update(&mut self, phase: &str, arousal: f32) -> Option<(f32, f32)> {
        let next = match phase {
            "inhale" => {
                if arousal > 0.5 {
                    BrainWaveState::Alpha
                } else {
                    BrainWaveState::Theta
                }
            }
            "exhale" => BrainWaveState::Theta,
            "hold" => {
                if arousal < 0.3 {
                    BrainWaveState::Delta
                } else {
                    BrainWaveState::Theta
                }
            }
            _ => self.current_state,
        };
        
        if next != self.current_state {
            self.current_state = next;
            Some(next.config())
        } else {
            None
        }
    }
    
    pub fn current_state(&self) -> BrainWaveState {
        self.current_state
    }
}

impl Default for BinauralEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_brain_wave_configs() {
        assert_eq!(BrainWaveState::Delta.config(), (200.0, 2.5));
        assert_eq!(BrainWaveState::Theta.config(), (200.0, 6.0));
        assert_eq!(BrainWaveState::Alpha.config(), (200.0, 10.0));
        assert_eq!(BrainWaveState::Beta.config(), (220.0, 18.0));
    }
    
    #[test]
    fn test_binaural_update() {
        let mut engine = BinauralEngine::new();
        
        // Initial state is Alpha, so high arousal inhale returns None (no change)
        let result = engine.update("inhale", 0.7);
        assert!(result.is_none());  // Already in Alpha state
        assert_eq!(engine.current_state(), BrainWaveState::Alpha);
        
        // Exhale -> Theta (state change)
        let result = engine.update("exhale", 0.7);
        assert!(result.is_some());
        assert_eq!(engine.current_state(), BrainWaveState::Theta);
        
        // Same state -> None
        let result = engine.update("exhale", 0.5);
        assert!(result.is_none());
    }
}
