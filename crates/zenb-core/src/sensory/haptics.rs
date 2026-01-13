//! Haptic pattern library

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HapticPattern {
    HeartbeatCalm,
    HeartbeatActive,
    BreathInWave,
    BreathOutWave,
    UiSuccess,
    UiWarn,
    UiError,
}

impl HapticPattern {
    /// Get timing pattern in milliseconds
    pub fn timings(&self) -> &'static [u32] {
        match self {
            Self::HeartbeatCalm => &[30, 200, 15],
            Self::HeartbeatActive => &[45, 150, 25],
            Self::BreathInWave => &[10, 50, 15, 45, 20, 40, 30],
            Self::BreathOutWave => &[50, 20, 40, 30, 30, 40, 20],
            Self::UiSuccess => &[15],
            Self::UiWarn => &[15, 50, 15],
            Self::UiError => &[15, 30, 15, 30, 50],
        }
    }

    /// Get total duration in milliseconds
    pub fn duration(&self) -> u32 {
        self.timings().iter().sum()
    }
}

pub struct HapticEngine {
    current_pattern: Option<HapticPattern>,
}

impl HapticEngine {
    pub fn new() -> Self {
        Self {
            current_pattern: None,
        }
    }

    pub fn trigger(&mut self, pattern: HapticPattern) -> &'static [u32] {
        self.current_pattern = Some(pattern);
        pattern.timings()
    }

    pub fn current_pattern(&self) -> Option<HapticPattern> {
        self.current_pattern
    }
}

impl Default for HapticEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_timings() {
        assert_eq!(HapticPattern::HeartbeatCalm.timings(), &[30, 200, 15]);
        assert_eq!(HapticPattern::UiSuccess.timings(), &[15]);
    }

    #[test]
    fn test_pattern_duration() {
        assert_eq!(HapticPattern::HeartbeatCalm.duration(), 245);
        assert_eq!(HapticPattern::UiSuccess.duration(), 15);
    }

    #[test]
    fn test_haptic_engine() {
        let mut engine = HapticEngine::new();

        let timings = engine.trigger(HapticPattern::UiSuccess);
        assert_eq!(timings, &[15]);
        assert_eq!(engine.current_pattern(), Some(HapticPattern::UiSuccess));
    }
}
