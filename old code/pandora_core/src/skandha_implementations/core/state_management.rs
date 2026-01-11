//! State management primitives for stateful skandhas.
//!
//! Provides common state types and decay implementations.

use super::traits::DecayableState;
use std::time::{Duration, Instant};

// ============================================================================
// MOOD STATE (for VedanaSkandha)
// ============================================================================

/// Mood state for emotional valence tracking.
///
/// Based on dimensional model of affect (Russell, 1980):
/// - **Valence**: Pleasant (+1.0) ↔ Unpleasant (-1.0)
/// - **Arousal**: Activated (+1.0) ↔ Deactivated (-1.0)
///
/// # Invariants
///
/// - Both dimensions bounded in [-1.0, 1.0]
/// - Neutral state is (0.0, 0.0)
/// - State decays toward neutral over time
#[derive(Debug, Clone, PartialEq)]
pub struct MoodState {
    /// Emotional valence: -1.0 (unpleasant) to +1.0 (pleasant)
    pub valence: f32,

    /// Emotional arousal: -1.0 (calm) to +1.0 (excited)
    pub arousal: f32,

    /// Last update timestamp for decay calculation
    pub last_update: Instant,

    /// Decay rate per second (0.0 = instant, 1.0 = no decay)
    pub decay_rate: f32,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            last_update: Instant::now(),
            decay_rate: 0.95, // 5% decay per second
        }
    }
}

impl MoodState {
    /// Create new mood state with custom decay rate.
    pub fn with_decay_rate(decay_rate: f32) -> Self {
        Self {
            decay_rate: decay_rate.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Update mood based on new feeling.
    ///
    /// # Algorithm
    ///
    /// 1. Decay current mood based on elapsed time
    /// 2. Blend new feeling with decayed mood (weighted average)
    /// 3. Clamp result to valid range
    ///
    /// # Arguments
    ///
    /// * `feeling_valence` - Valence of new feeling [-1.0, 1.0]
    /// * `feeling_intensity` - Weight of new feeling [0.0, 1.0]
    ///   - 0.0 = ignore new feeling (use current mood)
    ///   - 1.0 = replace mood completely with new feeling
    ///   - 0.3 = typical value (30% new, 70% existing mood)
    pub fn update(&mut self, feeling_valence: f32, feeling_intensity: f32) {
        // Apply decay first
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update);
        self.decay(elapsed);

        // Blend new feeling with existing mood
        let intensity = feeling_intensity.clamp(0.0, 1.0);
        self.valence = (self.valence * (1.0 - intensity) + feeling_valence * intensity)
            .clamp(-1.0, 1.0);

        // Arousal increases with feeling intensity
        let arousal_delta = feeling_intensity * 0.5; // Cap arousal increase
        self.arousal = (self.arousal + arousal_delta).clamp(-1.0, 1.0);

        self.last_update = now;
    }

    /// Get current mood as quadrant label.
    ///
    /// Returns:
    /// - "Pleasant-Activated" (happy, excited)
    /// - "Pleasant-Deactivated" (calm, content)
    /// - "Unpleasant-Activated" (angry, anxious)
    /// - "Unpleasant-Deactivated" (sad, depressed)
    /// - "Neutral" (arousal and valence near zero)
    pub fn quadrant(&self) -> &'static str {
        const THRESHOLD: f32 = 0.2;

        if self.valence.abs() < THRESHOLD && self.arousal.abs() < THRESHOLD {
            return "Neutral";
        }

        match (self.valence > 0.0, self.arousal > 0.0) {
            (true, true) => "Pleasant-Activated",
            (true, false) => "Pleasant-Deactivated",
            (false, true) => "Unpleasant-Activated",
            (false, false) => "Unpleasant-Deactivated",
        }
    }
}

impl DecayableState for MoodState {
    fn decay(&mut self, delta_time: Duration) {
        let factor = self.decay_rate.powf(delta_time.as_secs_f32());
        self.valence *= factor;
        self.arousal *= factor;
        self.last_update = Instant::now();
    }

    fn reset(&mut self) {
        self.valence = 0.0;
        self.arousal = 0.0;
        self.last_update = Instant::now();
    }

    fn decay_factor(&self) -> f32 {
        self.decay_rate
    }
}

// ============================================================================
// PATTERN MEMORY STATE (for SannaSkandha)
// ============================================================================

/// Pattern memory for frequency-based reinforcement.
///
/// Tracks how often patterns are perceived to:
/// - Strengthen frequently-seen patterns
/// - Weaken rarely-seen patterns via decay
/// - Enable pattern priming (faster recognition)
#[derive(Debug, Clone)]
pub struct PatternMemoryState {
    /// Pattern frequency map: pattern_hash → (count, last_seen, strength)
    ///
    /// Using fnv::FnvHashMap for speed (not crypto-secure, but fast).
    /// Limited to MAX_PATTERNS to prevent unbounded growth.
    pub patterns: fnv::FnvHashMap<u64, PatternEntry>,

    /// Maximum number of patterns to track
    pub max_patterns: usize,

    /// Decay rate for pattern strength
    pub decay_rate: f32,
}

/// Entry in pattern memory.
#[derive(Debug, Clone)]
pub struct PatternEntry {
    /// Number of times pattern was perceived
    pub count: u32,

    /// Last time pattern was perceived
    pub last_seen: Instant,

    /// Strength score (0.0 to 1.0), derived from recency + frequency
    pub strength: f32,
}

impl Default for PatternMemoryState {
    fn default() -> Self {
        Self {
            patterns: fnv::FnvHashMap::default(),
            max_patterns: 1000,
            decay_rate: 0.98, // 2% decay per second (slower than mood)
        }
    }
}

impl PatternMemoryState {
    /// Record perception of a pattern.
    ///
    /// # Algorithm
    ///
    /// 1. If pattern exists: increment count, update timestamp, boost strength
    /// 2. If new pattern and space available: insert with initial strength
    /// 3. If new pattern and no space: evict weakest pattern
    pub fn record_pattern(&mut self, pattern_hash: u64) {
        let now = Instant::now();

        if let Some(entry) = self.patterns.get_mut(&pattern_hash) {
            // Existing pattern: reinforce
            entry.count += 1;
            entry.last_seen = now;
            entry.strength = (entry.strength + 0.1).min(1.0); // Boost by 10%
        } else {
            // New pattern
            if self.patterns.len() >= self.max_patterns {
                // Evict weakest pattern
                if let Some((&weakest_hash, _)) = self.patterns
                    .iter()
                    .min_by(|a, b| a.1.strength.partial_cmp(&b.1.strength).unwrap())
                {
                    self.patterns.remove(&weakest_hash);
                }
            }

            self.patterns.insert(pattern_hash, PatternEntry {
                count: 1,
                last_seen: now,
                strength: 0.5, // Initial strength
            });
        }
    }

    /// Get strength of a pattern (0.0 if unknown).
    pub fn pattern_strength(&self, pattern_hash: u64) -> f32 {
        self.patterns
            .get(&pattern_hash)
            .map(|entry| entry.strength)
            .unwrap_or(0.0)
    }

    /// Get all strong patterns (strength > threshold).
    pub fn strong_patterns(&self, threshold: f32) -> Vec<u64> {
        self.patterns
            .iter()
            .filter(|(_, entry)| entry.strength > threshold)
            .map(|(&hash, _)| hash)
            .collect()
    }
}

impl DecayableState for PatternMemoryState {
    fn decay(&mut self, delta_time: Duration) {
        let factor = self.decay_rate.powf(delta_time.as_secs_f32());

        // Decay all pattern strengths
        self.patterns.values_mut().for_each(|entry| {
            entry.strength *= factor;
        });

        // Remove very weak patterns (garbage collection)
        self.patterns.retain(|_, entry| entry.strength > 0.01);
    }

    fn reset(&mut self) {
        self.patterns.clear();
    }

    fn decay_factor(&self) -> f32 {
        self.decay_rate
    }
}


// ============================================================================
// GENERIC STATE WRAPPER
// ============================================================================

/// Generic wrapper for any decayable state with automatic decay.
///
/// Useful for "set and forget" state management - decay happens
/// automatically on each access.
#[derive(Debug, Clone)]
pub struct AutoDecayState<T: DecayableState> {
    inner: T,
    last_decay: Instant,
}

impl<T: DecayableState + Default> Default for AutoDecayState<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
            last_decay: Instant::now(),
        }
    }
}

impl<T: DecayableState> AutoDecayState<T> {
    pub fn new(state: T) -> Self {
        Self {
            inner: state,
            last_decay: Instant::now(),
        }
    }

    /// Get reference to inner state (applies decay first).
    pub fn get(&mut self) -> &T {
        self.apply_decay();
        &self.inner
    }

    /// Get mutable reference to inner state (applies decay first).
    pub fn get_mut(&mut self) -> &mut T {
        self.apply_decay();
        &mut self.inner
    }

    fn apply_decay(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_decay);
        self.inner.decay(elapsed);
        self.last_decay = now;
    }
}

impl<T: DecayableState> DecayableState for AutoDecayState<T> {
    fn decay(&mut self, delta_time: Duration) {
        self.inner.decay(delta_time);
        self.last_decay = Instant::now();
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.last_decay = Instant::now();
    }

    fn decay_factor(&self) -> f32 {
        self.inner.decay_factor()
    }
}
