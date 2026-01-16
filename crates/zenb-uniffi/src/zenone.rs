//! ZenOne Runtime - Unified API for ZenOne Native Apps
//!
//! This module provides a high-level wrapper that combines:
//! - `Engine` (Active Inference control)
//! - `EnsembleProcessor` (rPPG signal processing)
//! - `PhaseMachine` (breath timing)
//!
//! # Platform Integration
//!
//! ```text
//! Platform (iOS/Android/Web)
//!     │
//!     ▼ Camera Frame RGB
//! ┌─────────────────┐
//! │  ZenOneRuntime  │
//! │  .process_frame()
//! └───────┬─────────┘
//!         │
//!         ▼
//!  ZenOneFrame {
//!    belief, phase, vitals
//!  }
//! ```

use std::time::Instant;

// Re-export from zenb-core
use zenb_core::{
    BeliefState, Engine, PhaseDurations, PhaseMachine,
    breath_patterns::{BreathPattern, builtin_patterns},
    phase_machine::Phase,
};

// Re-export from zenb-signals
use zenb_signals::rppg::EnsembleProcessor;

/// Configuration for ZenOneRuntime
#[derive(Debug, Clone)]
pub struct ZenOneConfig {
    /// Sample rate for rPPG processing (typically 30 FPS)
    pub sample_rate: f32,
    /// Default breathing pattern ID
    pub default_pattern: String,
    /// Enable safety monitors
    pub enable_safety: bool,
}

impl Default for ZenOneConfig {
    fn default() -> Self {
        Self {
            sample_rate: 30.0,
            default_pattern: "4-7-8".to_string(),
            enable_safety: true,
        }
    }
}

/// Vital signs output
#[derive(Debug, Clone, Default)]
pub struct Vitals {
    /// Heart rate in BPM (if available)
    pub heart_rate: Option<f32>,
    /// Signal quality (0.0 - 1.0)
    pub signal_quality: f32,
    /// SNR of the rPPG signal
    pub snr: f32,
}

/// Output from a single frame processing
#[derive(Debug, Clone)]
pub struct ZenOneFrame {
    /// Current breathing phase
    pub phase: String,
    /// Progress through current phase (0.0 - 1.0)
    pub phase_progress: f32,
    /// Progress through entire cycle (0.0 - 1.0)
    pub cycle_progress: f32,
    /// Vital signs
    pub vitals: Vitals,
    /// Completed cycles count
    pub cycles_completed: u64,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total session duration in seconds
    pub duration_sec: f32,
    /// Completed breathing cycles
    pub cycles_completed: u64,
    /// Pattern used
    pub pattern_id: String,
    /// Average heart rate during session
    pub avg_heart_rate: Option<f32>,
}

/// Internal session state
struct SessionState {
    start_time: Instant,
    pattern_id: String,
    hr_samples: Vec<f32>,
}

/// ZenOne Runtime - Main API for native apps
pub struct ZenOneRuntime {
    /// Core engine (Active Inference)
    engine: Engine,
    /// rPPG signal processor
    processor: EnsembleProcessor,
    /// Phase machine for breath timing
    phase_machine: PhaseMachine,
    /// Current breathing pattern
    current_pattern: BreathPattern,
    /// Active session (if any)
    session: Option<SessionState>,
    /// Last frame timestamp
    last_timestamp_us: i64,
    /// Configuration
    #[allow(dead_code)]
    config: ZenOneConfig,
}

impl ZenOneRuntime {
    /// Create a new ZenOneRuntime with configuration
    pub fn new(config: ZenOneConfig) -> Self {
        // Get default pattern
        let patterns = builtin_patterns();
        let current_pattern = patterns
            .get(&config.default_pattern)
            .cloned()
            .unwrap_or_else(|| patterns.get("4-7-8").unwrap().clone());
        
        // Initialize phase machine with pattern timings
        let durations = current_pattern.to_phase_durations();
        let phase_machine = PhaseMachine::new(durations);
        
        // Initialize engine with default breath rate
        let engine = Engine::new(6.0);
        
        // Initialize signal processor
        let processor = EnsembleProcessor::new();
        
        Self {
            engine,
            processor,
            phase_machine,
            current_pattern,
            session: None,
            last_timestamp_us: 0,
            config,
        }
    }
    
    /// Process a single camera frame (RGB mean only)
    ///
    /// # Arguments
    /// * `rgb_mean` - Mean RGB values from face ROI [R, G, B]
    /// * `timestamp_us` - Frame timestamp in microseconds
    ///
    /// # Returns
    /// `ZenOneFrame` with current state and vitals
    pub fn process_frame(
        &mut self,
        rgb_mean: [f32; 3],
        timestamp_us: i64,
    ) -> ZenOneFrame {
        // Calculate delta time
        let dt_us = if self.last_timestamp_us > 0 {
            (timestamp_us - self.last_timestamp_us).max(0) as u64
        } else {
            33_333 // ~30fps default
        };
        self.last_timestamp_us = timestamp_us;
        
        // 1. rPPG processing
        self.processor.add_sample(rgb_mean[0], rgb_mean[1], rgb_mean[2]);
        let ppg_result = self.processor.process();
        
        // 2. Update phase machine
        let (_phase_transitions, _cycles_in_tick) = self.phase_machine.tick(dt_us);
        
        // Update session stats if active
        if let Some(ref mut session) = self.session {
            if let Some(ref result) = ppg_result {
                session.hr_samples.push(result.bpm);
            }
        }
        
        // 3. Build vitals
        let vitals = Vitals {
            heart_rate: ppg_result.as_ref().map(|r| r.bpm),
            signal_quality: ppg_result.as_ref().map(|r| r.confidence).unwrap_or(0.0),
            snr: ppg_result.as_ref().map(|r| r.snr).unwrap_or(0.0),
        };
        
        // 4. Get current phase
        let phase_str = match self.phase_machine.phase {
            Phase::Inhale => "inhale",
            Phase::HoldIn => "holdIn",
            Phase::Exhale => "exhale",
            Phase::HoldOut => "holdOut",
        };
        
        ZenOneFrame {
            phase: phase_str.to_string(),
            phase_progress: self.phase_machine.cycle_phase_norm(),
            cycle_progress: self.phase_machine.cycle_phase_norm(),
            vitals,
            cycles_completed: self.phase_machine.cycle_index,
            timestamp_us,
        }
    }
    
    /// Load a breathing pattern by ID
    pub fn load_pattern(&mut self, pattern_id: &str) -> bool {
        let patterns = builtin_patterns();
        if let Some(pattern) = patterns.get(pattern_id) {
            self.current_pattern = pattern.clone();
            self.phase_machine = PhaseMachine::new(pattern.to_phase_durations());
            log::info!("Loaded pattern: {} ({})", pattern.label, pattern_id);
            true
        } else {
            log::warn!("Pattern not found: {}", pattern_id);
            false
        }
    }
    
    /// Start a new session
    pub fn start_session(&mut self) {
        self.session = Some(SessionState {
            start_time: Instant::now(),
            pattern_id: self.current_pattern.id.clone(),
            hr_samples: Vec::new(),
        });
        log::info!("Session started with pattern: {}", self.current_pattern.id);
    }
    
    /// Stop the current session and get statistics
    pub fn stop_session(&mut self) -> SessionStats {
        if let Some(session) = self.session.take() {
            let duration = session.start_time.elapsed();
            let avg_hr = if !session.hr_samples.is_empty() {
                Some(session.hr_samples.iter().sum::<f32>() / session.hr_samples.len() as f32)
            } else {
                None
            };
            
            log::info!("Session stopped. Duration: {:.1}s", duration.as_secs_f32());
            
            SessionStats {
                duration_sec: duration.as_secs_f32(),
                cycles_completed: self.phase_machine.cycle_index,
                pattern_id: session.pattern_id,
                avg_heart_rate: avg_hr,
            }
        } else {
            SessionStats::default()
        }
    }
    
    /// Check if a session is active
    pub fn is_session_active(&self) -> bool {
        self.session.is_some()
    }
    
    /// Get current pattern info
    pub fn current_pattern(&self) -> &BreathPattern {
        &self.current_pattern
    }
    
    /// Get list of available pattern IDs
    pub fn available_patterns() -> Vec<String> {
        builtin_patterns().keys().cloned().collect()
    }
}

impl Default for ZenOneRuntime {
    fn default() -> Self {
        Self::new(ZenOneConfig::default())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_runtime_creation() {
        let runtime = ZenOneRuntime::default();
        assert_eq!(runtime.current_pattern.id, "4-7-8");
        assert!(!runtime.is_session_active());
    }
    
    #[test]
    fn test_load_pattern() {
        let mut runtime = ZenOneRuntime::default();
        assert!(runtime.load_pattern("box"));
        assert_eq!(runtime.current_pattern.id, "box");
        
        assert!(!runtime.load_pattern("nonexistent"));
    }
    
    #[test]
    fn test_session_lifecycle() {
        let mut runtime = ZenOneRuntime::default();
        
        assert!(!runtime.is_session_active());
        runtime.start_session();
        assert!(runtime.is_session_active());
        
        let stats = runtime.stop_session();
        assert!(!runtime.is_session_active());
        assert_eq!(stats.pattern_id, "4-7-8");
    }
    
    #[test]
    fn test_available_patterns() {
        let patterns = ZenOneRuntime::available_patterns();
        assert_eq!(patterns.len(), 11);
        assert!(patterns.contains(&"4-7-8".to_string()));
        assert!(patterns.contains(&"box".to_string()));
        assert!(patterns.contains(&"wim-hof".to_string()));
    }
}
