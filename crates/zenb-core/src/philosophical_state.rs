//! Philosophical State Module - B.ONE V3 Integration
//!
//! Implements the meta-level consciousness state that modulates
//! how the Ngũ Uẩn (Five Skandhas) pipeline processes information.
//!
//! # B.ONE V3 Concept
//! > "Trạng thái của hệ thống (YÊN, ĐỘNG, HỖN LOẠN...) ảnh hưởng trực tiếp
//! > đến cách pipeline Ngũ Uẩn xử lý thông tin."
//!
//! # Three Philosophical States (Tam Trạng Thái)
//! - **YÊN (Tranquil)**: Homeostatic equilibrium, standard processing
//! - **ĐỘNG (Active)**: Engaged processing, enhanced attention
//! - **HỖN LOẠN (Chaotic)**: High entropy, protective measures activated
//!
//! # Usage
//! ```rust
//! use zenb_core::philosophical_state::{PhilosophicalState, PhilosophicalStateMonitor};
//!
//! let mut monitor = PhilosophicalStateMonitor::default();
//! let new_state = monitor.update(0.2, 0.9); // low free energy, high coherence
//! assert_eq!(new_state, PhilosophicalState::Yen);
//! ```

use serde::{Deserialize, Serialize};

/// The three philosophical states of consciousness.
///
/// These meta-level states affect how the entire cognitive pipeline operates,
/// modulating attention, processing depth, and safety measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum PhilosophicalState {
    /// YÊN (Tranquil/Stillness)
    ///
    /// System is in homeostatic equilibrium. Characterized by:
    /// - Low free energy (good predictions)
    /// - High coherence across consciousnesses
    /// - Standard processing mode
    /// - Regular sampling intervals
    #[default]
    Yen,

    /// ĐỘNG (Active/Movement)
    ///
    /// System is engaged in significant processing. Characterized by:
    /// - Moderate free energy
    /// - Moderate coherence
    /// - Enhanced attention mode
    /// - Higher sampling frequency
    /// - More detailed logging
    Dong,

    /// HỖN LOẠN (Chaotic/Turbulence)
    ///
    /// System experiencing high entropy. Characterized by:
    /// - High free energy (poor predictions)
    /// - Low coherence
    /// - Protective measures activated
    /// - Safe fallback behaviors
    /// - Trauma guard active
    HonLoan,
}

impl PhilosophicalState {
    /// Get Vietnamese name of the state.
    pub fn vietnamese_name(&self) -> &'static str {
        match self {
            Self::Yen => "YÊN",
            Self::Dong => "ĐỘNG",
            Self::HonLoan => "HỖN LOẠN",
        }
    }

    /// Get English description of the state.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Yen => "Tranquil - Homeostatic equilibrium",
            Self::Dong => "Active - Engaged processing",
            Self::HonLoan => "Chaotic - Protective measures active",
        }
    }

    /// Get processing multiplier for sampling frequency.
    /// Higher values mean more frequent sampling.
    pub fn sampling_multiplier(&self) -> f32 {
        match self {
            Self::Yen => 1.0,
            Self::Dong => 2.0,
            Self::HonLoan => 0.5, // Slow down in chaos
        }
    }

    /// Check if safe fallback should be activated.
    pub fn requires_safe_fallback(&self) -> bool {
        matches!(self, Self::HonLoan)
    }

    /// Get attention level (0.0 = minimal, 1.0 = maximum).
    pub fn attention_level(&self) -> f32 {
        match self {
            Self::Yen => 0.5,
            Self::Dong => 0.9,
            Self::HonLoan => 0.3, // Conserve resources in chaos
        }
    }
}

impl std::fmt::Display for PhilosophicalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.vietnamese_name(), self.description())
    }
}

/// Thresholds for state transitions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhilosophicalStateThresholds {
    /// Free energy threshold to enter ĐỘNG from YÊN
    pub fe_yen_to_dong: f32,
    /// Free energy threshold to enter HỖN LOẠN from ĐỘNG
    pub fe_dong_to_honloan: f32,
    /// Coherence threshold to stay in YÊN
    pub coherence_yen: f32,
    /// Coherence threshold to stay in ĐỘNG
    pub coherence_dong: f32,
    /// Hysteresis buffer to prevent rapid oscillation
    pub hysteresis: f32,
}

impl Default for PhilosophicalStateThresholds {
    fn default() -> Self {
        Self {
            fe_yen_to_dong: 0.3,
            fe_dong_to_honloan: 0.7,
            coherence_yen: 0.8,
            coherence_dong: 0.5,
            hysteresis: 0.05,
        }
    }
}

/// Monitor that tracks and transitions philosophical states.
///
/// # State Machine
/// ```text
///                    ┌──────────────────────────────┐
///                    │                              │
///        ┌───────────▼───────────┐                  │
///        │         YÊN          │                  │
///        │   (FE < 0.3, C > 0.8)│                  │
///        └───────────┬───────────┘                  │
///                    │ FE >= 0.3 or C <= 0.8       │
///        ┌───────────▼───────────┐                  │
///        │        ĐỘNG          │                  │
///        │ (0.3≤FE<0.7, 0.5≤C<0.8)│                │
///        └───────────┬───────────┘                  │
///                    │ FE >= 0.7 or C < 0.5        │
///        ┌───────────▼───────────┐                  │
///        │      HỖN LOẠN        │──────────────────┘
///        │   (FE >= 0.7, C < 0.5)│   C >= 0.5
///        └───────────────────────┘
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhilosophicalStateMonitor {
    /// Current philosophical state
    pub current_state: PhilosophicalState,
    /// Previous state (for transition detection)
    pub previous_state: PhilosophicalState,
    /// Thresholds for state transitions
    pub thresholds: PhilosophicalStateThresholds,
    /// Exponential moving average of free energy
    pub free_energy_ema: f32,
    /// Current coherence score
    pub coherence_score: f32,
    /// Number of cycles in current state
    pub cycles_in_state: u64,
    /// Total state transitions
    pub total_transitions: u64,
    /// Timestamp of last transition (microseconds)
    pub last_transition_us: i64,
}

impl Default for PhilosophicalStateMonitor {
    fn default() -> Self {
        Self {
            current_state: PhilosophicalState::Yen,
            previous_state: PhilosophicalState::Yen,
            thresholds: PhilosophicalStateThresholds::default(),
            free_energy_ema: 0.0,
            coherence_score: 1.0,
            cycles_in_state: 0,
            total_transitions: 0,
            last_transition_us: 0,
        }
    }
}

impl PhilosophicalStateMonitor {
    /// Create monitor with custom thresholds.
    pub fn with_thresholds(thresholds: PhilosophicalStateThresholds) -> Self {
        Self {
            thresholds,
            ..Default::default()
        }
    }

    /// Update state based on current free energy and coherence.
    ///
    /// # Arguments
    /// * `free_energy` - Current free energy (prediction error), range [0, ∞)
    /// * `coherence` - Current coherence score, range [0, 1]
    ///
    /// # Returns
    /// The new philosophical state after evaluation.
    pub fn update(&mut self, free_energy: f32, coherence: f32) -> PhilosophicalState {
        self.update_with_timestamp(free_energy, coherence, 0)
    }

    /// Update state with timestamp tracking.
    pub fn update_with_timestamp(
        &mut self,
        free_energy: f32,
        coherence: f32,
        timestamp_us: i64,
    ) -> PhilosophicalState {
        // Update metrics with EMA smoothing
        const EMA_ALPHA: f32 = 0.3;
        self.free_energy_ema = self.free_energy_ema * (1.0 - EMA_ALPHA) + free_energy * EMA_ALPHA;
        self.coherence_score = coherence;

        // Determine new state based on thresholds
        let new_state = self.evaluate_state();

        // Handle state transition
        if new_state != self.current_state {
            self.previous_state = self.current_state;
            self.current_state = new_state;
            self.cycles_in_state = 0;
            self.total_transitions += 1;
            self.last_transition_us = timestamp_us;

            log::info!(
                "PhilosophicalState: {} -> {} (FE={:.3}, C={:.3})",
                self.previous_state.vietnamese_name(),
                self.current_state.vietnamese_name(),
                self.free_energy_ema,
                self.coherence_score
            );
        } else {
            self.cycles_in_state += 1;
        }

        self.current_state
    }

    /// Evaluate current state based on thresholds (internal).
    fn evaluate_state(&self) -> PhilosophicalState {
        let fe = self.free_energy_ema;
        let c = self.coherence_score;
        let th = &self.thresholds;

        // Apply hysteresis based on current state
        let hysteresis = th.hysteresis;

        match self.current_state {
            PhilosophicalState::Yen => {
                // Transition to ĐỘNG if free energy rises or coherence drops
                if fe >= th.fe_yen_to_dong + hysteresis || c <= th.coherence_yen - hysteresis {
                    if fe >= th.fe_dong_to_honloan {
                        PhilosophicalState::HonLoan
                    } else {
                        PhilosophicalState::Dong
                    }
                } else {
                    PhilosophicalState::Yen
                }
            }
            PhilosophicalState::Dong => {
                // Transition to HỖN LOẠN if things get worse
                if fe >= th.fe_dong_to_honloan + hysteresis || c < th.coherence_dong - hysteresis {
                    PhilosophicalState::HonLoan
                }
                // Transition back to YÊN if things improve
                else if fe < th.fe_yen_to_dong - hysteresis && c > th.coherence_yen + hysteresis {
                    PhilosophicalState::Yen
                } else {
                    PhilosophicalState::Dong
                }
            }
            PhilosophicalState::HonLoan => {
                // Only recover when coherence improves significantly
                if c >= th.coherence_dong + hysteresis && fe < th.fe_dong_to_honloan - hysteresis {
                    PhilosophicalState::Dong
                } else {
                    PhilosophicalState::HonLoan
                }
            }
        }
    }

    /// Check if a state transition occurred in the last update.
    pub fn just_transitioned(&self) -> bool {
        self.cycles_in_state == 0 && self.total_transitions > 0
    }

    /// Get the direction of the last transition.
    pub fn transition_direction(&self) -> StateTransitionDirection {
        if self.previous_state == self.current_state {
            StateTransitionDirection::None
        } else {
            match (self.previous_state, self.current_state) {
                (PhilosophicalState::Yen, PhilosophicalState::Dong)
                | (PhilosophicalState::Dong, PhilosophicalState::HonLoan)
                | (PhilosophicalState::Yen, PhilosophicalState::HonLoan) => {
                    StateTransitionDirection::Degrading
                }
                (PhilosophicalState::HonLoan, PhilosophicalState::Dong)
                | (PhilosophicalState::Dong, PhilosophicalState::Yen)
                | (PhilosophicalState::HonLoan, PhilosophicalState::Yen) => {
                    StateTransitionDirection::Improving
                }
                _ => StateTransitionDirection::None,
            }
        }
    }

    /// Get processing configuration based on current state.
    pub fn get_processing_config(&self) -> PhilosophicalProcessingConfig {
        match self.current_state {
            PhilosophicalState::Yen => PhilosophicalProcessingConfig {
                sampling_multiplier: 1.0,
                attention_level: 0.5,
                enable_trauma_guard: false,
                enable_safe_fallback: false,
                logging_verbosity: LogVerbosity::Normal,
                refinement_enabled: false,
            },
            PhilosophicalState::Dong => PhilosophicalProcessingConfig {
                sampling_multiplier: 2.0,
                attention_level: 0.9,
                enable_trauma_guard: true,
                enable_safe_fallback: false,
                logging_verbosity: LogVerbosity::Detailed,
                refinement_enabled: true,
            },
            PhilosophicalState::HonLoan => PhilosophicalProcessingConfig {
                sampling_multiplier: 0.5,
                attention_level: 0.3,
                enable_trauma_guard: true,
                enable_safe_fallback: true,
                logging_verbosity: LogVerbosity::Critical,
                refinement_enabled: false, // Conserve resources
            },
        }
    }
}

/// Direction of state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateTransitionDirection {
    /// No transition occurred
    None,
    /// Moving toward worse state (YÊN -> ĐỘNG -> HỖN LOẠN)
    Degrading,
    /// Moving toward better state (HỖN LOẠN -> ĐỘNG -> YÊN)
    Improving,
}

/// Processing configuration derived from philosophical state.
#[derive(Debug, Clone, Copy)]
pub struct PhilosophicalProcessingConfig {
    /// Multiplier for sampling frequency
    pub sampling_multiplier: f32,
    /// Attention level for processing
    pub attention_level: f32,
    /// Whether trauma guard should be active
    pub enable_trauma_guard: bool,
    /// Whether to use safe fallback actions
    pub enable_safe_fallback: bool,
    /// Logging verbosity level
    pub logging_verbosity: LogVerbosity,
    /// Whether to enable uncertainty refinement
    pub refinement_enabled: bool,
}

/// Logging verbosity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogVerbosity {
    /// Minimal logging (YÊN state)
    Normal,
    /// Detailed logging (ĐỘNG state)
    Detailed,
    /// Only critical events (HỖN LOẠN state)
    Critical,
}

// ============================================================================
// COHERENCE COMPUTATION
// ============================================================================

/// Compute coherence score from three consciousness aspects.
///
/// # B.ONE V3 Concept
/// > "Coherence measures the consistency across the three consciousnesses
/// > (MINH GIỚI, GEM, PHÁ QUÂN) in their perception of the current state."
///
/// # Arguments
/// * `minh_gioi_conf` - Confidence from moral guardian (Vedanā)
/// * `gem_conf` - Confidence from pattern oracle (Saññā)
/// * `pha_quan_conf` - Confidence from strategic intent (Saṅkhāra)
///
/// # Returns
/// Coherence score in [0, 1], where 1 = perfect agreement
pub fn compute_coherence(
    minh_gioi_conf: f32,
    gem_conf: f32,
    pha_quan_conf: f32,
) -> f32 {
    let confidences = [minh_gioi_conf, gem_conf, pha_quan_conf];

    // Coherence = 1 - variance of confidences
    // High variance = low coherence (disagreement)
    // Low variance = high coherence (agreement)

    let mean: f32 = confidences.iter().sum::<f32>() / 3.0;
    let variance: f32 = confidences.iter()
        .map(|c| (c - mean).powi(2))
        .sum::<f32>() / 3.0;

    // Map variance to coherence (variance of [0,1] values is at most 0.25)
    // Coherence = 1 - 4*variance (scales variance to [0,1] range)
    (1.0 - 4.0 * variance).clamp(0.0, 1.0)
}

/// Compute coherence from belief engine agent votes.
pub fn compute_coherence_from_votes(votes: &[(String, crate::belief::AgentVote)]) -> f32 {
    if votes.is_empty() {
        return 1.0; // No votes = default coherence
    }

    let confidences: Vec<f32> = votes.iter()
        .map(|(_, vote)| vote.confidence)
        .collect();

    let mean: f32 = confidences.iter().sum::<f32>() / confidences.len() as f32;
    let variance: f32 = confidences.iter()
        .map(|c| (c - mean).powi(2))
        .sum::<f32>() / confidences.len() as f32;

    (1.0 - 4.0 * variance).clamp(0.0, 1.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state_is_yen() {
        let monitor = PhilosophicalStateMonitor::default();
        assert_eq!(monitor.current_state, PhilosophicalState::Yen);
    }

    #[test]
    fn test_stay_in_yen_with_low_fe_high_coherence() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // Low free energy, high coherence = stay in YÊN
        let state = monitor.update(0.1, 0.95);
        assert_eq!(state, PhilosophicalState::Yen);
    }

    #[test]
    fn test_transition_to_dong() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // Elevated free energy triggers transition to ĐỘNG
        for _ in 0..5 {
            monitor.update(0.5, 0.7);
        }
        assert_eq!(monitor.current_state, PhilosophicalState::Dong);
    }

    #[test]
    fn test_transition_to_honloan() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // High free energy, low coherence = HỖN LOẠN
        for _ in 0..5 {
            monitor.update(0.9, 0.3);
        }
        assert_eq!(monitor.current_state, PhilosophicalState::HonLoan);
    }

    #[test]
    fn test_recovery_from_honloan() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // First, go to HỖN LOẠN
        for _ in 0..5 {
            monitor.update(0.9, 0.3);
        }
        assert_eq!(monitor.current_state, PhilosophicalState::HonLoan);

        // Then recover to ĐỘNG
        for _ in 0..10 {
            monitor.update(0.4, 0.7);
        }
        assert_eq!(monitor.current_state, PhilosophicalState::Dong);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // At threshold boundary, hysteresis should prevent rapid switching
        let threshold = monitor.thresholds.fe_yen_to_dong;

        monitor.update(threshold - 0.01, 0.85);
        let state1 = monitor.current_state;

        monitor.update(threshold + 0.01, 0.85);
        let state2 = monitor.current_state;

        // With hysteresis, small fluctuations shouldn't cause transition
        assert_eq!(state1, state2);
    }

    #[test]
    fn test_transition_direction() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // Transition YÊN -> ĐỘNG (degrading)
        for _ in 0..5 {
            monitor.update(0.5, 0.6);
        }
        assert_eq!(monitor.transition_direction(), StateTransitionDirection::Degrading);
    }

    #[test]
    fn test_processing_config_yen() {
        let monitor = PhilosophicalStateMonitor::default();
        let config = monitor.get_processing_config();

        assert_eq!(config.sampling_multiplier, 1.0);
        assert!(!config.enable_safe_fallback);
        assert!(!config.enable_trauma_guard);
    }

    #[test]
    fn test_processing_config_honloan() {
        let mut monitor = PhilosophicalStateMonitor::default();

        // Force HỖN LOẠN state
        for _ in 0..5 {
            monitor.update(0.9, 0.2);
        }

        let config = monitor.get_processing_config();
        assert!(config.enable_safe_fallback);
        assert!(config.enable_trauma_guard);
        assert_eq!(config.logging_verbosity, LogVerbosity::Critical);
    }

    #[test]
    fn test_coherence_computation() {
        // Perfect agreement
        let coherence = compute_coherence(0.9, 0.9, 0.9);
        assert!(coherence > 0.99);

        // Some disagreement
        let coherence = compute_coherence(0.9, 0.5, 0.7);
        assert!(coherence < 0.9);
        assert!(coherence > 0.5);

        // High disagreement
        let coherence = compute_coherence(1.0, 0.0, 0.5);
        assert!(coherence < 0.5);
    }

    #[test]
    fn test_state_display() {
        let yen = PhilosophicalState::Yen;
        let display = format!("{}", yen);
        assert!(display.contains("YÊN"));
        assert!(display.contains("Tranquil"));
    }
}
