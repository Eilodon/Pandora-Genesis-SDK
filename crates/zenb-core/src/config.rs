use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

use crate::estimators::ukf::AukfConfig;
use crate::causal::PCConfig;
use crate::sensory::binaural::PsychoacousticCalibration;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),
    #[error("Config validation error: {0}")]
    Validation(String),
    #[error("Environment variable error: {0}")]
    EnvVar(#[from] std::env::VarError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenbConfig {
    pub fep: FepConfig,
    pub resonance: ResonanceConfig,
    pub safety: SafetyConfig,
    pub breath: BreathConfig,
    pub belief: BeliefConfig,
    pub performance: PerformanceConfig,
    pub sota: SotaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SotaConfig {
    /// Enable Hybrid UKF State Estimator
    pub use_ukf: bool,
    /// Automatically fall back to legacy estimator if UKF diverges
    pub ukf_fallback_enabled: bool,
    /// Adaptive UKF configuration (Sage-Husa)
    pub ukf_config: AukfConfig,
    
    /// Enable EFE-based policy selection
    pub use_efe_selection: bool,
    /// EFE precision (beta) - controls exploration vs exploitation
    /// If None, uses adaptive meta-learning
    pub efe_precision_beta: Option<f32>,
    
    /// Enable PC algorithm for causal structure learning
    pub pc_learning_enabled: bool,
    /// PC Algorithm configuration
    pub pc_config: PCConfig,

    /// Enable Automatic Scientist for hypothesis discovery
    #[serde(default)]
    pub scientist_enabled: Option<bool>,

    /// Psychoacoustic calibration profile
    pub audio_profile: AudioProfile,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AudioProfile {
    Standard,
    HearingImpaired,
    Audiophile,
}

impl Default for AudioProfile {
    fn default() -> Self {
        Self::Standard
    }
}

impl AudioProfile {
    pub fn to_calibration(&self) -> PsychoacousticCalibration {
        match self {
            Self::Standard => PsychoacousticCalibration::default(),
            Self::HearingImpaired => PsychoacousticCalibration {
                sensitivity_offset: 5.0,
                max_safe_level: 80.0,
                ..Default::default()
            },
            Self::Audiophile => PsychoacousticCalibration {
                sensitivity_offset: -3.0,
                reference_db_spl: 65.0,
                ..Default::default()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepConfig {
    pub process_noise: f32,
    pub base_obs_var: f32,
    pub lr_base: f32,
    pub lr_min: f32,
    pub lr_max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonanceConfig {
    pub window_size_sec: f32,
    pub coherence_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub trauma_hard_th: f32,
    pub trauma_soft_th: f32,
    pub trauma_decay_default: f32,
    /// DEPRECATED: This field is ignored at runtime.
    /// Test time bypass is now compile-time only. See `SafetyConfig::allow_test_time()`.
    #[serde(default)]
    #[deprecated(since = "0.2.0", note = "Use compile-time cfg(test) instead")]
    pub allow_test_time: bool,
}

impl SafetyConfig {
    /// Returns whether test-time bypass is allowed.
    /// This is a COMPILE-TIME decision, not runtime configurable.
    /// In release builds, this always returns false regardless of config.
    #[inline]
    pub fn allow_test_time() -> bool {
        cfg!(test)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathConfig {
    pub default_target_bpm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefConfig {
    /// Agent weights: [Gemini, MinhGioi, PhaQuan]
    #[serde(alias = "pathway_weights")]
    pub agent_weights: Vec<f32>,
    /// Prior logits for belief states
    pub prior_logits: [f32; 5],
    /// Smoothing time constant (seconds)
    pub smooth_tau_sec: f32,
    /// Hysteresis enter threshold
    pub enter_threshold: f32,
    /// Hysteresis exit threshold
    pub exit_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Batch size for event appends
    pub batch_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Maximum buffer size before forced flush
    pub max_buffer_size: usize,
    /// Enable burst filtering for sensor data
    pub enable_burst_filter: bool,
    /// Burst filter threshold in microseconds
    pub burst_filter_threshold_us: i64,
}

impl Default for ZenbConfig {
    fn default() -> Self {
        Self {
            fep: FepConfig::default(),
            resonance: ResonanceConfig::default(),
            safety: SafetyConfig::default(),
            breath: BreathConfig::default(),
            belief: BeliefConfig::default(),
            performance: PerformanceConfig::default(),
            sota: SotaConfig::default(),
        }
    }
}

impl Default for SotaConfig {
    fn default() -> Self {
        Self {
            use_ukf: false, // Default to false for safe rollout
            ukf_fallback_enabled: true,
            ukf_config: AukfConfig::default(),
            use_efe_selection: false,
            efe_precision_beta: None, // Enable adaptive by default if EFE is on
            pc_learning_enabled: false,
            pc_config: PCConfig::default(),
            scientist_enabled: None, // Default: disabled
            audio_profile: AudioProfile::default(),
        }
    }
}

impl Default for FepConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.02,
            base_obs_var: 0.25,
            lr_base: 0.6,
            lr_min: 0.05,
            lr_max: 0.8,
        }
    }
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            window_size_sec: 12.0,
            coherence_threshold: 0.35,
        }
    }
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            trauma_hard_th: 3.0,
            trauma_soft_th: 1.5,
            trauma_decay_default: 0.001,
            allow_test_time: false,
        }
    }
}

impl Default for BreathConfig {
    fn default() -> Self {
        Self {
            default_target_bpm: 6.0,
        }
    }
}

impl Default for BeliefConfig {
    fn default() -> Self {
        Self {
            agent_weights: vec![1.0, 0.8, 1.2],
            prior_logits: [0.0; 5],
            smooth_tau_sec: 4.0,
            enter_threshold: 0.6,
            exit_threshold: 0.4,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            batch_size: 20,
            flush_interval_ms: 80,
            max_buffer_size: 100,
            enable_burst_filter: true,
            burst_filter_threshold_us: 10_000, // 10ms
        }
    }
}

impl ZenbConfig {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: ZenbConfig = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with environment variable overrides
    /// Environment variables should be prefixed with ZENB_
    /// Example: ZENB_FEP_PROCESS_NOISE=0.03
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let mut config = Self::from_file(path)?;
        config.apply_env_overrides()?;
        config.validate()?;
        Ok(config)
    }

    /// Load from multiple sources with priority:
    /// 1. Environment variables (highest priority)
    /// 2. User config file (if exists)
    /// 3. Default config file
    /// 4. Built-in defaults (lowest priority)
    pub fn load_layered(
        default_path: Option<&Path>,
        user_path: Option<&Path>,
    ) -> Result<Self, ConfigError> {
        let mut config = ZenbConfig::default();

        // Layer 1: Default config file
        if let Some(path) = default_path {
            if path.exists() {
                config = Self::from_file(path)?;
            }
        }

        // Layer 2: User config file (overrides defaults)
        if let Some(path) = user_path {
            if path.exists() {
                let user_config = Self::from_file(path)?;
                config = config.merge(user_config);
            }
        }

        // Layer 3: Environment variables (highest priority)
        config.apply_env_overrides()?;
        config.validate()?;

        Ok(config)
    }

    /// Merge another config into this one (other takes priority)
    fn merge(mut self, other: ZenbConfig) -> Self {
        // For simplicity, just replace with other
        // In production, you might want field-by-field merging
        other
    }

    /// Apply environment variable overrides
    pub(crate) fn apply_env_overrides(&mut self) -> Result<(), ConfigError> {
        use std::env;

        // FEP overrides
        if let Ok(val) = env::var("ZENB_FEP_PROCESS_NOISE") {
            self.fep.process_noise = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_FEP_PROCESS_NOISE".to_string())
            })?;
        }
        if let Ok(val) = env::var("ZENB_FEP_BASE_OBS_VAR") {
            self.fep.base_obs_var = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_FEP_BASE_OBS_VAR".to_string())
            })?;
        }
        if let Ok(val) = env::var("ZENB_FEP_LR_BASE") {
            self.fep.lr_base = val
                .parse()
                .map_err(|_| ConfigError::Validation("Invalid ZENB_FEP_LR_BASE".to_string()))?;
        }

        // Resonance overrides
        if let Ok(val) = env::var("ZENB_RESONANCE_WINDOW_SIZE_SEC") {
            self.resonance.window_size_sec = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_RESONANCE_WINDOW_SIZE_SEC".to_string())
            })?;
        }

        // Safety overrides
        if let Ok(val) = env::var("ZENB_SAFETY_TRAUMA_HARD_TH") {
            self.safety.trauma_hard_th = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_SAFETY_TRAUMA_HARD_TH".to_string())
            })?;
        }

        // Breath overrides
        if let Ok(val) = env::var("ZENB_BREATH_DEFAULT_TARGET_BPM") {
            self.breath.default_target_bpm = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_BREATH_DEFAULT_TARGET_BPM".to_string())
            })?;
        }

        // Belief overrides
        if let Ok(val) = env::var("ZENB_BELIEF_SMOOTH_TAU_SEC") {
            self.belief.smooth_tau_sec = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_BELIEF_SMOOTH_TAU_SEC".to_string())
            })?;
        }

        // Performance overrides
        if let Ok(val) = env::var("ZENB_PERFORMANCE_BATCH_SIZE") {
            self.performance.batch_size = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_PERFORMANCE_BATCH_SIZE".to_string())
            })?;
        }
        if let Ok(val) = env::var("ZENB_PERFORMANCE_FLUSH_INTERVAL_MS") {
            self.performance.flush_interval_ms = val.parse().map_err(|_| {
                ConfigError::Validation("Invalid ZENB_PERFORMANCE_FLUSH_INTERVAL_MS".to_string())
            })?;
        }

        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        // FEP validation
        if self.fep.process_noise <= 0.0 || self.fep.process_noise > 1.0 {
            return Err(ConfigError::Validation(
                "fep.process_noise must be in (0, 1]".to_string(),
            ));
        }
        if self.fep.base_obs_var <= 0.0 {
            return Err(ConfigError::Validation(
                "fep.base_obs_var must be positive".to_string(),
            ));
        }
        if self.fep.lr_min < 0.0 || self.fep.lr_min > self.fep.lr_max {
            return Err(ConfigError::Validation(
                "fep.lr_min must be in [0, lr_max]".to_string(),
            ));
        }
        if self.fep.lr_max > 1.0 {
            return Err(ConfigError::Validation(
                "fep.lr_max must be <= 1.0".to_string(),
            ));
        }

        // Resonance validation
        if self.resonance.window_size_sec <= 0.0 {
            return Err(ConfigError::Validation(
                "resonance.window_size_sec must be positive".to_string(),
            ));
        }
        if self.resonance.coherence_threshold < 0.0 || self.resonance.coherence_threshold > 1.0 {
            return Err(ConfigError::Validation(
                "resonance.coherence_threshold must be in [0, 1]".to_string(),
            ));
        }

        // Safety validation
        if self.safety.trauma_hard_th <= self.safety.trauma_soft_th {
            return Err(ConfigError::Validation(
                "safety.trauma_hard_th must be > trauma_soft_th".to_string(),
            ));
        }
        if self.safety.trauma_decay_default < 0.0 || self.safety.trauma_decay_default > 1.0 {
            return Err(ConfigError::Validation(
                "safety.trauma_decay_default must be in [0, 1]".to_string(),
            ));
        }

        // Breath validation
        if self.breath.default_target_bpm < 1.0 || self.breath.default_target_bpm > 30.0 {
            return Err(ConfigError::Validation(
                "breath.default_target_bpm must be in [1, 30]".to_string(),
            ));
        }

        // Belief validation
        if self.belief.agent_weights.len() != 3 {
            return Err(ConfigError::Validation(
                "belief.agent_weights must have exactly 3 elements".to_string(),
            ));
        }
        if self.belief.agent_weights.iter().any(|&w| w < 0.0) {
            return Err(ConfigError::Validation(
                "belief.agent_weights must be non-negative".to_string(),
            ));
        }
        if self.belief.smooth_tau_sec <= 0.0 {
            return Err(ConfigError::Validation(
                "belief.smooth_tau_sec must be positive".to_string(),
            ));
        }
        if self.belief.enter_threshold <= self.belief.exit_threshold {
            return Err(ConfigError::Validation(
                "belief.enter_threshold must be > exit_threshold".to_string(),
            ));
        }

        // Performance validation
        if self.performance.batch_size == 0 {
            return Err(ConfigError::Validation(
                "performance.batch_size must be > 0".to_string(),
            ));
        }
        if self.performance.max_buffer_size < self.performance.batch_size {
            return Err(ConfigError::Validation(
                "performance.max_buffer_size must be >= batch_size".to_string(),
            ));
        }

        Ok(())
    }

    /// Export configuration to TOML string
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = self
            .to_toml_string()
            .map_err(|e| ConfigError::Validation(format!("TOML serialization error: {}", e)))?;
        fs::write(path, content)?;
        Ok(())
    }
}
