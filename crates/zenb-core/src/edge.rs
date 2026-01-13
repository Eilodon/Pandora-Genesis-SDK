//! Edge Optimization module for device-aware configuration.
//!
//! Ported from Pandora's `pandora_orchestrator::edge_optimization`.
//! Provides device-specific optimization configuration for AGOLOS deployment.
//!
//! # Usage
//! ```rust
//! use zenb_core::edge::{EdgeDeviceSpecs, EdgeDeviceType, EdgeOptimizer};
//!
//! let specs = EdgeDeviceSpecs::detect_current(); // Auto-detect
//! let optimizer = EdgeOptimizer::new(specs);
//!
//! // Get optimized config for this device
//! if optimizer.should_use_simplified_efe() {
//!     // Use simplified computation
//! }
//! ```

use serde::{Deserialize, Serialize};

// ============================================================================
// Device Types
// ============================================================================

/// Target edge device type for optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDeviceType {
    /// High-end smartphones with GPU (iPhone Pro, Samsung S series)
    Mobile,
    /// Low-power IoT devices with limited resources
    IoT,
    /// Single-board computers (Raspberry Pi, Jetson Nano)
    SingleBoardComputer,
    /// Microcontrollers (ESP32, ARM Cortex-M)
    Microcontroller,
    /// Desktop/Server - no resource constraints
    Desktop,
}

impl Default for EdgeDeviceType {
    fn default() -> Self {
        EdgeDeviceType::Mobile
    }
}

// ============================================================================
// Device Specifications
// ============================================================================

/// Device hardware specifications for optimization decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeDeviceSpecs {
    /// Device category
    pub device_type: EdgeDeviceType,
    /// Available RAM in MB
    pub available_ram_mb: f32,
    /// Current battery level (0.0 - 1.0), None if plugged in or desktop
    pub battery_level: Option<f32>,
    /// Whether device has GPU/Neural Engine
    pub has_gpu: bool,
    /// Number of CPU cores available
    pub cpu_cores: u8,
    /// Whether device is currently thermal throttling
    pub is_thermal_throttled: bool,
}

impl Default for EdgeDeviceSpecs {
    fn default() -> Self {
        Self {
            device_type: EdgeDeviceType::Mobile,
            available_ram_mb: 2048.0,
            battery_level: Some(0.8),
            has_gpu: true,
            cpu_cores: 4,
            is_thermal_throttled: false,
        }
    }
}

impl EdgeDeviceSpecs {
    /// Create specs for a mobile device (iOS/Android).
    pub fn mobile(ram_mb: f32, battery: f32, has_gpu: bool) -> Self {
        Self {
            device_type: EdgeDeviceType::Mobile,
            available_ram_mb: ram_mb,
            battery_level: Some(battery),
            has_gpu,
            cpu_cores: 4,
            is_thermal_throttled: false,
        }
    }

    /// Create specs for a low-power IoT device.
    pub fn iot() -> Self {
        Self {
            device_type: EdgeDeviceType::IoT,
            available_ram_mb: 64.0,
            battery_level: None, // Usually always powered
            has_gpu: false,
            cpu_cores: 1,
            is_thermal_throttled: false,
        }
    }

    /// Create specs for a Raspberry Pi.
    pub fn raspberry_pi() -> Self {
        Self {
            device_type: EdgeDeviceType::SingleBoardComputer,
            available_ram_mb: 1024.0,
            battery_level: None,
            has_gpu: false,
            cpu_cores: 4,
            is_thermal_throttled: false,
        }
    }

    /// Create specs for desktop (no constraints).
    pub fn desktop() -> Self {
        Self {
            device_type: EdgeDeviceType::Desktop,
            available_ram_mb: 16384.0,
            battery_level: None,
            has_gpu: true,
            cpu_cores: 8,
            is_thermal_throttled: false,
        }
    }

    /// Is this a low-resource device?
    pub fn is_constrained(&self) -> bool {
        matches!(
            self.device_type,
            EdgeDeviceType::IoT | EdgeDeviceType::Microcontroller
        )
    }

    /// Is battery low? (< 20%)
    pub fn is_battery_low(&self) -> bool {
        self.battery_level.map(|b| b < 0.2).unwrap_or(false)
    }
}

// ============================================================================
// Optimization Configuration
// ============================================================================

/// Configuration derived from device specs for optimal performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Use simplified EFE calculation (faster, less accurate)
    pub use_simplified_efe: bool,
    /// Use simplified belief engine (fewer agents)
    pub use_simplified_belief: bool,
    /// Enable Vajra-001 advanced features
    pub enable_vajra_features: bool,
    /// Enable Holographic Memory (RAM intensive)
    pub enable_holographic_memory: bool,
    /// Maximum observation buffer size
    pub max_observation_buffer: usize,
    /// Target poll interval in milliseconds
    pub recommended_poll_interval_ms: u32,
    /// Enable sensor anomaly detection
    pub enable_anomaly_detection: bool,
    /// Maximum computation time per tick in microseconds
    pub max_compute_time_us: u32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_simplified_efe: false,
            use_simplified_belief: false,
            enable_vajra_features: true,
            enable_holographic_memory: true,
            max_observation_buffer: 100,
            recommended_poll_interval_ms: 500,
            enable_anomaly_detection: true,
            max_compute_time_us: 10_000, // 10ms
        }
    }
}

// ============================================================================
// Edge Optimizer
// ============================================================================

/// Optimizer that provides device-specific configuration.
#[derive(Debug, Clone)]
pub struct EdgeOptimizer {
    specs: EdgeDeviceSpecs,
    config: OptimizationConfig,
}

impl EdgeOptimizer {
    /// Create optimizer for given device specs.
    pub fn new(specs: EdgeDeviceSpecs) -> Self {
        let config = Self::compute_config(&specs);
        Self { specs, config }
    }

    /// Compute optimal configuration based on device specs.
    fn compute_config(specs: &EdgeDeviceSpecs) -> OptimizationConfig {
        match specs.device_type {
            EdgeDeviceType::Desktop => OptimizationConfig {
                use_simplified_efe: false,
                use_simplified_belief: false,
                enable_vajra_features: true,
                enable_holographic_memory: true,
                max_observation_buffer: 500,
                recommended_poll_interval_ms: 200,
                enable_anomaly_detection: true,
                max_compute_time_us: 50_000, // 50ms allowed
            },
            EdgeDeviceType::Mobile => {
                let low_battery = specs.is_battery_low();
                OptimizationConfig {
                    use_simplified_efe: low_battery,
                    use_simplified_belief: low_battery,
                    enable_vajra_features: !low_battery,
                    enable_holographic_memory: specs.available_ram_mb > 1024.0 && !low_battery,
                    max_observation_buffer: if low_battery { 50 } else { 100 },
                    recommended_poll_interval_ms: if low_battery { 1000 } else { 500 },
                    enable_anomaly_detection: true,
                    max_compute_time_us: if low_battery { 5_000 } else { 10_000 },
                }
            }
            EdgeDeviceType::SingleBoardComputer => OptimizationConfig {
                use_simplified_efe: false,
                use_simplified_belief: false,
                enable_vajra_features: specs.available_ram_mb > 512.0,
                enable_holographic_memory: specs.available_ram_mb > 512.0,
                max_observation_buffer: 75,
                recommended_poll_interval_ms: 750,
                enable_anomaly_detection: true,
                max_compute_time_us: 20_000,
            },
            EdgeDeviceType::IoT => OptimizationConfig {
                use_simplified_efe: true,
                use_simplified_belief: true,
                enable_vajra_features: false,
                enable_holographic_memory: false,
                max_observation_buffer: 20,
                recommended_poll_interval_ms: 2000,
                enable_anomaly_detection: false, // Too expensive
                max_compute_time_us: 2_000,
            },
            EdgeDeviceType::Microcontroller => OptimizationConfig {
                use_simplified_efe: true,
                use_simplified_belief: true,
                enable_vajra_features: false,
                enable_holographic_memory: false,
                max_observation_buffer: 10,
                recommended_poll_interval_ms: 5000,
                enable_anomaly_detection: false,
                max_compute_time_us: 1_000, // 1ms max
            },
        }
    }

    /// Get current optimization config.
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }

    /// Get device specs.
    pub fn specs(&self) -> &EdgeDeviceSpecs {
        &self.specs
    }

    /// Update device specs and recompute config.
    pub fn update_specs(&mut self, specs: EdgeDeviceSpecs) {
        self.config = Self::compute_config(&specs);
        self.specs = specs;
    }

    /// Should we use simplified EFE?
    #[inline]
    pub fn should_use_simplified_efe(&self) -> bool {
        self.config.use_simplified_efe
    }

    /// Should we enable Vajra features?
    #[inline]
    pub fn should_enable_vajra(&self) -> bool {
        self.config.enable_vajra_features
    }

    /// Get recommended poll interval.
    #[inline]
    pub fn recommended_poll_interval_ms(&self) -> u32 {
        self.config.recommended_poll_interval_ms
    }

    /// Get max observation buffer size.
    #[inline]
    pub fn max_observation_buffer(&self) -> usize {
        self.config.max_observation_buffer
    }

    /// Get device tier (0 = lowest, 3 = highest).
    pub fn device_tier(&self) -> u8 {
        match self.specs.device_type {
            EdgeDeviceType::Microcontroller => 0,
            EdgeDeviceType::IoT => 1,
            EdgeDeviceType::SingleBoardComputer => 2,
            EdgeDeviceType::Mobile if self.specs.is_battery_low() => 2,
            EdgeDeviceType::Mobile => 3,
            EdgeDeviceType::Desktop => 3,
        }
    }
}

impl Default for EdgeOptimizer {
    fn default() -> Self {
        Self::new(EdgeDeviceSpecs::default())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desktop_config() {
        let optimizer = EdgeOptimizer::new(EdgeDeviceSpecs::desktop());
        let config = optimizer.config();

        assert!(!config.use_simplified_efe);
        assert!(config.enable_vajra_features);
        assert!(config.enable_holographic_memory);
    }

    #[test]
    fn test_mobile_low_battery() {
        let mut specs = EdgeDeviceSpecs::mobile(2048.0, 0.15, true);
        let optimizer = EdgeOptimizer::new(specs.clone());

        assert!(specs.is_battery_low());
        assert!(optimizer.should_use_simplified_efe());
        assert!(!optimizer.should_enable_vajra());
    }

    #[test]
    fn test_mobile_full_battery() {
        let specs = EdgeDeviceSpecs::mobile(2048.0, 0.9, true);
        let optimizer = EdgeOptimizer::new(specs);

        assert!(!optimizer.should_use_simplified_efe());
        assert!(optimizer.should_enable_vajra());
    }

    #[test]
    fn test_iot_constrained() {
        let specs = EdgeDeviceSpecs::iot();
        let optimizer = EdgeOptimizer::new(specs.clone());

        assert!(specs.is_constrained());
        assert!(optimizer.should_use_simplified_efe());
        assert!(!optimizer.config().enable_anomaly_detection);
    }

    #[test]
    fn test_device_tier() {
        assert_eq!(
            EdgeOptimizer::new(EdgeDeviceSpecs::desktop()).device_tier(),
            3
        );
        assert_eq!(EdgeOptimizer::new(EdgeDeviceSpecs::iot()).device_tier(), 1);
    }

    #[test]
    fn test_update_specs() {
        let mut optimizer = EdgeOptimizer::new(EdgeDeviceSpecs::mobile(2048.0, 0.9, true));
        assert!(optimizer.should_enable_vajra());

        // Battery dropped
        optimizer.update_specs(EdgeDeviceSpecs::mobile(2048.0, 0.1, true));
        assert!(!optimizer.should_enable_vajra());
    }
}
