//! Hardware Bridge - Real device state for HamiltonianGuard
//!
//! Provides platform-agnostic interface to:
//! - Battery level and charging state
//! - CPU temperature  
//! - CPU/Memory usage
//!
//! Two implementations:
//! - `MobileHardwareProvider`: For Android/iOS (state pushed from platform)
//! - `NativeHardwareProvider`: For desktop (reads via sysinfo crate)

use std::sync::Mutex;
use zenb_core::safety::PhysicalState;

// ============================================================================
// Hardware State Provider Trait
// ============================================================================

/// Hardware state provider trait
pub trait HardwareStateProvider: Send + Sync {
    /// Get current physical state
    fn get_physical_state(&self) -> PhysicalState;
    
    /// Check if hardware info is available
    fn is_available(&self) -> bool;
}

// ============================================================================
// Mobile Hardware Provider (Android/iOS)
// ============================================================================

/// Mobile hardware provider (receives state from Android/iOS via FFI)
///
/// This is the primary provider for mobile platforms. The native platform
/// layer (Kotlin/Swift) pushes hardware state updates to this provider
/// via the `update()` method.
pub struct MobileHardwareProvider {
    state: Mutex<PhysicalState>,
    last_update_us: Mutex<i64>,
}

impl Default for MobileHardwareProvider {
    fn default() -> Self {
        Self {
            state: Mutex::new(PhysicalState::mock_normal()),
            last_update_us: Mutex::new(0),
        }
    }
}

impl MobileHardwareProvider {
    /// Create a new mobile hardware provider
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update state from mobile platform
    ///
    /// Called by Android/iOS layer via FFI to push real hardware data.
    ///
    /// # Arguments
    /// * `battery_level` - Battery percentage 0.0-1.0
    /// * `temperature_c` - Device temperature in Celsius
    /// * `is_charging` - Whether device is currently charging
    pub fn update(&self, battery_level: f32, temperature_c: f32, is_charging: bool) {
        if let Ok(mut state) = self.state.lock() {
            state.battery_level = battery_level.clamp(0.0, 1.0);
            state.temperature_c = temperature_c.clamp(-40.0, 100.0); // Reasonable range
            state.is_charging = is_charging;
        }
        
        if let Ok(mut ts) = self.last_update_us.lock() {
            *ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
        }
    }
    
    /// Update full state including CPU/memory
    ///
    /// For platforms that can provide CPU/memory info (Android with system permissions)
    pub fn update_full(
        &self,
        battery_level: f32,
        temperature_c: f32,
        is_charging: bool,
        cpu_usage: f32,
        memory_usage: f32,
    ) {
        if let Ok(mut state) = self.state.lock() {
            state.battery_level = battery_level.clamp(0.0, 1.0);
            state.temperature_c = temperature_c.clamp(-40.0, 100.0);
            state.is_charging = is_charging;
            state.cpu_usage = cpu_usage.clamp(0.0, 1.0);
            state.memory_usage = memory_usage.clamp(0.0, 1.0);
        }
        
        if let Ok(mut ts) = self.last_update_us.lock() {
            *ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
        }
    }
    
    /// Check if state is stale (not updated in 60 seconds)
    pub fn is_stale(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
            
        let last = self.last_update_us.lock().map(|t| *t).unwrap_or(0);
        
        // Stale if no update in 60 seconds
        (now - last) > 60_000_000
    }
}

impl HardwareStateProvider for MobileHardwareProvider {
    fn get_physical_state(&self) -> PhysicalState {
        self.state.lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| PhysicalState::mock_normal())
    }
    
    fn is_available(&self) -> bool {
        !self.is_stale()
    }
}

// ============================================================================
// Native Hardware Provider (Desktop/Server)
// ============================================================================

#[cfg(feature = "sysinfo")]
mod native {
    use super::*;
    use sysinfo::{Components, System};
    
    /// Native hardware provider using sysinfo crate
    ///
    /// For desktop/server platforms where we can directly read system info.
    /// Requires the `sysinfo` feature to be enabled.
    pub struct NativeHardwareProvider {
        system: Mutex<System>,
    }

    impl NativeHardwareProvider {
        pub fn new() -> Self {
            let mut system = System::new_all();
            system.refresh_all();
            Self {
                system: Mutex::new(system),
            }
        }

        pub fn refresh(&self) {
            if let Ok(mut sys) = self.system.lock() {
                sys.refresh_all();
            }
        }
    }

    impl Default for NativeHardwareProvider {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HardwareStateProvider for NativeHardwareProvider {
        fn get_physical_state(&self) -> PhysicalState {
            let sys = match self.system.lock() {
                Ok(s) => s,
                Err(_) => return PhysicalState::mock_normal(),
            };

            // CPU usage (average across all cores)
            let cpu_usage = sys.global_cpu_usage() / 100.0;

            // Memory usage
            let used_mem = sys.used_memory() as f32;
            let total_mem = sys.total_memory() as f32;
            let memory_usage = if total_mem > 0.0 {
                used_mem / total_mem
            } else {
                0.5
            };

            // Temperature (from components if available)
            let mut temperature_c = 35.0; // Default
            let components = Components::new_with_refreshed_list();
            for component in &components {
                if component.label().contains("CPU") || component.label().contains("Core") {
                    temperature_c = component.temperature();
                    break;
                }
            }

            // Battery: Not directly available via sysinfo on all platforms
            // Will use default values (can be overridden by platform layer)
            let battery_level = 0.8;
            let is_charging = false;

            PhysicalState {
                battery_level,
                temperature_c,
                cpu_usage,
                memory_usage,
                is_charging,
            }
        }

        fn is_available(&self) -> bool {
            true
        }
    }
}

#[cfg(feature = "sysinfo")]
pub use native::NativeHardwareProvider;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_provider_default() {
        let provider = MobileHardwareProvider::new();
        let state = provider.get_physical_state();
        
        assert_eq!(state.battery_level, 0.8);
        assert_eq!(state.temperature_c, 35.0);
        assert!(!state.is_charging);
    }

    #[test]
    fn test_mobile_provider_update() {
        let provider = MobileHardwareProvider::new();
        
        provider.update(0.25, 45.0, true);
        
        let state = provider.get_physical_state();
        assert_eq!(state.battery_level, 0.25);
        assert_eq!(state.temperature_c, 45.0);
        assert!(state.is_charging);
    }

    #[test]
    fn test_mobile_provider_clamps_values() {
        let provider = MobileHardwareProvider::new();
        
        // Out of range values should be clamped
        provider.update(1.5, 150.0, false);
        
        let state = provider.get_physical_state();
        assert_eq!(state.battery_level, 1.0);
        assert_eq!(state.temperature_c, 100.0);
    }

    #[test]
    fn test_mobile_provider_full_update() {
        let provider = MobileHardwareProvider::new();
        
        provider.update_full(0.5, 38.0, true, 0.75, 0.60);
        
        let state = provider.get_physical_state();
        assert_eq!(state.battery_level, 0.5);
        assert_eq!(state.temperature_c, 38.0);
        assert!(state.is_charging);
        assert_eq!(state.cpu_usage, 0.75);
        assert_eq!(state.memory_usage, 0.60);
    }

    #[test]
    fn test_is_available() {
        let provider = MobileHardwareProvider::new();
        
        // Fresh provider has no updates, so is_stale() returns true
        assert!(provider.is_stale());
        
        // After update, should be available (not stale)
        provider.update(0.5, 35.0, false);
        assert!(!provider.is_stale());
        assert!(provider.is_available());
    }
}
