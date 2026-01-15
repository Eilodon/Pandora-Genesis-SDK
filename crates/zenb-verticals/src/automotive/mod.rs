//! Automotive Vertical: Driver Monitoring System

pub mod dms;
pub mod drowsiness;
pub mod distraction;
pub mod cardiac_monitor;

pub use dms::{DriverMonitoringSystem, DmsConfig, DriverState, DriverAction};
pub use drowsiness::{DrowsinessDetector, DrowsinessResult, DrowsinessLevel};
pub use distraction::{DistractionDetector, DistractionResult, DistractionLevel};
pub use cardiac_monitor::{CardiacMonitor, CardiacAlert, CardiacAlertType};
