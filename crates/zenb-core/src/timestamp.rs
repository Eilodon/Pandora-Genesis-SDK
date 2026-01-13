//! Timestamp tracking and verification
//!
//! Consolidates time-related state to enforce monotonicity and prevent "time travel" bugs.
//! Extracted from Engine during Phase 2.4 refactoring.

use serde::{Deserialize, Serialize};

/// High-resolution timestamp log with strict monotonicity enforcement.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TimestampLog {
    /// Last general update timestamp
    pub last_update: Option<i64>,
    
    /// Last sensor ingestion timestamp
    pub last_ingest: Option<i64>,
    
    /// Last control tick timestamp
    pub last_control: Option<i64>,
    
    /// Session start timestamp
    pub session_start: Option<i64>,
}

impl TimestampLog {
    /// Create a new empty timestamp log
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update ingestion timestamp, verifying monotonicity.
    ///
    /// # Arguments
    /// * `ts_us` - New timestamp in microseconds
    ///
    /// # Returns
    /// * `Ok(dt_sec)` - Time delta since last ingest in seconds (0.0 if first)
    /// * `Err(msg)` - If timestamp flowed backwards
    pub fn update_ingest(&mut self, ts_us: i64) -> Result<f32, String> {
        self.check_monotonicity(self.last_ingest, ts_us, "ingest")?;
        
        let dt = if let Some(last) = self.last_ingest {
            crate::domain::dt_sec(ts_us, last)
        } else {
            0.0
        };
        
        self.last_ingest = Some(ts_us);
        self.last_update = Some(ts_us);
        Ok(dt)
    }
    
    /// Update control timestamp, verifying monotonicity.
    ///
    /// # Arguments
    /// * `ts_us` - New timestamp in microseconds
    ///
    /// # Returns
    /// * `Ok(dt_sec)` - Time delta since last control tick in seconds (0.0 if first)
    /// * `Err(msg)` - If timestamp flowed backwards
    pub fn update_control(&mut self, ts_us: i64) -> Result<f32, String> {
        self.check_monotonicity(self.last_control, ts_us, "control")?;
        
        let dt = if let Some(last) = self.last_control {
            crate::domain::dt_sec(ts_us, last)
        } else {
            0.0
        };
        
        self.last_control = Some(ts_us);
        self.last_update = Some(ts_us); // Control tick also updates general last_ts
        Ok(dt)
    }
    
    /// Start a new session if not running.
    pub fn start_session(&mut self, ts_us: i64) {
        if self.session_start.is_none() {
            self.session_start = Some(ts_us);
        }
    }
    
    /// Get current session duration in seconds.
    pub fn session_duration(&self, now_us: i64) -> f32 {
        if let Some(start) = self.session_start {
            crate::domain::dt_sec(now_us, start)
        } else {
            0.0
        }
    }
    
    /// Clear session start (end session).
    pub fn end_session(&mut self) {
        self.session_start = None;
    }
    
    /// Internal monotonicity check.
    fn check_monotonicity(&self, last_opt: Option<i64>, now: i64, context: &str) -> Result<(), String> {
        if let Some(last) = last_opt {
            if now < last {
                return Err(format!(
                    "Timestamp regression in {}: now={} < last={} (delta={}us)",
                    context, now, last, now - last
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonicity_check() {
        let mut log = TimestampLog::new();
        
        // Initial update OK
        assert!(log.update_ingest(1000).is_ok());
        
        // Proper forward step OK
        assert!(log.update_ingest(2000).is_ok());
        
        // Backward step should fail
        let res = log.update_ingest(1500);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("Timestamp regression"));
    }
    
    #[test]
    fn test_dt_calculation() {
        let mut log = TimestampLog::new();
        
        // First update: dt = 0.0
        let dt1 = log.update_control(1_000_000).unwrap();
        assert_eq!(dt1, 0.0);
        
        // Second update: dt = 0.5s
        let dt2 = log.update_control(1_500_000).unwrap();
        assert_eq!(dt2, 0.5);
    }
    
    #[test]
    fn test_session_duration() {
        let mut log = TimestampLog::new();
        log.start_session(1_000_000);
        
        assert_eq!(log.session_duration(1_000_000), 0.0);
        assert_eq!(log.session_duration(2_500_000), 1.5);
        
        // End session
        log.end_session();
        assert_eq!(log.session_duration(3_000_000), 0.0);
    }
}
