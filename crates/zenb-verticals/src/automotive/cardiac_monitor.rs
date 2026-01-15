//! Cardiac Event Monitor
//!
//! Detects potential cardiac emergencies from HR/HRV patterns.

use zenb_signals::physio::HrvResult;

/// Cardiac alert
#[derive(Debug, Clone)]
pub struct CardiacAlert {
    pub is_emergency: bool,
    pub alert_type: CardiacAlertType,
    pub message: String,
    pub confidence: f32,
}

/// Types of cardiac alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardiacAlertType {
    Bradycardia,
    Tachycardia,
    Arrhythmia,
    LowHrv,
}

/// Cardiac Monitor
pub struct CardiacMonitor {
    hr_history: Vec<f32>,
    hrv_history: Vec<f32>,

    bradycardia_threshold: f32,
    tachycardia_threshold: f32,
    hr_change_threshold: f32,
    low_hrv_threshold: f32,
}

impl CardiacMonitor {
    pub fn new() -> Self {
        Self {
            hr_history: Vec::with_capacity(60),
            hrv_history: Vec::with_capacity(60),
            bradycardia_threshold: 50.0,
            tachycardia_threshold: 120.0,
            hr_change_threshold: 30.0,
            low_hrv_threshold: 10.0,
        }
    }

    pub fn set_hr_change_threshold(&mut self, threshold: f32) {
        self.hr_change_threshold = threshold.max(1.0);
    }

    /// Check for cardiac anomalies
    pub fn check(
        &mut self,
        heart_rate: Option<f32>,
        hrv: &Option<HrvResult>,
    ) -> Option<CardiacAlert> {
        let hr = heart_rate?;

        self.hr_history.push(hr);
        if self.hr_history.len() > 60 {
            self.hr_history.remove(0);
        }

        if let Some(h) = hrv {
            if let Some(metrics) = h.metrics.as_ref() {
                self.hrv_history.push(metrics.rmssd_ms);
                if self.hrv_history.len() > 60 {
                    self.hrv_history.remove(0);
                }
            }
        }

        if hr < self.bradycardia_threshold {
            return Some(CardiacAlert {
                is_emergency: hr < 40.0,
                alert_type: CardiacAlertType::Bradycardia,
                message: format!("Low heart rate: {:.0} BPM", hr),
                confidence: 0.8,
            });
        }

        if hr > self.tachycardia_threshold {
            return Some(CardiacAlert {
                is_emergency: hr > 150.0,
                alert_type: CardiacAlertType::Tachycardia,
                message: format!("High heart rate: {:.0} BPM", hr),
                confidence: 0.8,
            });
        }

        if self.hr_history.len() >= 10 {
            let recent_avg: f32 = self.hr_history[self.hr_history.len() - 5..]
                .iter()
                .sum::<f32>()
                / 5.0;
            let older_avg: f32 = self.hr_history[self.hr_history.len() - 10..self.hr_history.len() - 5]
                .iter()
                .sum::<f32>()
                / 5.0;

            let change = (recent_avg - older_avg).abs();
            if change > self.hr_change_threshold {
                return Some(CardiacAlert {
                    is_emergency: change > 50.0,
                    alert_type: CardiacAlertType::Arrhythmia,
                    message: format!("Sudden HR change: {:.0} BPM", change),
                    confidence: 0.7,
                });
            }
        }

        if let Some(h) = hrv {
            if let Some(metrics) = h.metrics.as_ref() {
                if metrics.rmssd_ms < self.low_hrv_threshold {
                    return Some(CardiacAlert {
                        is_emergency: false,
                        alert_type: CardiacAlertType::LowHrv,
                        message: format!("Very low HRV: {:.1} ms", metrics.rmssd_ms),
                        confidence: 0.6,
                    });
                }
            }
        }

        None
    }

    pub fn reset(&mut self) {
        self.hr_history.clear();
        self.hrv_history.clear();
    }
}

impl Default for CardiacMonitor {
    fn default() -> Self {
        Self::new()
    }
}
