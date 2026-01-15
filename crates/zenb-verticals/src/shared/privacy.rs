//! Privacy Compliance Utilities
//!
//! Helpers for GDPR, CCPA, BIPA compliance.

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum retention period (seconds)
    pub max_retention_sec: u64,
    /// Auto-delete after session
    pub delete_on_session_end: bool,
    /// Anonymize after period
    pub anonymize_after_sec: Option<u64>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_retention_sec: 86_400,
            delete_on_session_end: true,
            anonymize_after_sec: Some(3_600),
        }
    }
}

/// Consent status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentStatus {
    NotRequested,
    Pending,
    Granted,
    Denied,
    Withdrawn,
}

/// Privacy-safe data wrapper
#[derive(Debug, Clone)]
pub struct PrivacyWrapped<T> {
    data: T,
    consent: ConsentStatus,
    created_at_us: i64,
    retention: RetentionPolicy,
}

impl<T> PrivacyWrapped<T> {
    pub fn new(data: T, consent: ConsentStatus, timestamp_us: i64) -> Self {
        Self {
            data,
            consent,
            created_at_us: timestamp_us,
            retention: RetentionPolicy::default(),
        }
    }

    /// Access data only if consent granted
    pub fn access(&self) -> Option<&T> {
        if self.consent == ConsentStatus::Granted {
            Some(&self.data)
        } else {
            None
        }
    }

    /// Check if data should be deleted
    pub fn should_delete(&self, current_us: i64) -> bool {
        let age_sec = (current_us - self.created_at_us) as u64 / 1_000_000;
        age_sec > self.retention.max_retention_sec
    }

    /// Check if data should be anonymized
    pub fn should_anonymize(&self, current_us: i64) -> bool {
        if let Some(anon_sec) = self.retention.anonymize_after_sec {
            let age_sec = (current_us - self.created_at_us) as u64 / 1_000_000;
            age_sec > anon_sec
        } else {
            false
        }
    }
}

/// Aggregate-only analytics (privacy-preserving)
#[derive(Debug, Clone, Default)]
pub struct AggregateAnalytics {
    pub sample_count: u64,
    pub avg_engagement: f32,
    pub avg_valence: f32,
    pub emotion_distribution: [u32; 7],
}

impl AggregateAnalytics {
    pub fn update(&mut self, engagement: f32, valence: f32, emotion_idx: usize) {
        self.sample_count += 1;

        let n = self.sample_count as f32;
        self.avg_engagement = self.avg_engagement * (n - 1.0) / n + engagement / n;
        self.avg_valence = self.avg_valence * (n - 1.0) / n + valence / n;

        if emotion_idx < 7 {
            self.emotion_distribution[emotion_idx] += 1;
        }
    }
}
