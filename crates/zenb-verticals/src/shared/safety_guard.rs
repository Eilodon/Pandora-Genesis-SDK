//! Safety Guard Framework
//!
//! Ensures safe operation across all vertical modules.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Safety configuration
#[derive(Debug, Clone)]
pub struct SafetyConfig {
    /// Maximum requests per minute per user
    pub rate_limit_per_minute: u32,
    /// Nonce validity window (seconds)
    pub nonce_window_sec: u64,
    /// Minimum confidence for high-stakes decisions
    pub min_confidence_threshold: f32,
    /// Enable fail-safe mode
    pub fail_safe_enabled: bool,
    /// Default action on error
    pub default_on_error: SafeDefault,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDefault {
    /// Deny access on error (secure)
    Deny,
    /// Allow access on error (permissive)
    Allow,
    /// Require manual review
    ManualReview,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            rate_limit_per_minute: 60,
            nonce_window_sec: 300,
            min_confidence_threshold: 0.7,
            fail_safe_enabled: true,
            default_on_error: SafeDefault::Deny,
        }
    }
}

/// Safety Guard
pub struct SafetyGuard {
    config: SafetyConfig,
    request_counts: HashMap<String, (Instant, u32)>,
    used_nonces: HashMap<String, Instant>,
}

impl SafetyGuard {
    pub fn new() -> Self {
        Self::with_config(SafetyConfig::default())
    }

    pub fn with_config(config: SafetyConfig) -> Self {
        Self {
            config,
            request_counts: HashMap::new(),
            used_nonces: HashMap::new(),
        }
    }

    /// Check rate limit for user
    pub fn check_rate_limit(&mut self, user_id: &str) -> bool {
        let now = Instant::now();

        if let Some((start, count)) = self.request_counts.get_mut(user_id) {
            if now.duration_since(*start) > Duration::from_secs(60) {
                *start = now;
                *count = 1;
                true
            } else if *count >= self.config.rate_limit_per_minute {
                false
            } else {
                *count += 1;
                true
            }
        } else {
            self.request_counts.insert(user_id.to_string(), (now, 1));
            true
        }
    }

    /// Validate nonce (anti-replay)
    pub fn validate_nonce(&mut self, nonce: &str) -> bool {
        let now = Instant::now();

        self.used_nonces.retain(|_, t| {
            now.duration_since(*t) < Duration::from_secs(self.config.nonce_window_sec)
        });

        if self.used_nonces.contains_key(nonce) {
            false
        } else {
            self.used_nonces.insert(nonce.to_string(), now);
            true
        }
    }

    /// Check if confidence meets threshold for action
    pub fn meets_confidence_threshold(&self, confidence: f32) -> bool {
        confidence >= self.config.min_confidence_threshold
    }

    /// Get fail-safe default action
    pub fn get_fail_safe_action(&self) -> SafeDefault {
        if self.config.fail_safe_enabled {
            self.config.default_on_error
        } else {
            SafeDefault::Allow
        }
    }

    /// Validate request with all checks
    pub fn validate_request(
        &mut self,
        user_id: &str,
        nonce: &str,
        confidence: f32,
    ) -> Result<(), SafetyError> {
        if !self.check_rate_limit(user_id) {
            return Err(SafetyError::RateLimitExceeded);
        }

        if !self.validate_nonce(nonce) {
            return Err(SafetyError::NonceReused);
        }

        if !self.meets_confidence_threshold(confidence) {
            return Err(SafetyError::LowConfidence);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum SafetyError {
    RateLimitExceeded,
    NonceReused,
    LowConfidence,
    InvalidInput,
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self::new()
    }
}
