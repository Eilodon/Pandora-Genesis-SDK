use thiserror::Error;

mod context;
pub use context::ErrorContext;

#[derive(Error, Debug)]
pub enum PandoraError {
    // === Configuration Errors ===
    #[error("Configuration error: {message}")]
    Config {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Missing required configuration: {field}")]
    MissingConfig { field: String },

    // === Skill Execution Errors ===
    #[error("Skill '{skill_name}' execution failed: {message}")]
    SkillExecution {
        skill_name: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Skill '{skill_name}' not found")]
    SkillNotFound { skill_name: String },

    #[error("Invalid skill input for '{skill_name}': {message}")]
    InvalidSkillInput { skill_name: String, message: String },

    // === Timeout & Circuit Breaker Errors ===
    #[error("Timeout while executing '{operation}' after {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },

    #[error("Circuit breaker open for '{resource}'")]
    CircuitOpen { resource: String },

    #[error("Rate limit exceeded for '{resource}': {limit} requests per {window_secs}s")]
    RateLimitExceeded {
        resource: String,
        limit: u64,
        window_secs: u64,
    },

    // === Resource Errors ===
    #[error("Resource '{resource}' not found")]
    ResourceNotFound { resource: String },
    #[error("Resource '{resource}' already exists")]
    ResourceExists { resource: String },
    #[error("Insufficient resources: {message}")]
    InsufficientResources { message: String },

    // === I/O Errors ===
    #[error("I/O error: {message}")]
    Io {
        message: String,
        #[source]
        source: std::io::Error,
    },

    // === Serialization Errors ===
    #[error("Serialization error: {message}")]
    Serialization {
        message: String,
        #[source]
        source: Option<serde_json::Error>,
    },

    // === FFI Errors ===
    #[error("FFI interface error: {message}")]
    Ffi {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    // === Unknown/Catch-all ===
    #[error("Unknown error: {0}")]
    Unknown(String),

    // === Internal Errors ===
    #[error("Internal error: {message} (this is a bug, please report)")]
    Internal { message: String },
}

impl PandoraError {
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
            source: None,
        }
    }

    pub fn config_with_source<S, E>(message: S, source: E) -> Self
    where
        S: Into<String>,
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Config {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn skill_exec<S1, S2>(skill_name: S1, message: S2) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
    {
        Self::SkillExecution {
            skill_name: skill_name.into(),
            message: message.into(),
            source: None,
        }
    }

    pub fn skill_exec_with_source<S1, S2, E>(skill_name: S1, message: S2, source: E) -> Self
    where
        S1: Into<String>,
        S2: Into<String>,
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::SkillExecution {
            skill_name: skill_name.into(),
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "CONFIG_ERROR",
            Self::MissingConfig { .. } => "MISSING_CONFIG",
            Self::SkillExecution { .. } => "SKILL_EXEC_ERROR",
            Self::SkillNotFound { .. } => "SKILL_NOT_FOUND",
            Self::InvalidSkillInput { .. } => "INVALID_SKILL_INPUT",
            Self::Timeout { .. } => "TIMEOUT",
            Self::CircuitOpen { .. } => "CIRCUIT_OPEN",
            Self::RateLimitExceeded { .. } => "RATE_LIMIT_EXCEEDED",
            Self::ResourceNotFound { .. } => "RESOURCE_NOT_FOUND",
            Self::ResourceExists { .. } => "RESOURCE_EXISTS",
            Self::InsufficientResources { .. } => "INSUFFICIENT_RESOURCES",
            Self::Io { .. } => "IO_ERROR",
            Self::Serialization { .. } => "SERIALIZATION_ERROR",
            Self::Ffi { .. } => "FFI_ERROR",
            Self::Unknown(_) => "UNKNOWN_ERROR",
            Self::Internal { .. } => "INTERNAL_ERROR",
        }
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. }
                | Self::CircuitOpen { .. }
                | Self::Io { .. }
                | Self::InsufficientResources { .. }
        )
    }

    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. }
                | Self::RateLimitExceeded { .. }
                | Self::InsufficientResources { .. }
        )
    }
}

impl From<std::io::Error> for PandoraError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            source: err,
        }
    }
}

impl From<serde_json::Error> for PandoraError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            message: err.to_string(),
            source: Some(err),
        }
    }
}
