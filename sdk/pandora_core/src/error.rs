use thiserror::Error;

#[derive(Error, Debug)]
pub enum PandoraError {
    #[error("Deserialization failed: {0}")]
    Deserialize(String),

    #[error("GridWorld error: {0}")]
    GridWorld(String),

    #[error("Orchestrator error: {0}")]
    Orchestrator(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Invalid state: {0}")]
    State(String),

    #[error("Prediction failed: {0}")]
    PredictionFailed(String),

    #[error("Skill verification failed: {0}")]
    SkillVerificationFailed(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

pub type PResult<T> = Result<T, PandoraError>;

impl From<std::io::Error> for PandoraError {
    fn from(e: std::io::Error) -> Self {
        PandoraError::Io(e.to_string())
    }
}
