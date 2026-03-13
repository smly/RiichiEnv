#[derive(Debug, thiserror::Error)]
pub enum RiichiError {
    /// Parse error for tile strings or hand strings
    #[error("Parse error on '{input}': {message}")]
    Parse { input: String, message: String },
    /// Validation error for action construction or encoding
    #[error("Invalid action: {message}")]
    InvalidAction { message: String },
    /// Game state inconsistency (e.g. replay desync)
    #[error("Invalid state: {message}")]
    InvalidState { message: String },
    /// Serialization or deserialization failure
    #[error("Serialization error: {message}")]
    Serialization { message: String },
}

pub type RiichiResult<T> = Result<T, RiichiError>;

#[cfg(feature = "python")]
impl From<RiichiError> for pyo3::PyErr {
    fn from(err: RiichiError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
