use std::fmt;

#[derive(Debug)]
pub enum RiichiError {
    /// 牌文字列・手牌文字列のパースエラー
    Parse { input: String, message: String },
    /// アクション構成・エンコードのバリデーションエラー
    InvalidAction { message: String },
    /// ゲーム状態の不整合（リプレイ同期ずれ等）
    InvalidState { message: String },
    /// シリアライズ/デシリアライズの失敗
    Serialization { message: String },
}

impl fmt::Display for RiichiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiichiError::Parse { input, message } => {
                write!(f, "Parse error on '{}': {}", input, message)
            }
            RiichiError::InvalidAction { message } => {
                write!(f, "Invalid action: {}", message)
            }
            RiichiError::InvalidState { message } => {
                write!(f, "Invalid state: {}", message)
            }
            RiichiError::Serialization { message } => {
                write!(f, "Serialization error: {}", message)
            }
        }
    }
}

impl std::error::Error for RiichiError {}

pub type RiichiResult<T> = Result<T, RiichiError>;

#[cfg(feature = "python")]
impl From<RiichiError> for pyo3::PyErr {
    fn from(err: RiichiError) -> pyo3::PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}
