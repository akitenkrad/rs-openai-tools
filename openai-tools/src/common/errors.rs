use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpenAIToolError {
    #[error("Request error: {0}")]
    RequestError(#[from] request::Error),
    #[error("JSON serialization/deserialization error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Error from anyhow: {0}")]
    AnyhowError(#[from] anyhow::Error),
    #[error("Error: {0}")]
    Error(String),
}

pub type Result<T> = std::result::Result<T, OpenAIToolError>;
