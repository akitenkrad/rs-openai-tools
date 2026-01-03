use serde::Deserialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OpenAIToolError {
    #[error("Request error: {0}")]
    RequestError(#[from] request::Error),
    #[error("JSON serialization/deserialization error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("Error from anyhow: {0}")]
    AnyhowError(#[from] anyhow::Error),
    #[error("WebSocket error: {0}")]
    WebSocketError(String),
    #[error("Realtime API error: {code} - {message}")]
    RealtimeError { code: String, message: String },
    #[error("Error: {0}")]
    Error(String),
}

pub type Result<T> = std::result::Result<T, OpenAIToolError>;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ErrorMessage {
    pub message: Option<String>,
    pub type_name: Option<String>,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorMessage,
}
