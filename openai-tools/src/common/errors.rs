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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_api_key"
            }
        }"#;

        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            error_resp.error.message,
            Some("Incorrect API key provided".to_string())
        );
        assert_eq!(
            error_resp.error.code,
            Some("invalid_api_key".to_string())
        );
    }

    #[test]
    fn test_error_response_rate_limit() {
        let json = r#"{
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "param": null,
                "code": "rate_limit_exceeded"
            }
        }"#;

        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            error_resp.error.message,
            Some("Rate limit exceeded".to_string())
        );
        assert_eq!(
            error_resp.error.code,
            Some("rate_limit_exceeded".to_string())
        );
    }

    #[test]
    fn test_error_response_with_missing_fields() {
        let json = r#"{"error": {"message": "Error occurred"}}"#;

        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            error_resp.error.message,
            Some("Error occurred".to_string())
        );
        assert!(error_resp.error.code.is_none());
        assert!(error_resp.error.param.is_none());
        assert!(error_resp.error.type_name.is_none());
    }

    #[test]
    fn test_error_response_empty_error() {
        let json = r#"{"error": {}}"#;

        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert!(error_resp.error.message.is_none());
        assert!(error_resp.error.code.is_none());
    }

    #[test]
    fn test_error_response_with_param() {
        let json = r#"{
            "error": {
                "message": "Invalid value for 'model'",
                "type": "invalid_request_error",
                "param": "model",
                "code": null
            }
        }"#;

        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            error_resp.error.message,
            Some("Invalid value for 'model'".to_string())
        );
        assert_eq!(error_resp.error.param, Some("model".to_string()));
        assert!(error_resp.error.code.is_none());
    }

    #[test]
    fn test_error_message_unwrap_or_default() {
        // Test that unwrap_or_default works correctly for None case
        let json = r#"{"error": {}}"#;
        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        let message = error_resp.error.message.unwrap_or_default();
        assert_eq!(message, "");

        // Test that unwrap_or_default works correctly for Some case
        let json = r#"{"error": {"message": "Test error"}}"#;
        let error_resp: ErrorResponse = serde_json::from_str(json).unwrap();
        let message = error_resp.error.message.unwrap_or_default();
        assert_eq!(message, "Test error");
    }
}
