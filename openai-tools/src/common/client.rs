//! HTTP client utilities for OpenAI API requests.
//!
//! This module provides helper functions for creating HTTP clients with
//! configurable timeout settings.

use crate::common::errors::{OpenAIToolError, Result};
use std::time::Duration;

/// Creates an HTTP client with optional timeout configuration.
///
/// This function creates a `reqwest::Client` instance with an optional request
/// timeout. If no timeout is specified, the client will have no timeout limit
/// (the default reqwest behavior).
///
/// # Arguments
///
/// * `timeout` - Optional request timeout duration. If `None`, no timeout is set.
///
/// # Returns
///
/// * `Ok(request::Client)` - A configured HTTP client
/// * `Err(OpenAIToolError)` - If client creation fails
///
/// # Example
///
/// ```rust
/// use std::time::Duration;
/// use openai_tools::common::client::create_http_client;
///
/// // With timeout (30 seconds)
/// let client = create_http_client(Some(Duration::from_secs(30))).unwrap();
///
/// // Without timeout (default behavior)
/// let client = create_http_client(None).unwrap();
/// ```
pub fn create_http_client(timeout: Option<Duration>) -> Result<request::Client> {
    let mut builder = request::Client::builder();

    if let Some(duration) = timeout {
        builder = builder.timeout(duration);
    }

    builder.build().map_err(|e| OpenAIToolError::Error(format!("Failed to create HTTP client: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_http_client_without_timeout() {
        let result = create_http_client(None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_http_client_with_timeout() {
        let result = create_http_client(Some(Duration::from_secs(30)));
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_http_client_with_millisecond_timeout() {
        let result = create_http_client(Some(Duration::from_millis(500)));
        assert!(result.is_ok());
    }
}
