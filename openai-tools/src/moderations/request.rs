//! OpenAI Moderations API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Moderations API.
//! It allows you to classify text inputs to determine if they violate content policies.
//!
//! # Key Features
//!
//! - **Single Text Moderation**: Check a single text string
//! - **Batch Moderation**: Check multiple texts at once
//! - **Model Selection**: Choose between omni-moderation and text-moderation models
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::moderations::request::Moderations;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let moderations = Moderations::new()?;
//!
//!     // Check a text for policy violations
//!     let response = moderations.moderate_text("Hello, world!", None).await?;
//!     if response.results[0].flagged {
//!         println!("Content was flagged!");
//!     } else {
//!         println!("Content is safe");
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::moderations::response::ModerationResponse;
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

const BASE_URL: &str = "https://api.openai.com/v1/moderations";

/// Moderation model options.
///
/// The model to use for content moderation. Newer omni-moderation models
/// support more categorization options and multi-modal inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModerationModel {
    /// Latest omni-moderation model with multi-modal support
    #[serde(rename = "omni-moderation-latest")]
    OmniModerationLatest,
    /// Legacy text-only moderation model
    #[serde(rename = "text-moderation-latest")]
    TextModerationLatest,
}

impl Default for ModerationModel {
    fn default() -> Self {
        Self::OmniModerationLatest
    }
}

impl ModerationModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OmniModerationLatest => "omni-moderation-latest",
            Self::TextModerationLatest => "text-moderation-latest",
        }
    }
}

impl std::fmt::Display for ModerationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Request payload for moderation endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModerationRequest {
    /// The input to classify
    input: ModerationInput,
    /// The model to use for classification
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

/// Input types for moderation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ModerationInput {
    /// Single text string
    Single(String),
    /// Multiple text strings
    Multiple(Vec<String>),
}

/// Client for interacting with the OpenAI Moderations API.
///
/// This struct provides methods to classify text content for potential
/// policy violations. Use [`Moderations::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::moderations::request::{Moderations, ModerationModel};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let moderations = Moderations::new()?;
///
///     // Check content with a specific model
///     let response = moderations
///         .moderate_text("Some text to check", Some(ModerationModel::OmniModerationLatest))
///         .await?;
///
///     for result in &response.results {
///         println!("Flagged: {}", result.flagged);
///     }
///
///     Ok(())
/// }
/// ```
pub struct Moderations {
    /// OpenAI API key for authentication
    api_key: String,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Moderations {
    /// Creates a new Moderations client.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Moderations)` - A new Moderations client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::moderations::request::Moderations;
    ///
    /// let moderations = Moderations::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").map_err(|e| {
            OpenAIToolError::Error(format!("OPENAI_API_KEY not set in environment: {}", e))
        })?;
        Ok(Self { api_key, timeout: None })
    }

    /// Sets the request timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for a response
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Creates the HTTP client with default headers.
    fn create_client(&self) -> Result<(request::Client, request::header::HeaderMap)> {
        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        headers.insert(
            "Content-Type",
            request::header::HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "User-Agent",
            request::header::HeaderValue::from_static("openai-tools-rust"),
        );
        Ok((client, headers))
    }

    /// Moderates a single text string.
    ///
    /// Classifies the input text to determine if it violates OpenAI's content policy.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to moderate
    /// * `model` - Optional model to use (defaults to `omni-moderation-latest`)
    ///
    /// # Returns
    ///
    /// * `Ok(ModerationResponse)` - The moderation results
    /// * `Err(OpenAIToolError)` - If the request fails or response parsing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::moderations::request::Moderations;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let moderations = Moderations::new()?;
    ///     let response = moderations.moderate_text("Hello, world!", None).await?;
    ///
    ///     let result = &response.results[0];
    ///     if result.flagged {
    ///         println!("Content was flagged!");
    ///         println!("Hate score: {}", result.category_scores.hate);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn moderate_text(
        &self,
        text: &str,
        model: Option<ModerationModel>,
    ) -> Result<ModerationResponse> {
        let request_body = ModerationRequest {
            input: ModerationInput::Single(text.to_string()),
            model: model.map(|m| m.as_str().to_string()),
        };

        self.send_request(&request_body).await
    }

    /// Moderates multiple text strings.
    ///
    /// Classifies multiple input texts in a single request.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of text strings to moderate
    /// * `model` - Optional model to use (defaults to `omni-moderation-latest`)
    ///
    /// # Returns
    ///
    /// * `Ok(ModerationResponse)` - The moderation results (one result per input)
    /// * `Err(OpenAIToolError)` - If the request fails or response parsing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::moderations::request::Moderations;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let moderations = Moderations::new()?;
    ///     let texts = vec![
    ///         "First text to check".to_string(),
    ///         "Second text to check".to_string(),
    ///     ];
    ///     let response = moderations.moderate_texts(texts, None).await?;
    ///
    ///     for (i, result) in response.results.iter().enumerate() {
    ///         println!("Text {}: flagged = {}", i + 1, result.flagged);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn moderate_texts(
        &self,
        texts: Vec<String>,
        model: Option<ModerationModel>,
    ) -> Result<ModerationResponse> {
        let request_body = ModerationRequest {
            input: ModerationInput::Multiple(texts),
            model: model.map(|m| m.as_str().to_string()),
        };

        self.send_request(&request_body).await
    }

    /// Sends the moderation request to the API.
    async fn send_request(&self, request_body: &ModerationRequest) -> Result<ModerationResponse> {
        let (client, headers) = self.create_client()?;

        let body =
            serde_json::to_string(request_body).map_err(OpenAIToolError::SerdeJsonError)?;

        let response = client
            .post(BASE_URL)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<ModerationResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
