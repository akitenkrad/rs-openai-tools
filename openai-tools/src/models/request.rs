//! OpenAI Models API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Models API.
//! It allows you to list, retrieve, and delete models available in the OpenAI platform.
//!
//! # Key Features
//!
//! - **List Models**: Retrieve all available models
//! - **Retrieve Model**: Get details of a specific model
//! - **Delete Model**: Delete a fine-tuned model (only for models you own)
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::models::request::Models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let models = Models::new()?;
//!
//!     // List all available models
//!     let response = models.list().await?;
//!     for model in &response.data {
//!         println!("{}: owned by {}", model.id, model.owned_by);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::models::response::{DeleteResponse, Model, ModelsListResponse};
use dotenvy::dotenv;
use std::env;
use std::time::Duration;

const BASE_URL: &str = "https://api.openai.com/v1/models";

/// Client for interacting with the OpenAI Models API.
///
/// This struct provides methods to list, retrieve, and delete models.
/// Use [`Models::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::models::request::Models;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let models = Models::new()?;
///
///     // Get details of a specific model
///     let model = models.retrieve("gpt-4o-mini").await?;
///     println!("Model: {} (created: {})", model.id, model.created);
///
///     Ok(())
/// }
/// ```
pub struct Models {
    /// OpenAI API key for authentication
    api_key: String,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Models {
    /// Creates a new Models client.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Models)` - A new Models client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::models::request::Models;
    ///
    /// let models = Models::new().expect("API key should be set");
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
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use std::time::Duration;
    /// use openai_tools::models::request::Models;
    ///
    /// let mut models = Models::new().unwrap();
    /// models.timeout(Duration::from_secs(30));
    /// ```
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
            "User-Agent",
            request::header::HeaderValue::from_static("openai-tools-rust"),
        );
        Ok((client, headers))
    }

    /// Lists all available models.
    ///
    /// Returns a list of models that are currently available in the OpenAI API.
    ///
    /// # Returns
    ///
    /// * `Ok(ModelsListResponse)` - The list of available models
    /// * `Err(OpenAIToolError)` - If the request fails or response parsing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::models::request::Models;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let models = Models::new()?;
    ///     let response = models.list().await?;
    ///
    ///     println!("Found {} models", response.data.len());
    ///     for model in &response.data {
    ///         println!("- {}", model.id);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn list(&self) -> Result<ModelsListResponse> {
        let (client, headers) = self.create_client()?;

        let response = client
            .get(BASE_URL)
            .headers(headers)
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

        serde_json::from_str::<ModelsListResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Retrieves details of a specific model.
    ///
    /// Gets information about a model by its ID, including when it was created
    /// and who owns it.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to retrieve (e.g., "gpt-4o-mini")
    ///
    /// # Returns
    ///
    /// * `Ok(Model)` - The model details
    /// * `Err(OpenAIToolError)` - If the model is not found or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::models::request::Models;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let models = Models::new()?;
    ///     let model = models.retrieve("gpt-4o-mini").await?;
    ///
    ///     println!("Model: {}", model.id);
    ///     println!("Owned by: {}", model.owned_by);
    ///     println!("Created: {}", model.created);
    ///     Ok(())
    /// }
    /// ```
    pub async fn retrieve(&self, model_id: &str) -> Result<Model> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", BASE_URL, model_id);

        let response = client
            .get(&url)
            .headers(headers)
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

        serde_json::from_str::<Model>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Deletes a fine-tuned model.
    ///
    /// You must have the Owner role in your organization or be allowed to delete models.
    /// This only works for fine-tuned models that you have created.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the fine-tuned model to delete
    ///
    /// # Returns
    ///
    /// * `Ok(DeleteResponse)` - Confirmation of deletion
    /// * `Err(OpenAIToolError)` - If the model cannot be deleted or the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::models::request::Models;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let models = Models::new()?;
    ///
    ///     // Delete a fine-tuned model
    ///     let result = models.delete("ft:gpt-4o-mini:my-org:my-model:abc123").await?;
    ///     if result.deleted {
    ///         println!("Model {} was deleted", result.id);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn delete(&self, model_id: &str) -> Result<DeleteResponse> {
        let (client, headers) = self.create_client()?;
        let url = format!("{}/{}", BASE_URL, model_id);

        let response = client
            .delete(&url)
            .headers(headers)
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

        serde_json::from_str::<DeleteResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
