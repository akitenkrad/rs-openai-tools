//! OpenAI Embeddings API Request Module
//!
//! This module provides the functionality to build and send requests to the OpenAI Embeddings API.
//! It offers a builder pattern for constructing embedding requests, allowing you to convert text
//! into numerical vector representations that capture semantic meaning.
//!
//! # Key Features
//!
//! - **Builder Pattern**: Fluent API for constructing embedding requests
//! - **Single & Batch Input**: Support for single text or multiple texts at once
//! - **Encoding Formats**: Support for `float` and `base64` output formats
//! - **Error Handling**: Robust error management and validation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the embedding client
//!     let mut embedding = Embedding::new()?;
//!     
//!     // Generate embedding for a single text
//!     let response = embedding
//!         .model("text-embedding-3-small")
//!         .input_text("Hello, world!")
//!         .embed()
//!         .await?;
//!         
//!     let vector = response.data[0].embedding.as_1d().unwrap();
//!     println!("Embedding dimension: {}", vector.len());
//!     Ok(())
//! }
//! ```
//!
//! # Batch Processing
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!     
//!     // Embed multiple texts in a single request
//!     let texts = vec!["First text", "Second text", "Third text"];
//!     
//!     let response = embedding
//!         .model("text-embedding-3-small")
//!         .input_text_array(texts)
//!         .embed()
//!         .await?;
//!         
//!     for data in &response.data {
//!         println!("Index {}: {} dimensions",
//!                  data.index,
//!                  data.embedding.as_1d().unwrap().len());
//!     }
//!     Ok(())
//! }
//! ```

use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::common::models::EmbeddingModel;
use crate::embedding::response::Response;
use core::str;
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

/// Internal structure for handling input text in embedding requests.
///
/// This struct supports two input formats:
/// - Single text string (`input_text`)
/// - Array of text strings (`input_text_array`)
///
/// The custom `Serialize` implementation ensures proper JSON formatting
/// based on which input type is provided.
#[derive(Debug, Clone, Deserialize, Default)]
struct Input {
    /// Single input text for embedding
    #[serde(skip_serializing_if = "String::is_empty")]
    input_text: String,
    /// Array of input texts for batch embedding
    #[serde(skip_serializing_if = "Vec::is_empty")]
    input_text_array: Vec<String>,
}

impl Input {
    /// Creates an Input from a single text string.
    ///
    /// # Arguments
    ///
    /// * `input_text` - The text to embed
    ///
    /// # Returns
    ///
    /// A new `Input` instance with the single text set
    pub fn from_text(input_text: &str) -> Self {
        Self { input_text: input_text.to_string(), input_text_array: vec![] }
    }

    /// Creates an Input from an array of text strings.
    ///
    /// # Arguments
    ///
    /// * `input_text_array` - Vector of texts to embed
    ///
    /// # Returns
    ///
    /// A new `Input` instance with the text array set
    pub fn from_text_array(input_text_array: Vec<String>) -> Self {
        Self { input_text: String::new(), input_text_array }
    }
}

/// Custom serialization for Input to match OpenAI API format.
///
/// The OpenAI Embeddings API accepts either a single string or an array of strings
/// for the `input` field. This implementation serializes to the appropriate format
/// based on which field is populated.
impl Serialize for Input {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if !self.input_text.is_empty() && self.input_text_array.is_empty() {
            self.input_text.serialize(serializer)
        } else if self.input_text.is_empty() && !self.input_text_array.is_empty() {
            self.input_text_array.serialize(serializer)
        } else {
            // Default to empty string if both are empty
            "".serialize(serializer)
        }
    }
}

/// Request body structure for the OpenAI Embeddings API.
///
/// Contains all parameters that can be sent to the API endpoint.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct Body {
    /// The model to use for embedding generation
    model: EmbeddingModel,
    /// The input text(s) to embed
    input: Input,
    /// The format for the output embeddings ("float" or "base64")
    encoding_format: Option<String>,
}

/// Default API endpoint for Embeddings
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";

/// Main struct for building and sending embedding requests to the OpenAI API.
///
/// This struct provides a builder pattern interface for constructing embedding
/// requests with various parameters. Use [`Embedding::new()`] to create a new
/// instance, then chain methods to configure the request before calling [`embed()`].
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::embedding::request::Embedding;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut embedding = Embedding::new()?;
///
///     let response = embedding
///         .model("text-embedding-3-small")
///         .input_text("Sample text")
///         .embed()
///         .await?;
///
///     Ok(())
/// }
/// ```
pub struct Embedding {
    /// The API endpoint URL
    endpoint: String,
    /// OpenAI API key for authentication
    api_key: String,
    /// Request body containing model and input parameters
    body: Body,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Embedding {
    /// Creates a new Embedding instance.
    ///
    /// Initializes the embedding client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Embedding)` - A new embedding instance ready for configuration
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::embedding::request::Embedding;
    ///
    /// let embedding = Embedding::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").map_err(|e| OpenAIToolError::Error(format!("OPENAI_API_KEY not set in environment: {}", e)))?;
        let body = Body::default();
        Ok(Self { endpoint: DEFAULT_ENDPOINT.to_string(), api_key, body, timeout: None })
    }

    /// Sets a custom API endpoint URL
    ///
    /// Use this to point to alternative OpenAI-compatible APIs.
    ///
    /// # Arguments
    ///
    /// * `url` - The full URL of the API endpoint
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn base_url<T: AsRef<str>>(&mut self, url: T) -> &mut Self {
        self.endpoint = url.as_ref().to_string();
        self
    }

    /// Sets the model to use for embedding generation.
    ///
    /// # Arguments
    ///
    /// * `model` - The embedding model to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::embedding::request::Embedding;
    /// use openai_tools::common::models::EmbeddingModel;
    ///
    /// let mut embedding = Embedding::new().unwrap();
    /// embedding.model(EmbeddingModel::TextEmbedding3Small);
    /// ```
    pub fn model(&mut self, model: EmbeddingModel) -> &mut Self {
        self.body.model = model;
        self
    }

    /// Sets the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`] with `EmbeddingModel` enum for type safety.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The model identifier string (e.g., "text-embedding-3-small")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    #[deprecated(since = "0.2.0", note = "Use `model(EmbeddingModel)` instead for type safety")]
    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.body.model = EmbeddingModel::from(model_id.as_ref());
        self
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
    /// use openai_tools::embedding::request::Embedding;
    ///
    /// let mut embedding = Embedding::new().unwrap();
    /// embedding.model("text-embedding-3-small")
    ///     .timeout(Duration::from_secs(30));
    /// ```
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets a single text input for embedding.
    ///
    /// Use this method when you want to embed a single piece of text.
    /// For multiple texts, use [`input_text_array`] instead.
    ///
    /// # Arguments
    ///
    /// * `input_text` - The text to convert into an embedding vector
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// embedding.input_text("Hello, world!");
    /// ```
    pub fn input_text<T: AsRef<str>>(&mut self, input_text: T) -> &mut Self {
        self.body.input = Input::from_text(input_text.as_ref());
        self
    }

    /// Sets multiple text inputs for batch embedding.
    ///
    /// Use this method when you want to embed multiple texts in a single API call.
    /// This is more efficient than making separate requests for each text.
    ///
    /// # Arguments
    ///
    /// * `input_text_array` - Vector of texts to convert into embedding vectors
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// let texts = vec!["First text", "Second text", "Third text"];
    /// embedding.input_text_array(texts);
    /// ```
    pub fn input_text_array<T: AsRef<str>>(&mut self, input_text_array: Vec<T>) -> &mut Self {
        let input_strings = input_text_array.into_iter().map(|s| s.as_ref().to_string()).collect();
        self.body.input = Input::from_text_array(input_strings);
        self
    }

    /// Sets the encoding format for the output embeddings.
    ///
    /// # Arguments
    ///
    /// * `encoding_format` - Either "float" (default) or "base64"
    ///   - `"float"`: Returns embeddings as arrays of floating point numbers
    ///   - `"base64"`: Returns embeddings as base64-encoded strings (more compact)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Panics
    ///
    /// Panics if `encoding_format` is not "float" or "base64"
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # let mut embedding = Embedding::new().unwrap();
    /// embedding.encoding_format("float");
    /// ```
    pub fn encoding_format<T: AsRef<str>>(&mut self, encoding_format: T) -> &mut Self {
        let encoding_format = encoding_format.as_ref();
        assert!(encoding_format == "float" || encoding_format == "base64", "encoding_format must be either 'float' or 'base64'");
        self.body.encoding_format = Some(encoding_format.to_string());
        self
    }

    /// Sends the embedding request to the OpenAI API.
    ///
    /// This method validates the request parameters, constructs the HTTP request,
    /// sends it to the OpenAI Embeddings API endpoint, and parses the response.
    ///
    /// # Returns
    ///
    /// * `Ok(Response)` - The embedding response containing vectors and metadata
    /// * `Err(OpenAIToolError)` - If validation fails, the request fails, or parsing fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API key is not set
    /// - Model ID is not set
    /// - Input text is not set
    /// - Network request fails
    /// - Response parsing fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::embedding::request::Embedding;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut embedding = Embedding::new()?;
    /// let response = embedding
    ///     .model("text-embedding-3-small")
    ///     .input_text("Hello, world!")
    ///     .embed()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn embed(&self) -> Result<Response> {
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        // Note: Model defaults to EmbeddingModel::TextEmbedding3Small, so no need to check for empty
        if self.body.input.input_text.is_empty() && self.body.input.input_text_array.is_empty() {
            return Err(OpenAIToolError::Error("Input text is not set.".into()));
        }

        let body = serde_json::to_string(&self.body)?;

        let client = create_http_client(self.timeout)?;
        let mut header = request::header::HeaderMap::new();
        header.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        header.insert("Authorization", request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap());
        header.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));

        if cfg!(test) {
            // Replace API key with a placeholder in debug mode
            let body_for_debug = serde_json::to_string_pretty(&self.body).unwrap().replace(&self.api_key, "*************");
            tracing::info!("Request body: {}", body_for_debug);
        }

        let response = client.post(&self.endpoint).headers(header).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
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

        serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
