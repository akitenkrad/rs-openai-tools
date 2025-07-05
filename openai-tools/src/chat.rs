//! # Chat Completion Module
//!
//! This module provides functionality for interacting with OpenAI's Chat Completion API.
//! It includes structures for request configuration, response handling, and a high-level
//! client for making chat completion requests.
//!
//! ## Features
//!
//! - **Simple Chat**: Basic text-to-text conversations
//! - **Structured Output**: JSON schema-based responses for structured data
//! - **Flexible Configuration**: Support for all OpenAI chat completion parameters
//! - **Type Safety**: Strongly typed request and response structures
//!
//! ## Examples
//!
//! ### Basic Usage
//!
//! ```rust,no_run
//! use openai_tools::chat::ChatCompletion;
//! use openai_tools::common::{Message, Role};
//! use openai_tools::errors::Result;
//!
//! # async fn example() -> Result<()> {
//! let mut chat = ChatCompletion::new();
//! let messages = vec![
//!     Message::from_string(Role::User, "Hello, world!".to_string())
//! ];
//!
//! chat.model_id("gpt-4o-mini".to_string())
//!     .messages(messages)
//!     .temperature(0.7);
//!
//! let response = chat.chat().await?;
//! println!("{}", response.choices[0].message.content);
//! # Ok(())
//! # }
//! ```
//!
//! ### Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::chat::{ChatCompletion, ChatCompletionResponseFormat};
//! use openai_tools::structured_output::Schema;
//! use openai_tools::common::{Message, Role};
//! use openai_tools::errors::Result;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct WeatherInfo {
//!     location: String,
//!     temperature: f32,
//!     condition: String,
//! }
//!
//! # async fn example() -> Result<()> {
//! let mut schema = Schema::chat_json_schema("weather".to_string());
//! schema.add_property("location".to_string(), "string".to_string(), None);
//! schema.add_property("temperature".to_string(), "number".to_string(), None);
//! schema.add_property("condition".to_string(), "string".to_string(), None);
//!
//! let mut chat = ChatCompletion::new();
//! chat.model_id("gpt-4o-mini".to_string())
//!     .messages(vec![Message::from_string(Role::User, "What's the weather in Tokyo?".to_string())])
//!     .response_format(ChatCompletionResponseFormat::new("json_schema".to_string(), schema));
//!
//! let response = chat.chat().await?;
//! let weather: WeatherInfo = serde_json::from_str(&response.choices[0].message.content)?;
//! # Ok(())
//! # }
//! ```

use crate::common::{Message, Usage};
use crate::errors::{OpenAIToolError, Result};
use crate::structured_output::Schema;
use core::str;
use derive_new::new;
use dotenvy::dotenv;
use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::env;

/// Specifies the format that the model must output for chat completions.
///
/// This structure is used to configure structured outputs, particularly for JSON schema-based
/// responses that ensure the model returns data in a specific format.
///
/// # Fields
///
/// * `type_name` - The type of response format (e.g., "json_schema")
/// * `json_schema` - The JSON schema definition for structured output
///
/// # Example
///
/// ```rust
/// use openai_tools::chat::ChatCompletionResponseFormat;
/// use openai_tools::structured_output::Schema;
///
/// let mut schema = Schema::chat_json_schema("person".to_string());
/// schema.add_property("name".to_string(), "string".to_string(), Some("Person's name".to_string()));
/// schema.add_property("age".to_string(), "integer".to_string(), Some("Person's age".to_string()));
///
/// let format = ChatCompletionResponseFormat::new("json_schema".to_string(), schema);
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponseFormat {
    #[serde(rename = "type")]
    pub type_name: String,
    pub json_schema: Schema,
}

impl ChatCompletionResponseFormat {
    /// Creates a new `ChatCompletionResponseFormat` instance.
    ///
    /// # Arguments
    ///
    /// * `type_name` - The type of response format (typically "json_schema")
    /// * `json_schema` - The JSON schema definition for structured output
    ///
    /// # Returns
    ///
    /// A new `ChatCompletionResponseFormat` instance configured with the specified parameters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::chat::ChatCompletionResponseFormat;
    /// use openai_tools::structured_output::Schema;
    ///
    /// let schema = Schema::chat_json_schema("example".to_string());
    /// let format = ChatCompletionResponseFormat::new("json_schema".to_string(), schema);
    /// ```
    pub fn new(type_name: String, json_schema: Schema) -> Self {
        Self {
            type_name,
            json_schema,
        }
    }
}

/// Request body structure for OpenAI Chat Completion API.
///
/// This structure contains all the parameters that can be sent to the OpenAI Chat Completion
/// endpoint. Most fields are optional and will be omitted from the JSON if not set.
///
/// # Fields
///
/// * `model` - ID of the model to use (required)
/// * `messages` - List of conversation messages (required)
/// * `store` - Whether to store the output for the user
/// * `frequency_penalty` - Penalty for token frequency (-2.0 to 2.0)
/// * `logit_bias` - Modify likelihood of specified tokens
/// * `logprobs` - Whether to return log probabilities
/// * `top_logprobs` - Number of most likely tokens to return (0-20)
/// * `max_completion_tokens` - Maximum tokens to generate
/// * `n` - Number of completion choices to generate
/// * `modalities` - Output types the model should generate
/// * `presence_penalty` - Penalty for token presence (-2.0 to 2.0)
/// * `temperature` - Sampling temperature (0-2)
/// * `response_format` - Output format specification
#[derive(Debug, Clone, Deserialize, Serialize, Default, new)]
pub struct ChatCompletionRequestBody {
    /// ID of the model to use. (https://platform.openai.com/docs/models#model-endpoint-compatibility)
    pub model: String,
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<Message>,
    /// Whether or not to store the output of this chat completion request for user. false by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// -2.0 ~ 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens to an associated bias value from 100 to 100.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<FxHashMap<String, i32>>,
    /// Whether to return log probabilities of the output tokens or not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// 0 ~ 20. Specify the number of most likely tokens to return at each token position, each with an associated log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// An upper bound for the number of tokens that can be generated for a completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    /// How many chat completion choices to generate for each input message. 1 by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Output types that you would like the model to generate for this request. ["text"] for most models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    /// -2.0 ~ 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// 0 ~ 2. What sampling temperature to use. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An object specifying the format that the model must output. (https://platform.openai.com/docs/guides/structured-outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatCompletionResponseFormat>,
}

/// Represents a message in the chat completion response.
///
/// This structure contains the message content and metadata returned by the AI model
/// as part of a chat completion response. It includes the actual text content,
/// the role of the message sender, and optional fields for refusal and annotations.
///
/// # Fields
///
/// * `content` - The text content of the message generated by the AI
/// * `role` - The role of the message sender (typically "assistant" for AI responses)
/// * `refusal` - Optional refusal message if the AI declined to respond to the request
/// * `annotations` - Optional annotations or metadata associated with the message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatResponseMessage {
    pub content: String,
    pub role: String,
    pub refusal: Option<String>,
    pub annotations: Option<Vec<String>>,
}

/// Represents a single choice in the chat completion response.
///
/// Each chat completion request can generate multiple choices (controlled by the `n` parameter).
/// This structure represents one such choice with its associated message and metadata.
///
/// # Fields
///
/// * `index` - The index of this choice in the list of generated choices
/// * `message` - The generated message content
/// * `finish_reason` - The reason why the generation stopped
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

/// Complete response from the OpenAI Chat Completion API.
///
/// This structure contains all the information returned by a successful chat completion
/// request, including metadata about the request and the generated choices.
///
/// # Fields
///
/// * `id` - Unique identifier for the chat completion
/// * `object` - Object type, always "chat.completion"
/// * `created` - Unix timestamp of when the completion was created
/// * `model` - The model used for the completion
/// * `system_fingerprint` - System fingerprint for the completion
/// * `choices` - List of generated completion choices
/// * `usage` - Token usage statistics for the request
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// High-level client for OpenAI Chat Completion API.
///
/// This struct provides a convenient interface for making chat completion requests
/// to the OpenAI API. It handles API key management, request building, and response
/// parsing automatically.
///
/// # Features
///
/// - **Fluent API**: Builder pattern for easy configuration
/// - **Automatic Authentication**: Reads API key from environment variables
/// - **Error Handling**: Comprehensive error messages for common issues
/// - **Type Safety**: Strongly typed parameters and responses
///
/// # Environment Variables
///
/// The client requires the `OPENAI_API_KEY` environment variable to be set.
/// This can be done through a `.env` file or system environment variables.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::chat::ChatCompletion;
/// use openai_tools::common::{Message, Role};
///
/// # async fn example() -> anyhow::Result<()> {
/// let mut chat = ChatCompletion::new();
///
/// let response = chat
///     .model_id("gpt-4o-mini".to_string())
///     .messages(vec![Message::from_string(Role::User, "Hello!".to_string())])
///     .temperature(0.7)
///     .chat()
///     .await?;
///
/// println!("{}", response.choices[0].message.content);
/// # Ok(())
/// # }
/// ```
pub struct ChatCompletion {
    api_key: String,
    pub request_body: ChatCompletionRequestBody,
}

impl ChatCompletion {
    /// Creates a new `ChatCompletion` client instance.
    ///
    /// This constructor automatically loads environment variables from a `.env` file
    /// (if present) and retrieves the OpenAI API key from the `OPENAI_API_KEY`
    /// environment variable.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Returns
    ///
    /// A new `ChatCompletion` instance ready for configuration and use.
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        return Self {
            api_key,
            request_body: ChatCompletionRequestBody::default(),
        };
    }

    /// Sets the model ID for the chat completion request.
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use openai_tools::chat::ChatCompletion;
    /// let mut chat = ChatCompletion::new();
    /// chat.model_id("gpt-4o-mini".to_string());
    /// ```
    pub fn model_id(&mut self, model_id: String) -> &mut Self {
        self.request_body.model = String::from(model_id);
        return self;
    }

    /// Sets the conversation messages for the chat completion request.
    ///
    /// # Arguments
    ///
    /// * `messages` - A vector of `Message` objects representing the conversation history
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use openai_tools::chat::ChatCompletion;
    /// # use openai_tools::common::{Message, Role};
    /// let mut chat = ChatCompletion::new();
    /// let messages = vec![
    ///     Message::from_string(Role::System, "You are a helpful assistant.".to_string()),
    ///     Message::from_string(Role::User, "Hello!".to_string()),
    /// ];
    /// chat.messages(messages);
    /// ```
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages = messages;
        return self;
    }

    /// Sets whether to store the output of this chat completion request.
    ///
    /// # Arguments
    ///
    /// * `store` - Whether to store the output for the user
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn store(&mut self, store: bool) -> &mut Self {
        self.request_body.store = Option::from(store);
        return self;
    }

    /// Sets the frequency penalty for the chat completion request.
    ///
    /// Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Number between -2.0 and 2.0
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use openai_tools::chat::ChatCompletion;
    /// let mut chat = ChatCompletion::new();
    /// chat.frequency_penalty(0.5); // Reduce repetition
    /// ```
    pub fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.request_body.frequency_penalty = Option::from(frequency_penalty);
        return self;
    }

    /// Sets logit bias to modify the likelihood of specified tokens appearing.
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - Maps tokens (as strings) to bias values from -100 to 100
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn logit_bias(&mut self, logit_bias: FxHashMap<String, i32>) -> &mut Self {
        self.request_body.logit_bias = Option::from(logit_bias);
        return self;
    }

    /// Sets whether to return log probabilities of the output tokens.
    ///
    /// # Arguments
    ///
    /// * `logprobs` - Whether to return log probabilities
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn logprobs(&mut self, logprobs: bool) -> &mut Self {
        self.request_body.logprobs = Option::from(logprobs);
        return self;
    }

    /// Sets the number of most likely tokens to return at each token position.
    ///
    /// # Arguments
    ///
    /// * `top_logprobs` - Number between 0 and 20
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn top_logprobs(&mut self, top_logprobs: u8) -> &mut Self {
        self.request_body.top_logprobs = Option::from(top_logprobs);
        return self;
    }

    /// Sets the maximum number of tokens that can be generated for a completion.
    ///
    /// # Arguments
    ///
    /// * `max_completion_tokens` - Maximum number of tokens to generate
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.request_body.max_completion_tokens = Option::from(max_completion_tokens);
        return self;
    }

    /// Sets how many chat completion choices to generate for each input message.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of choices to generate (default is 1)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn n(&mut self, n: u32) -> &mut Self {
        self.request_body.n = Option::from(n);
        return self;
    }

    /// Sets the output types that the model should generate for this request.
    ///
    /// # Arguments
    ///
    /// * `modalities` - Vector of output types (e.g., ["text"] for most models)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn modalities(&mut self, modalities: Vec<String>) -> &mut Self {
        self.request_body.modalities = Option::from(modalities);
        return self;
    }

    /// Sets the presence penalty for the chat completion request.
    ///
    /// Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Number between -2.0 and 2.0
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        self.request_body.presence_penalty = Option::from(presence_penalty);
        return self;
    }

    /// Sets the sampling temperature for the chat completion request.
    ///
    /// Higher values like 0.8 will make the output more random, while lower values
    /// like 0.2 will make it more focused and deterministic.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Number between 0 and 2
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        self.request_body.temperature = Option::from(temperature);
        return self;
    }

    /// Sets the response format for structured outputs.
    ///
    /// This is used to ensure the model returns data in a specific JSON schema format,
    /// enabling structured data extraction from the model's responses.
    ///
    /// # Arguments
    ///
    /// * `response_format` - The response format specification including JSON schema
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    pub fn response_format(&mut self, response_format: ChatCompletionResponseFormat) -> &mut Self {
        self.request_body.response_format = Option::from(response_format);
        return self;
    }

    /// Executes the chat completion request.
    ///
    /// This method sends the configured request to the OpenAI Chat Completion API
    /// and returns the response. It performs validation on required fields before
    /// making the request.
    ///
    /// # Returns
    ///
    /// * `Ok(ChatCompletionResponse)` - The successful response from the API
    /// * `Err(anyhow::Error)` - An error if the request fails or validation fails
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The API key is not set or empty
    /// - The model ID is not set or empty
    /// - No messages have been provided
    /// - The HTTP request fails
    /// - The response cannot be parsed as JSON
    /// - The API returns a non-success status code
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::chat::ChatCompletion;
    /// # use openai_tools::common::{Message, Role};
    /// # use openai_tools::errors::Result;
    /// # async fn example() -> Result<()> {
    /// let mut chat = ChatCompletion::new();
    ///
    /// let response = chat
    ///     .model_id("gpt-4o-mini".to_string())
    ///     .messages(vec![Message::from_string(Role::User, "Hello!".to_string())])
    ///     .chat()
    ///     .await?;
    ///
    /// println!("Response: {}", response.choices[0].message.content);
    /// println!("Tokens used: {}", response.usage.total_tokens.unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn chat(&mut self) -> Result<ChatCompletionResponse> {
        // Check if the API key is set & body is built.
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        if self.request_body.model.is_empty() {
            return Err(OpenAIToolError::Error("Model ID is not set.".into()));
        }
        if self.request_body.messages.is_empty() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        }

        let body = serde_json::to_string(&self.request_body)?;
        let url = "https://api.openai.com/v1/chat/completions";

        let client = request::Client::new();
        let mut header = request::header::HeaderMap::new();
        header.insert(
            "Content-Type",
            request::header::HeaderValue::from_static("application/json"),
        );
        header.insert(
            "Authorization",
            request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap(),
        );
        header.insert(
            "User-Agent",
            request::header::HeaderValue::from_static("openai-tools-rust/0.1.0"),
        );
        let response = client
            .post(url)
            .headers(header)
            .body(body)
            .send()
            .await
            .map_err(|e| OpenAIToolError::RequestError(e))?;
        let content = response
            .text()
            .await
            .map_err(|e| OpenAIToolError::RequestError(e))?;

        serde_json::from_str::<ChatCompletionResponse>(&content)
            .map_err(|e| OpenAIToolError::SerdeJsonError(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Role;
    use serde::{Deserialize, Serialize};
    use serde_json;
    use std::sync::Once;
    use tracing_subscriber::EnvFilter;

    static INIT: Once = Once::new();

    fn init_tracing() {
        INIT.call_once(|| {
            // `RUST_LOG` 環境変数があればそれを使い、なければ "info"
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer() // `cargo test` / nextest 用
                .init();
        });
    }
    #[tokio::test]
    async fn test_chat_completion() {
        init_tracing();
        let mut chat = ChatCompletion::new();
        let messages = vec![Message::from_string(Role::User, String::from("Hi there!"))];

        chat.model_id(String::from("gpt-4o-mini"))
            .messages(messages)
            .temperature(1.0);

        let mut counter = 3;
        loop {
            match chat.chat().await {
                Ok(response) => {
                    tracing::info!("{}", &response.choices[0].message.content);
                    assert!(true);
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[tokio::test]
    async fn test_chat_completion_2() {
        init_tracing();
        let mut chat = ChatCompletion::new();
        let messages = vec![Message::from_string(
            Role::User,
            String::from("トンネルを抜けると？"),
        )];

        chat.model_id(String::from("gpt-4o-mini"))
            .messages(messages)
            .temperature(1.5);

        let mut counter = 3;
        loop {
            match chat.chat().await {
                Ok(response) => {
                    println!("{}", &response.choices[0].message.content);
                    assert!(true);
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct Weather {
        #[serde(default = "String::new")]
        location: String,
        #[serde(default = "String::new")]
        date: String,
        #[serde(default = "String::new")]
        weather: String,
        #[serde(default = "String::new")]
        error: String,
    }

    #[tokio::test]
    async fn test_chat_completion_with_json_schema() {
        init_tracing();
        let mut openai = ChatCompletion::new();
        let messages = vec![Message::from_string(
            Role::User,
            String::from(
                "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.",
            ),
        )];

        let mut json_schema = Schema::chat_json_schema(String::from("weather"));
        json_schema.add_property(
            String::from("location"),
            String::from("string"),
            Option::from(String::from("The location to check the weather for.")),
        );
        json_schema.add_property(
            String::from("date"),
            String::from("string"),
            Option::from(String::from("The date to check the weather for.")),
        );
        json_schema.add_property(
            String::from("weather"),
            String::from("string"),
            Option::from(String::from("The weather for the location and date.")),
        );
        json_schema.add_property(
            String::from("error"),
            String::from("string"),
            Option::from(String::from(
                "Error message. If there is no error, leave this field empty.",
            )),
        );
        openai
            .model_id(String::from("gpt-4o-mini"))
            .messages(messages)
            .temperature(1.0)
            .response_format(ChatCompletionResponseFormat::new(
                String::from("json_schema"),
                json_schema,
            ));

        let mut counter = 3;
        loop {
            match openai.chat().await {
                Ok(response) => {
                    println!("{:#?}", response);
                    match serde_json::from_str::<Weather>(&response.choices[0].message.content) {
                        Ok(weather) => {
                            println!("{:#?}", weather);
                            assert!(true);
                        }
                        Err(e) => {
                            println!("{:#?}", e);
                            assert!(false);
                        }
                    }
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    #[derive(Deserialize)]
    struct Summary {
        pub is_survey: bool,
        pub research_question: String,
        pub contributions: String,
        pub dataset: String,
        pub proposed_method: String,
        pub experiment_results: String,
        pub comparison_with_related_works: String,
        pub future_works: String,
    }
    #[tokio::test]
    async fn test_summarize() {
        init_tracing();
        let mut openai = ChatCompletion::new();
        let instruction = std::fs::read_to_string("src/test_rsc/sample_instruction.txt").unwrap();

        let messages = vec![Message::from_string(Role::User, instruction.clone())];

        let mut json_schema = Schema::chat_json_schema(String::from("summary"));
        json_schema.add_property(
            String::from("is_survey"),
            String::from("boolean"),
            Option::from(String::from(
                "この論文がサーベイ論文かどうかをtrue/falseで判定．",
            )),
        );
        json_schema.add_property(
        String::from("research_question"),
        String::from("string"),
        Option::from(String::from("この論文のリサーチクエスチョンの説明．この論文の背景や既存研究との関連も含めて記述する．")),
    );
        json_schema.add_property(
            String::from("contributions"),
            String::from("string"),
            Option::from(String::from(
                "この論文のコントリビューションをリスト形式で記述する．",
            )),
        );
        json_schema.add_property(
            String::from("dataset"),
            String::from("string"),
            Option::from(String::from(
                "この論文で使用されているデータセットをリストアップする．",
            )),
        );
        json_schema.add_property(
            String::from("proposed_method"),
            String::from("string"),
            Option::from(String::from("提案手法の詳細な説明．")),
        );
        json_schema.add_property(
            String::from("experiment_results"),
            String::from("string"),
            Option::from(String::from("実験の結果の詳細な説明．")),
        );
        json_schema.add_property(
        String::from("comparison_with_related_works"),
        String::from("string"),
        Option::from(String::from("関連研究と比較した場合のこの論文の新規性についての説明．可能な限り既存研究を参照しながら記述すること．")),
    );
        json_schema.add_property(
            String::from("future_works"),
            String::from("string"),
            Option::from(String::from(
                "未解決の課題および将来の研究の方向性について記述．",
            )),
        );

        openai
            .model_id(String::from("gpt-4o-mini"))
            .messages(messages)
            .temperature(1.0)
            .response_format(ChatCompletionResponseFormat::new(
                String::from("json_schema"),
                json_schema,
            ));

        let mut counter = 3;
        loop {
            match openai.chat().await {
                Ok(response) => {
                    println!("{:#?}", response);
                    match serde_json::from_str::<Summary>(&response.choices[0].message.content) {
                        Ok(summary) => {
                            tracing::info!("Summary.is_survey: {}", summary.is_survey);
                            tracing::info!(
                                "Summary.research_question: {}",
                                summary.research_question
                            );
                            tracing::info!("Summary.contributions: {}", summary.contributions);
                            tracing::info!("Summary.dataset: {}", summary.dataset);
                            tracing::info!("Summary.proposed_method: {}", summary.proposed_method);
                            tracing::info!(
                                "Summary.experiment_results: {}",
                                summary.experiment_results
                            );
                            tracing::info!(
                                "Summary.comparison_with_related_works: {}",
                                summary.comparison_with_related_works
                            );
                            tracing::info!("Summary.future_works: {}", summary.future_works);
                            assert!(true);
                        }
                        Err(e) => {
                            tracing::error!("Error: {}", e);
                            assert!(false);
                        }
                    }
                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            assert!(false, "Chat completion failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        assert!(false, "Chat completion failed");
                    }
                },
            };
        }
    }

    // #[tokio::test]
    // async fn test_chat_completion_with_long_arguments() {
    //     init_tracing();
    //     let mut openai = ChatCompletion::new();
    //     let text = std::fs::read_to_string("src/test_rsc/long_text.txt").unwrap();
    //     let messages = vec![Message::from_string(Role::User, text)];

    //     let token_count = messages
    //         .iter()
    //         .map(|m| m.get_input_token_count())
    //         .sum::<usize>();
    //     tracing::info!("Token count: {}", token_count);

    //     openai
    //         .model_id(String::from("gpt-4o-mini"))
    //         .messages(messages)
    //         .temperature(1.0);

    //     let mut counter = 3;
    //     loop {
    //         match openai.chat().await {
    //             Ok(response) => {
    //                 println!("{:#?}", response);
    //                 assert!(true);
    //                 break;
    //             }
    //             Err(e) => match e {
    //                 OpenAIToolError::RequestError(e) => {
    //                     tracing::warn!("Request error: {} (retrying... {})", e, counter);
    //                     counter -= 1;
    //                     if counter == 0 {
    //                         assert!(false, "Chat completion failed (retry limit reached)");
    //                     }
    //                     continue;
    //                 }
    //                 _ => {
    //                     tracing::error!("Error: {}", e);
    //                     assert!(false, "Chat completion failed");
    //                 }
    //             },
    //         };
    //     }
    // }
}
