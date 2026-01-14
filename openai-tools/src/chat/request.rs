//! OpenAI Chat Completions API Request Module
//!
//! This module provides the functionality to build and send requests to the OpenAI Chat Completions API.
//! It offers a builder pattern for constructing requests with various parameters and options,
//! making it easy to interact with OpenAI's conversational AI models.
//!
//! # Key Features
//!
//! - **Builder Pattern**: Fluent API for constructing requests
//! - **Structured Output**: Support for JSON schema-based responses
//! - **Function Calling**: Tool integration for extended model capabilities
//! - **Comprehensive Parameters**: Full support for all OpenAI API parameters
//! - **Error Handling**: Robust error management and validation
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the chat completion client
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Create a simple conversation
//!     let messages = vec![
//!         Message::from_string(Role::User, "Hello! How are you?")
//!     ];
//!
//!     // Send the request and get a response
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .temperature(0.7)
//!         .chat()
//!         .await?;
//!         
//!     println!("AI Response: {}",
//!              response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! # Advanced Usage
//!
//! ## Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::structured_output::Schema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize)]
//! struct PersonInfo {
//!     name: String,
//!     age: u32,
//!     occupation: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Define JSON schema for structured output
//!     let mut schema = Schema::chat_json_schema("person_info");
//!     schema.add_property("name", "string", "Person's full name");
//!     schema.add_property("age", "number", "Person's age in years");
//!     schema.add_property("occupation", "string", "Person's job or profession");
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "Extract information about: John Smith, 30 years old, software engineer")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .json_schema(schema)
//!         .chat()
//!         .await?;
//!         
//!     // Parse structured response
//!     let person: PersonInfo = serde_json::from_str(
//!         response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap()
//!     )?;
//!     
//!     println!("Extracted: {} (age: {}, job: {})",
//!              person.name, person.age, person.occupation);
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::tool::Tool;
//! use openai_tools::common::parameters::ParameterProperty;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Define a weather checking tool
//!     let weather_tool = Tool::function(
//!         "get_weather",
//!         "Get current weather information for a location",
//!         vec![
//!             ("location", ParameterProperty::from_string("The city and country")),
//!             ("unit", ParameterProperty::from_string("Temperature unit (celsius/fahrenheit)")),
//!         ],
//!         false,
//!     );
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "What's the weather like in Tokyo today?")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .tools(vec![weather_tool])
//!         .temperature(0.1)
//!         .chat()
//!         .await?;
//!         
//!     // Handle tool calls
//!     if let Some(tool_calls) = &response.choices[0].message.tool_calls {
//!         for call in tool_calls {
//!             println!("Tool called: {}", call.function.name);
//!             if let Ok(args) = call.function.arguments_as_map() {
//!                 println!("Arguments: {:?}", args);
//!             }
//!             // Execute the function and continue the conversation...
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! # Environment Setup
//!
//! Before using this module, ensure you have set up your OpenAI API key:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```
//!
//! Or create a `.env` file in your project root:
//!
//! ```text
//! OPENAI_API_KEY=your-api-key-here
//! ```
//!
//!
//! # Error Handling
//!
//! All methods return a `Result` type for proper error handling:
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::errors::OpenAIToolError;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut chat = ChatCompletion::new();
//!     
//!     match chat.model_id("gpt-4o-mini").chat().await {
//!         Ok(response) => {
//!             if let Some(content) = &response.choices[0].message.content {
//!                 if let Some(text) = &content.text {
//!                     println!("Success: {}", text);
//!                 }
//!             }
//!         }
//!         Err(OpenAIToolError::RequestError(e)) => {
//!             eprintln!("Network error: {}", e);
//!         }
//!         Err(OpenAIToolError::SerdeJsonError(e)) => {
//!             eprintln!("JSON parsing error: {}", e);
//!         }
//!         Err(e) => {
//!             eprintln!("Other error: {}", e);
//!         }
//!     }
//! }
//! ```

use crate::chat::response::Response;
use crate::common::{
    client::create_http_client,
    errors::{ErrorResponse, OpenAIToolError, Result},
    message::Message,
    models::ChatModel,
    structured_output::Schema,
    tool::Tool,
};
use core::str;
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::time::Duration;

/// Response format structure for OpenAI API requests
///
/// This structure is used for structured output when JSON schema is specified.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct Format {
    #[serde(rename = "type")]
    type_name: String,
    json_schema: Schema,
}

impl Format {
    /// Creates a new Format structure
    ///
    /// # Arguments
    ///
    /// * `type_name` - The type name for the response format
    /// * `json_schema` - The JSON schema definition
    ///
    /// # Returns
    ///
    /// A new Format structure instance
    pub fn new<T: AsRef<str>>(type_name: T, json_schema: Schema) -> Self {
        Self { type_name: type_name.as_ref().to_string(), json_schema }
    }
}

/// Request body structure for OpenAI Chat Completions API
///
/// This structure represents the parameters that will be sent in the request body
/// to the OpenAI API. Each field corresponds to the API specification.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct Body {
    model: ChatModel,
    messages: Vec<Message>,
    /// Whether to store the request and response at OpenAI
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    /// Frequency penalty parameter to reduce repetition (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    /// Logit bias to adjust the probability of specific tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, i32>>,
    /// Whether to include probability information for each token
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    /// Number of top probabilities to return for each token (0-20)
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    /// Maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    /// Number of responses to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    /// Available modalities for the response (e.g., text, audio)
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<String>>,
    /// Presence penalty to encourage new topics (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    /// Temperature parameter to control response randomness (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Response format specification (e.g., JSON schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Format>,
    /// Optional tools that can be used by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
}

/// OpenAI Chat Completions API client
///
/// This structure manages interactions with the OpenAI Chat Completions API.
/// It handles API key management, request parameter configuration, and API calls.
///
/// # Example
///
/// ```rust
/// use openai_tools::chat::request::ChatCompletion;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut chat = ChatCompletion::new();
/// let messages = vec![Message::from_string(Role::User, "Hello!")];
///
/// let response = chat
///     .model_id("gpt-4o-mini")
///     .messages(messages)
///     .temperature(1.0)
///     .chat()
///     .await?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// # }
/// ```
/// Default API endpoint for Chat Completions
const DEFAULT_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatCompletion {
    /// The API endpoint URL
    endpoint: String,
    api_key: String,
    request_body: Body,
    /// Optional request timeout duration
    #[serde(skip)]
    timeout: Option<Duration>,
}

impl ChatCompletion {
    /// Creates a new ChatCompletion instance
    ///
    /// Loads the API key from the `OPENAI_API_KEY` environment variable.
    /// If a `.env` file exists, it will also be loaded.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Returns
    ///
    /// A new ChatCompletion instance
    pub fn new() -> Self {
        dotenv().ok();
        let api_key =
            env::var("OPENAI_API_KEY").map_err(|e| OpenAIToolError::Error(format!("OPENAI_API_KEY not set in environment: {}", e))).unwrap();
        Self { endpoint: DEFAULT_ENDPOINT.to_string(), api_key, request_body: Body::default(), timeout: None }
    }

    /// Sets a custom API endpoint URL
    ///
    /// Use this to point to alternative OpenAI-compatible APIs (e.g., Azure OpenAI,
    /// local LLM servers, or proxy servers).
    ///
    /// # Arguments
    ///
    /// * `url` - The full URL of the API endpoint
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.base_url("https://my-proxy.example.com/v1/chat/completions");
    /// ```
    pub fn base_url<T: AsRef<str>>(&mut self, url: T) -> &mut Self {
        self.endpoint = url.as_ref().to_string();
        self
    }

    /// Sets the model to use for chat completion.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to use (e.g., `ChatModel::Gpt4oMini`, `ChatModel::Gpt4o`)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::models::ChatModel;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model(ChatModel::Gpt4oMini);
    /// ```
    pub fn model(&mut self, model: ChatModel) -> &mut Self {
        self.request_body.model = model;
        self
    }

    /// Sets the model using a string ID (for backward compatibility).
    ///
    /// Prefer using [`model`] with `ChatModel` enum for type safety.
    ///
    /// # Arguments
    ///
    /// * `model_id` - OpenAI model ID string (e.g., "gpt-4o-mini")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model_id("gpt-4o-mini");
    /// ```
    #[deprecated(since = "0.2.0", note = "Use `model(ChatModel)` instead for type safety")]
    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.request_body.model = ChatModel::from(model_id.as_ref());
        self
    }

    /// Sets the request timeout duration
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
    /// use openai_tools::chat::request::ChatCompletion;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.model_id("gpt-4o-mini")
    ///     .timeout(Duration::from_secs(30));
    /// ```
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the chat message history
    ///
    /// # Arguments
    ///
    /// * `messages` - Vector of chat messages representing the conversation history
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages = messages;
        self
    }

    /// Adds a single message to the conversation history
    ///
    /// This method appends a new message to the existing conversation history.
    /// It's useful for building conversations incrementally.
    ///
    /// # Arguments
    ///
    /// * `message` - The message to add to the conversation
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let mut chat = ChatCompletion::new();
    /// chat.add_message(Message::from_string(Role::User, "Hello!"))
    ///     .add_message(Message::from_string(Role::Assistant, "Hi there!"))
    ///     .add_message(Message::from_string(Role::User, "How are you?"));
    /// ```
    pub fn add_message(&mut self, message: Message) -> &mut Self {
        self.request_body.messages.push(message);
        self
    }
    /// Sets whether to store the request and response at OpenAI
    ///
    /// # Arguments
    ///
    /// * `store` - `true` to store, `false` to not store
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn store(&mut self, store: bool) -> &mut Self {
        self.request_body.store = Option::from(store);
        self
    }

    /// Sets the frequency penalty
    ///
    /// A parameter that penalizes based on word frequency to reduce repetition.
    /// Positive values decrease repetition, negative values increase it.
    ///
    /// # Arguments
    ///
    /// * `frequency_penalty` - Frequency penalty value (range: -2.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.request_body.frequency_penalty = Option::from(frequency_penalty);
        self
    }

    /// Sets logit bias to adjust the probability of specific tokens
    ///
    /// # Arguments
    ///
    /// * `logit_bias` - A map of token IDs to adjustment values
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn logit_bias<T: AsRef<str>>(&mut self, logit_bias: HashMap<T, i32>) -> &mut Self {
        self.request_body.logit_bias =
            Option::from(logit_bias.into_iter().map(|(k, v)| (k.as_ref().to_string(), v)).collect::<HashMap<String, i32>>());
        self
    }

    /// Sets whether to include probability information for each token
    ///
    /// # Arguments
    ///
    /// * `logprobs` - `true` to include probability information
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn logprobs(&mut self, logprobs: bool) -> &mut Self {
        self.request_body.logprobs = Option::from(logprobs);
        self
    }

    /// Sets the number of top probabilities to return for each token
    ///
    /// # Arguments
    ///
    /// * `top_logprobs` - Number of top probabilities (range: 0-20)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn top_logprobs(&mut self, top_logprobs: u8) -> &mut Self {
        self.request_body.top_logprobs = Option::from(top_logprobs);
        self
    }

    /// Sets the maximum number of tokens to generate
    ///
    /// # Arguments
    ///
    /// * `max_completion_tokens` - Maximum number of tokens
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.request_body.max_completion_tokens = Option::from(max_completion_tokens);
        self
    }

    /// Sets the number of responses to generate
    ///
    /// # Arguments
    ///
    /// * `n` - Number of responses to generate
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn n(&mut self, n: u32) -> &mut Self {
        self.request_body.n = Option::from(n);
        self
    }

    /// Sets the available modalities for the response
    ///
    /// # Arguments
    ///
    /// * `modalities` - List of modalities (e.g., `["text", "audio"]`)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn modalities<T: AsRef<str>>(&mut self, modalities: Vec<T>) -> &mut Self {
        self.request_body.modalities = Option::from(modalities.into_iter().map(|m| m.as_ref().to_string()).collect::<Vec<String>>());
        self
    }

    /// Sets the presence penalty
    ///
    /// A parameter that controls the tendency to include new content in the document.
    /// Positive values encourage talking about new topics, negative values encourage
    /// staying on existing topics.
    ///
    /// # Arguments
    ///
    /// * `presence_penalty` - Presence penalty value (range: -2.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        self.request_body.presence_penalty = Option::from(presence_penalty);
        self
    }

    /// Sets the temperature parameter to control response randomness
    ///
    /// Higher values (e.g., 1.0) produce more creative and diverse outputs,
    /// while lower values (e.g., 0.2) produce more deterministic and consistent outputs.
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature parameter (range: 0.0 to 2.0)
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        self.request_body.temperature = Option::from(temperature);
        self
    }

    /// Sets structured output using JSON schema
    ///
    /// Enables receiving responses in a structured JSON format according to the
    /// specified JSON schema.
    ///
    /// # Arguments
    ///
    /// * `json_schema` - JSON schema defining the response structure
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn json_schema(&mut self, json_schema: Schema) -> &mut Self {
        self.request_body.response_format = Option::from(Format::new(String::from("json_schema"), json_schema));
        self
    }

    /// Sets the tools that can be called by the model
    ///
    /// Enables function calling by providing a list of tools that the model can choose to call.
    /// When tools are provided, the model may generate tool calls instead of or in addition to
    /// regular text responses.
    ///
    /// # Arguments
    ///
    /// * `tools` - Vector of tools available for the model to use
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Option::from(tools);
        self
    }

    /// Gets the current message history
    ///
    /// # Returns
    ///
    /// A vector containing the message history
    pub fn get_message_history(&self) -> Vec<Message> {
        self.request_body.messages.clone()
    }

    /// Checks if the model is a reasoning model that doesn't support custom temperature
    ///
    /// Reasoning models (o1, o3, o4 series) only support the default temperature value of 1.0.
    /// This method checks if the current model is one of these reasoning models.
    ///
    /// # Returns
    ///
    /// `true` if the model is a reasoning model, `false` otherwise
    ///
    /// # Supported Reasoning Models
    ///
    /// - `o1`, `o1-pro`, and variants
    /// - `o3`, `o3-mini`, and variants
    /// - `o4-mini` and variants
    fn is_reasoning_model(&self) -> bool {
        self.request_body.model.is_reasoning_model()
    }

    /// Sends the chat completion request to OpenAI API
    ///
    /// This method validates the request parameters, constructs the HTTP request,
    /// and sends it to the OpenAI Chat Completions endpoint.
    ///
    /// # Returns
    ///
    /// A `Result` containing the API response on success, or an error on failure.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - API key is not set
    /// - Model ID is not set
    /// - Messages are empty
    /// - Network request fails
    /// - Response parsing fails
    ///
    /// # Note
    ///
    /// For reasoning models (o1, o3, etc.), the `temperature` parameter is automatically
    /// ignored if set to a value other than the default (1.0), as these models only
    /// support the default temperature. A warning will be logged when this occurs.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::chat::request::ChatCompletion;
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>>
    /// # {
    /// let mut chat = ChatCompletion::new();
    /// let messages = vec![Message::from_string(Role::User, "Hello!")];
    ///
    /// let response = chat
    ///     .model_id("gpt-4o-mini")
    ///     .messages(messages)
    ///     .temperature(1.0)
    ///     .chat()
    ///     .await?;
    ///
    /// println!("{}", response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// # }
    /// ```
    pub async fn chat(&mut self) -> Result<Response> {
        // Check if the API key is set & body is built.
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        // Note: Model defaults to ChatModel::Gpt4oMini, so no need to check for empty
        if self.request_body.messages.is_empty() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        }

        // Handle reasoning models that don't support certain parameters
        // See: https://platform.openai.com/docs/guides/reasoning
        if self.is_reasoning_model() {
            let model = &self.request_body.model;

            // Temperature: only default (1.0) is supported
            if let Some(temp) = self.request_body.temperature {
                if (temp - 1.0).abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support custom temperature. \
                         Ignoring temperature={} and using default (1.0).",
                        model, temp
                    );
                    self.request_body.temperature = None;
                }
            }

            // Frequency penalty: only 0 is supported
            if let Some(fp) = self.request_body.frequency_penalty {
                if fp.abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support frequency_penalty. \
                         Ignoring frequency_penalty={} and using default (0).",
                        model, fp
                    );
                    self.request_body.frequency_penalty = None;
                }
            }

            // Presence penalty: only 0 is supported
            if let Some(pp) = self.request_body.presence_penalty {
                if pp.abs() > f32::EPSILON {
                    tracing::warn!(
                        "Reasoning model '{}' does not support presence_penalty. \
                         Ignoring presence_penalty={} and using default (0).",
                        model, pp
                    );
                    self.request_body.presence_penalty = None;
                }
            }

            // Logprobs: not supported
            if self.request_body.logprobs.is_some() {
                tracing::warn!(
                    "Reasoning model '{}' does not support logprobs. Ignoring logprobs parameter.",
                    model
                );
                self.request_body.logprobs = None;
            }

            // Top logprobs: not supported
            if self.request_body.top_logprobs.is_some() {
                tracing::warn!(
                    "Reasoning model '{}' does not support top_logprobs. Ignoring top_logprobs parameter.",
                    model
                );
                self.request_body.top_logprobs = None;
            }

            // Logit bias: not supported
            if self.request_body.logit_bias.is_some() {
                tracing::warn!(
                    "Reasoning model '{}' does not support logit_bias. Ignoring logit_bias parameter.",
                    model
                );
                self.request_body.logit_bias = None;
            }

            // N: only 1 is supported
            if let Some(n) = self.request_body.n {
                if n != 1 {
                    tracing::warn!(
                        "Reasoning model '{}' does not support n != 1. \
                         Ignoring n={} and using default (1).",
                        model, n
                    );
                    self.request_body.n = None;
                }
            }
        }

        let body = serde_json::to_string(&self.request_body)?;

        let client = create_http_client(self.timeout)?;
        let mut header = request::header::HeaderMap::new();
        header.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        header.insert("Authorization", request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap());
        header.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));

        if cfg!(debug_assertions) {
            // Replace API key with a placeholder in debug mode
            let body_for_debug = serde_json::to_string_pretty(&self.request_body).unwrap().replace(&self.api_key, "*************");
            tracing::info!("Request body: {}", body_for_debug);
        }

        let response = client.post(&self.endpoint).headers(header).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(debug_assertions) {
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
