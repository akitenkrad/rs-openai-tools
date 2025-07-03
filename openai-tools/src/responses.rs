//! OpenAI Responses API client implementation.
//!
//! This module provides a convenient interface for interacting with OpenAI's Responses API,
//! supporting both simple text conversations and complex function calling scenarios.
//!
//! # Basic Usage
//!
//! ```rust,no_run
//! use openai_tools::responses::Responses;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut responses = Responses::new();
//!     responses
//!         .model_id("gpt-4o-mini".into())
//!         .plain_text_input("Hello, world!".into());
//!
//!     let result = responses.complete().await?;
//!     println!("AI: {}", result.output[0].content.as_ref().unwrap()[0].text);
//!     Ok(())
//! }
//! ```
//!
//! # Function Calling
//!
//! ```rust,no_run
//! use openai_tools::responses::{Responses, Tool};
//! use openai_tools::common::Message;
//! use fxhash::FxHashMap;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut responses = Responses::new();
//!     
//!     let messages = vec![Message::from_string("user".to_string(), "Calculate 5 + 3".to_string())];
//!     let mut params = FxHashMap::default();
//!     params.insert("a".to_string(), "number".to_string());
//!     params.insert("b".to_string(), "number".to_string());
//!     
//!     let tool = Tool::function("calculator".into(), Some(params));
//!     
//!     responses
//!         .model_id("gpt-4o-mini".into())
//!         .messages(messages)
//!         .tools(vec![tool]);
//!
//!     let result = responses.complete().await?;
//!     // Handle function calls in result.output
//!     Ok(())
//! }
//! ```

use crate::{
    common::{Message, Usage},
    errors::{OpenAIToolError, Result},
    structured_output::Schema,
};
use derive_new::new;
use dotenvy::dotenv;
use fxhash::FxHashMap;
use request;
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use std::{env, vec};

/// Type alias for parameter names in tool definitions.
pub type ParameterName = String;

/// Type alias for parameter types in tool definitions (e.g., "string", "number", "boolean").
pub type ParameterType = String;

/// Represents the type definition for a tool parameter.
///
/// Used to specify the data type of function parameters in tool definitions.
/// Common types include "string", "number", "boolean", "array", "object".
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ToolParameterType {
    /// The type name (e.g., "string", "number", "boolean")
    #[serde(rename = "type")]
    pub type_name: String,
}

/// Defines the parameters schema for a tool function.
///
/// This follows the JSON Schema format for defining function parameters,
/// including type definitions, required fields, and whether additional properties are allowed.
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ToolParameters {
    /// The type of the parameters object, typically "object"
    #[serde(rename = "type")]
    pub type_name: String,
    /// Map of parameter names to their type definitions
    pub properties: Option<FxHashMap<ParameterName, ToolParameterType>>,
    /// List of required parameter names
    pub required: Option<Vec<ParameterName>>,
    /// Whether additional properties beyond those defined are allowed
    #[serde(rename = "additionalProperties")]
    pub additional_properties: Option<bool>,
}

/// Represents a tool that can be used by the AI model.
///
/// Tools enable the AI to call external functions or connect to MCP (Model Context Protocol) servers.
/// There are two main types of tools:
/// - **Function tools**: Direct function calls with defined parameters
/// - **MCP tools**: Connections to external MCP servers
///
/// # Examples
///
/// ## Function Tool
/// ```rust
/// use openai_tools::responses::Tool;
/// use fxhash::FxHashMap;
///
/// let mut params = FxHashMap::default();
/// params.insert("query".to_string(), "string".to_string());
///
/// let search_tool = Tool::function("web_search".into(), Some(params));
/// ```
///
/// ## MCP Tool
/// ```rust
/// use openai_tools::responses::Tool;
/// use fxhash::FxHashMap;
///
/// let mut params = FxHashMap::default();
/// params.insert("location".to_string(), "string".to_string());
///
/// let weather_tool = Tool::mcp(
///     "weather-server".to_string(),
///     "http://localhost:3000".to_string(),
///     "always".to_string(),
///     Some(vec!["get_weather".to_string()]),
///     Some(params),
/// );
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct Tool {
    /// The type of tool: "function" for direct function calls, "mcp" for MCP servers
    #[serde(rename = "type")]
    pub type_name: String,
    /// The name of the function (for function tools)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Label identifying the MCP server (for MCP tools)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    /// URL of the MCP server (for MCP tools)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    /// When to require approval: "always", "never", or "on_error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<String>,
    /// List of allowed tool names on the MCP server (for MCP tools)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
    /// Parameter schema for the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<ToolParameters>,
}

impl Tool {
    /// Creates a new MCP (Model Context Protocol) tool.
    ///
    /// MCP tools allow the AI to connect to external services through the MCP protocol.
    ///
    /// # Parameters
    ///
    /// * `server_label` - A human-readable label for the MCP server
    /// * `server_url` - The URL endpoint of the MCP server
    /// * `require_approval` - When to require user approval ("always", "never", "on_error")
    /// * `allowed_tools` - Optional list of specific tools allowed on this server
    /// * `parameters` - Optional parameter definitions for the MCP tools
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::responses::Tool;
    /// use fxhash::FxHashMap;
    ///
    /// let mut params = FxHashMap::default();
    /// params.insert("city".to_string(), "string".to_string());
    /// params.insert("units".to_string(), "string".to_string());
    ///
    /// let weather_tool = Tool::mcp(
    ///     "Weather API".to_string(),
    ///     "https://weather.example.com/mcp".to_string(),
    ///     "never".to_string(),
    ///     Some(vec!["get_weather".to_string(), "get_forecast".to_string()]),
    ///     Some(params),
    /// );
    /// ```
    pub fn mcp(
        server_label: String,
        server_url: String,
        require_approval: String,
        allowed_tools: Option<Vec<String>>,
        parameters: Option<FxHashMap<ParameterName, ParameterType>>,
    ) -> Self {
        Self {
            type_name: "mcp".into(),
            name: Some("".into()),
            server_label: Some(server_label),
            server_url: Some(server_url),
            require_approval: Some(require_approval),
            allowed_tools: Some(allowed_tools.unwrap_or_default()),
            parameters: match parameters.clone() {
                Some(p) => Some(ToolParameters {
                    type_name: "object".into(),
                    properties: Some(
                        p.into_iter()
                            .map(|(k, v)| (k, ToolParameterType { type_name: v }))
                            .collect(),
                    ),
                    required: parameters.as_ref().map(|p| p.keys().cloned().collect()),
                    additional_properties: Some(false),
                }),
                None => None,
            },
        }
    }

    /// Creates a new function tool.
    ///
    /// Function tools allow the AI to call specific functions with defined parameters.
    /// When the AI determines a function should be called, it will return a function_call
    /// output with the function name and arguments.
    ///
    /// # Parameters
    ///
    /// * `name` - The name of the function to be called
    /// * `parameters` - Optional parameter definitions (parameter name -> type)
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::responses::Tool;
    /// use fxhash::FxHashMap;
    ///
    /// let mut params = FxHashMap::default();
    /// params.insert("a".to_string(), "number".to_string());
    /// params.insert("b".to_string(), "number".to_string());
    /// params.insert("operation".to_string(), "string".to_string());
    ///
    /// let calculator = Tool::function("calculator".to_string(), Some(params));
    /// ```
    pub fn function(
        name: String,
        parameters: Option<FxHashMap<ParameterName, ParameterType>>,
    ) -> Self {
        Self {
            type_name: "function".into(),
            name: Some(name),
            server_label: Some("".into()),
            server_url: Some("".into()),
            require_approval: Some("".into()),
            allowed_tools: Some(vec![]),
            parameters: match parameters.clone() {
                Some(p) => Some(ToolParameters {
                    type_name: "object".into(),
                    properties: Some(
                        p.into_iter()
                            .map(|(k, v)| (k, ToolParameterType { type_name: v }))
                            .collect(),
                    ),
                    required: parameters.as_ref().map(|p| p.keys().cloned().collect()),
                    additional_properties: Some(false),
                }),
                None => None,
            },
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, new)]
pub struct ResponsesFormat {
    pub format: Schema,
}
/// Request body for the OpenAI Responses API.
///
/// This struct represents the payload sent to the API, containing the model configuration,
/// input data, and any tools that should be available to the AI.
#[derive(Debug, Clone, Default, new)]
pub struct ResponsesBody {
    /// The ID of the model to use (e.g., "gpt-4o-mini", "gpt-4")
    pub model: String,
    /// Optional instructions to guide the AI's behavior
    pub instructions: Option<String>,
    /// Plain text input (mutually exclusive with messages_input)
    pub plain_text_input: Option<String>,
    /// Structured message input (mutually exclusive with plain_text_input)
    pub messages_input: Option<Vec<Message>>,
    /// Optional tools available for the AI to use
    pub tools: Option<Vec<Tool>>,
    /// Optional response format configuration
    pub text: Option<ResponsesFormat>,
}

impl Serialize for ResponsesBody {
    fn serialize<S>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let input = if self.plain_text_input.is_some() {
            self.plain_text_input.clone().unwrap()
        } else if self.messages_input.is_some() {
            serde_json::to_string(&self.messages_input).unwrap()
        } else {
            return Err(serde::ser::Error::custom(
                "Either plain_text_input or messages_input must be set.",
            ));
        };
        let mut state = serializer.serialize_struct("ResponsesBody", 4)?;
        state.serialize_field("model", &self.model)?;
        state.serialize_field("instructions", &self.instructions)?;
        state.serialize_field("input", &input)?;
        if self.tools.is_some() {
            state.serialize_field("tools", &self.tools)?;
        }
        if self.text.is_some() {
            state.serialize_field("text", &self.text)?;
        }
        state.end()
    }
}

/// Content within a response output.
///
/// Represents textual content returned by the AI, including any annotations or log probabilities.
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ResponsesContent {
    /// The type of content, typically "text"
    #[serde(rename = "type")]
    pub type_name: String,
    /// The actual text content
    pub text: String,
    /// Any annotations associated with the content
    pub annotations: Vec<String>,
    /// Log probabilities for the content tokens
    pub logprobs: Vec<String>,
}

/// Individual output item from the AI response.
///
/// This can represent different types of outputs:
/// - Text responses from the AI
/// - Function calls that should be executed
/// - Other structured outputs
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ResponsesOutput {
    /// Unique identifier for this output
    pub id: String,
    /// The type of output: "text", "function_call", etc.
    #[serde(rename = "type")]
    pub type_name: String,
    /// The role (e.g., "assistant") for text outputs
    pub role: Option<String>,
    /// Status of the output
    pub status: Option<String>,
    /// Text content (for text outputs)
    pub content: Option<Vec<ResponsesContent>>,
    /// Function arguments as JSON string (for function_call outputs)
    pub arguments: Option<String>,
    /// Unique identifier for the function call
    pub call_id: Option<String>,
    /// Function name (for function_call outputs)
    pub name: Option<String>,
}

/// Reasoning information from the AI model.
///
/// Provides insight into the AI's reasoning process and effort level.
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ResponsesReasoning {
    /// The effort level used in reasoning
    pub effort: Option<String>,
    /// Summary of the reasoning process
    pub summary: Option<String>,
}

/// Text formatting configuration for responses.
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ResponsesText {
    /// Format configuration
    pub format: Schema,
}

/// Complete response from the OpenAI Responses API.
///
/// This struct contains all the information returned by the API, including the AI's outputs,
/// usage statistics, and metadata about the request processing.
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct ResponsesResponse {
    /// Unique identifier for this response
    pub id: String,
    /// Object type, typically "response"
    pub object: String,
    /// Unix timestamp when the response was created
    pub created_at: usize,
    /// Status of the response processing
    pub status: String,
    /// Whether the response was processed in the background
    pub background: bool,
    /// Error message if the request failed
    pub error: Option<String>,
    /// Details about incomplete responses
    pub incomplete_details: Option<String>,
    /// Instructions that were used for this response
    pub instructions: Option<String>,
    /// Maximum number of output tokens that were allowed
    pub max_output_tokens: Option<usize>,
    /// Maximum number of tool calls that were allowed
    pub max_tool_calls: Option<usize>,
    /// The model that was used to generate the response
    pub model: String,
    /// List of outputs from the AI (text, function calls, etc.)
    pub output: Vec<ResponsesOutput>,
    /// Whether parallel tool calls were enabled
    pub parallel_tool_calls: bool,
    /// ID of the previous response in a conversation chain
    pub previous_response_id: Option<String>,
    /// Reasoning information from the AI
    pub reasoning: ResponsesReasoning,
    /// Service tier used for processing
    pub service_tier: Option<String>,
    /// Whether the response should be stored
    pub store: Option<bool>,
    /// Temperature setting used for generation
    pub temperature: Option<f64>,
    /// Text formatting configuration
    pub text: ResponsesText,
    /// Tool choice configuration that was used
    pub tool_choice: Option<String>,
    /// Tools that were available during generation
    pub tools: Option<Vec<Tool>>,
    /// Number of top log probabilities returned
    pub top_logprobs: Option<usize>,
    /// Top-p (nucleus sampling) parameter used
    pub top_p: Option<f64>,
    /// Truncation strategy that was applied
    pub truncation: Option<String>,
    /// Token usage statistics
    pub usage: Option<Usage>,
    /// User identifier associated with the request
    pub user: Option<String>,
    /// Additional metadata as key-value pairs
    pub metadata: FxHashMap<String, String>,
}

/// Main client for interacting with the OpenAI Responses API.
///
/// This struct provides a high-level interface for making requests to the OpenAI Responses API.
/// It supports both simple text-based conversations and complex function calling scenarios.
///
/// # Environment Requirements
///
/// The `OPENAI_API_KEY` environment variable must be set with a valid OpenAI API key.
///
/// # Usage Patterns
///
/// ## Simple Text Conversation
/// ```rust,no_run
/// use openai_tools::responses::Responses;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut responses = Responses::new();
///     responses
///         .model_id("gpt-4o-mini".into())
///         .plain_text_input("What is the capital of France?".into());
///
///     let result = responses.complete().await?;
///     if let Some(content) = &result.output[0].content {
///         println!("AI: {}", content[0].text);
///     }
///     Ok(())
/// }
/// ```
///
/// ## Structured Messages
/// ```rust,no_run
/// use openai_tools::{responses::Responses, common::Message};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut responses = Responses::new();
///     
///     let messages = vec![
///         Message::from_string("system".to_string(), "You are a helpful assistant.".to_string()),
///         Message::from_string("user".to_string(), "Hello!".to_string()),
///     ];
///     
///     responses
///         .model_id("gpt-4o-mini".into())
///         .messages(messages);
///
///     let result = responses.complete().await?;
///     // Process result...
///     Ok(())
/// }
/// ```
///
/// ## Function Calling
/// ```rust,no_run
/// use openai_tools::responses::{Responses, Tool};
/// use openai_tools::common::Message;
/// use fxhash::FxHashMap;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut responses = Responses::new();
///     
///     let messages = vec![Message::from_string("user".to_string(), "Calculate 15 * 8".to_string())];
///     
///     let mut params = FxHashMap::default();
///     params.insert("a".to_string(), "number".to_string());
///     params.insert("b".to_string(), "number".to_string());
///     
///     let calculator = Tool::function("multiply".to_string(), Some(params));
///     
///     responses
///         .model_id("gpt-4o-mini".into())
///         .messages(messages)
///         .tools(vec![calculator]);
///
///     let result = responses.complete().await?;
///     
///     for output in &result.output {
///         match output.type_name.as_str() {
///             "function_call" => {
///                 println!("Function called: {}", output.name.as_ref().unwrap());
///                 println!("Arguments: {}", output.arguments.as_ref().unwrap());
///             }
///             "text" => {
///                 if let Some(content) = &output.content {
///                     println!("AI: {}", content[0].text);
///                 }
///             }
///             _ => {}
///         }
///     }
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone, Default, Serialize)]
pub struct Responses {
    /// OpenAI API key (loaded from environment)
    api_key: String,
    /// Request body that will be sent to the API
    pub request_body: ResponsesBody,
}

impl Responses {
    /// Creates a new Responses instance.
    ///
    /// This constructor loads the OpenAI API key from the `OPENAI_API_KEY` environment variable.
    /// Make sure this environment variable is set before calling this method.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::responses::Responses;
    ///
    /// // Make sure OPENAI_API_KEY is set in your environment
    /// let responses = Responses::new();
    /// ```
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        Self {
            api_key,
            request_body: ResponsesBody::default(),
        }
    }

    /// Sets the model ID to use for the request.
    ///
    /// # Parameters
    ///
    /// * `model_id` - The ID of the model to use (e.g., "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::Responses;
    /// let mut responses = Responses::new();
    /// responses.model_id("gpt-4o-mini".into());
    /// ```
    pub fn model_id(&mut self, model_id: String) -> &mut Self {
        self.request_body.model = model_id;
        self
    }

    /// Sets custom instructions to guide the AI's behavior.
    ///
    /// Instructions help shape how the AI responds and behaves during the conversation.
    ///
    /// # Parameters
    ///
    /// * `instructions` - Custom instructions for the AI
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::Responses;
    /// let mut responses = Responses::new();
    /// responses
    ///     .model_id("gpt-4o-mini".into())
    ///     .instructions("You are a helpful coding assistant. Provide clear, concise code examples.".into());
    /// ```
    pub fn instructions(&mut self, instructions: String) -> &mut Self {
        self.request_body.instructions = Some(instructions);
        self
    }

    /// Sets plain text input for the request.
    ///
    /// Use this method for simple text-based conversations. This is mutually exclusive
    /// with `messages_input()` - you should use one or the other, not both.
    ///
    /// # Parameters
    ///
    /// * `input` - The plain text input to send to the AI
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::Responses;
    /// let mut responses = Responses::new();
    /// responses
    ///     .model_id("gpt-4o-mini".into())
    ///     .plain_text_input("Explain quantum computing in simple terms.".into());
    /// ```
    pub fn plain_text_input(&mut self, input: String) -> &mut Self {
        self.request_body.plain_text_input = Some(input);
        self
    }

    /// Sets structured message input for the request.
    ///
    /// Use this method for conversations with multiple messages, including system prompts
    /// and conversation history. This is mutually exclusive with `plain_text_input()`.
    ///
    /// # Parameters
    ///
    /// * `messages` - Vector of messages representing the conversation
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::Responses;
    /// # use openai_tools::common::Message;
    /// let mut responses = Responses::new();
    ///
    /// let messages = vec![
    ///     Message::from_string("system".to_string(), "You are a helpful assistant.".to_string()),
    ///     Message::from_string("user".to_string(), "What's the weather like?".to_string()),
    /// ];
    ///
    /// responses
    ///     .model_id("gpt-4o-mini".into())
    ///     .messages(messages);
    /// ```
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages_input = Some(messages);
        self
    }

    /// Sets the tools available for the AI to use.
    ///
    /// Tools enable function calling, allowing the AI to call external functions or
    /// connect to MCP servers when appropriate.
    ///
    /// # Parameters
    ///
    /// * `tools` - Vector of tools available to the AI
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::{Responses, Tool};
    /// # use openai_tools::common::Message;
    /// # use fxhash::FxHashMap;
    /// let mut responses = Responses::new();
    ///
    /// let mut params = FxHashMap::default();
    /// params.insert("query".to_string(), "string".to_string());
    ///
    /// let search_tool = Tool::function("web_search".to_string(), Some(params));
    ///
    /// responses
    ///     .model_id("gpt-4o-mini".into())
    ///     .plain_text_input("Search for information about Rust programming.".into())
    ///     .tools(vec![search_tool]);
    /// ```
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Some(tools);
        self
    }

    pub fn text(&mut self, text_format: Schema) -> &mut Self {
        self.request_body.text = Option::from(ResponsesFormat::new(text_format));
        return self;
    }

    /// Sends the request to the OpenAI API and returns the response.
    ///
    /// This method validates the request parameters, sends the HTTP request to the
    /// OpenAI Responses API, and parses the response.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `ResponsesResponse` on success, or an error on failure.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The API key is not set or empty
    /// - The model ID is not set or empty  
    /// - Neither `plain_text_input` nor `messages_input` is set
    /// - Both `plain_text_input` and `messages_input` are set
    /// - The HTTP request fails
    /// - The API returns an error status
    /// - The response cannot be parsed as JSON
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use openai_tools::responses::Responses;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let mut responses = Responses::new();
    /// responses
    ///     .model_id("gpt-4o-mini".into())
    ///     .plain_text_input("Hello, world!".into());
    ///
    /// match responses.complete().await {
    ///     Ok(response) => {
    ///         println!("Success! Got {} outputs", response.output.len());
    ///         for output in &response.output {
    ///             match output.type_name.as_str() {
    ///                 "text" => {
    ///                     if let Some(content) = &output.content {
    ///                         println!("AI: {}", content[0].text);
    ///                     }
    ///                 }
    ///                 "function_call" => {
    ///                     println!("Function call: {}", output.name.as_ref().unwrap());
    ///                 }
    ///                 _ => {
    ///                     println!("Other output type: {}", output.type_name);
    ///                 }
    ///             }
    ///         }
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Request failed: {}", e);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn complete(&self) -> Result<ResponsesResponse> {
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        if self.request_body.model.is_empty() {
            return Err(OpenAIToolError::Error("Model ID is not set.".into()));
        }
        if self.request_body.messages_input.is_none()
            && self.request_body.plain_text_input.is_none()
        {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        } else if !self.request_body.plain_text_input.is_some()
            && !self.request_body.messages_input.is_some()
        {
            return Err(OpenAIToolError::Error(
                "Both plain text input and messages are set. Please use one of them.".into(),
            ));
        }

        let body = serde_json::to_string(&self.request_body)?;
        let url = "https://api.openai.com/v1/responses".to_string();

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
        let content = response.text().await.expect("Failed to read response text");

        println!("Response content: {}", content);

        serde_json::from_str::<ResponsesResponse>(&content)
            .map_err(|e| OpenAIToolError::SerdeJsonError(e))
    }
}

#[cfg(test)]
mod tests {
    use crate::common::MessageContent;

    use super::*;
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
    async fn test_responses_with_plain_text() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini".into());
        responses.instructions("test instructions".into());
        responses.plain_text_input("Hello world!".into());

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_messages() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini".into());
        responses.instructions("test instructions".into());
        let messages = vec![Message::from_string(
            String::from("user"),
            String::from("Hello world!"),
        )];
        responses.messages(messages);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_tools() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini".into());
        responses.instructions("test instructions".into());
        let messages = vec![Message::from_string(
            String::from("user"),
            String::from("Calculate 2 + 2 using a calculator tool."),
        )];
        responses.messages(messages);

        let tool = Tool::function(
            "calculator".into(),
            Some(
                [("a".into(), "number".into()), ("b".into(), "number".into())]
                    .iter()
                    .cloned()
                    .collect::<FxHashMap<String, String>>(),
            ),
        );
        responses.tools(vec![tool]);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        println!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert_eq!(res.output[0].type_name, "function_call");
                    assert_eq!(res.output[0].name.as_ref().unwrap(), "calculator");
                    assert!(res.output[0].call_id.as_ref().unwrap().len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[derive(Debug, Deserialize)]
    struct TestResponse {
        pub capital: String,
    }
    #[tokio::test]
    async fn test_responses_with_json_schema() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini".into());

        let messages = vec![Message::from_string(
            String::from("user"),
            String::from("What is the capital of France?"),
        )];
        responses.messages(messages);

        let mut schema = Schema::responses_json_schema("capital".into());
        schema.add_property(
            "capital".into(),
            "string".into(),
            Some("The capital city of France".into()),
        );
        responses.text(schema);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    let res = serde_json::from_str::<TestResponse>(
                        res.output[0].content.as_ref().unwrap()[0].text.as_str(),
                    )
                    .unwrap();
                    assert_eq!(res.capital, "Paris");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_responses_with_image_input() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini".into());
        responses.instructions("test instructions".into());

        let message = Message::from_message_array(
            "user".into(),
            vec![
                MessageContent::from_text("Do you find a clock in this image?".into()),
                MessageContent::from_image_file("src/test_rsc/sample_image.jpg".into()),
            ],
        );
        responses.messages(vec![message]);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
                    break;
                }
                Err(e) => {
                    tracing::error!("Error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Failed to complete responses after 3 attempts");
                    }
                }
            }
        }
    }
}
