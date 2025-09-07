use crate::{
    common::{
        errors::{OpenAIToolError, Result},
        message::Message,
        structured_output::Schema,
        tool::Tool,
    },
    responses::response::Response,
};
use derive_new::new;
use dotenvy::dotenv;
use request;
use serde::{ser::SerializeStruct, Serialize};
use std::env;

/// Represents the format configuration for structured output in responses
///
/// This struct is used to specify the schema format for structured text output
/// when making requests to the OpenAI Responses API.
#[derive(Debug, Clone, Default, Serialize, new)]
pub struct Format {
    /// The schema definition that specifies the structure of the expected output
    pub format: Schema,
}

/// Represents the body of a request to the OpenAI Responses API
///
/// This struct contains all the necessary parameters for making a request to generate responses.
/// It supports both plain text input and structured message input, along with optional tools
/// and structured output formatting.
#[derive(Debug, Clone, Default, new)]
pub struct Body {
    /// The ID of the model to use for generating responses (e.g., "gpt-4", "gpt-3.5-turbo")
    pub model: String,
    /// Optional instructions to guide the model's behavior and response style
    pub instructions: Option<String>,
    /// Plain text input for simple text-based requests
    pub plain_text_input: Option<String>,
    /// Structured message input for conversation-style interactions
    pub messages_input: Option<Vec<Message>>,
    /// Optional tools that the model can use during response generation
    pub tools: Option<Vec<Tool>>,
    /// Optional structured output format specification
    pub text: Option<Format>,
}

impl Serialize for Body {
    /// Custom serialization implementation for the request body
    ///
    /// This implementation handles the conversion of either plain text input
    /// or messages input into the appropriate "input" field format required
    /// by the OpenAI API. It also conditionally includes optional fields
    /// like tools and text formatting.
    ///
    /// # Errors
    ///
    /// Returns a serialization error if neither plain_text_input nor
    /// messages_input is set, as one of them is required.
    fn serialize<S>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let input = if self.plain_text_input.is_some() {
            self.plain_text_input.clone().unwrap()
        } else if self.messages_input.is_some() {
            serde_json::to_string(&self.messages_input).unwrap()
        } else {
            return Err(serde::ser::Error::custom("Either plain_text_input or messages_input must be set."));
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

/// Client for making requests to the OpenAI Responses API
///
/// This struct provides a convenient interface for building and executing requests
/// to the OpenAI Responses API. It handles authentication, request formatting,
/// and response parsing automatically.
///
/// # Examples
///
/// ```rust,no_run
/// use openai_tools::responses::request::Responses;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Responses::new();
/// let response = client
///     .model_id("gpt-4")
///     .instructions("You are a helpful assistant.")
///     .plain_text_input("Hello, how are you?")
///     .complete()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Default, Serialize)]
pub struct Responses {
    /// The OpenAI API key used for authentication
    api_key: String,
    /// The request body containing all parameters for the API call
    pub request_body: Body,
}

impl Responses {
    /// Creates a new instance of the Responses client
    ///
    /// This method initializes a new client by loading the OpenAI API key from
    /// the `OPENAI_API_KEY` environment variable. Make sure to set this environment
    /// variable before calling this method.
    ///
    /// # Panics
    ///
    /// Panics if the `OPENAI_API_KEY` environment variable is not set.
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        Self { api_key, request_body: Body::default() }
    }

    /// Sets the model ID for the request
    ///
    /// # Arguments
    ///
    /// * `model_id` - The ID of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.request_body.model = model_id.as_ref().to_string();
        self
    }

    /// Sets instructions to guide the model's behavior
    ///
    /// # Arguments
    ///
    /// * `instructions` - Instructions that define how the model should behave
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn instructions<T: AsRef<str>>(&mut self, instructions: T) -> &mut Self {
        self.request_body.instructions = Some(instructions.as_ref().to_string());
        self
    }

    /// Sets plain text input for simple text-based requests
    ///
    /// This method is mutually exclusive with `messages()`. Use this for simple
    /// text-based interactions where you don't need conversation history.
    ///
    /// # Arguments
    ///
    /// * `input` - The plain text input to send to the model
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn plain_text_input<T: AsRef<str>>(&mut self, input: T) -> &mut Self {
        self.request_body.plain_text_input = Some(input.as_ref().to_string());
        self
    }

    /// Sets structured message input for conversation-style interactions
    ///
    /// This method is mutually exclusive with `plain_text_input()`. Use this
    /// for complex conversations with message history and different roles.
    ///
    /// # Arguments
    ///
    /// * `messages` - A vector of messages representing the conversation history
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages_input = Some(messages);
        self
    }

    /// Sets tools that the model can use during response generation
    ///
    /// # Arguments
    ///
    /// * `tools` - A vector of tools available to the model
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Some(tools);
        self
    }

    /// Sets structured output format specification
    ///
    /// This allows you to specify the exact format and structure of the
    /// model's response output.
    ///
    /// # Arguments
    ///
    /// * `text_format` - The schema defining the expected output structure
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn structured_output(&mut self, text_format: Schema) -> &mut Self {
        self.request_body.text = Option::from(Format::new(text_format));
        self
    }

    /// Executes the request and returns the response
    ///
    /// This method sends the configured request to the OpenAI Responses API
    /// and returns the parsed response. It performs validation of required
    /// fields before sending the request.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Response` on success, or an `OpenAIToolError` on failure
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The API key is not set or is empty
    /// - The model ID is not set or is empty  
    /// - Neither messages nor plain text input is provided
    /// - Both messages and plain text input are provided (mutually exclusive)
    /// - The HTTP request fails
    /// - The response cannot be parsed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Responses::new();
    /// let response = client
    ///     .model_id("gpt-4")
    ///     .plain_text_input("Hello!")
    ///     .complete()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn complete(&self) -> Result<Response> {
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        if self.request_body.model.is_empty() {
            return Err(OpenAIToolError::Error("Model ID is not set.".into()));
        }
        if self.request_body.messages_input.is_none() && self.request_body.plain_text_input.is_none() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        } else if self.request_body.plain_text_input.is_none() && self.request_body.messages_input.is_none() {
            return Err(OpenAIToolError::Error("Both plain text input and messages are set. Please use one of them.".into()));
        }

        let body = serde_json::to_string(&self.request_body)?;
        let url = "https://api.openai.com/v1/responses".to_string();

        let client = request::Client::new();
        let mut header = request::header::HeaderMap::new();
        header.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        header.insert("Authorization", request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap());
        header.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust/0.1.0"));

        if cfg!(debug_assertions) {
            // Replace API key with a placeholder for security
            let body_for_debug = serde_json::to_string_pretty(&self.request_body).unwrap().replace(&self.api_key, "*************");
            // Log the request body for debugging purposes
            tracing::info!("Request body: {}", body_for_debug);
        }

        let response = client.post(url).headers(header).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(debug_assertions) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
