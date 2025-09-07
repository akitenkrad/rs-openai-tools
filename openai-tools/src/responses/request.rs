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
use std::collections::HashMap;
use std::env;
use strum::{Display, EnumString};

/// Specifies additional data to include in the response output
///
/// This enum defines various types of additional information that can be
/// included in the API response output, such as web search results, code
/// interpreter outputs, image URLs, and other metadata.
///
/// # API Reference
///
/// Corresponds to the `include` parameter in the OpenAI Responses API:
/// <https://platform.openai.com/docs/api-reference/responses/create>
#[derive(Debug, Clone, EnumString, Display, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Include {
    /// Include web search call results in the output  
    ///
    /// When included, the response will contain information about web search
    /// results that were used during the response generation process.
    #[strum(serialize = "web_search_call.results")]
    #[serde(rename = "web_search_call.results")]
    WebSearchCall,

    /// Include code interpreter call outputs in the output
    ///
    /// When included, the response will contain outputs from any code
    /// that was executed during the response generation process.
    #[strum(serialize = "code_interpreter_call.outputs")]
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCall,

    /// Include computer call output image URLs in the output
    ///
    /// When included, the response will contain image URLs from any
    /// computer interaction calls that were made.
    #[strum(serialize = "computer_call_output.output.image_url")]
    #[serde(rename = "computer_call_output.output.image_url")]
    ImageUrlInComputerCallOutput,

    /// Include file search call results in the output
    ///
    /// When included, the response will contain results from any
    /// file search operations that were performed.
    #[strum(serialize = "file_search_call.results")]
    #[serde(rename = "file_search_call.results")]
    FileSearchCall,

    /// Include image URLs from input messages in the output
    ///
    /// When included, the response will contain image URLs that were
    /// present in the input messages.
    #[strum(serialize = "message.input_image.image_url")]
    #[serde(rename = "message.input_image.image_url")]
    ImageUrlInInputMessages,

    /// Include log probabilities in the output
    ///
    /// When included, the response will contain log probability information
    /// for the generated text tokens.
    #[strum(serialize = "message.output_text.logprobs")]
    #[serde(rename = "message.output_text.logprobs")]
    LogprobsInOutput,

    /// Include reasoning encrypted content in the output
    ///
    /// When included, the response will contain encrypted reasoning
    /// content that shows the model's internal reasoning process.
    #[strum(serialize = "reasoning.encrypted_content")]
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

/// Defines the level of reasoning effort the model should apply
///
/// This enum controls how much computational effort the model invests
/// in reasoning through complex problems before generating a response.
///
/// # API Reference
///
/// Corresponds to the `reasoning.effort` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Minimal reasoning effort - fastest response time
    ///
    /// Use this for simple queries that don't require deep analysis.
    #[strum(serialize = "minimal")]
    #[serde(rename = "minimal")]
    Minimal,

    /// Low reasoning effort - balanced performance
    ///
    /// Use this for moderately complex queries that benefit from some reasoning.
    #[strum(serialize = "low")]
    #[serde(rename = "low")]
    Low,

    /// Medium reasoning effort - comprehensive analysis
    ///
    /// Use this for complex queries that require thorough consideration.
    #[strum(serialize = "medium")]
    #[serde(rename = "medium")]
    Medium,

    /// High reasoning effort - maximum thoughtfulness
    ///
    /// Use this for very complex queries requiring deep, careful analysis.
    #[strum(serialize = "high")]
    #[serde(rename = "high")]
    High,
}

/// Defines the format of reasoning summary to include in the response
///
/// This enum controls how the model's reasoning process is summarized
/// and presented in the response output.
///
/// # API Reference
///
/// Corresponds to the `reasoning.summary` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    /// Automatically determine the appropriate summary format
    ///
    /// The model will choose the most suitable summary format based on the query.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Provide a concise summary of the reasoning process
    ///
    /// Use this for shorter, more focused reasoning explanations.
    #[strum(serialize = "concise")]
    #[serde(rename = "concise")]
    Concise,

    /// Provide a detailed summary of the reasoning process
    ///
    /// Use this for comprehensive reasoning explanations with full detail.
    #[strum(serialize = "detailed")]
    #[serde(rename = "detailed")]
    Detailed,
}

/// Configuration for reasoning behavior in responses
///
/// This struct allows you to control how the model approaches reasoning
/// for complex queries, including the effort level and summary format.
///
/// # API Reference
///
/// Corresponds to the `reasoning` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct Reasoning {
    /// The level of reasoning effort to apply
    pub effort: Option<ReasoningEffort>,
    /// The format for the reasoning summary
    pub summary: Option<ReasoningSummary>,
}

/// Defines how the model should choose and use tools
///
/// This enum controls the model's behavior regarding tool usage during
/// response generation.
///
/// # API Reference
///
/// Corresponds to the `tool_choice` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    /// Disable tool usage completely
    ///
    /// The model will not use any tools and will generate responses
    /// based solely on its training data.
    #[strum(serialize = "none")]
    #[serde(rename = "none")]
    None,

    /// Automatically decide when to use tools
    ///
    /// The model will automatically determine when tools are needed
    /// and which tools to use based on the query context.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Require the use of tools
    ///
    /// The model must use at least one of the provided tools in its response.
    #[strum(serialize = "required")]
    #[serde(rename = "required")]
    Required,
}

/// Controls truncation behavior for long inputs
///
/// This enum defines how the system should handle inputs that exceed
/// the maximum context length.
///
/// # API Reference
///
/// Corresponds to the `truncation` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize, EnumString, Display, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    /// Automatically truncate inputs to fit context length
    ///
    /// The system will automatically trim inputs to ensure they fit
    /// within the model's context window.
    #[strum(serialize = "auto")]
    #[serde(rename = "auto")]
    Auto,

    /// Disable truncation - return error if input is too long
    ///
    /// The system will return an error rather than truncating
    /// inputs that exceed the context length.
    #[strum(serialize = "disabled")]
    #[serde(rename = "disabled")]
    Disabled,
}

/// Options for streaming responses
///
/// This struct configures how streaming responses should behave,
/// including whether to include obfuscated content.
///
/// # API Reference
///
/// Corresponds to the `stream_options` parameter in the OpenAI Responses API.
#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    /// Whether to include obfuscated content in streaming responses
    ///
    /// When enabled, streaming responses may include placeholder or
    /// obfuscated content that gets replaced as the real content is generated.
    pub include_obfuscation: bool,
}
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
/// This struct contains all the parameters for making requests to the OpenAI Responses API.
/// It supports both plain text and structured message input, along with extensive configuration
/// options for tools, reasoning, output formatting, and response behavior.
///
/// # Required Parameters
///
/// - `model`: The ID of the model to use
/// - Either `plain_text_input` OR `messages_input` (mutually exclusive)
///
/// # API Reference
///
/// Based on the OpenAI Responses API specification:
/// <https://platform.openai.com/docs/api-reference/responses/create>
///
/// # Examples
///
/// ## Simple Text Input
///
/// ```rust
/// use openai_tools::responses::request::Body;
///
/// let body = Body {
///     model: "gpt-4".to_string(),
///     plain_text_input: Some("What is the weather like?".to_string()),
///     ..Default::default()
/// };
/// ```
///
/// ## With Messages and Tools
///
/// ```rust
/// use openai_tools::responses::request::Body;
/// use openai_tools::common::message::Message;
/// use openai_tools::common::role::Role;
///
/// let messages = vec![
///     Message::from_string(Role::User, "Help me with coding")
/// ];
///
/// let body = Body {
///     model: "gpt-4".to_string(),
///     messages_input: Some(messages),
///     instructions: Some("You are a helpful coding assistant".to_string()),
///     max_output_tokens: Some(1000),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, new)]
pub struct Body {
    /// The ID of the model to use for generating responses
    ///
    /// Specifies which OpenAI model to use for response generation.
    /// Common values include "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo".
    ///
    /// # Required
    ///
    /// This field is required for all requests.
    ///
    /// # Examples
    ///
    /// - `"gpt-4"` - Latest GPT-4 model
    /// - `"gpt-4-turbo"` - GPT-4 Turbo for faster responses
    /// - `"gpt-3.5-turbo"` - More cost-effective option
    pub model: String,

    /// Optional instructions to guide the model's behavior and response style
    ///
    /// Provides system-level instructions that define how the model should
    /// behave, its personality, response format, or any other behavioral guidance.
    ///
    /// # Examples
    ///
    /// - `"You are a helpful assistant that provides concise answers"`
    /// - `"Respond only with JSON formatted data"`
    /// - `"Act as a professional code reviewer"`
    pub instructions: Option<String>,

    /// Plain text input for simple text-based requests
    ///
    /// Use this for straightforward text input when you don't need the structure
    /// of messages with roles. This is mutually exclusive with `messages_input`.
    ///
    /// # Mutually Exclusive
    ///
    /// Cannot be used together with `messages_input`. Choose one based on your needs:
    /// - Use `plain_text_input` for simple, single-turn interactions
    /// - Use `messages_input` for conversation history or role-based interactions
    ///
    /// # Examples
    ///
    /// - `"What is the capital of France?"`
    /// - `"Summarize this article: [article content]"`
    /// - `"Write a haiku about programming"`
    pub plain_text_input: Option<String>,

    /// Structured message input for conversation-style interactions
    ///
    /// Use this when you need conversation history, different message roles
    /// (user, assistant, system), or structured dialogue. This is mutually
    /// exclusive with `plain_text_input`.
    ///
    /// # Mutually Exclusive
    ///
    /// Cannot be used together with `plain_text_input`.
    ///
    /// # Message Roles
    ///
    /// - `System`: Instructions for the model's behavior
    /// - `User`: User input or questions
    /// - `Assistant`: Previous model responses (for conversation history)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let messages = vec![
    ///     Message::from_string(Role::System, "You are a helpful assistant"),
    ///     Message::from_string(Role::User, "Hello!"),
    ///     Message::from_string(Role::Assistant, "Hi there! How can I help you?"),
    ///     Message::from_string(Role::User, "What's 2+2?"),
    /// ];
    /// ```
    pub messages_input: Option<Vec<Message>>,

    /// Optional tools that the model can use during response generation
    ///
    /// Provides the model with access to external tools like web search,
    /// code execution, file access, or custom functions. The model will
    /// automatically decide when and how to use these tools based on the query.
    ///
    /// # Tool Types
    ///
    /// - Web search tools for finding current information
    /// - Code interpreter for running and analyzing code
    /// - File search tools for accessing document collections
    /// - Custom function tools for specific business logic
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::tool::Tool;
    /// use openai_tools::common::parameters::ParameterProperty;
    ///
    /// let tools = vec![
    ///     Tool::function("search", "Search the web", Vec::<(&str, ParameterProperty)>::new(), false),
    ///     Tool::function("calculate", "Perform calculations", Vec::<(&str, ParameterProperty)>::new(), false),
    /// ];
    /// ```
    pub tools: Option<Vec<Tool>>,
    /// Optional tool choice configuration
    // TODO: Implement ToolChoice
    // pub tool_choice: Option<ToolChoice>,

    /// Optional structured output format specification
    ///
    /// Defines the structure and format for the model's response output.
    /// Use this when you need the response in a specific JSON schema format
    /// or other structured format for programmatic processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::structured_output::Schema;
    /// use openai_tools::responses::request::Format;
    ///
    /// let format = Format::new(Schema::responses_json_schema("response_schema"));
    /// ```
    pub structured_output: Option<Format>,

    /// Optional sampling temperature for controlling response randomness
    ///
    /// Controls the randomness and creativity of the model's responses.
    /// Higher values make the output more random and creative, while lower
    /// values make it more focused, deterministic, and consistent.
    ///
    /// # Range
    ///
    /// - **Range**: 0.0 to 2.0
    /// - **Default**: 1.0 (if not specified)
    /// - **Minimum**: 0.0 (most deterministic, least creative)
    /// - **Maximum**: 2.0 (most random, most creative)
    ///
    /// # Recommended Values
    ///
    /// - **0.0 - 0.3**: Highly focused and deterministic
    ///   - Best for: Factual questions, code generation, translations
    ///   - Behavior: Very consistent, predictable responses
    ///
    /// - **0.3 - 0.7**: Balanced creativity and consistency
    ///   - Best for: General conversation, explanations, analysis
    ///   - Behavior: Good balance between creativity and reliability
    ///
    /// - **0.7 - 1.2**: More creative and varied responses
    ///   - Best for: Creative writing, brainstorming, ideation
    ///   - Behavior: More diverse and interesting outputs
    ///
    /// - **1.2 - 2.0**: Highly creative and unpredictable
    ///   - Best for: Experimental creative tasks, humor, unconventional ideas
    ///   - Behavior: Very diverse but potentially less coherent
    ///
    /// # Usage Guidelines
    ///
    /// - **Start with 0.7** for most applications as a good default
    /// - **Use 0.0-0.3** when you need consistent, reliable responses
    /// - **Use 0.8-1.2** for creative tasks that still need coherence
    /// - **Avoid values above 1.5** unless you specifically want very random outputs
    ///
    /// # API Reference
    ///
    /// Corresponds to the `temperature` parameter in the OpenAI Responses API:
    /// <https://platform.openai.com/docs/api-reference/responses/create>
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// // Deterministic, factual responses
    /// let mut client_factual = Responses::new();
    /// client_factual.temperature(0.2);
    ///
    /// // Balanced creativity and consistency
    /// let mut client_balanced = Responses::new();
    /// client_balanced.temperature(0.7);
    ///
    /// // High creativity for brainstorming
    /// let mut client_creative = Responses::new();
    /// client_creative.temperature(1.1);
    /// ```
    pub temperature: Option<f64>,

    /// Optional maximum number of tokens to generate in the response
    ///
    /// Controls the maximum length of the generated response. The actual response
    /// may be shorter if the model naturally concludes or hits other stopping conditions.
    ///
    /// # Range
    ///
    /// - Minimum: 1
    /// - Maximum: Depends on the model (typically 4096-8192 for most models)
    ///
    /// # Default Behavior
    ///
    /// If not specified, the model will use its default maximum output length.
    ///
    /// # Examples
    ///
    /// - `Some(100)` - Short responses, good for summaries or brief answers
    /// - `Some(1000)` - Medium responses, suitable for detailed explanations
    /// - `Some(4000)` - Long responses, for comprehensive analysis or long-form content
    pub max_output_tokens: Option<usize>,

    /// Optional maximum number of tool calls to make
    ///
    /// Limits how many tools the model can invoke during response generation.
    /// This helps control cost and response time when using multiple tools.
    ///
    /// # Range
    ///
    /// - Minimum: 0 (no tool calls allowed)
    /// - Maximum: Implementation-dependent
    ///
    /// # Use Cases
    ///
    /// - Set to `Some(1)` for single tool usage
    /// - Set to `Some(0)` to disable tool usage entirely
    /// - Leave as `None` for unlimited tool usage (subject to other constraints)
    pub max_tool_calls: Option<usize>,

    /// Optional metadata to include with the request
    ///
    /// Arbitrary key-value pairs that can be attached to the request for
    /// tracking, logging, or passing additional context that doesn't affect
    /// the model's behavior.
    ///
    /// # Common Use Cases
    ///
    /// - Request tracking: `{"request_id": "req_123", "user_id": "user_456"}`
    /// - A/B testing: `{"experiment": "variant_a", "test_group": "control"}`
    /// - Analytics: `{"session_id": "sess_789", "feature": "chat"}`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use serde_json::Value;
    ///
    /// let mut metadata = HashMap::new();
    /// metadata.insert("user_id".to_string(), Value::String("user123".to_string()));
    /// metadata.insert("session_id".to_string(), Value::String("sess456".to_string()));
    /// metadata.insert("priority".to_string(), Value::Number(serde_json::Number::from(1)));
    /// ```
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Optional flag to enable parallel tool calls
    ///
    /// When enabled, the model can make multiple tool calls simultaneously
    /// rather than sequentially. This can significantly improve response time
    /// when multiple independent tools need to be used.
    ///
    /// # Default
    ///
    /// If not specified, defaults to the model's default behavior (usually `true`).
    ///
    /// # When to Use
    ///
    /// - `Some(true)`: Enable when tools are independent and can run in parallel
    /// - `Some(false)`: Disable when tools have dependencies or order matters
    ///
    /// # Examples
    ///
    /// - Weather + Stock prices: Can run in parallel (`true`)
    /// - File read + File analysis: Should run sequentially (`false`)
    pub parallel_tool_calls: Option<bool>,

    /// Optional fields to include in the output
    ///
    /// Specifies additional metadata and information to include in the response
    /// beyond the main generated content. This can include tool call details,
    /// reasoning traces, log probabilities, and more.
    ///
    /// # Available Inclusions
    ///
    /// - Web search call sources and results
    /// - Code interpreter execution outputs
    /// - Image URLs from various sources
    /// - Log probabilities for generated tokens
    /// - Reasoning traces and encrypted content
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Include;
    ///
    /// let includes = vec![
    ///     Include::WebSearchCall,
    ///     Include::LogprobsInOutput,
    ///     Include::ReasoningEncryptedContent,
    /// ];
    /// ```
    pub include: Option<Vec<Include>>,

    /// Optional flag to enable background processing
    ///
    /// When enabled, allows the request to be processed in the background,
    /// potentially improving throughput for non-urgent requests.
    ///
    /// # Use Cases
    ///
    /// - `Some(true)`: Batch processing, non-interactive requests
    /// - `Some(false)` or `None`: Real-time, interactive requests
    ///
    /// # Trade-offs
    ///
    /// - Background processing may have lower latency guarantees
    /// - May be more cost-effective for bulk operations
    /// - May have different rate limiting behavior
    pub background: Option<bool>,

    /// Optional conversation ID for tracking
    ///
    /// Identifier for grouping related requests as part of the same conversation
    /// or session. This helps with context management and analytics.
    ///
    /// # Format
    ///
    /// Typically a UUID or other unique identifier string.
    ///
    /// # Examples
    ///
    /// - `Some("conv_123e4567-e89b-12d3-a456-426614174000".to_string())`
    /// - `Some("user123_session456".to_string())`
    pub conversation: Option<String>,

    /// Optional ID of the previous response for context
    ///
    /// References a previous response in the same conversation to maintain
    /// context and enable features like response chaining or follow-up handling.
    ///
    /// # Use Cases
    ///
    /// - Multi-turn conversations with context preservation
    /// - Follow-up questions or clarifications
    /// - Response refinement or iteration
    ///
    /// # Examples
    ///
    /// - `Some("resp_abc123def456".to_string())`
    pub previous_response_id: Option<String>,

    /// Optional reasoning configuration
    ///
    /// Controls how the model approaches complex reasoning tasks, including
    /// the effort level and format of reasoning explanations.
    ///
    /// # Use Cases
    ///
    /// - Complex problem-solving requiring deep analysis
    /// - Mathematical or logical reasoning tasks
    /// - When you need insight into the model's reasoning process
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::{Reasoning, ReasoningEffort, ReasoningSummary};
    ///
    /// let reasoning = Reasoning {
    ///     effort: Some(ReasoningEffort::High),
    ///     summary: Some(ReasoningSummary::Detailed),
    /// };
    /// ```
    pub reasoning: Option<Reasoning>,

    /// Optional safety identifier
    ///
    /// Identifier for safety and content filtering configurations.
    /// Used to specify which safety policies should be applied to the request.
    ///
    /// # Examples
    ///
    /// - `Some("strict".to_string())` - Apply strict content filtering
    /// - `Some("moderate".to_string())` - Apply moderate content filtering
    /// - `Some("permissive".to_string())` - Apply permissive content filtering
    pub safety_identifier: Option<String>,

    /// Optional service tier specification
    ///
    /// Specifies the service tier for the request, which may affect
    /// processing priority, rate limits, and pricing.
    ///
    /// # Common Values
    ///
    /// - `Some("default".to_string())` - Standard service tier
    /// - `Some("scale".to_string())` - High-throughput tier
    /// - `Some("premium".to_string())` - Premium service tier with enhanced features
    pub service_tier: Option<String>,

    /// Optional flag to store the conversation
    ///
    /// When enabled, the conversation may be stored for future reference,
    /// training, or analytics purposes (subject to privacy policies).
    ///
    /// # Privacy Considerations
    ///
    /// - `Some(true)`: Allow storage (check privacy policies)
    /// - `Some(false)`: Explicitly opt-out of storage
    /// - `None`: Use default storage policy
    pub store: Option<bool>,

    /// Optional flag to enable streaming responses
    ///
    /// When enabled, the response will be streamed back in chunks as it's
    /// generated, allowing for real-time display of partial results.
    ///
    /// # Use Cases
    ///
    /// - `Some(true)`: Real-time chat interfaces, live text generation
    /// - `Some(false)`: Batch processing, when you need the complete response
    ///
    /// # Considerations
    ///
    /// - Streaming responses require different handling in client code
    /// - May affect some response features or formatting options
    pub stream: Option<bool>,

    /// Optional streaming configuration options
    ///
    /// Additional options for controlling streaming response behavior,
    /// such as whether to include obfuscated placeholder content.
    ///
    /// # Only Relevant When Streaming
    ///
    /// This field is only meaningful when `stream` is `Some(true)`.
    pub stream_options: Option<StreamOptions>,

    /// Optional number of top log probabilities to include
    ///
    /// Specifies how many of the most likely alternative tokens to include
    /// with their log probabilities for each generated token.
    ///
    /// # Range
    ///
    /// - Minimum: 0 (no log probabilities)
    /// - Maximum: Model-dependent (typically 5-20)
    ///
    /// # Use Cases
    ///
    /// - Model analysis and debugging
    /// - Confidence estimation
    /// - Alternative response exploration
    ///
    /// # Examples
    ///
    /// - `Some(1)` - Include the top alternative for each token
    /// - `Some(5)` - Include top 5 alternatives for detailed analysis
    pub top_logprobs: Option<usize>,

    /// Optional nucleus sampling parameter
    ///
    /// Controls the randomness of the model's responses by limiting the
    /// cumulative probability of considered tokens.
    ///
    /// # Range
    ///
    /// - 0.0 to 1.0
    /// - Lower values (e.g., 0.1) make responses more focused and deterministic
    /// - Higher values (e.g., 0.9) make responses more diverse and creative
    ///
    /// # Default
    ///
    /// If not specified, uses the model's default value (typically around 1.0).
    ///
    /// # Examples
    ///
    /// - `Some(0.1)` - Very focused, deterministic responses
    /// - `Some(0.7)` - Balanced creativity and focus
    /// - `Some(0.95)` - High creativity and diversity
    pub top_p: Option<f64>,

    /// Optional truncation behavior configuration
    ///
    /// Controls how the system handles inputs that exceed the maximum
    /// context length supported by the model.
    ///
    /// # Options
    ///
    /// - `Some(Truncation::Auto)` - Automatically truncate long inputs
    /// - `Some(Truncation::Disabled)` - Return error for long inputs
    /// - `None` - Use system default behavior
    ///
    /// # Use Cases
    ///
    /// - `Auto`: When you want to handle long documents gracefully
    /// - `Disabled`: When you need to ensure complete input processing
    pub truncation: Option<Truncation>,
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
        let mut state = serializer.serialize_struct("ResponsesBody", 4)?;
        state.serialize_field("model", &self.model)?;

        // Set input
        let input = if self.plain_text_input.is_some() {
            self.plain_text_input.clone().unwrap()
        } else if self.messages_input.is_some() {
            serde_json::to_string(&self.messages_input).unwrap()
        } else {
            return Err(serde::ser::Error::custom("Either plain_text_input or messages_input must be set."));
        };
        state.serialize_field("input", &input)?;

        // Optional fields
        if self.temperature.is_some() {
            state.serialize_field("temperature", &self.temperature)?;
        }
        if self.instructions.is_some() {
            state.serialize_field("instructions", &self.instructions)?;
        }
        if self.tools.is_some() {
            state.serialize_field("tools", &self.tools)?;
        }
        if self.structured_output.is_some() {
            state.serialize_field("text", &self.structured_output)?;
        }
        if self.max_output_tokens.is_some() {
            state.serialize_field("max_output_tokens", &self.max_output_tokens)?;
        }
        if self.max_tool_calls.is_some() {
            state.serialize_field("max_tool_calls", &self.max_tool_calls)?;
        }
        if self.metadata.is_some() {
            state.serialize_field("metadata", &self.metadata)?;
        }
        if self.parallel_tool_calls.is_some() {
            state.serialize_field("parallel_tool_calls", &self.parallel_tool_calls)?;
        }
        if self.include.is_some() {
            state.serialize_field("include", &self.include)?;
        }
        if self.background.is_some() {
            state.serialize_field("background", &self.background)?;
        }
        if self.conversation.is_some() {
            state.serialize_field("conversation", &self.conversation)?;
        }
        if self.previous_response_id.is_some() {
            state.serialize_field("previous_response_id", &self.previous_response_id)?;
        }
        if self.reasoning.is_some() {
            state.serialize_field("reasoning", &self.reasoning)?;
        }
        if self.safety_identifier.is_some() {
            state.serialize_field("safety_identifier", &self.safety_identifier)?;
        }
        if self.service_tier.is_some() {
            state.serialize_field("service_tier", &self.service_tier)?;
        }
        if self.store.is_some() {
            state.serialize_field("store", &self.store)?;
        }
        if self.stream.is_some() {
            state.serialize_field("stream", &self.stream)?;
        }
        if self.stream_options.is_some() {
            state.serialize_field("stream_options", &self.stream_options)?;
        }
        if self.top_logprobs.is_some() {
            state.serialize_field("top_logprobs", &self.top_logprobs)?;
        }
        if self.top_p.is_some() {
            state.serialize_field("top_p", &self.top_p)?;
        }
        if self.truncation.is_some() {
            state.serialize_field("truncation", &self.truncation)?;
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
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut client = Responses::new();
/// let response = client
///     .model_id("gpt-4")
///     .instructions("You are a helpful assistant.")
///     .str_message("Hello, how are you?")
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
    pub fn str_message<T: AsRef<str>>(&mut self, input: T) -> &mut Self {
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
        self.request_body.structured_output = Option::from(Format::new(text_format));
        self
    }

    /// Sets the sampling temperature for controlling response randomness
    ///
    /// Controls the randomness and creativity of the model's responses.
    /// Higher values make the output more random and creative, while lower
    /// values make it more focused and deterministic.
    ///
    /// # Arguments
    ///
    /// * `temperature` - The temperature value (0.0 to 2.0)
    ///   - 0.0: Most deterministic and focused responses
    ///   - 1.0: Default balanced behavior
    ///   - 2.0: Most random and creative responses
    ///
    /// # Panics
    ///
    /// This method will panic if the temperature value is outside the valid
    /// range of 0.0 to 2.0, as this would result in an API error.
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::responses::request::Responses;
    ///
    /// // Deterministic responses for factual queries
    /// let mut client = Responses::new();
    /// client.temperature(0.2);
    ///
    /// // Creative responses for brainstorming
    /// let mut client = Responses::new();
    /// client.temperature(1.1);
    /// ```
    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        assert!((0.0..=2.0).contains(&temperature), "Temperature must be between 0.0 and 2.0, got {}", temperature);
        self.request_body.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(&mut self, max_tokens: usize) -> &mut Self {
        self.request_body.max_output_tokens = Some(max_tokens);
        self
    }

    pub fn max_tool_calls(&mut self, max_tokens: usize) -> &mut Self {
        self.request_body.max_tool_calls = Some(max_tokens);
        self
    }

    pub fn metadata(&mut self, key: String, value: serde_json::Value) -> &mut Self {
        if self.request_body.metadata.is_none() {
            self.request_body.metadata = Some(HashMap::new());
        }
        if self.request_body.metadata.as_ref().unwrap().keys().any(|k| k == &key) {
            self.request_body.metadata.as_mut().unwrap().remove(&key);
        }
        self.request_body.metadata.as_mut().unwrap().insert(key, value);
        self
    }

    pub fn parallel_tool_calls(&mut self, enable: bool) -> &mut Self {
        self.request_body.parallel_tool_calls = Some(enable);
        self
    }

    pub fn include(&mut self, includes: Vec<Include>) -> &mut Self {
        self.request_body.include = Some(includes);
        self
    }

    pub fn background(&mut self, enable: bool) -> &mut Self {
        self.request_body.background = Some(enable);
        self
    }

    pub fn conversation<T: AsRef<str>>(&mut self, conversation_id: T) -> &mut Self {
        self.request_body.conversation = Some(conversation_id.as_ref().to_string());
        self
    }

    pub fn previous_response_id<T: AsRef<str>>(&mut self, response_id: T) -> &mut Self {
        self.request_body.previous_response_id = Some(response_id.as_ref().to_string());
        self
    }

    pub fn reasoning(&mut self, effort: ReasoningEffort, summary: ReasoningSummary) -> &mut Self {
        self.request_body.reasoning = Some(Reasoning { effort: Some(effort), summary: Some(summary) });
        self
    }

    pub fn safety_identifier<T: AsRef<str>>(&mut self, safety_id: T) -> &mut Self {
        self.request_body.safety_identifier = Some(safety_id.as_ref().to_string());
        self
    }

    pub fn service_tier<T: AsRef<str>>(&mut self, tier: T) -> &mut Self {
        self.request_body.service_tier = Some(tier.as_ref().to_string());
        self
    }

    pub fn store(&mut self, enable: bool) -> &mut Self {
        self.request_body.store = Some(enable);
        self
    }

    pub fn stream(&mut self, enable: bool) -> &mut Self {
        self.request_body.stream = Some(enable);
        self
    }

    pub fn stream_options(&mut self, include_obfuscation: bool) -> &mut Self {
        self.request_body.stream_options = Some(StreamOptions { include_obfuscation });
        self
    }

    pub fn top_logprobs(&mut self, n: usize) -> &mut Self {
        self.request_body.top_logprobs = Some(n);
        self
    }

    pub fn top_p(&mut self, p: f64) -> &mut Self {
        self.request_body.top_p = Some(p);
        self
    }

    pub fn truncation(&mut self, truncation: Truncation) -> &mut Self {
        self.request_body.truncation = Some(truncation);
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
    /// ```rust,no_run
    /// use openai_tools::responses::request::Responses;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut client = Responses::new();
    /// let response = client
    ///     .model_id("gpt-4")
    ///     .str_message("Hello!")
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
