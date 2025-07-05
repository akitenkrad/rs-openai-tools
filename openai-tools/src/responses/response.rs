use crate::common::{structured_output::Schema, tool::Tool, usage::Usage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Content within a response output.
///
/// Represents textual content returned by the AI, including any annotations or log probabilities.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Content {
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Output {
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
    pub content: Option<Vec<Content>>,
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Reasoning {
    /// The effort level used in reasoning
    pub effort: Option<String>,
    /// Summary of the reasoning process
    pub summary: Option<String>,
}

/// Text formatting configuration for responses.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Text {
    /// Format configuration
    pub format: Schema,
}

/// Complete response from the OpenAI Responses API.
///
/// This struct contains all the information returned by the API, including the AI's outputs,
/// usage statistics, and metadata about the request processing.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Response {
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
    pub output: Vec<Output>,
    /// Whether parallel tool calls were enabled
    pub parallel_tool_calls: bool,
    /// ID of the previous response in a conversation chain
    pub previous_response_id: Option<String>,
    /// Reasoning information from the AI
    pub reasoning: Reasoning,
    /// Service tier used for processing
    pub service_tier: Option<String>,
    /// Whether the response should be stored
    pub store: Option<bool>,
    /// Temperature setting used for generation
    pub temperature: Option<f64>,
    /// Text formatting configuration
    pub text: Text,
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
    pub metadata: HashMap<String, String>,
}
