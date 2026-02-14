use crate::common::{structured_output::Schema, tool::Tool, usage::Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Content within a response output.
///
/// Represents textual content returned by the AI, including any annotations or log probabilities.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Content {
    /// The type of content, typically "text"
    #[serde(rename = "type")]
    pub type_name: Option<String>,
    /// The actual text content
    pub text: Option<String>,
    /// Any annotations associated with the content
    pub annotations: Option<Vec<String>>,
    /// Log probabilities for the content tokens
    pub logprobs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FileSearchCallResult {
    /// Set of 16 key-value pairs that can be attached to an object
    pub attributes: Option<HashMap<String, String>>,
    /// The unique ID of the file
    pub file_id: Option<String>,
    /// The name of the file
    pub filename: Option<String>,
    /// The relevance score of the file - a value between 0 and 1
    pub score: Option<f64>,
    /// The text that was retrieved from the file
    pub text: Option<String>,
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
    pub id: Option<String>,
    /// The type of output: "text", "function_call", etc.
    #[serde(rename = "type")]
    pub type_name: Option<String>,
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
    /// The queries used to search for files (for file_search_call outputs)
    pub queries: Option<Vec<String>>,
    /// The results of the file search tool call (for file_search_call outputs)
    pub results: Option<Vec<FileSearchCallResult>>,
    // TODO: implement the action structure
    /// An object describing the specific action taken in this web search call (for web_search_call outputs)
    pub action: Option<Value>,
    // TODO: implement the tool_call structure
    /// The pending safety checks for the computer call (for computer_call outputs)
    pub pending_safety_checks: Option<Value>,
    /// Reasoning summary content (for reasoning outputs)
    pub summary: Option<Value>,
    /// The encrypted content of the reasoning item (for reasoning outputs)
    pub encrypted_content: Option<String>,
    // TODO: implement Image generation call
    // TODO: implement Code interpreter tool call
    // TODO: implement Local shell call
    // TODO: implement MCP tool call
    // TODO: implement MCP list tools
    // TODO: implement MCP approval request
    // TODO: implement Custom tool call
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
    pub id: Option<String>,
    /// Object type, typically "response"
    pub object: Option<String>,
    /// Unix timestamp when the response was created
    pub created_at: Option<usize>,
    /// Unix timestamp when the response completed
    pub completed_at: Option<u64>,
    /// Status of the response processing
    pub status: Option<String>,
    /// Whether the response was processed in the background
    pub background: Option<bool>,
    /// Error details if the request failed
    pub error: Option<Value>,
    /// Details about incomplete responses (e.g., `{"reason": "max_output_tokens"}`)
    pub incomplete_details: Option<Value>,
    /// Instructions that were used for this response
    pub instructions: Option<String>,
    /// Maximum number of output tokens that were allowed
    pub max_output_tokens: Option<usize>,
    /// Maximum number of tool calls that were allowed
    pub max_tool_calls: Option<usize>,
    /// The model that was used to generate the response
    pub model: Option<String>,
    /// List of outputs from the AI (text, function calls, etc.)
    pub output: Option<Vec<Output>>,
    /// Whether parallel tool calls were enabled
    pub parallel_tool_calls: Option<bool>,
    /// ID of the previous response in a conversation chain
    pub previous_response_id: Option<String>,
    /// Reasoning information from the AI
    pub reasoning: Option<Reasoning>,
    /// Service tier used for processing
    pub service_tier: Option<String>,
    /// Whether the response should be stored
    pub store: Option<bool>,
    /// Temperature setting used for generation
    pub temperature: Option<f64>,
    /// Text formatting configuration
    pub text: Option<Text>,
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
    pub metadata: Option<HashMap<String, String>>,
}

impl Response {
    pub fn output_text(&self) -> Option<String> {
        let output = if let Some(outputs) = &self.output {
            let outputs =
                outputs.iter().filter_map(|o| if o.type_name.as_deref() == Some("message") { Some(o.clone()) } else { None }).collect::<Vec<_>>();
            if outputs.is_empty() {
                tracing::warn!("No message outputs found in response");
                return None;
            }
            outputs.first().unwrap().clone()
        } else {
            return None;
        };
        let content = if let Some(contents) = &output.content {
            let contents = contents
                .iter()
                .filter_map(|c| if c.type_name.as_deref() == Some("output_text") { Some(c.clone()) } else { None })
                .collect::<Vec<_>>();
            if contents.is_empty() {
                tracing::warn!("No output_text contents found in message output");
                return None;
            }
            contents.first().unwrap().clone()
        } else {
            return None;
        };
        content.text.clone()
    }
}

/// Response for delete operations
///
/// Returned when a response is successfully deleted via the DELETE endpoint.
///
/// # API Reference
///
/// <https://platform.openai.com/docs/api-reference/responses/delete>
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DeleteResponseResult {
    /// The ID of the deleted response
    pub id: String,
    /// Object type, typically "response.deleted"
    pub object: String,
    /// Whether the deletion was successful
    pub deleted: bool,
}

/// Input item in a response
///
/// Represents a single input item that was part of the request.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponseInputItem {
    /// Unique identifier for this input item
    pub id: String,
    /// The type of input item (e.g., "message", "function_call_output")
    #[serde(rename = "type")]
    pub item_type: String,
    /// The role of the input item (e.g., "user", "assistant")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The content of the input item
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    /// The status of the input item
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// Response for listing input items
///
/// Returned when listing input items for a response via the GET endpoint.
///
/// # API Reference
///
/// <https://platform.openai.com/docs/api-reference/responses/list-input-items>
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InputItemsListResponse {
    /// Object type, typically "list"
    pub object: String,
    /// The list of input items
    pub data: Vec<ResponseInputItem>,
    /// ID of the first item in the list
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_id: Option<String>,
    /// ID of the last item in the list
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_id: Option<String>,
    /// Whether there are more items to fetch
    pub has_more: bool,
}

/// Response for compact operation
///
/// Returned when a response is compacted to reduce its size.
///
/// # API Reference
///
/// <https://platform.openai.com/docs/api-reference/responses/compact>
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CompactedResponse {
    /// Unique identifier for the compacted response
    pub id: String,
    /// Object type, typically "response"
    pub object: String,
    /// Unix timestamp when the response was created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<u64>,
    /// The compacted output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<Value>>,
    /// Token usage statistics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Response for input token counting
///
/// Returned when counting input tokens for a potential request.
///
/// # API Reference
///
/// <https://platform.openai.com/docs/api-reference/responses/input-tokens>
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InputTokensResponse {
    /// Object type, typically "input_tokens"
    pub object: String,
    /// The number of input tokens
    pub input_tokens: u64,
}
