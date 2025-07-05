use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Token usage statistics for OpenAI API requests.
///
/// This structure contains detailed information about token consumption during
/// API requests, including both input (prompt) and output (completion) tokens.
/// Different fields may be populated depending on the specific API endpoint
/// and model used.
///
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: Option<usize>,
    pub input_tokens_details: Option<HashMap<String, usize>>,
    pub output_tokens: Option<usize>,
    pub output_tokens_details: Option<HashMap<String, usize>>,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
    pub completion_tokens_details: Option<HashMap<String, usize>>,
}
