//! # Common Types and Structures
//!
//! This module contains common data structures and types used across the OpenAI Tools library.
//! These structures represent core concepts like messages, token usage, and other shared
//! components that are used by multiple API endpoints.
//!
//! ## Key Components
//!
//! - **Message**: Represents a single message in a conversation
//! - **Usage**: Token usage statistics for API requests
//!
//! ## Example
//!
//! ```rust
//! use openai_tools::common::{Message, Usage};
//!
//! // Create a user message
//! let message = Message::new("user".to_string(), "Hello, world!".to_string());
//!
//! // Usage is typically returned by API responses
//! let usage = Usage::new(
//!     Some(10),    // input_tokens
//!     None,        // input_tokens_details
//!     Some(20),    // output_tokens
//!     None,        // output_tokens_details
//!     Some(10),    // prompt_tokens
//!     Some(20),    // completion_tokens
//!     Some(30),    // total_tokens
//!     None,        // completion_tokens_details
//! );
//!
//! println!("Total tokens used: {:?}", usage.total_tokens);
//! ```

use derive_new::new;
use fxhash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Token usage statistics for OpenAI API requests.
///
/// This structure contains detailed information about token consumption during
/// API requests, including both input (prompt) and output (completion) tokens.
/// Different fields may be populated depending on the specific API endpoint
/// and model used.
///
/// # Fields
///
/// * `input_tokens` - Number of tokens in the input/prompt
/// * `input_tokens_details` - Detailed breakdown of input token usage by category
/// * `output_tokens` - Number of tokens in the output/completion
/// * `output_tokens_details` - Detailed breakdown of output token usage by category
/// * `prompt_tokens` - Legacy field for input tokens (may be deprecated)
/// * `completion_tokens` - Legacy field for output tokens (may be deprecated)
/// * `total_tokens` - Total number of tokens used (input + output)
/// * `completion_tokens_details` - Detailed breakdown of completion token usage
///
/// # Note
///
/// Not all fields will be populated for every request. The availability of
/// detailed token breakdowns depends on the model and API endpoint being used.
///
/// # Example
///
/// ```rust
/// use openai_tools::common::Usage;
///
/// // Create usage statistics manually (typically done by API response parsing)
/// let usage = Usage::new(
///     Some(25),    // input tokens
///     None,        // no detailed input breakdown
///     Some(50),    // output tokens
///     None,        // no detailed output breakdown
///     Some(25),    // prompt tokens (legacy)
///     Some(50),    // completion tokens (legacy)
///     Some(75),    // total tokens
///     None,        // no detailed completion breakdown
/// );
///
/// if let Some(total) = usage.total_tokens {
///     println!("Request used {} tokens total", total);
/// }
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize, new)]
pub struct Usage {
    pub input_tokens: Option<usize>,
    pub input_tokens_details: Option<FxHashMap<String, usize>>,
    pub output_tokens: Option<usize>,
    pub output_tokens_details: Option<FxHashMap<String, usize>>,
    pub prompt_tokens: Option<usize>,
    pub completion_tokens: Option<usize>,
    pub total_tokens: Option<usize>,
    pub completion_tokens_details: Option<FxHashMap<String, usize>>,
}

/// Represents a single message in a conversation with an AI model.
///
/// Messages are the fundamental building blocks of conversations in chat-based
/// AI interactions. Each message has a role (indicating who sent it) and content
/// (the actual message text). Messages can also contain refusal information
/// when the AI model declines to respond to certain requests.
///
/// # Roles
///
/// Common roles include:
/// - **"system"**: System messages that set the behavior or context for the AI
/// - **"user"**: Messages from the human user
/// - **"assistant"**: Messages from the AI assistant
/// - **"function"**: Messages related to function/tool calls (for advanced use cases)
///
/// # Fields
///
/// * `role` - The role of the message sender
/// * `content` - The text content of the message
/// * `refusal` - Optional refusal message if the AI declined to respond
///
/// # Example
///
/// ```rust
/// use openai_tools::common::Message;
///
/// // Create a system message to set context
/// let system_msg = Message::new(
///     "system".to_string(),
///     "You are a helpful assistant that explains complex topics simply.".to_string()
/// );
///
/// // Create a user message
/// let user_msg = Message::new(
///     "user".to_string(),
///     "What is quantum computing?".to_string()
/// );
///
/// // Create an assistant response
/// let assistant_msg = Message::new(
///     "assistant".to_string(),
///     "Quantum computing is a type of computation that uses quantum mechanics...".to_string()
/// );
///
/// // Messages are typically used in vectors for conversation history
/// let conversation = vec![system_msg, user_msg, assistant_msg];
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl Message {
    /// Creates a new `Message` with the specified role and content.
    ///
    /// This is the primary constructor for creating message instances.
    /// The `refusal` field is automatically set to `None` and can be
    /// modified separately if needed.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender (e.g., "user", "assistant", "system")
    /// * `message` - The text content of the message
    ///
    /// # Returns
    ///
    /// A new `Message` instance with the specified role and content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::Message;
    ///
    /// // Create various types of messages
    /// let system_message = Message::new(
    ///     "system".to_string(),
    ///     "You are a helpful AI assistant.".to_string()
    /// );
    ///
    /// let user_message = Message::new(
    ///     "user".to_string(),
    ///     "Hello! How are you today?".to_string()
    /// );
    ///
    /// let assistant_message = Message::new(
    ///     "assistant".to_string(),
    ///     "Hello! I'm doing well, thank you for asking.".to_string()
    /// );
    ///
    /// // Verify the message was created correctly
    /// assert_eq!(user_message.role, "user");
    /// assert_eq!(user_message.content, "Hello! How are you today?");
    /// assert_eq!(user_message.refusal, None);
    /// ```
    pub fn new(role: String, message: String) -> Self {
        Self {
            role: String::from(role),
            content: String::from(message),
            refusal: None,
        }
    }
}
