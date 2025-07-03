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
//! let message = Message::from_string("user".to_string(), "Hello, world!".to_string());
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

use base64::prelude::*;
use derive_new::new;
use fxhash::FxHashMap;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

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

/// Represents the content of a message, which can be either text or an image.
///
/// This structure is used to encapsulate different types of message content
/// that can be sent to or received from AI models. It supports both text-based
/// content and image content (either as URLs or base64-encoded data).
///
/// # Fields
///
/// * `type_name` - The type of content ("input_text" for text, "input_image" for images)
/// * `text` - Optional text content when the message contains text
/// * `image_url` - Optional image URL or base64-encoded image data when the message contains an image
///
/// # Example
///
/// ```rust
/// use openai_tools::common::MessageContent;
///
/// // Create text content
/// let text_content = MessageContent::from_text("Hello, world!".to_string());
///
/// // Create image content from URL
/// // let image_content = MessageContent::from_image_url("https://example.com/image.jpg".to_string());
///
/// // Create image content from file
/// // let file_content = MessageContent::from_image_file("path/to/image.jpg".to_string());
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MessageContent {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl MessageContent {
    /// Creates a new `MessageContent` containing text.
    ///
    /// This constructor creates a message content instance specifically for text-based
    /// messages. The content type is automatically set to "input_text" and the
    /// image_url field is set to None.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to include in the message
    ///
    /// # Returns
    ///
    /// A new `MessageContent` instance configured for text content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::MessageContent;
    ///
    /// let content = MessageContent::from_text("Hello, AI assistant!".to_string());
    /// assert_eq!(content.type_name, "input_text");
    /// assert_eq!(content.text, Some("Hello, AI assistant!".to_string()));
    /// assert_eq!(content.image_url, None);
    /// ```
    pub fn from_text(text: String) -> Self {
        Self {
            type_name: "input_text".to_string(),
            text: Some(text),
            image_url: None,
        }
    }

    /// Creates a new `MessageContent` containing an image from a URL.
    ///
    /// This constructor creates a message content instance for image-based messages
    /// using an existing image URL. The content type is automatically set to
    /// "input_image" and the text field is set to None.
    ///
    /// # Arguments
    ///
    /// * `image_url` - The URL or base64-encoded data URI of the image
    ///
    /// # Returns
    ///
    /// A new `MessageContent` instance configured for image content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::MessageContent;
    ///
    /// let content = MessageContent::from_image_url("https://example.com/image.jpg".to_string());
    /// assert_eq!(content.type_name, "input_image");
    /// assert_eq!(content.text, None);
    /// assert_eq!(content.image_url, Some("https://example.com/image.jpg".to_string()));
    /// ```
    pub fn from_image_url(image_url: String) -> Self {
        Self {
            type_name: "input_image".to_string(),
            text: None,
            image_url: Some(image_url),
        }
    }

    /// Creates a new `MessageContent` containing an image loaded from a file.
    ///
    /// This constructor reads an image file from the local filesystem, encodes it
    /// as base64, and creates a data URI suitable for use with AI models. The
    /// content type is automatically set to "input_image" and the text field
    /// is set to None.
    ///
    /// # Arguments
    ///
    /// * `file_path` - The path to the image file to load
    ///
    /// # Returns
    ///
    /// A new `MessageContent` instance configured for image content with base64-encoded data.
    ///
    /// # Supported Formats
    ///
    /// - PNG (.png)
    /// - JPEG (.jpg, .jpeg)
    /// - GIF (.gif)
    ///
    pub fn from_image_file(file_path: String) -> Self {
        let ext = file_path.clone();
        let ext = std::path::Path::new(&ext)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap();
        let img = image::ImageReader::open(file_path.clone())
            .expect("Failed to open image file")
            .decode()
            .expect("Failed to decode image");
        let img_fmt = match ext {
            "png" => image::ImageFormat::Png,
            "jpg" | "jpeg" => image::ImageFormat::Jpeg,
            "gif" => image::ImageFormat::Gif,
            _ => panic!("Unsupported image format"),
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, img_fmt)
            .expect("Failed to write image to buffer");
        let base64_string = BASE64_STANDARD.encode(buf.into_inner());
        let image_url = format!("data:image/{};base64,{}", ext, base64_string);
        Self {
            type_name: "input_image".to_string(),
            text: None,
            image_url: Some(image_url),
        }
    }
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
/// let system_msg = Message::from_string(
///     "system".to_string(),
///     "You are a helpful assistant that explains complex topics simply.".to_string()
/// );
///
/// // Create a user message
/// let user_msg = Message::from_string(
///     "user".to_string(),
///     "What is quantum computing?".to_string()
/// );
///
/// // Create an assistant response
/// let assistant_msg = Message::from_string(
///     "assistant".to_string(),
///     "Quantum computing is a type of computation that uses quantum mechanics...".to_string()
/// );
///
/// // Messages are typically used in vectors for conversation history
/// let conversation = vec![system_msg, user_msg, assistant_msg];
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    role: String,
    content: Option<MessageContent>,
    contents: Option<Vec<MessageContent>>,
}

impl Serialize for Message {
    /// Custom serialization implementation for `Message`.
    ///
    /// This method ensures that messages are serialized correctly by enforcing
    /// that either `content` or `contents` is present, but not both. This prevents
    /// invalid message structures from being serialized.
    ///
    /// # Arguments
    ///
    /// * `serializer` - The serializer to use for output
    ///
    /// # Returns
    ///
    /// Result of the serialization operation
    ///
    /// # Errors
    ///
    /// Returns a serialization error if:
    /// - Both `content` and `contents` are present
    /// - Neither `content` nor `contents` are present
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Message", 3)?;
        state.serialize_field("role", &self.role)?;

        // Ensure that either content or contents is present, but not both
        if (self.content.is_none() && self.contents.is_none())
            || (self.content.is_some() && self.contents.is_some())
        {
            return Err(serde::ser::Error::custom(
                "Message must have either content or contents",
            ));
        }

        // Serialize content or contents based on which one is present
        if let Some(content) = &self.content {
            state.serialize_field("content", &content.text)?;
        }
        if let Some(contents) = &self.contents {
            state.serialize_field("content", contents)?;
        }
        state.end()
    }
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
    /// # use openai_tools::common::Message;
    /// # pub fn main() {
    /// // Create various types of messages
    /// let system_message = Message::from_string(
    ///     "system".to_string(),
    ///     "You are a helpful AI assistant.".to_string()
    /// );
    ///
    /// let user_message = Message::from_string(
    ///     "user".to_string(),
    ///     "Hello! How are you today?".to_string()
    /// );
    ///
    /// let assistant_message = Message::from_string(
    ///     "assistant".to_string(),
    ///     "Hello! I'm doing well, thank you for asking.".to_string()
    /// );
    /// # }
    /// ```
    /// Creates a new `Message` from a role and text string.
    ///
    /// This constructor creates a message with a single text content. It's the
    /// most common way to create simple text-based messages for conversations
    /// with AI models.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender (e.g., "user", "assistant", "system")
    /// * `message` - The text content of the message
    ///
    /// # Returns
    ///
    /// A new `Message` instance with the specified role and text content.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::Message;
    ///
    /// let user_message = Message::from_string(
    ///     "user".to_string(),
    ///     "What is the weather like today?".to_string()
    /// );
    ///
    /// let system_message = Message::from_string(
    ///     "system".to_string(),
    ///     "You are a helpful weather assistant.".to_string()
    /// );
    /// ```
    pub fn from_string(role: String, message: String) -> Self {
        Self {
            role: String::from(role),
            content: Some(MessageContent::from_text(String::from(message))),
            contents: None,
        }
    }

    /// Creates a new `Message` from a role and an array of message contents.
    ///
    /// This constructor allows creating messages with multiple content types,
    /// such as messages that contain both text and images. This is useful for
    /// multimodal conversations where a single message may include various
    /// types of content.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender (e.g., "user", "assistant", "system")
    /// * `contents` - A vector of `MessageContent` instances representing different content types
    ///
    /// # Returns
    ///
    /// A new `Message` instance with the specified role and multiple content elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::{Message, MessageContent};
    ///
    /// let contents = vec![
    ///     MessageContent::from_text("Please analyze this image:".to_string()),
    ///     MessageContent::from_image_url("https://example.com/image.jpg".to_string()),
    /// ];
    ///
    /// let multimodal_message = Message::from_message_array(
    ///     "user".to_string(),
    ///     contents
    /// );
    /// ```
    pub fn from_message_array(role: String, contents: Vec<MessageContent>) -> Self {
        Self {
            role: String::from(role),
            content: None,
            contents: Some(contents),
        }
    }

    /// Calculates the number of input tokens for this message.
    ///
    /// This method uses the OpenAI tiktoken tokenizer (o200k_base) to count
    /// the number of tokens in the text content of the message. This is useful
    /// for estimating API costs and ensuring messages don't exceed token limits.
    ///
    /// # Returns
    ///
    /// The number of tokens in the message's text content. Returns 0 if:
    /// - The message has no content
    /// - The message content has no text (e.g., image-only messages)
    ///
    /// # Note
    ///
    /// This method only counts tokens for text content. Image content tokens
    /// are not included in the count as they are calculated differently by
    /// the OpenAI API.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::Message;
    ///
    /// let message = Message::from_string(
    ///     "user".to_string(),
    ///     "Hello, how are you today?".to_string()
    /// );
    ///
    /// let token_count = message.get_input_token_count();
    /// println!("Message contains {} tokens", token_count);
    /// ```
    pub fn get_input_token_count(&self) -> usize {
        let bpe = tiktoken_rs::o200k_base().unwrap();
        if let Some(content) = &self.content {
            return bpe
                .encode_with_special_tokens(&content.clone().text.unwrap())
                .len();
        } else if let Some(contents) = &self.contents {
            let mut total_tokens = 0;
            for content in contents {
                if let Some(text) = &content.text {
                    total_tokens += bpe.encode_with_special_tokens(text).len();
                }
            }
            return total_tokens;
        } else {
            return 0; // No content to count tokens for
        }
    }
}
