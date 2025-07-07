//! Message module for OpenAI tools.
//!
//! This module provides data structures and functionality for handling OpenAI API messages,
//! including text content, images, tool calls, and multi-modal interactions. It serves as
//! the foundation for communication between users and OpenAI models.
//!
//! ## Core Components
//!
//! - [`Message`] - The main message structure containing role, content, and metadata
//! - [`Content`] - Represents different types of content (text, images, etc.)
//! - [`ToolCall`] - Represents function calls made by OpenAI models
//!
//! ## Supported Content Types
//!
//! The module supports various content types for rich interactions:
//!
//! - **Text content**: Plain text messages
//! - **Image content**: Images from URLs or local files (PNG, JPEG, GIF)
//! - **Multi-modal content**: Combining text and images in a single message
//!
//! ## Usage in the Library
//!
//! This module is used throughout the OpenAI tools library:
//!
//! - In [`crate::chat::request`] - For Chat Completion API message handling
//! - In [`crate::responses::request`] - For Responses API message processing
//! - In [`crate::chat::response`] - For parsing OpenAI API responses
//!
//! ## Examples
//!
//! ### Basic Text Message
//!
//! ```rust,no_run
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! # fn main() {
//! let message = Message::from_string(Role::User, "Hello, how are you?");
//! # }
//! ```
//!
//! ### Multi-modal Message with Text and Image
//!
//! ```rust,no_run
//! use openai_tools::common::message::{Message, Content};
//! use openai_tools::common::role::Role;
//!
//! # fn main() {
//! let contents = vec![
//!     Content::from_text("What's in this image?"),
//!     Content::from_image_file("path/to/image.png"),
//! ];
//! let message = Message::from_message_array(Role::User, contents);
//! # }
//! ```
//!

use crate::common::{function::Function, role::Role};
use base64::prelude::*;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

/// Represents a tool call made by an OpenAI model.
///
/// Tool calls are generated when an OpenAI model decides to invoke a function
/// or tool as part of its response. This structure contains the metadata
/// necessary to identify and execute the requested function.
///
/// # Fields
///
/// * `id` - Unique identifier for this tool call
/// * `type_name` - The type of tool call (typically "function")
/// * `function` - The function details including name and arguments
///
/// # Examples
///
/// ```rust,no_run
/// use openai_tools::common::message::ToolCall;
/// use openai_tools::common::function::Function;
///
/// // Tool calls are typically received from OpenAI API responses
/// // and contain function invocation details
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,
    /// The type of tool call (e.g., "function")
    #[serde(rename = "type")]
    pub type_name: String,
    /// The function to be called with its arguments
    pub function: Function,
}

/// Represents different types of content that can be included in a message.
///
/// Content can be either text or images, supporting multi-modal interactions
/// with OpenAI models. Images can be provided as URLs or loaded from local files
/// and are automatically encoded as base64 data URLs.
///
/// # Supported Image Formats
///
/// * PNG
/// * JPEG/JPG
/// * GIF
///
/// # Fields
///
/// * `type_name` - The type of content ("input_text" or "input_image")
/// * `text` - Optional text content
/// * `image_url` - Optional image URL or base64 data URL
///
/// # Examples
///
/// ```rust,no_run
/// use openai_tools::common::message::Content;
///
/// // Create text content
/// let text_content = Content::from_text("Hello, world!");
///
/// // Create image content from URL
/// let image_content = Content::from_image_url("https://example.com/image.png");
///
/// // Create image content from local file
/// let file_content = Content::from_image_file("path/to/image.png");
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Content {
    /// The type of content ("input_text" or "input_image")
    #[serde(rename = "type")]
    pub type_name: String,
    /// Optional text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Optional image URL or base64 data URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl Content {
    /// Creates a new Content instance with text content.
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to include
    ///
    /// # Returns
    ///
    /// A new Content instance with type "input_text"
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::common::message::Content;
    ///
    /// let content = Content::from_text("Hello, world!");
    /// assert_eq!(content.type_name, "input_text");
    /// ```
    pub fn from_text<T: AsRef<str>>(text: T) -> Self {
        Self { type_name: "input_text".to_string(), text: Some(text.as_ref().to_string()), image_url: None }
    }

    /// Creates a new Content instance with an image URL.
    ///
    /// # Arguments
    ///
    /// * `image_url` - The URL of the image
    ///
    /// # Returns
    ///
    /// A new Content instance with type "input_image"
    ///
    /// # Examples
    ///
    /// ```rust
    /// use openai_tools::common::message::Content;
    ///
    /// let content = Content::from_image_url("https://example.com/image.png");
    /// assert_eq!(content.type_name, "input_image");
    /// ```
    pub fn from_image_url<T: AsRef<str>>(image_url: T) -> Self {
        Self { type_name: "input_image".to_string(), text: None, image_url: Some(image_url.as_ref().to_string()) }
    }

    /// Creates a new Content instance from a local image file.
    ///
    /// This method reads an image file from the filesystem, encodes it as base64,
    /// and creates a data URL suitable for use with OpenAI APIs.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A new Content instance with type "input_image" and base64-encoded image data
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The file cannot be opened
    /// - The image cannot be decoded
    /// - The image format is unsupported
    /// - The image cannot be encoded to the buffer
    ///
    /// # Supported Formats
    ///
    /// * PNG
    /// * JPEG/JPG
    /// * GIF
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::common::message::Content;
    ///
    /// let content = Content::from_image_file("path/to/image.png");
    /// assert_eq!(content.type_name, "input_image");
    /// ```
    pub fn from_image_file<T: AsRef<str>>(file_path: T) -> Self {
        let ext = file_path.as_ref();
        let ext = std::path::Path::new(&ext).extension().and_then(|s| s.to_str()).unwrap();
        let img = image::ImageReader::open(file_path.as_ref()).expect("Failed to open image file").decode().expect("Failed to decode image");
        let img_fmt = match ext {
            "png" => image::ImageFormat::Png,
            "jpg" | "jpeg" => image::ImageFormat::Jpeg,
            "gif" => image::ImageFormat::Gif,
            _ => panic!("Unsupported image format"),
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, img_fmt).expect("Failed to write image to buffer");
        let base64_string = BASE64_STANDARD.encode(buf.into_inner());
        let image_url = format!("data:image/{ext};base64,{base64_string}");
        Self { type_name: "input_image".to_string(), text: None, image_url: Some(image_url) }
    }
}

/// Represents a message in an OpenAI conversation.
///
/// Messages are the core communication unit between users and OpenAI models.
/// They can contain various types of content including text, images, tool calls,
/// and metadata like refusals and annotations.
///
/// # Content Types
///
/// A message can contain either:
/// - Single content (`content` field) - for simple text messages
/// - Multiple content items (`content_list` field) - for multi-modal messages
///
/// # Fields
///
/// * `role` - The role of the message sender (User, Assistant, System, etc.)
/// * `content` - Optional single content item
/// * `content_list` - Optional list of content items for multi-modal messages
/// * `tool_calls` - Optional list of tool calls made by the assistant
/// * `refusal` - Optional refusal message if the model declined to respond
/// * `annotations` - Optional list of annotations or metadata
///
/// # Examples
///
/// ```rust,no_run
/// use openai_tools::common::message::{Message, Content};
/// use openai_tools::common::role::Role;
///
/// // Simple text message
/// let message = Message::from_string(Role::User, "Hello!");
///
/// // Multi-modal message with text and image
/// let contents = vec![
///     Content::from_text("What's in this image?"),
///     Content::from_image_url("https://example.com/image.png"),
/// ];
/// let message = Message::from_message_array(Role::User, contents);
/// ```
#[derive(Debug, Clone)]
pub struct Message {
    /// The role of the message sender
    pub role: Role,
    /// Optional single content item
    pub content: Option<Content>,
    /// Optional list of content items for multi-modal messages
    pub content_list: Option<Vec<Content>>,
    /// Optional list of tool calls made by the assistant
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional refusal message if the model declined to respond
    pub refusal: Option<String>,
    /// Optional list of annotations or metadata
    pub annotations: Option<Vec<String>>,
}

/// Custom serialization implementation for Message.
///
/// This implementation ensures that messages are serialized correctly for the OpenAI API,
/// handling the mutually exclusive nature of `content` and `content_list` fields.
/// Either `content` or `content_list` must be present, but not both.
impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Message", 3)?;
        state.serialize_field("role", &self.role)?;

        // Ensure that either content or contents is present, but not both
        if (self.content.is_none() && self.content_list.is_none()) || (self.content.is_some() && self.content_list.is_some()) {
            return Err(serde::ser::Error::custom("Message must have either content or contents"));
        }

        // Serialize content or contents based on which one is present
        if let Some(content) = &self.content {
            state.serialize_field("content", &content.text)?;
        }
        if let Some(contents) = &self.content_list {
            state.serialize_field("content", contents)?;
        }
        state.end()
    }
}

/// Custom deserialization implementation for Message.
///
/// This implementation handles the deserialization of messages from OpenAI API responses,
/// converting string content to Content objects and handling optional fields.
impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Message, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MessageData {
            role: Role,
            content: Option<String>,
            tool_calls: Option<Vec<ToolCall>>,
            refusal: Option<String>,
            annotations: Option<Vec<String>>,
        }

        let data = MessageData::deserialize(deserializer)?;
        let content = if let Some(text) = data.content { Some(Content::from_text(text)) } else { None };

        Ok(Message {
            role: data.role,
            content,
            content_list: None,
            tool_calls: data.tool_calls,
            refusal: data.refusal,
            annotations: data.annotations,
        })
    }
}

impl Message {
    /// Creates a new Message with a single text content.
    ///
    /// This is a convenience method for creating simple text messages.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender
    /// * `message` - The text content of the message
    ///
    /// # Returns
    ///
    /// A new Message instance with the specified role and text content
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let message = Message::from_string(Role::User, "Hello, how are you?");
    /// ```
    pub fn from_string<T: AsRef<str>>(role: Role, message: T) -> Self {
        Self { role, content: Some(Content::from_text(message.as_ref())), content_list: None, tool_calls: None, refusal: None, annotations: None }
    }

    /// Creates a new Message with multiple content items.
    ///
    /// This method is used for multi-modal messages that contain multiple
    /// types of content such as text and images.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message sender
    /// * `contents` - Vector of content items to include in the message
    ///
    /// # Returns
    ///
    /// A new Message instance with the specified role and content list
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::common::message::{Message, Content};
    /// use openai_tools::common::role::Role;
    ///
    /// let contents = vec![
    ///     Content::from_text("What's in this image?"),
    ///     Content::from_image_url("https://example.com/image.png"),
    /// ];
    /// let message = Message::from_message_array(Role::User, contents);
    /// ```
    pub fn from_message_array(role: Role, contents: Vec<Content>) -> Self {
        Self { role, content: None, content_list: Some(contents), tool_calls: None, refusal: None, annotations: None }
    }

    /// Calculates the approximate token count for the message content.
    ///
    /// This method uses the tiktoken library to estimate the number of tokens
    /// that would be consumed by this message when sent to OpenAI's API.
    /// Only text content is counted; images are not included in the calculation.
    ///
    /// # Returns
    ///
    /// The estimated number of tokens for the text content in this message
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use openai_tools::common::message::Message;
    /// use openai_tools::common::role::Role;
    ///
    /// let message = Message::from_string(Role::User, "Hello, world!");
    /// let token_count = message.get_input_token_count();
    /// ```
    pub fn get_input_token_count(&self) -> usize {
        let bpe = tiktoken_rs::o200k_base().unwrap();
        if let Some(content) = &self.content {
            bpe.encode_with_special_tokens(&content.clone().text.unwrap()).len()
        } else if let Some(contents) = &self.content_list {
            let mut total_tokens = 0;
            for content in contents {
                if let Some(text) = &content.text {
                    total_tokens += bpe.encode_with_special_tokens(text).len();
                }
            }
            total_tokens
        } else {
            0 // No content to count tokens for
        }
    }
}
