//! OpenAI Conversations API Response Types
//!
//! This module defines the response structures for the OpenAI Conversations API.
//! The Conversations API allows you to create and manage long-running conversations
//! with the Responses API.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents an OpenAI Conversation object.
///
/// A conversation stores items (messages, tool calls, tool outputs, etc.)
/// and can be used with the Responses API for multi-turn interactions.
///
/// # Example
///
/// ```rust
/// use openai_tools::conversations::response::Conversation;
///
/// // Example conversation from API response
/// let json = r#"{
///     "id": "conv_abc123",
///     "object": "conversation",
///     "created_at": 1741900000,
///     "metadata": {"topic": "demo"}
/// }"#;
///
/// let conversation: Conversation = serde_json::from_str(json).unwrap();
/// assert_eq!(conversation.id, "conv_abc123");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// The unique ID of the conversation (e.g., "conv_abc123")
    pub id: String,
    /// Object type, always "conversation"
    pub object: String,
    /// Unix timestamp (in seconds) when the conversation was created
    pub created_at: i64,
    /// Set of key-value pairs for storing additional information
    ///
    /// Keys are strings with a maximum length of 64 characters.
    /// Values are strings with a maximum length of 512 characters.
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

/// Response structure for listing conversations.
///
/// Contains a list of conversation objects with pagination information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationListResponse {
    /// Object type, always "list"
    pub object: String,
    /// Array of conversation objects
    pub data: Vec<Conversation>,
    /// The ID of the first object in the list
    #[serde(default)]
    pub first_id: Option<String>,
    /// The ID of the last object in the list
    #[serde(default)]
    pub last_id: Option<String>,
    /// Whether there are more objects available
    #[serde(default)]
    pub has_more: bool,
}

/// Represents a conversation item.
///
/// Items can be messages, tool calls, tool outputs, reasoning, or other types
/// that form the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    /// The unique ID of the item
    pub id: String,
    /// Object type (may not be present in API responses)
    #[serde(default)]
    pub object: Option<String>,
    /// The type of item (e.g., "message", "tool_call", "tool_output")
    #[serde(rename = "type")]
    pub item_type: String,
    /// The role of the item (e.g., "user", "assistant", "system")
    #[serde(default)]
    pub role: Option<String>,
    /// The content of the item (structure varies by type)
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    /// The status of the item
    #[serde(default)]
    pub status: Option<String>,
}

/// Response structure for listing conversation items.
///
/// Contains a list of conversation item objects with pagination information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemListResponse {
    /// Object type, always "list"
    pub object: String,
    /// Array of conversation item objects
    pub data: Vec<ConversationItem>,
    /// The ID of the first item in the list
    #[serde(default)]
    pub first_id: Option<String>,
    /// The ID of the last item in the list
    #[serde(default)]
    pub last_id: Option<String>,
    /// Whether there are more items available
    #[serde(default)]
    pub has_more: bool,
}

/// Response structure for conversation deletion.
///
/// Returned when a conversation is successfully deleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteConversationResponse {
    /// The conversation ID that was deleted
    pub id: String,
    /// Object type, always "conversation.deleted"
    pub object: String,
    /// Whether the conversation was successfully deleted
    pub deleted: bool,
}

/// Input item for creating conversation items.
///
/// Used when adding new items to a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputItem {
    /// The type of item (e.g., "message")
    #[serde(rename = "type")]
    pub item_type: String,
    /// The role of the item (e.g., "user", "assistant")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The content of the item (can be a string or structured content)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
}

impl InputItem {
    /// Creates a new message item.
    ///
    /// # Arguments
    ///
    /// * `role` - The role of the message (e.g., "user", "assistant")
    /// * `content` - The text content of the message
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::conversations::response::InputItem;
    ///
    /// let item = InputItem::message("user", "Hello, how are you?");
    /// assert_eq!(item.item_type, "message");
    /// assert_eq!(item.role, Some("user".to_string()));
    /// ```
    pub fn message(role: &str, content: &str) -> Self {
        Self { item_type: "message".to_string(), role: Some(role.to_string()), content: Some(serde_json::Value::String(content.to_string())) }
    }

    /// Creates a new user message item.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::conversations::response::InputItem;
    ///
    /// let item = InputItem::user_message("Hello!");
    /// assert_eq!(item.role, Some("user".to_string()));
    /// ```
    pub fn user_message(content: &str) -> Self {
        Self::message("user", content)
    }

    /// Creates a new assistant message item.
    ///
    /// # Arguments
    ///
    /// * `content` - The text content of the message
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::conversations::response::InputItem;
    ///
    /// let item = InputItem::assistant_message("Hi there!");
    /// assert_eq!(item.role, Some("assistant".to_string()));
    /// ```
    pub fn assistant_message(content: &str) -> Self {
        Self::message("assistant", content)
    }
}
