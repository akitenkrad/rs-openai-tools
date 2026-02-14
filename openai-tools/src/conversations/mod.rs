//! # OpenAI Conversations API
//!
//! The Conversations API allows you to create and manage long-running conversations
//! with the Responses API. This is the recommended approach for building conversational
//! applications with OpenAI's models.
//!
//! ## Overview
//!
//! Conversations provide a way to:
//! - Store conversation state persistently
//! - Manage conversation items (messages, tool calls, etc.)
//! - Use with the Responses API for multi-turn interactions
//!
//! ## Key Features
//!
//! - **Create Conversations**: Initialize new conversations with optional metadata
//! - **Manage Items**: Add and retrieve conversation items
//! - **Metadata**: Attach key-value pairs for tracking and filtering
//! - **Pagination**: Navigate through conversation items efficiently
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::conversations::request::Conversations;
//! use openai_tools::conversations::response::InputItem;
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let conversations = Conversations::new()?;
//!
//!     // Create a conversation with metadata
//!     let mut metadata = HashMap::new();
//!     metadata.insert("user_id".to_string(), "user123".to_string());
//!
//!     let conv = conversations.create(Some(metadata), None).await?;
//!     println!("Created conversation: {}", conv.id);
//!
//!     // Add items to the conversation
//!     let items = vec![
//!         InputItem::user_message("Hello, how are you?"),
//!     ];
//!     conversations.create_items(&conv.id, items).await?;
//!
//!     // List conversation items
//!     let items = conversations.list_items(&conv.id, None, None, None, None).await?;
//!     for item in &items.data {
//!         println!("Item: {} ({})", item.id, item.item_type);
//!     }
//!
//!     // Delete the conversation when done
//!     conversations.delete(&conv.id).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Using with Responses API
//!
//! Conversations integrate with the Responses API to maintain context:
//!
//! ```rust,no_run
//! use openai_tools::conversations::request::Conversations;
//! use openai_tools::responses::request::Responses;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a conversation
//!     let conversations = Conversations::new()?;
//!     let conv = conversations.create(None, None).await?;
//!
//!     // Use the conversation with Responses API
//!     let mut client = Responses::new();
//!     let response = client
//!         .model_id("gpt-4.1")
//!         .conversation(&conv.id)  // Pass conversation ID
//!         .str_message("Hello!")
//!         .complete()
//!         .await?;
//!
//!     // Items are automatically added to the conversation
//!     Ok(())
//! }
//! ```
//!
//! ## API Endpoints
//!
//! | Method | Endpoint | Description |
//! |--------|----------|-------------|
//! | POST | `/v1/conversations` | Create a conversation |
//! | GET | `/v1/conversations/{id}` | Retrieve a conversation |
//! | POST | `/v1/conversations/{id}` | Update a conversation |
//! | DELETE | `/v1/conversations/{id}` | Delete a conversation |
//! | POST | `/v1/conversations/{id}/items` | Create items |
//! | GET | `/v1/conversations/{id}/items` | List items |
//!
//! ## Related Modules
//!
//! - [`crate::responses`] - Use conversations with the Responses API
//! - [`crate::chat`] - Alternative for simpler chat interactions

pub mod request;
pub mod response;

pub use request::{ConversationInclude, Conversations};
pub use response::{Conversation, ConversationItem, ConversationItemListResponse, ConversationListResponse, DeleteConversationResponse, InputItem};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_conversation_deserialize() {
        let json = r#"{
            "id": "conv_abc123",
            "object": "conversation",
            "created_at": 1741900000,
            "metadata": {"topic": "demo"}
        }"#;

        let conversation: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conversation.id, "conv_abc123");
        assert_eq!(conversation.object, "conversation");
        assert_eq!(conversation.created_at, 1741900000);
        assert!(conversation.metadata.is_some());
        assert_eq!(conversation.metadata.as_ref().unwrap().get("topic"), Some(&"demo".to_string()));
    }

    #[test]
    fn test_conversation_without_metadata() {
        let json = r#"{
            "id": "conv_xyz789",
            "object": "conversation",
            "created_at": 1741900000
        }"#;

        let conversation: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conversation.id, "conv_xyz789");
        assert!(conversation.metadata.is_none());
    }

    #[test]
    fn test_conversation_list_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "id": "conv_1",
                    "object": "conversation",
                    "created_at": 1741900000,
                    "metadata": {}
                },
                {
                    "id": "conv_2",
                    "object": "conversation",
                    "created_at": 1741900001,
                    "metadata": null
                }
            ],
            "first_id": "conv_1",
            "last_id": "conv_2",
            "has_more": true
        }"#;

        let response: ConversationListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.first_id, Some("conv_1".to_string()));
        assert_eq!(response.last_id, Some("conv_2".to_string()));
        assert!(response.has_more);
    }

    #[test]
    fn test_conversation_item_deserialize() {
        // Test with object field (legacy format)
        let json = r#"{
            "id": "item_abc123",
            "object": "conversation.item",
            "type": "message",
            "role": "user",
            "content": "Hello!",
            "status": "completed"
        }"#;

        let item: ConversationItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.id, "item_abc123");
        assert_eq!(item.object, Some("conversation.item".to_string()));
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, Some("user".to_string()));
        assert_eq!(item.status, Some("completed".to_string()));
    }

    #[test]
    fn test_conversation_item_deserialize_api_format() {
        // Test without object field (actual API response format)
        let json = r#"{
            "id": "msg_abc123",
            "type": "message",
            "status": "completed",
            "content": [{"type": "input_text", "text": "Hello!"}],
            "role": "user"
        }"#;

        let item: ConversationItem = serde_json::from_str(json).unwrap();
        assert_eq!(item.id, "msg_abc123");
        assert_eq!(item.object, None);
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, Some("user".to_string()));
        assert_eq!(item.status, Some("completed".to_string()));
        assert!(item.content.is_some());
    }

    #[test]
    fn test_conversation_item_list_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "id": "item_1",
                    "object": "conversation.item",
                    "type": "message",
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "id": "item_2",
                    "object": "conversation.item",
                    "type": "message",
                    "role": "assistant",
                    "content": "Hi there!"
                }
            ],
            "first_id": "item_1",
            "last_id": "item_2",
            "has_more": false
        }"#;

        let response: ConversationItemListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].role, Some("user".to_string()));
        assert_eq!(response.data[1].role, Some("assistant".to_string()));
        assert!(!response.has_more);
    }

    #[test]
    fn test_delete_conversation_response() {
        let json = r#"{
            "id": "conv_abc123",
            "object": "conversation.deleted",
            "deleted": true
        }"#;

        let response: DeleteConversationResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "conv_abc123");
        assert_eq!(response.object, "conversation.deleted");
        assert!(response.deleted);
    }

    #[test]
    fn test_input_item_message() {
        let item = InputItem::message("user", "Hello!");
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, Some("user".to_string()));

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_input_item_user_message() {
        let item = InputItem::user_message("How are you?");
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, Some("user".to_string()));
    }

    #[test]
    fn test_input_item_assistant_message() {
        let item = InputItem::assistant_message("I'm doing well!");
        assert_eq!(item.item_type, "message");
        assert_eq!(item.role, Some("assistant".to_string()));
    }

    #[test]
    fn test_conversation_include_as_str() {
        assert_eq!(ConversationInclude::WebSearchCallSources.as_str(), "web_search_call.action.sources");
        assert_eq!(ConversationInclude::CodeInterpreterCallOutputs.as_str(), "code_interpreter_call.outputs");
        assert_eq!(ConversationInclude::FileSearchCallResults.as_str(), "file_search_call.results");
        assert_eq!(ConversationInclude::MessageInputImageUrl.as_str(), "message.input_image.image_url");
        assert_eq!(ConversationInclude::ReasoningEncryptedContent.as_str(), "reasoning.encrypted_content");
    }

    #[test]
    fn test_create_request_serialization() {
        let mut metadata = HashMap::new();
        metadata.insert("topic".to_string(), "test".to_string());

        let items = vec![InputItem::user_message("Hello!")];

        // Test that we can serialize with both metadata and items
        let body = serde_json::json!({
            "metadata": metadata,
            "items": items
        });

        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("\"topic\":\"test\""));
        assert!(json.contains("\"type\":\"message\""));
    }

    #[test]
    fn test_conversation_with_empty_metadata() {
        let json = r#"{
            "id": "conv_empty",
            "object": "conversation",
            "created_at": 1741900000,
            "metadata": {}
        }"#;

        let conversation: Conversation = serde_json::from_str(json).unwrap();
        assert_eq!(conversation.id, "conv_empty");
        assert!(conversation.metadata.is_some());
        assert!(conversation.metadata.as_ref().unwrap().is_empty());
    }
}
