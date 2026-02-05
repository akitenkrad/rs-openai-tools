//! Integration tests for the OpenAI Conversations API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test conversations_integration

use openai_tools::conversations::request::Conversations;
use openai_tools::conversations::response::InputItem;
use std::collections::HashMap;

/// Test creating a conversation and deleting it.
#[tokio::test]
async fn test_create_and_delete_conversation() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create a conversation with metadata
    let mut metadata = HashMap::new();
    metadata.insert("test".to_string(), "integration".to_string());

    let conv = conversations.create(Some(metadata), None).await.expect("Should create conversation");

    // Verify conversation structure
    assert!(conv.id.starts_with("conv_"), "Conversation ID should start with 'conv_'");
    assert_eq!(conv.object, "conversation");
    assert!(conv.created_at > 0, "Created timestamp should be positive");
    assert!(conv.metadata.is_some(), "Metadata should be present");

    println!("Created conversation: {}", conv.id);

    // Delete the conversation
    let delete_result = conversations.delete(&conv.id).await.expect("Should delete conversation");

    assert!(delete_result.deleted, "Conversation should be deleted");
    assert_eq!(delete_result.id, conv.id);

    println!("Deleted conversation: {}", delete_result.id);
}

/// Test creating a conversation with initial items.
#[tokio::test]
async fn test_create_conversation_with_items() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create conversation with initial items
    let items = vec![InputItem::user_message("Hello, this is a test!")];

    let conv = conversations.create(None, Some(items)).await.expect("Should create conversation with items");

    assert!(conv.id.starts_with("conv_"));
    println!("Created conversation with items: {}", conv.id);

    // Cleanup
    conversations.delete(&conv.id).await.expect("Should delete conversation");
}

/// Test retrieving a conversation.
#[tokio::test]
async fn test_retrieve_conversation() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create a conversation
    let mut metadata = HashMap::new();
    metadata.insert("purpose".to_string(), "retrieve-test".to_string());

    let created = conversations.create(Some(metadata), None).await.expect("Should create conversation");

    // Retrieve the conversation
    let retrieved = conversations.retrieve(&created.id).await.expect("Should retrieve conversation");

    assert_eq!(retrieved.id, created.id);
    assert_eq!(retrieved.object, "conversation");
    assert_eq!(retrieved.created_at, created.created_at);

    println!("Retrieved conversation: {} (created at {})", retrieved.id, retrieved.created_at);

    // Cleanup
    conversations.delete(&created.id).await.ok();
}

/// Test updating a conversation's metadata.
#[tokio::test]
async fn test_update_conversation() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create a conversation
    let mut initial_metadata = HashMap::new();
    initial_metadata.insert("status".to_string(), "initial".to_string());

    let created = conversations.create(Some(initial_metadata), None).await.expect("Should create conversation");

    // Update the conversation
    let mut new_metadata = HashMap::new();
    new_metadata.insert("status".to_string(), "updated".to_string());
    new_metadata.insert("extra".to_string(), "new-field".to_string());

    let updated = conversations.update(&created.id, new_metadata).await.expect("Should update conversation");

    assert_eq!(updated.id, created.id);
    assert!(updated.metadata.is_some());

    let metadata = updated.metadata.as_ref().unwrap();
    assert_eq!(metadata.get("status"), Some(&"updated".to_string()));
    assert_eq!(metadata.get("extra"), Some(&"new-field".to_string()));

    println!("Updated conversation: {} with metadata: {:?}", updated.id, metadata);

    // Cleanup
    conversations.delete(&created.id).await.ok();
}

/// Test adding items to a conversation.
#[tokio::test]
async fn test_create_items() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create a conversation
    let conv = conversations.create(None, None).await.expect("Should create conversation");

    // Add items
    let items = vec![InputItem::user_message("What is the capital of France?")];

    let result = conversations.create_items(&conv.id, items).await.expect("Should create items");

    assert_eq!(result.object, "list");
    assert!(!result.data.is_empty(), "Should have added items");

    println!("Added {} items to conversation {}", result.data.len(), conv.id);

    // Cleanup
    conversations.delete(&conv.id).await.ok();
}

/// Test listing items in a conversation.
#[tokio::test]
async fn test_list_items() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    // Create a conversation with initial items
    let items = vec![InputItem::user_message("First message"), InputItem::user_message("Second message")];

    let conv = conversations.create(None, Some(items)).await.expect("Should create conversation");

    // List items
    let result = conversations.list_items(&conv.id, Some(10), None, None, None).await.expect("Should list items");

    assert_eq!(result.object, "list");
    println!("Conversation {} has {} items", conv.id, result.data.len());

    for item in &result.data {
        println!("  - {} ({}): {:?}", item.id, item.item_type, item.role);
    }

    // Cleanup
    conversations.delete(&conv.id).await.ok();
}

/// Test that retrieving a non-existent conversation fails.
#[tokio::test]
async fn test_retrieve_nonexistent_conversation() {
    let conversations = Conversations::new().expect("Should create Conversations client");

    let result = conversations.retrieve("conv_nonexistent12345").await;

    assert!(result.is_err(), "Should fail for non-existent conversation");
    println!("Expected error for non-existent conversation: {:?}", result.err());
}
