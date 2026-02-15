//! Integration tests for the OpenAI Models API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test models_integration

use openai_tools::models::request::Models;

/// Test listing all available models.
#[tokio::test]
async fn test_list_models() {
    let models = Models::new().expect("Should create Models client");
    let response = models.list().await.expect("Should list models");

    // Verify response structure
    assert_eq!(response.object, "list");
    assert!(!response.data.is_empty(), "Should have at least one model");

    // Verify model structure
    for model in &response.data {
        assert!(!model.id.is_empty(), "Model ID should not be empty");
        assert_eq!(model.object, "model");
        assert!(model.created > 0, "Created timestamp should be positive");
        assert!(!model.owned_by.is_empty(), "owned_by should not be empty");
    }

    println!("Found {} models", response.data.len());
}

/// Test retrieving a specific model.
#[tokio::test]
async fn test_retrieve_model() {
    let models = Models::new().expect("Should create Models client");

    // Retrieve a well-known model
    let model = models.retrieve("gpt-4o-mini").await.expect("Should retrieve gpt-4o-mini");

    assert_eq!(model.id, "gpt-4o-mini");
    assert_eq!(model.object, "model");
    assert!(model.created > 0);
    assert!(!model.owned_by.is_empty());

    println!("Retrieved model: {} (owned by {})", model.id, model.owned_by);
}

/// Test retrieving multiple models.
#[tokio::test]
async fn test_retrieve_multiple_models() {
    let models = Models::new().expect("Should create Models client");

    let model_ids = vec!["gpt-4o-mini", "text-embedding-3-small"];

    for model_id in model_ids {
        let model = models.retrieve(model_id).await.expect(&format!("Should retrieve {}", model_id));

        assert_eq!(model.id, model_id);
        println!("Retrieved: {} (created: {})", model.id, model.created);
    }
}

/// Test that retrieving a non-existent model fails.
#[tokio::test]
async fn test_retrieve_nonexistent_model() {
    let models = Models::new().expect("Should create Models client");

    let result = models.retrieve("nonexistent-model-12345").await;

    // Should fail with an error (the API returns an error for non-existent models)
    assert!(result.is_err(), "Should fail for non-existent model");
}
