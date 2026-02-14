//! Integration tests for the OpenAI Batch API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test batch_integration

use openai_tools::batch::request::Batches;

/// Test listing batches
#[tokio::test]
async fn test_list_batches() {
    let batches = Batches::new().expect("Failed to create Batches client");

    let response = batches.list(Some(10), None).await;
    assert!(response.is_ok(), "Failed to list batches: {:?}", response.err());

    let list = response.unwrap();
    assert_eq!(list.object, "list");
    // The data array may be empty if no batches exist
    println!("Found {} batches", list.data.len());
}

/// Test listing batches with pagination
#[tokio::test]
async fn test_list_batches_with_pagination() {
    let batches = Batches::new().expect("Failed to create Batches client");

    // First page
    let response = batches.list(Some(5), None).await;
    assert!(response.is_ok(), "Failed to list batches: {:?}", response.err());

    let first_page = response.unwrap();
    println!("First page: {} batches, has_more: {}", first_page.data.len(), first_page.has_more);

    // If there are more, get next page
    if first_page.has_more {
        if let Some(last_id) = &first_page.last_id {
            let second_page = batches.list(Some(5), Some(last_id)).await;
            assert!(second_page.is_ok(), "Failed to get second page: {:?}", second_page.err());
            println!("Second page: {} batches", second_page.unwrap().data.len());
        }
    }
}

/// Test retrieving a non-existent batch (should fail gracefully)
#[tokio::test]
async fn test_retrieve_nonexistent_batch() {
    let batches = Batches::new().expect("Failed to create Batches client");

    // This should fail with an error, not panic
    let result = batches.retrieve("batch_nonexistent123").await;
    assert!(result.is_err(), "Expected error for non-existent batch");
}

// Note: The following tests are commented out because they require
// a valid input file to be uploaded first, which incurs API costs.

/*
/// Test creating a batch job
/// This test requires a valid input file ID
#[tokio::test]
async fn test_create_batch() {
    use openai_tools::batch::request::{CreateBatchRequest, BatchEndpoint};
    use openai_tools::files::request::{Files, FilePurpose};

    // First, upload a batch input file
    let files = Files::new().expect("Failed to create Files client");

    // Create a simple JSONL batch input
    let input = r#"{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "Hello"}]}}"#;

    let file = files
        .upload_bytes(input.as_bytes(), "batch_input.jsonl", FilePurpose::Batch)
        .await
        .expect("Failed to upload batch input file");

    // Create batch
    let batches = Batches::new().expect("Failed to create Batches client");
    let request = CreateBatchRequest::new(&file.id, BatchEndpoint::ChatCompletions);

    let batch = batches.create(request).await;
    assert!(batch.is_ok(), "Failed to create batch: {:?}", batch.err());

    let batch = batch.unwrap();
    println!("Created batch: {} ({:?})", batch.id, batch.status);
    assert_eq!(batch.object, "batch");
    assert!(!batch.id.is_empty());

    // Cleanup: The batch will process automatically
    // You can cancel it if needed:
    // batches.cancel(&batch.id).await.ok();
}
*/
