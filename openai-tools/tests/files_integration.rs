//! Integration tests for the OpenAI Files API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test files_integration

use openai_tools::files::request::{FilePurpose, Files};

/// Test listing all files.
#[tokio::test]
async fn test_list_files() {
    let files = Files::new().expect("Should create Files client");
    let response = files.list(None).await.expect("Should list files");

    // Verify response structure
    assert_eq!(response.object, "list");

    println!("Found {} files", response.data.len());

    // If there are files, verify their structure
    for file in &response.data {
        assert!(!file.id.is_empty(), "File ID should not be empty");
        assert_eq!(file.object, "file");
        assert!(!file.filename.is_empty(), "Filename should not be empty");
        assert!(!file.purpose.is_empty(), "Purpose should not be empty");
    }
}

/// Test listing files filtered by purpose.
#[tokio::test]
async fn test_list_files_by_purpose() {
    let files = Files::new().expect("Should create Files client");

    // Test listing by different purposes
    let purposes = vec![FilePurpose::FineTune, FilePurpose::Batch, FilePurpose::Assistants];

    for purpose in purposes {
        let response = files.list(Some(purpose)).await.expect(&format!("Should list files with purpose {:?}", purpose));

        assert_eq!(response.object, "list");
        println!("Found {} files with purpose {:?}", response.data.len(), purpose);
    }
}

/// Test uploading, retrieving, and deleting a file.
#[tokio::test]
async fn test_upload_retrieve_delete_file() {
    let files = Files::new().expect("Should create Files client");

    // Create a simple JSONL content for fine-tuning
    let content = r#"{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}"#;

    // Upload the file
    let uploaded = files.upload_bytes(content.as_bytes(), "test_upload.jsonl", FilePurpose::FineTune).await.expect("Should upload file");

    assert!(!uploaded.id.is_empty(), "File ID should not be empty");
    assert_eq!(uploaded.object, "file");
    assert_eq!(uploaded.filename, "test_upload.jsonl");
    assert_eq!(uploaded.purpose, "fine-tune");
    assert!(uploaded.bytes > 0, "File size should be positive");

    println!("Uploaded file: {} ({} bytes)", uploaded.id, uploaded.bytes);

    // Retrieve the file
    let retrieved = files.retrieve(&uploaded.id).await.expect("Should retrieve file");

    assert_eq!(retrieved.id, uploaded.id);
    assert_eq!(retrieved.filename, uploaded.filename);
    assert_eq!(retrieved.bytes, uploaded.bytes);

    println!("Retrieved file: {}", retrieved.id);

    // Delete the file
    let deleted = files.delete(&uploaded.id).await.expect("Should delete file");

    assert_eq!(deleted.id, uploaded.id);
    assert!(deleted.deleted, "File should be deleted");

    println!("Deleted file: {}", deleted.id);
}

/// Test that retrieving a non-existent file fails.
#[tokio::test]
async fn test_retrieve_nonexistent_file() {
    let files = Files::new().expect("Should create Files client");

    let result = files.retrieve("file-nonexistent12345").await;

    // Should fail with an error
    assert!(result.is_err(), "Should fail for non-existent file");
}

/// Test FilePurpose enum string representations.
#[test]
fn test_file_purpose_strings() {
    assert_eq!(FilePurpose::Assistants.as_str(), "assistants");
    assert_eq!(FilePurpose::AssistantsOutput.as_str(), "assistants_output");
    assert_eq!(FilePurpose::Batch.as_str(), "batch");
    assert_eq!(FilePurpose::BatchOutput.as_str(), "batch_output");
    assert_eq!(FilePurpose::FineTune.as_str(), "fine-tune");
    assert_eq!(FilePurpose::FineTuneResults.as_str(), "fine-tune-results");
    assert_eq!(FilePurpose::Vision.as_str(), "vision");
    assert_eq!(FilePurpose::UserData.as_str(), "user_data");
}
