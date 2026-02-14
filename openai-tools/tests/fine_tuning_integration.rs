//! Integration tests for the OpenAI Fine-tuning API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test fine_tuning_integration

use openai_tools::fine_tuning::request::FineTuning;

/// Test listing fine-tuning jobs
#[tokio::test]
async fn test_list_fine_tuning_jobs() {
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    let response = fine_tuning.list(Some(10), None).await;
    assert!(response.is_ok(), "Failed to list fine-tuning jobs: {:?}", response.err());

    let list = response.unwrap();
    assert_eq!(list.object, "list");
    // The data array may be empty if no jobs exist
    println!("Found {} fine-tuning jobs", list.data.len());
}

/// Test listing fine-tuning jobs with pagination
#[tokio::test]
async fn test_list_fine_tuning_jobs_with_pagination() {
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    // First page
    let response = fine_tuning.list(Some(5), None).await;
    assert!(response.is_ok(), "Failed to list jobs: {:?}", response.err());

    let first_page = response.unwrap();
    println!("First page: {} jobs, has_more: {}", first_page.data.len(), first_page.has_more);

    // If there are more, we could get next page
    // (but we don't want to make too many API calls in tests)
}

/// Test retrieving a non-existent job (should fail gracefully)
#[tokio::test]
async fn test_retrieve_nonexistent_job() {
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    // This should fail with an error, not panic
    let result = fine_tuning.retrieve("ftjob-nonexistent123").await;
    assert!(result.is_err(), "Expected error for non-existent job");
}

/// Test listing events for a non-existent job (should fail gracefully)
#[tokio::test]
async fn test_list_events_nonexistent_job() {
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    // This should fail with an error, not panic
    let result = fine_tuning.list_events("ftjob-nonexistent123", Some(10), None).await;
    assert!(result.is_err(), "Expected error for non-existent job events");
}

/// Test listing checkpoints for a non-existent job (should fail gracefully)
#[tokio::test]
async fn test_list_checkpoints_nonexistent_job() {
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    // This should fail with an error, not panic
    let result = fine_tuning.list_checkpoints("ftjob-nonexistent123", None, None).await;
    assert!(result.is_err(), "Expected error for non-existent job checkpoints");
}

// Note: The following tests are commented out because they require
// uploading training data and incur significant API costs.

/*
/// Test creating a fine-tuning job
/// This test requires a valid training file ID
#[tokio::test]
async fn test_create_fine_tuning_job() {
    use openai_tools::fine_tuning::request::CreateFineTuningJobRequest;
    use openai_tools::fine_tuning::response::Hyperparameters;
    use openai_tools::files::request::{Files, FilePurpose};

    // First, upload a training file
    let files = Files::new().expect("Failed to create Files client");

    // Create training data (at least 10 examples required for fine-tuning)
    let training_data = (1..=10)
        .map(|i| {
            format!(
                r#"{{"messages": [{{"role": "user", "content": "Test {i}"}}, {{"role": "assistant", "content": "Response {i}"}}]}}"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let file = files
        .upload_bytes(training_data.as_bytes(), "training.jsonl", FilePurpose::FineTune)
        .await
        .expect("Failed to upload training file");

    // Create fine-tuning job
    let fine_tuning = FineTuning::new().expect("Failed to create FineTuning client");

    let hyperparams = Hyperparameters {
        n_epochs: Some(1),
        ..Default::default()
    };

    let request = CreateFineTuningJobRequest::new("gpt-4.1-mini-2025-04-14", &file.id)
        .with_suffix("integration-test")
        .with_supervised_method(Some(hyperparams));

    let job = fine_tuning.create(request).await;
    assert!(job.is_ok(), "Failed to create fine-tuning job: {:?}", job.err());

    let job = job.unwrap();
    println!("Created job: {} ({:?})", job.id, job.status);
    assert_eq!(job.object, "fine_tuning.job");
    assert!(!job.id.is_empty());

    // Cleanup: Cancel the job to avoid incurring costs
    let _ = fine_tuning.cancel(&job.id).await;
}
*/
