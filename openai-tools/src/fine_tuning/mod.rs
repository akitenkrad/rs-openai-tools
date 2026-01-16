//! OpenAI Fine-tuning API Module
//!
//! This module provides access to OpenAI's Fine-tuning API, which allows you to
//! customize models for your specific use case by training them on your data.
//!
//! # Overview
//!
//! Fine-tuning lets you:
//! - Improve model performance on specific tasks
//! - Reduce prompt size by training behaviors into the model
//! - Get more consistent output format
//! - Customize tone and style
//!
//! # Supported Methods
//!
//! - **Supervised Fine-tuning (SFT)**: Train on input-output pairs
//! - **Direct Preference Optimization (DPO)**: Train on preference pairs
//! - **Reinforcement Fine-tuning (RFT)**: Train using graded outputs (reasoning models)
//!
//! # Supported Models
//!
//! - GPT-4o and GPT-4o-mini (recommended)
//! - GPT-4 Turbo
//! - GPT-3.5 Turbo
//!
//! # Workflow
//!
//! 1. Prepare training data in JSONL format
//! 2. Upload the file using the Files API with purpose "fine-tune"
//! 3. Create a fine-tuning job
//! 4. Monitor progress through events
//! 5. Use the fine-tuned model for inference
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::fine_tuning::request::{FineTuning, CreateFineTuningJobRequest};
//! use openai_tools::fine_tuning::response::{FineTuningJobStatus, Hyperparameters};
//! use openai_tools::common::models::FineTuningModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let fine_tuning = FineTuning::new()?;
//!
//!     // Create a fine-tuning job with custom hyperparameters
//!     let hyperparams = Hyperparameters {
//!         n_epochs: Some(3),
//!         batch_size: None,
//!         learning_rate_multiplier: None,
//!     };
//!
//!     let request = CreateFineTuningJobRequest::new(
//!             FineTuningModel::Gpt4oMini_2024_07_18,
//!             "file-abc123"
//!         )
//!         .with_suffix("my-model")
//!         .with_supervised_method(Some(hyperparams));
//!
//!     let job = fine_tuning.create(request).await?;
//!     println!("Created job: {}", job.id);
//!
//!     // Check job status
//!     let job = fine_tuning.retrieve(&job.id).await?;
//!     match job.status {
//!         FineTuningJobStatus::Succeeded => {
//!             println!("Training complete! Model: {:?}", job.fine_tuned_model);
//!         }
//!         FineTuningJobStatus::Failed => {
//!             println!("Training failed: {:?}", job.error);
//!         }
//!         _ => {
//!             println!("Job is {:?}", job.status);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Training Data Format
//!
//! ## Supervised Fine-tuning (Chat Format)
//!
//! ```json
//! {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
//! {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
//! ```
//!
//! ## DPO Format
//!
//! ```json
//! {"input": [{"role": "user", "content": "Translate to French: Hello"}], "preferred_output": [{"role": "assistant", "content": "Bonjour"}], "non_preferred_output": [{"role": "assistant", "content": "Hello en français"}]}
//! ```
//!
//! # Monitoring Training
//!
//! ```rust,no_run
//! use openai_tools::fine_tuning::request::FineTuning;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let fine_tuning = FineTuning::new()?;
//!
//!     // Get training events
//!     let events = fine_tuning.list_events("ftjob-abc123", Some(20), None).await?;
//!     for event in &events.data {
//!         println!("[{}] {}", event.level, event.message);
//!     }
//!
//!     // Get checkpoints (saved at end of each epoch)
//!     let checkpoints = fine_tuning.list_checkpoints("ftjob-abc123", None, None).await?;
//!     for cp in &checkpoints.data {
//!         println!("Step {}: train_loss={:.4}", cp.step_number, cp.metrics.train_loss);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod request;
pub mod response;

pub use request::{CreateFineTuningJobRequest, FineTuning};
pub use response::{
    CheckpointMetrics, DpoConfig, FineTuningCheckpoint, FineTuningCheckpointListResponse,
    FineTuningError, FineTuningEvent, FineTuningEventListResponse, FineTuningJob,
    FineTuningJobListResponse, FineTuningJobStatus, Hyperparameters, Integration, MethodConfig,
    SupervisedConfig,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::models::FineTuningModel;

    #[test]
    fn test_create_fine_tuning_job_request_new() {
        let request =
            CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123");
        assert_eq!(request.model, FineTuningModel::Gpt4oMini_2024_07_18);
        assert_eq!(request.training_file, "file-abc123");
        assert!(request.validation_file.is_none());
        assert!(request.suffix.is_none());
        assert!(request.seed.is_none());
        assert!(request.method.is_none());
    }

    #[test]
    fn test_create_fine_tuning_job_request_with_options() {
        let request =
            CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123")
                .with_validation_file("file-def456")
                .with_suffix("my-model")
                .with_seed(42);

        assert_eq!(request.validation_file, Some("file-def456".to_string()));
        assert_eq!(request.suffix, Some("my-model".to_string()));
        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn test_create_fine_tuning_job_request_with_supervised_method() {
        let hyperparams = Hyperparameters {
            n_epochs: Some(3),
            batch_size: Some(4),
            learning_rate_multiplier: Some(0.1),
        };

        let request =
            CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123")
                .with_supervised_method(Some(hyperparams));

        assert!(request.method.is_some());
        let method = request.method.unwrap();
        assert_eq!(method.method_type, "supervised");
        assert!(method.supervised.is_some());
        assert!(method.dpo.is_none());
    }

    #[test]
    fn test_create_fine_tuning_job_request_with_dpo_method() {
        let request =
            CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123")
                .with_dpo_method(None);

        assert!(request.method.is_some());
        let method = request.method.unwrap();
        assert_eq!(method.method_type, "dpo");
        assert!(method.supervised.is_none());
        assert!(method.dpo.is_some());
    }

    #[test]
    fn test_create_fine_tuning_job_request_serialization() {
        let request =
            CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123")
                .with_suffix("test-model");

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o-mini-2024-07-18\""));
        assert!(json.contains("\"training_file\":\"file-abc123\""));
        assert!(json.contains("\"suffix\":\"test-model\""));
        assert!(!json.contains("validation_file")); // None should be skipped
    }

    #[test]
    fn test_fine_tuning_job_status_deserialization() {
        let json = r#"{"id":"ftjob-123","object":"fine_tuning.job","model":"gpt-4o-mini","created_at":1234567890,"organization_id":"org-123","result_files":[],"status":"running","training_file":"file-123","hyperparameters":{},"seed":0}"#;
        let job: FineTuningJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.status, FineTuningJobStatus::Running);

        let json = r#"{"id":"ftjob-123","object":"fine_tuning.job","model":"gpt-4o-mini","created_at":1234567890,"organization_id":"org-123","result_files":[],"status":"succeeded","training_file":"file-123","hyperparameters":{},"seed":0}"#;
        let job: FineTuningJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.status, FineTuningJobStatus::Succeeded);

        let json = r#"{"id":"ftjob-123","object":"fine_tuning.job","model":"gpt-4o-mini","created_at":1234567890,"organization_id":"org-123","result_files":[],"status":"failed","training_file":"file-123","hyperparameters":{},"seed":0}"#;
        let job: FineTuningJob = serde_json::from_str(json).unwrap();
        assert_eq!(job.status, FineTuningJobStatus::Failed);
    }

    #[test]
    fn test_hyperparameters_default() {
        let hp = Hyperparameters::default();
        assert!(hp.n_epochs.is_none());
        assert!(hp.batch_size.is_none());
        assert!(hp.learning_rate_multiplier.is_none());
    }

    #[test]
    fn test_fine_tuning_event_deserialization() {
        let json = r#"{"id":"ftevent-123","object":"fine_tuning.job.event","created_at":1234567890,"level":"info","message":"Training started","type":"message"}"#;
        let event: FineTuningEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.id, "ftevent-123");
        assert_eq!(event.level, "info");
        assert_eq!(event.message, "Training started");
        assert_eq!(event.event_type, "message");
    }

    #[test]
    fn test_fine_tuning_checkpoint_deserialization() {
        let json = r#"{"id":"ftckpt-123","object":"fine_tuning.job.checkpoint","created_at":1234567890,"fine_tuning_job_id":"ftjob-123","fine_tuned_model_checkpoint":"ft:gpt-4o-mini:org:suffix:ckpt-1","step_number":100,"metrics":{"step":100,"train_loss":0.5,"train_mean_token_accuracy":0.8}}"#;
        let checkpoint: FineTuningCheckpoint = serde_json::from_str(json).unwrap();
        assert_eq!(checkpoint.id, "ftckpt-123");
        assert_eq!(checkpoint.step_number, 100);
        assert_eq!(checkpoint.metrics.train_loss, 0.5);
        assert_eq!(checkpoint.metrics.train_mean_token_accuracy, 0.8);
    }

    #[test]
    fn test_fine_tuning_job_list_response_deserialization() {
        let json = r#"{"object":"list","data":[],"has_more":false}"#;
        let response: FineTuningJobListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
        assert!(!response.has_more);
    }

    #[test]
    fn test_fine_tuning_error_deserialization() {
        let json = r#"{"code":"invalid_training_file","message":"The training file is invalid","param":"training_file"}"#;
        let error: FineTuningError = serde_json::from_str(json).unwrap();
        assert_eq!(error.code, "invalid_training_file");
        assert_eq!(error.message, "The training file is invalid");
        assert_eq!(error.param, Some("training_file".to_string()));
    }
}
