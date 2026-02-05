//! OpenAI Batch API Module
//!
//! This module provides access to OpenAI's Batch API, which allows you to
//! send asynchronous groups of requests with 50% lower costs and higher rate limits.
//! Batches are processed within a 24-hour window but often complete much faster.
//!
//! # Overview
//!
//! The Batch API is ideal for:
//! - Processing large volumes of requests where immediate responses aren't required
//! - Cost optimization (50% discount compared to synchronous API calls)
//! - Workloads that can tolerate up to 24-hour latency
//!
//! # Supported Endpoints
//!
//! The following endpoints can be used with batch processing:
//! - Chat Completions (`/v1/chat/completions`)
//! - Embeddings (`/v1/embeddings`)
//! - Completions (`/v1/completions`)
//! - Responses (`/v1/responses`)
//! - Moderations (`/v1/moderations`)
//!
//! # Workflow
//!
//! 1. Create a JSONL file with your requests (each line is a request)
//! 2. Upload the file using the Files API with purpose "batch"
//! 3. Create a batch job with the file ID
//! 4. Poll for completion or wait for webhook
//! 5. Download results from the output file
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::batch::request::{Batches, CreateBatchRequest, BatchEndpoint};
//! use openai_tools::batch::response::BatchStatus;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let batches = Batches::new()?;
//!
//!     // Create a batch job
//!     let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
//!     let batch = batches.create(request).await?;
//!
//!     println!("Created batch: {}", batch.id);
//!
//!     // Check status
//!     let batch = batches.retrieve(&batch.id).await?;
//!     match batch.status {
//!         BatchStatus::Completed => {
//!             println!("Batch completed! Output file: {:?}", batch.output_file_id);
//!         }
//!         BatchStatus::Failed => {
//!             println!("Batch failed. Error file: {:?}", batch.error_file_id);
//!         }
//!         _ => {
//!             println!("Batch is {:?}", batch.status);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Input File Format
//!
//! Each line in the JSONL input file should have this structure:
//!
//! ```json
//! {
//!   "custom_id": "request-1",
//!   "method": "POST",
//!   "url": "/v1/chat/completions",
//!   "body": {
//!     "model": "gpt-4o-mini",
//!     "messages": [
//!       {"role": "system", "content": "You are a helpful assistant."},
//!       {"role": "user", "content": "Hello!"}
//!     ]
//!   }
//! }
//! ```
//!
//! # Output File Format
//!
//! Each line in the output JSONL contains the response:
//!
//! ```json
//! {
//!   "id": "batch_req_wnaDys",
//!   "custom_id": "request-1",
//!   "response": {
//!     "status_code": 200,
//!     "body": {
//!       "id": "chatcmpl-xxx",
//!       "object": "chat.completion",
//!       ...
//!     }
//!   }
//! }
//! ```
//!
//! # Related Modules
//!
//! - [`crate::files`] - Upload batch input files with `FilePurpose::Batch`
//! - [`crate::chat`] - Individual chat completions (for comparison)
//! - [`crate::embedding`] - Individual embeddings (for comparison)

pub mod request;
pub mod response;

pub use request::{BatchEndpoint, Batches, CompletionWindow, CreateBatchRequest};
pub use response::{BatchError, BatchErrors, BatchListResponse, BatchObject, BatchStatus, RequestCounts};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_endpoint_serialization() {
        let endpoint = BatchEndpoint::ChatCompletions;
        let json = serde_json::to_string(&endpoint).unwrap();
        assert_eq!(json, "\"/v1/chat/completions\"");

        let endpoint = BatchEndpoint::Embeddings;
        let json = serde_json::to_string(&endpoint).unwrap();
        assert_eq!(json, "\"/v1/embeddings\"");
    }

    #[test]
    fn test_batch_endpoint_as_str() {
        assert_eq!(BatchEndpoint::ChatCompletions.as_str(), "/v1/chat/completions");
        assert_eq!(BatchEndpoint::Embeddings.as_str(), "/v1/embeddings");
        assert_eq!(BatchEndpoint::Completions.as_str(), "/v1/completions");
        assert_eq!(BatchEndpoint::Responses.as_str(), "/v1/responses");
        assert_eq!(BatchEndpoint::Moderations.as_str(), "/v1/moderations");
    }

    #[test]
    fn test_completion_window_serialization() {
        let window = CompletionWindow::Hours24;
        let json = serde_json::to_string(&window).unwrap();
        assert_eq!(json, "\"24h\"");
    }

    #[test]
    fn test_create_batch_request_new() {
        let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
        assert_eq!(request.input_file_id, "file-abc123");
        assert_eq!(request.endpoint, BatchEndpoint::ChatCompletions);
        assert_eq!(request.completion_window, CompletionWindow::Hours24);
        assert!(request.metadata.is_none());
    }

    #[test]
    fn test_create_batch_request_with_metadata() {
        use std::collections::HashMap;

        let mut metadata = HashMap::new();
        metadata.insert("customer_id".to_string(), "user_123".to_string());

        let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions).with_metadata(metadata.clone());

        assert_eq!(request.metadata, Some(metadata));
    }

    #[test]
    fn test_create_batch_request_serialization() {
        let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
        let json = serde_json::to_string(&request).unwrap();

        assert!(json.contains("\"input_file_id\":\"file-abc123\""));
        assert!(json.contains("\"endpoint\":\"/v1/chat/completions\""));
        assert!(json.contains("\"completion_window\":\"24h\""));
        assert!(!json.contains("metadata")); // None should be skipped
    }

    #[test]
    fn test_batch_status_deserialization() {
        let json = r#"{"id":"batch_123","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-123","completion_window":"24h","status":"in_progress","created_at":1234567890,"has_more":false}"#;
        let batch: BatchObject = serde_json::from_str(json).unwrap();
        assert_eq!(batch.status, BatchStatus::InProgress);

        let json = r#"{"id":"batch_123","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-123","completion_window":"24h","status":"completed","created_at":1234567890,"has_more":false}"#;
        let batch: BatchObject = serde_json::from_str(json).unwrap();
        assert_eq!(batch.status, BatchStatus::Completed);

        let json = r#"{"id":"batch_123","object":"batch","endpoint":"/v1/chat/completions","input_file_id":"file-123","completion_window":"24h","status":"failed","created_at":1234567890,"has_more":false}"#;
        let batch: BatchObject = serde_json::from_str(json).unwrap();
        assert_eq!(batch.status, BatchStatus::Failed);
    }

    #[test]
    fn test_request_counts_deserialization() {
        let json = r#"{"total":100,"completed":50,"failed":2}"#;
        let counts: RequestCounts = serde_json::from_str(json).unwrap();
        assert_eq!(counts.total, 100);
        assert_eq!(counts.completed, 50);
        assert_eq!(counts.failed, 2);
    }

    #[test]
    fn test_batch_list_response_deserialization() {
        let json = r#"{"object":"list","data":[],"first_id":null,"last_id":null,"has_more":false}"#;
        let response: BatchListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.object, "list");
        assert!(response.data.is_empty());
        assert!(!response.has_more);
    }

    #[test]
    fn test_batch_errors_deserialization() {
        let json = r#"{"object":"list","data":[{"code":"invalid_request","message":"Invalid input","param":"messages","line":5}]}"#;
        let errors: BatchErrors = serde_json::from_str(json).unwrap();
        assert_eq!(errors.data.len(), 1);
        assert_eq!(errors.data[0].code, "invalid_request");
        assert_eq!(errors.data[0].line, Some(5));
    }
}
