//! # Files Module
//!
//! This module provides functionality for interacting with the OpenAI Files API.
//! It allows you to upload, list, retrieve, delete files, and access file content.
//!
//! ## Key Features
//!
//! - **Upload Files**: Upload files from paths or bytes for various purposes
//! - **List Files**: Retrieve all uploaded files, optionally filtered by purpose
//! - **Retrieve File**: Get detailed information about a specific file
//! - **Delete File**: Remove an uploaded file
//! - **Get Content**: Retrieve the content of an uploaded file
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::files::request::{Files, FilePurpose};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a Files client
//!     let files = Files::new()?;
//!
//!     // List all files
//!     let response = files.list(None).await?;
//!     println!("Found {} files", response.data.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Upload a File for Fine-tuning
//!
//! ```rust,no_run
//! use openai_tools::files::request::{Files, FilePurpose};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let files = Files::new()?;
//!
//!     // Upload a JSONL file for fine-tuning
//!     let file = files.upload_path("training_data.jsonl", FilePurpose::FineTune).await?;
//!     println!("Uploaded file ID: {}", file.id);
//!     println!("Size: {} bytes", file.bytes);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Upload File from Bytes
//!
//! ```rust,no_run
//! use openai_tools::files::request::{Files, FilePurpose};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let files = Files::new()?;
//!
//!     let content = r#"{"messages": [{"role": "user", "content": "Hello"}]}"#;
//!     let file = files.upload_bytes(
//!         content.as_bytes(),
//!         "training.jsonl",
//!         FilePurpose::FineTune
//!     ).await?;
//!
//!     println!("Uploaded: {}", file.id);
//!     Ok(())
//! }
//! ```
//!
//! ### List Files by Purpose
//!
//! ```rust,no_run
//! use openai_tools::files::request::{Files, FilePurpose};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let files = Files::new()?;
//!
//!     // List only fine-tuning files
//!     let response = files.list(Some(FilePurpose::FineTune)).await?;
//!     for file in &response.data {
//!         println!("{}: {} ({} bytes)", file.id, file.filename, file.bytes);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Retrieve and Download File Content
//!
//! ```rust,no_run
//! use openai_tools::files::request::Files;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let files = Files::new()?;
//!
//!     // Get file info
//!     let file = files.retrieve("file-abc123").await?;
//!     println!("File: {} ({} bytes)", file.filename, file.bytes);
//!
//!     // Download content
//!     let content = files.content("file-abc123").await?;
//!     let text = String::from_utf8(content)?;
//!     println!("Content: {}", text);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Delete a File
//!
//! ```rust,no_run
//! use openai_tools::files::request::Files;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let files = Files::new()?;
//!
//!     let result = files.delete("file-abc123").await?;
//!     if result.deleted {
//!         println!("Successfully deleted: {}", result.id);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## File Purposes
//!
//! | Purpose | Description |
//! |---------|-------------|
//! | `Assistants` | For use with Assistants and Message files |
//! | `AssistantsOutput` | Output files from Assistants |
//! | `Batch` | For Batch API input |
//! | `BatchOutput` | Output files from Batch API |
//! | `FineTune` | For fine-tuning training data |
//! | `FineTuneResults` | Results from fine-tuning |
//! | `Vision` | For vision features |
//! | `UserData` | User-uploaded data |
//!
//! ## File Limits
//!
//! - Individual files can be up to 512 MB
//! - Total organization storage up to 100 GB

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::files::request::FilePurpose;
    use crate::files::response::{DeleteResponse, File, FileListResponse};

    #[test]
    fn test_file_deserialization() {
        let json = r#"{
            "id": "file-abc123",
            "object": "file",
            "bytes": 140,
            "created_at": 1613779657,
            "filename": "mydata.jsonl",
            "purpose": "fine-tune"
        }"#;

        let file: File = serde_json::from_str(json).expect("Should deserialize File");
        assert_eq!(file.id, "file-abc123");
        assert_eq!(file.object, "file");
        assert_eq!(file.bytes, 140);
        assert_eq!(file.created_at, 1613779657);
        assert_eq!(file.filename, "mydata.jsonl");
        assert_eq!(file.purpose, "fine-tune");
    }

    #[test]
    fn test_file_with_status_deserialization() {
        let json = r#"{
            "id": "file-abc123",
            "object": "file",
            "bytes": 140,
            "created_at": 1613779657,
            "filename": "mydata.jsonl",
            "purpose": "fine-tune",
            "status": "processed",
            "status_details": null
        }"#;

        let file: File = serde_json::from_str(json).expect("Should deserialize File with status");
        assert_eq!(file.status, Some("processed".to_string()));
        assert_eq!(file.status_details, None);
    }

    #[test]
    fn test_file_list_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "id": "file-abc123",
                    "object": "file",
                    "bytes": 140,
                    "created_at": 1613779657,
                    "filename": "mydata.jsonl",
                    "purpose": "fine-tune"
                },
                {
                    "id": "file-def456",
                    "object": "file",
                    "bytes": 250,
                    "created_at": 1613779700,
                    "filename": "batch.jsonl",
                    "purpose": "batch"
                }
            ]
        }"#;

        let response: FileListResponse = serde_json::from_str(json).expect("Should deserialize FileListResponse");
        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "file-abc123");
        assert_eq!(response.data[1].purpose, "batch");
    }

    #[test]
    fn test_delete_response_deserialization() {
        let json = r#"{
            "id": "file-abc123",
            "object": "file",
            "deleted": true
        }"#;

        let response: DeleteResponse = serde_json::from_str(json).expect("Should deserialize DeleteResponse");
        assert_eq!(response.id, "file-abc123");
        assert_eq!(response.object, "file");
        assert!(response.deleted);
    }

    #[test]
    fn test_file_purpose_as_str() {
        assert_eq!(FilePurpose::Assistants.as_str(), "assistants");
        assert_eq!(FilePurpose::AssistantsOutput.as_str(), "assistants_output");
        assert_eq!(FilePurpose::Batch.as_str(), "batch");
        assert_eq!(FilePurpose::BatchOutput.as_str(), "batch_output");
        assert_eq!(FilePurpose::FineTune.as_str(), "fine-tune");
        assert_eq!(FilePurpose::FineTuneResults.as_str(), "fine-tune-results");
        assert_eq!(FilePurpose::Vision.as_str(), "vision");
        assert_eq!(FilePurpose::UserData.as_str(), "user_data");
    }

    #[test]
    fn test_file_purpose_display() {
        assert_eq!(format!("{}", FilePurpose::FineTune), "fine-tune");
        assert_eq!(format!("{}", FilePurpose::Batch), "batch");
    }

    #[test]
    fn test_file_serialization() {
        let file = File {
            id: "file-abc123".to_string(),
            object: "file".to_string(),
            bytes: 140,
            created_at: 1613779657,
            filename: "mydata.jsonl".to_string(),
            purpose: "fine-tune".to_string(),
            status: None,
            status_details: None,
        };

        let json = serde_json::to_string(&file).expect("Should serialize File");
        assert!(json.contains("\"id\":\"file-abc123\""));
        assert!(json.contains("\"purpose\":\"fine-tune\""));
        // status should not be in the JSON since it's None with skip_serializing_if
        assert!(!json.contains("\"status\""));
    }
}
