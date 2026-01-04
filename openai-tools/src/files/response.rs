//! OpenAI Files API Response Types
//!
//! This module defines the response structures for the OpenAI Files API.

use serde::{Deserialize, Serialize};

/// Represents an uploaded file in the OpenAI platform.
///
/// Files can be used for various purposes including fine-tuning, batch processing,
/// assistants, and vision tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct File {
    /// The file identifier, which can be referenced in the API endpoints
    pub id: String,
    /// Object type, always "file"
    pub object: String,
    /// The size of the file, in bytes
    pub bytes: i64,
    /// Unix timestamp (in seconds) when the file was created
    pub created_at: i64,
    /// The name of the file
    pub filename: String,
    /// The intended purpose of the file
    /// (e.g., "assistants", "batch", "fine-tune", "vision")
    pub purpose: String,
    /// The current status of the file (deprecated for some file types)
    /// Can be "uploaded", "processed", or "error"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Additional details about the file status (deprecated for some file types)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_details: Option<String>,
}

/// Response structure for listing files.
///
/// Contains a list of file objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileListResponse {
    /// Object type, always "list"
    pub object: String,
    /// Array of file objects
    pub data: Vec<File>,
    /// Whether there are more files to retrieve
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_more: Option<bool>,
}

/// Response structure for file deletion.
///
/// Returned when a file is successfully deleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// The file identifier that was deleted
    pub id: String,
    /// Object type, always "file"
    pub object: String,
    /// Whether the file was successfully deleted
    pub deleted: bool,
}
