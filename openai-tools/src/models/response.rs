//! OpenAI Models API Response Types
//!
//! This module defines the response structures for the OpenAI Models API.

use serde::{Deserialize, Serialize};

/// Response structure for listing all available models.
///
/// Contains a list of model objects that are currently available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsListResponse {
    /// Object type, always "list"
    pub object: String,
    /// Array of model objects
    pub data: Vec<Model>,
}

/// Represents an OpenAI model.
///
/// Contains basic information about a model including its ID, creation time,
/// and ownership details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// The model identifier, which can be referenced in the API endpoints
    pub id: String,
    /// Object type, always "model"
    pub object: String,
    /// Unix timestamp (in seconds) when the model was created
    pub created: i64,
    /// The organization that owns the model
    pub owned_by: String,
}

/// Response structure for model deletion.
///
/// Returned when a fine-tuned model is successfully deleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    /// The model identifier that was deleted
    pub id: String,
    /// Object type, always "model"
    pub object: String,
    /// Whether the model was successfully deleted
    pub deleted: bool,
}
