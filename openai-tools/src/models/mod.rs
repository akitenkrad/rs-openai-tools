//! # Models Module
//!
//! This module provides functionality for interacting with the OpenAI Models API.
//! It allows you to list, retrieve, and delete models available in the OpenAI platform.
//!
//! ## Key Features
//!
//! - **List Models**: Retrieve all available models in your organization
//! - **Retrieve Model**: Get detailed information about a specific model
//! - **Delete Model**: Remove fine-tuned models that you own
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::models::request::Models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a Models client
//!     let models = Models::new()?;
//!
//!     // List all available models
//!     let response = models.list().await?;
//!     println!("Found {} models", response.data.len());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### List All Models
//!
//! ```rust,no_run
//! use openai_tools::models::request::Models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let models = Models::new()?;
//!     let response = models.list().await?;
//!
//!     for model in &response.data {
//!         println!("{}: owned by {}", model.id, model.owned_by);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Retrieve a Specific Model
//!
//! ```rust,no_run
//! use openai_tools::models::request::Models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let models = Models::new()?;
//!
//!     // Get details of gpt-4o-mini
//!     let model = models.retrieve("gpt-4o-mini").await?;
//!     println!("Model: {}", model.id);
//!     println!("Owned by: {}", model.owned_by);
//!     println!("Created: {}", model.created);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Delete a Fine-tuned Model
//!
//! ```rust,no_run
//! use openai_tools::models::request::Models;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let models = Models::new()?;
//!
//!     // Delete a fine-tuned model (must be owned by you)
//!     let result = models.delete("ft:gpt-4o-mini:my-org:my-suffix:id").await?;
//!     if result.deleted {
//!         println!("Successfully deleted: {}", result.id);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Response Structure
//!
//! ### Model Object
//!
//! Each model object contains:
//! - `id`: Model identifier (e.g., "gpt-4o-mini")
//! - `object`: Always "model"
//! - `created`: Unix timestamp of creation
//! - `owned_by`: Organization that owns the model
//!
//! ### Delete Response
//!
//! When deleting a model:
//! - `id`: The deleted model's ID
//! - `object`: Always "model"
//! - `deleted`: Boolean indicating success

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::models::response::{DeleteResponse, Model, ModelsListResponse};

    #[test]
    fn test_model_deserialization() {
        let json = r#"{
            "id": "gpt-4o-mini",
            "object": "model",
            "created": 1686935002,
            "owned_by": "openai"
        }"#;

        let model: Model = serde_json::from_str(json).expect("Should deserialize Model");
        assert_eq!(model.id, "gpt-4o-mini");
        assert_eq!(model.object, "model");
        assert_eq!(model.created, 1686935002);
        assert_eq!(model.owned_by, "openai");
    }

    #[test]
    fn test_models_list_response_deserialization() {
        let json = r#"{
            "object": "list",
            "data": [
                {
                    "id": "gpt-4o-mini",
                    "object": "model",
                    "created": 1686935002,
                    "owned_by": "openai"
                },
                {
                    "id": "text-embedding-3-small",
                    "object": "model",
                    "created": 1705948997,
                    "owned_by": "openai"
                }
            ]
        }"#;

        let response: ModelsListResponse =
            serde_json::from_str(json).expect("Should deserialize ModelsListResponse");
        assert_eq!(response.object, "list");
        assert_eq!(response.data.len(), 2);
        assert_eq!(response.data[0].id, "gpt-4o-mini");
        assert_eq!(response.data[1].id, "text-embedding-3-small");
    }

    #[test]
    fn test_delete_response_deserialization() {
        let json = r#"{
            "id": "ft:gpt-4o-mini:my-org:my-suffix:abc123",
            "object": "model",
            "deleted": true
        }"#;

        let response: DeleteResponse =
            serde_json::from_str(json).expect("Should deserialize DeleteResponse");
        assert_eq!(response.id, "ft:gpt-4o-mini:my-org:my-suffix:abc123");
        assert_eq!(response.object, "model");
        assert!(response.deleted);
    }

    #[test]
    fn test_model_serialization() {
        let model = Model {
            id: "gpt-4o-mini".to_string(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "openai".to_string(),
        };

        let json = serde_json::to_string(&model).expect("Should serialize Model");
        assert!(json.contains("\"id\":\"gpt-4o-mini\""));
        assert!(json.contains("\"object\":\"model\""));
    }
}
