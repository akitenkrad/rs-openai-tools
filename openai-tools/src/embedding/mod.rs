//! # Embedding Module
//!
//! This module provides functionality for interacting with the OpenAI Embeddings API.
//! It allows you to convert text into numerical vector representations (embeddings)
//! that capture semantic meaning, enabling various NLP tasks such as semantic search,
//! clustering, and similarity comparison.
//!
//! ## Key Features
//!
//! - **Text Embedding Generation**: Convert single or multiple texts into vector embeddings
//! - **Multiple Input Formats**: Support for single text strings or arrays of texts
//! - **Flexible Encoding**: Support for both `float` and `base64` encoding formats
//! - **Various Model Support**: Compatible with OpenAI's embedding models (e.g., `text-embedding-3-small`, `text-embedding-3-large`)
//! - **Multi-dimensional Output**: Support for 1D, 2D, and 3D embedding vectors
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the embedding client
//!     let mut embedding = Embedding::new()?;
//!
//!     // Configure the model and input text
//!     embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text("Hello, world!");
//!
//!     // Generate embedding
//!     let response = embedding.embed().await?;
//!
//!     // Access the embedding vector
//!     let vector = response.data[0].embedding.as_1d().unwrap();
//!     println!("Embedding dimension: {}", vector.len());
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Single Text Embedding
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!
//!     embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text("The quick brown fox jumps over the lazy dog.");
//!
//!     let response = embedding.embed().await?;
//!
//!     // The response contains embedding data
//!     assert_eq!(response.object, "list");
//!     assert_eq!(response.data.len(), 1);
//!
//!     let vector = response.data[0].embedding.as_1d().unwrap();
//!     println!("Generated embedding with {} dimensions", vector.len());
//!     Ok(())
//! }
//! ```
//!
//! ### Batch Text Embedding
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!
//!     // Embed multiple texts at once
//!     let texts = vec![
//!         "Hello, world!",
//!         "こんにちは，世界！",
//!         "Bonjour le monde!",
//!     ];
//!
//!     embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text_array(texts);
//!
//!     let response = embedding.embed().await?;
//!
//!     // Each input text gets its own embedding
//!     for (i, data) in response.data.iter().enumerate() {
//!         let vector = data.embedding.as_1d().unwrap();
//!         println!("Text {}: {} dimensions", i, vector.len());
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Using Different Encoding Formats
//!
//! ```rust,no_run
//! use openai_tools::embedding::request::Embedding;
//! use openai_tools::common::models::EmbeddingModel;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!
//!     embedding
//!         .model(EmbeddingModel::TextEmbedding3Small)
//!         .input_text("Sample text for embedding")
//!         .encoding_format("float"); // or "base64"
//!
//!     let response = embedding.embed().await?;
//!     println!("Model used: {}", response.model);
//!     println!("Token usage: {:?}", response.usage);
//!     Ok(())
//! }
//! ```
//!
//! ## Supported Models
//!
//! | Model | Dimensions | Description |
//! |-------|------------|-------------|
//! | `text-embedding-3-small` | 1536 | Efficient model for most use cases |
//! | `text-embedding-3-large` | 3072 | Higher quality embeddings for demanding tasks |
//! | `text-embedding-ada-002` | 1536 | Legacy model (still supported) |
//!
//! ## Response Structure
//!
//! The embedding response contains:
//! - `object`: Always "list" for embedding responses
//! - `data`: Array of embedding objects, each containing:
//!   - `object`: Type identifier ("embedding")
//!   - `embedding`: The vector representation (1D, 2D, or 3D)
//!   - `index`: Position in the input array
//! - `model`: The model used for embedding
//! - `usage`: Token usage information

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::common::models::EmbeddingModel;
    use crate::embedding::request::Embedding;

    #[test]
    fn test_embedding_builder_model() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.model(EmbeddingModel::TextEmbedding3Small);
        // Model is set internally, we can verify by serialization
    }

    #[test]
    fn test_embedding_builder_input_text() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.input_text("Hello, world!");
        // Input is set internally
    }

    #[test]
    fn test_embedding_builder_input_text_array() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        let texts = vec!["Text 1", "Text 2", "Text 3"];
        embedding.input_text_array(texts);
        // Input array is set internally
    }

    #[test]
    fn test_embedding_builder_encoding_format_float() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.encoding_format("float");
        // Encoding format is set internally
    }

    #[test]
    fn test_embedding_builder_encoding_format_base64() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.encoding_format("base64");
        // Encoding format is set internally
    }

    #[test]
    #[should_panic(expected = "encoding_format must be either 'float' or 'base64'")]
    fn test_embedding_builder_encoding_format_invalid() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.encoding_format("invalid"); // Should panic
    }

    #[test]
    fn test_embedding_builder_chain() {
        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.model(EmbeddingModel::TextEmbedding3Small).input_text("Hello!").encoding_format("float");
        // Method chaining works
    }
}
