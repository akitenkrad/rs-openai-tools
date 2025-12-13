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
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the embedding client
//!     let mut embedding = Embedding::new()?;
//!     
//!     // Configure the model and input text
//!     embedding
//!         .model("text-embedding-3-small")
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
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!     
//!     embedding
//!         .model("text-embedding-3-small")
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
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!     
//!     // Embed multiple texts at once
//!     let texts = vec![
//!         "Hello, world!",
//!         "こんにちは、世界！",
//!         "Bonjour le monde!",
//!     ];
//!     
//!     embedding
//!         .model("text-embedding-3-small")
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
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut embedding = Embedding::new()?;
//!     
//!     embedding
//!         .model("text-embedding-3-small")
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
    use crate::common::errors::OpenAIToolError;
    use crate::embedding::request::Embedding;
    use std::sync::Once;
    use tracing_subscriber::EnvFilter;

    static TRACING_INIT: Once = Once::new();

    fn init_tracing() {
        TRACING_INIT.call_once(|| {
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
        });
    }

    #[tokio::test]
    #[test_log::test]
    async fn test_embedding_with_text() {
        init_tracing();

        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        embedding.model("text-embedding-3-small").input_text("Hello, world!");

        let mut counter = 3;
        loop {
            match embedding.embed().await {
                Ok(response) => {
                    tracing::info!("Embedding model: {}", &response.model);
                    tracing::info!("Embedding data count: {}", response.data.len());
                    tracing::info!("Embedding usage: {:?}", &response.usage);

                    assert_eq!(response.object, "list");
                    assert_eq!(response.data.len(), 1);
                    assert!(response.data[0].embedding.is_1d());

                    let embedding_vec = response.data[0].embedding.as_1d().expect("Embedding should be 1D");
                    tracing::info!("Embedding dimension: {}", embedding_vec.len());
                    assert_eq!(embedding_vec.len(), 1536); // text-embedding-3-small outputs 1536 dimensions

                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            panic!("Embedding request failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        panic!("Embedding request failed: {}", e);
                    }
                },
            };
        }
    }

    #[tokio::test]
    #[test_log::test]
    async fn test_embedding_with_text_array() {
        init_tracing();

        let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
        let texts = vec!["Hello, world!", "こんにちは、世界！", "Bonjour le monde!"];
        embedding.model("text-embedding-3-small").input_text_array(texts.clone());

        let mut counter = 3;
        loop {
            match embedding.embed().await {
                Ok(response) => {
                    tracing::info!("Embedding model: {}", &response.model);
                    tracing::info!("Embedding data count: {}", response.data.len());
                    tracing::info!("Embedding usage: {:?}", &response.usage);

                    assert_eq!(response.object, "list");
                    assert_eq!(response.data.len(), texts.len());

                    for (i, data) in response.data.iter().enumerate() {
                        assert!(data.embedding.is_1d());
                        let embedding_vec = data.embedding.as_1d().expect("Embedding should be 1D");
                        tracing::info!("Embedding[{}] dimension: {}", i, embedding_vec.len());
                        assert_eq!(embedding_vec.len(), 1536); // text-embedding-3-small outputs 1536 dimensions
                        assert_eq!(data.index, i);
                    }

                    break;
                }
                Err(e) => match e {
                    OpenAIToolError::RequestError(e) => {
                        tracing::warn!("Request error: {} (retrying... {})", e, counter);
                        counter -= 1;
                        if counter == 0 {
                            panic!("Embedding request failed (retry limit reached)");
                        }
                        continue;
                    }
                    _ => {
                        tracing::error!("Error: {}", e);
                        panic!("Embedding request failed: {}", e);
                    }
                },
            };
        }
    }
}
