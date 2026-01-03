//! Integration tests for Embedding API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test embedding_integration

use openai_tools::common::errors::OpenAIToolError;
use openai_tools::embedding::request::Embedding;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

fn init_tracing() {
    TRACING_INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
    });
}

#[tokio::test]
async fn test_embedding_with_text() {
    init_tracing();

    let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
    embedding
        .model("text-embedding-3-small")
        .input_text("Hello, world!");

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

                let embedding_vec = response.data[0]
                    .embedding
                    .as_1d()
                    .expect("Embedding should be 1D");
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
async fn test_embedding_with_text_array() {
    init_tracing();

    let mut embedding = Embedding::new().expect("Embedding initialization should succeed");
    let texts = vec!["Hello, world!", "こんにちは、世界！", "Bonjour le monde!"];
    embedding
        .model("text-embedding-3-small")
        .input_text_array(texts.clone());

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
