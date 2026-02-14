//! Integration tests for model-specific functionality
//!
//! Tests GPT-5 series and GPT-4 series models with Chat and Responses APIs.
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test models_by_version_integration

use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::{errors::OpenAIToolError, message::Message, role::Role};
use openai_tools::responses::request::Responses;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
    });
}

/// Helper function to run chat completion test with retry logic
async fn test_chat_with_model(model_id: &str) -> Result<(), String> {
    let mut chat = ChatCompletion::new();
    let messages = vec![Message::from_string(Role::User, "Say 'Hello' in one word.")];

    chat.model_id(model_id).messages(messages).temperature(1.0).safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match chat.chat().await {
            Ok(response) => {
                let content = response.choices[0].message.content.clone().expect("Response content should not be empty");
                tracing::info!("[{}] Chat response: {:?}", model_id, content);
                return Ok(());
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(ref msg) => {
                    tracing::warn!("[{}] Request error: {} (retrying... {})", model_id, msg, counter);
                    counter -= 1;
                    if counter == 0 {
                        return Err(format!("[{}] Chat completion failed (retry limit reached): {}", model_id, e));
                    }
                    continue;
                }
                _ => {
                    return Err(format!("[{}] Chat completion failed: {}", model_id, e));
                }
            },
        };
    }
}

/// Helper function to run responses API test with retry logic
async fn test_responses_with_model(model_id: &str) -> Result<(), String> {
    let mut responses = Responses::new();
    responses.model_id(model_id);
    responses.str_message("Say 'Hello' in one word.");
    responses.safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                let output = res.output_text().unwrap_or_default();
                tracing::info!("[{}] Responses output: {}", model_id, output);
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("[{}] Error: {} (retrying... {})", model_id, e, counter);
                counter -= 1;
                if counter == 0 {
                    return Err(format!("[{}] Responses API failed (retry limit reached): {}", model_id, e));
                }
            }
        }
    }
}

// =============================================================================
// GPT-5 Series Tests - Chat API
// =============================================================================

#[tokio::test]
async fn test_chat_gpt5() {
    init_tracing();
    let result = test_chat_with_model("gpt-5-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_chat_gpt5_mini() {
    init_tracing();
    let result = test_chat_with_model("gpt-5-mini-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_chat_gpt5_nano() {
    init_tracing();
    let result = test_chat_with_model("gpt-5-nano-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_chat_gpt52() {
    init_tracing();
    let result = test_chat_with_model("gpt-5.2-2025-12-11").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

/// Note: gpt-5.2-pro does not support Chat Completions API, only Responses API
#[tokio::test]
#[ignore = "gpt-5.2-pro does not support Chat Completions API"]
async fn test_chat_gpt52_pro() {
    init_tracing();
    let result = test_chat_with_model("gpt-5.2-pro-2025-12-11").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

// =============================================================================
// GPT-5 Series Tests - Responses API
// =============================================================================

#[tokio::test]
async fn test_responses_gpt5() {
    init_tracing();
    let result = test_responses_with_model("gpt-5-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_responses_gpt5_mini() {
    init_tracing();
    let result = test_responses_with_model("gpt-5-mini-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_responses_gpt5_nano() {
    init_tracing();
    let result = test_responses_with_model("gpt-5-nano-2025-08-07").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_responses_gpt52() {
    init_tracing();
    let result = test_responses_with_model("gpt-5.2-2025-12-11").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_responses_gpt52_pro() {
    init_tracing();
    let result = test_responses_with_model("gpt-5.2-pro-2025-12-11").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

// =============================================================================
// GPT-4 Series Tests - Chat API
// =============================================================================

#[tokio::test]
async fn test_chat_gpt4() {
    init_tracing();
    let result = test_chat_with_model("gpt-4").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_chat_gpt4_turbo() {
    init_tracing();
    let result = test_chat_with_model("gpt-4-turbo").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

// =============================================================================
// GPT-4 Series Tests - Responses API
// =============================================================================

#[tokio::test]
async fn test_responses_gpt4() {
    init_tracing();
    let result = test_responses_with_model("gpt-4").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}

#[tokio::test]
async fn test_responses_gpt4_turbo() {
    init_tracing();
    let result = test_responses_with_model("gpt-4-turbo").await;
    assert!(result.is_ok(), "{}", result.unwrap_err());
}
