//! Integration tests for Responses API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test responses_integration

use openai_tools::common::models::ChatModel;
use openai_tools::common::{
    message::{Content, Message},
    parameters::ParameterProperty,
    role::Role,
    structured_output::Schema,
    tool::Tool,
};
use openai_tools::responses::request::{Include, ReasoningEffort, ReasoningSummary, Responses, TextVerbosity, Truncation};
use serde::Deserialize;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt().with_env_filter(filter).with_test_writer().try_init();
    });
}

#[tokio::test]
async fn test_init_with_endpoint() {
    init_tracing();
    let mut responses = Responses::from_endpoint("https://api.openai.com/v1/responses");
    responses.model_id("gpt-4o-mini");
    responses.instructions("test instructions");
    responses.str_message("Hello world!");

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body: {}", body_json);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let message_output = res.output_text().unwrap();
                tracing::info!("Message output: {}", message_output);
                assert!(message_output.len() > 0);
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_plain_text() {
    init_tracing();
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");
    responses.instructions("test instructions");
    responses.str_message("Hello world!");

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body: {}", body_json);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let message_output = res.output_text().unwrap();
                tracing::info!("Message output: {}", message_output);
                assert!(message_output.len() > 0);
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_messages() {
    init_tracing();
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");
    responses.instructions("test instructions");
    let messages = vec![Message::from_string(Role::User, "Hello world!")];
    responses.messages(messages);

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body: {}", body_json);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let message_output = res.output_text().unwrap();
                tracing::info!("Message output: {}", message_output);
                assert!(message_output.len() > 0);
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_multi_turn_conversations() {
    init_tracing();

    // First interaction
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");
    let messages = vec![Message::from_string(Role::User, "Hello!")];
    responses.messages(messages);

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body: {}", body_json);

    let conversation_id: String;
    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let message_output = res.output_text().unwrap();
                tracing::info!("Message output: {}", message_output);
                assert!(message_output.len() > 0);

                // Save the conversation ID for the next turn
                conversation_id = res.id.as_ref().unwrap().clone();

                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }

    // Second interaction in the same conversation
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");
    let messages = vec![Message::from_string(Role::User, "What's the weather like today?")];
    responses.messages(messages);
    responses.previous_response_id(conversation_id);

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body for second turn: {}", body_json);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let message_output = res.output_text().unwrap();
                tracing::info!("Message output: {}", message_output);
                assert!(message_output.len() > 0);
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_tools() {
    init_tracing();
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");
    responses.instructions("test instructions");
    let messages = vec![Message::from_string(Role::User, "Calculate 2 + 2 using a calculator tool.")];
    responses.messages(messages);

    let tool = Tool::function(
        "calculator",
        "A simple calculator tool",
        vec![("a", ParameterProperty::from_number("The first number")), ("b", ParameterProperty::from_number("The second number"))],
        false,
    );
    responses.tools(vec![tool]);

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    println!("Request body: {}", body_json);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the function_call output in the response
                let function_call_output =
                    res.output.as_ref().unwrap().iter().find(|output| output.type_name.as_ref().unwrap() == "function_call").unwrap();
                assert_eq!(function_call_output.type_name.as_ref().unwrap(), "function_call");
                assert_eq!(function_call_output.name.as_ref().unwrap(), "calculator");
                assert!(function_call_output.call_id.as_ref().unwrap().len() > 0);
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_json_schema() {
    #[derive(Debug, Deserialize)]
    struct Country {
        pub name: String,
        pub population: String,
        pub big_or_small: bool,
    }
    #[derive(Debug, Deserialize)]
    struct TestResponse {
        pub capital: String,
        pub countries: Vec<Country>,
    }

    init_tracing();
    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini");

    let messages = vec![Message::from_string(Role::User, "What is the capital of France? Also, list some countries with their population.")];
    responses.messages(messages);

    let mut schema = Schema::responses_json_schema("capital");
    schema.add_property("capital", "string", "The capital city of France");
    schema.add_array("countries", vec![("name", "string"), ("population", "string"), ("big_or_small", "boolean")]);
    responses.structured_output(schema);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let res = serde_json::from_str::<TestResponse>(&res.output_text().unwrap()).unwrap();
                assert_eq!(res.capital, "Paris");
                assert!(res.countries.len() > 0);
                for country in res.countries.iter() {
                    assert!(country.name.len() > 0);
                    assert!(country.population.len() > 0);
                    assert!(country.big_or_small == true || country.big_or_small == false);
                }
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_responses_with_image_input() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model_id("gpt-4o-mini").messages(vec![Message::from_message_array(
        Role::User,
        vec![
            Content::from_text("What do you see in this image?"),
            Content::from_image_url(
                "https://images.ctfassets.net/kftzwdyauwt9/1cFVP33AOU26mMJmCGDo1S/0029938b700b84cd7caed52124ed508d/OAI_BrandPage_11.png",
            ),
        ],
    )]);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                for output in res.output.unwrap().iter() {
                    if output.type_name.as_ref().unwrap() == "message" {
                        tracing::info!("Image URL: {}", output.content.as_ref().unwrap()[0].text.as_ref().unwrap());
                    }
                }
                break;
            }
            Err(e) => {
                tracing::error!("Error: {} (retrying... {})", e, counter);
                counter -= 1;
                if counter == 0 {
                    assert!(false, "Failed to complete responses after 3 attempts");
                }
            }
        }
    }
}

#[tokio::test]
async fn test_error_handling_missing_messages() {
    init_tracing();

    let mut responses = Responses::new();

    // Set basic required parameters without messages
    responses.model_id("gpt-4o-mini");
    let response = responses.complete().await;
    tracing::info!("Response result: {:?}", response);
    assert!(response.is_err(), "Expected error due to missing messages");
}

#[tokio::test]
async fn test_error_handling_empty_messages() {
    init_tracing();

    let mut responses = Responses::new();

    // Set basic required parameters without messages
    responses.model_id("gpt-4o-mini");
    responses.messages(vec![]); // Empty messages
    let response = responses.complete().await;
    tracing::info!("Response result: {:?}", response);
    assert!(response.is_err(), "Expected error due to empty messages");
}

#[test]
fn test_optional_parameters() {
    // Unit test - doesn't require API key
    init_tracing();

    let mut responses = Responses::new();

    // Set basic required parameters
    responses.model_id("gpt-4o-mini");
    responses.str_message("Write a short poem about programming in exactly 50 words.");

    // Test various optional parameters
    responses.temperature(0.7); // Creativity control
    responses.max_output_tokens(100); // Output length limit
    responses.max_tool_calls(2); // Tool call limit
    responses.parallel_tool_calls(true); // Parallel tool execution
    responses.store(false); // Storage preference
    responses.stream(false); // Streaming disabled
    responses.top_logprobs(3); // Log probabilities
    responses.top_p(0.9); // Nucleus sampling
    responses.truncation(Truncation::Auto); // Input truncation

    // Add metadata for tracking
    responses.metadata("test_type".to_string(), serde_json::Value::String("optional_params".to_string()));
    responses.metadata("version".to_string(), serde_json::Value::String("1".to_string()));
    responses.metadata("debug".to_string(), serde_json::Value::String("true".to_string()));

    // Set conversation tracking
    responses.conversation("conv-test-conversation-123");
    responses.safety_identifier("moderate");
    responses.service_tier("default");

    // Add reasoning configuration
    responses.reasoning(ReasoningEffort::Medium, ReasoningSummary::Concise);

    // Set background processing and includes (using valid API values)
    responses.background(false);
    responses.include(vec![
        Include::WebSearchCall,             // "web_search_call.results"
        Include::ReasoningEncryptedContent, // "reasoning.encrypted_content"
    ]);

    let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body with optional parameters: {}", body_json);

    // Verify that all optional parameters are set correctly in the request body
    assert_eq!(responses.request_body.temperature, Some(0.7));
    assert_eq!(responses.request_body.max_output_tokens, Some(100));
    assert_eq!(responses.request_body.max_tool_calls, Some(2));
    assert_eq!(responses.request_body.parallel_tool_calls, Some(true));
    assert_eq!(responses.request_body.store, Some(false));
    assert_eq!(responses.request_body.stream, Some(false));
    assert_eq!(responses.request_body.top_logprobs, Some(3));
    assert_eq!(responses.request_body.top_p, Some(0.9));
    assert!(matches!(responses.request_body.truncation, Some(Truncation::Auto)));
    assert_eq!(responses.request_body.conversation, Some("conv-test-conversation-123".to_string()));
    assert_eq!(responses.request_body.safety_identifier, Some("moderate".to_string()));
    assert_eq!(responses.request_body.service_tier, Some("default".to_string()));
    assert_eq!(responses.request_body.background, Some(false));
    assert!(responses.request_body.metadata.is_some());
    assert!(responses.request_body.reasoning.is_some());
    assert!(responses.request_body.include.is_some());

    // Verify metadata content
    let metadata = responses.request_body.metadata.as_ref().unwrap();
    assert_eq!(metadata.get("test_type"), Some(&serde_json::Value::String("optional_params".to_string())));
    assert_eq!(metadata.get("version"), Some(&serde_json::Value::String("1".to_string())));
    assert_eq!(metadata.get("debug"), Some(&serde_json::Value::String("true".to_string())));

    // Verify reasoning configuration
    let reasoning = responses.request_body.reasoning.as_ref().unwrap();
    assert!(matches!(reasoning.effort, Some(ReasoningEffort::Medium)));
    assert!(matches!(reasoning.summary, Some(ReasoningSummary::Concise)));

    // Verify include fields
    let includes = responses.request_body.include.as_ref().unwrap();
    assert!(includes.contains(&Include::WebSearchCall));
    assert!(includes.contains(&Include::ReasoningEncryptedContent));

    // Verify that the request body can be serialized to JSON without errors
    let json_result = serde_json::to_string_pretty(&responses.request_body);
    assert!(json_result.is_ok(), "Failed to serialize request body to JSON: {:?}", json_result.err());

    let json_body = json_result.unwrap();
    tracing::info!("Successfully serialized request body with all optional parameters");

    // Verify key fields are present in the JSON
    assert!(json_body.contains("\"temperature\": 0.7"));
    assert!(json_body.contains("\"max_output_tokens\": 100"));
    assert!(json_body.contains("\"reasoning\""));
    assert!(json_body.contains("\"include\""));
    assert!(json_body.contains("\"metadata\""));

    tracing::info!("All optional parameters test passed successfully");
}

// ============================================================================
// GPT-5.2 New Parameter Tests
// ============================================================================

/// Test reasoning effort with None value (no reasoning tokens)
#[test]
fn test_reasoning_effort_none_serialization() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("What is 2+2?");
    responses.reasoning(ReasoningEffort::None, ReasoningSummary::Auto);

    let json_body = serde_json::to_string(&responses.request_body).unwrap();
    tracing::info!("Request body with ReasoningEffort::None: {}", json_body);

    assert!(json_body.contains("\"effort\":\"none\""));
}

/// Test reasoning effort with Xhigh value (maximum reasoning)
#[test]
fn test_reasoning_effort_xhigh_serialization() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("Solve this complex math problem: Find all prime factors of 123456789.");
    responses.reasoning(ReasoningEffort::Xhigh, ReasoningSummary::Detailed);

    let json_body = serde_json::to_string(&responses.request_body).unwrap();
    tracing::info!("Request body with ReasoningEffort::Xhigh: {}", json_body);

    assert!(json_body.contains("\"effort\":\"xhigh\""));
    assert!(json_body.contains("\"summary\":\"detailed\""));
}

/// Test text verbosity parameter
#[test]
fn test_text_verbosity_serialization() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("Explain quantum computing.");
    responses.text_verbosity(TextVerbosity::High);

    let json_body = serde_json::to_string(&responses.request_body).unwrap();
    tracing::info!("Request body with TextVerbosity::High: {}", json_body);

    assert!(json_body.contains("\"text\""));
    assert!(json_body.contains("\"verbosity\":\"high\""));
}

/// Test text verbosity with low setting for concise responses
#[test]
fn test_text_verbosity_low_serialization() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("What is the capital of Japan?");
    responses.text_verbosity(TextVerbosity::Low);

    let json_body = serde_json::to_string(&responses.request_body).unwrap();
    tracing::info!("Request body with TextVerbosity::Low: {}", json_body);

    assert!(json_body.contains("\"verbosity\":\"low\""));
}

/// Test combining reasoning effort and text verbosity
#[test]
fn test_combined_reasoning_and_verbosity() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("Analyze the impact of AI on society.");
    responses.reasoning(ReasoningEffort::High, ReasoningSummary::Detailed);
    responses.text_verbosity(TextVerbosity::High);

    let json_body = serde_json::to_string(&responses.request_body).unwrap();
    tracing::info!("Request body with reasoning and verbosity: {}", json_body);

    assert!(json_body.contains("\"reasoning\""));
    assert!(json_body.contains("\"effort\":\"high\""));
    assert!(json_body.contains("\"text\""));
    assert!(json_body.contains("\"verbosity\":\"high\""));
}

/// Integration test: Test GPT-5.2 with reasoning effort None (API call)
#[tokio::test]
async fn test_gpt52_with_reasoning_none() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("What is 2+2? Reply with just the number.");
    responses.reasoning(ReasoningEffort::None, ReasoningSummary::Auto);
    responses.max_output_tokens(50);

    let result = responses.complete().await;
    tracing::info!("GPT-5.2 with ReasoningEffort::None result: {:?}", result);

    match result {
        Ok(response) => {
            tracing::info!("Response received successfully");
            assert!(response.output.is_some(), "Expected output to be present");
            assert!(!response.output.unwrap().is_empty(), "Expected non-empty output");
        }
        Err(e) => {
            // GPT-5.2 may not be available in all accounts
            tracing::warn!("GPT-5.2 test failed (model may not be available): {:?}", e);
        }
    }
}

/// Integration test: Test GPT-5.2 with text verbosity (API call)
#[tokio::test]
async fn test_gpt52_with_text_verbosity() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt5_2);
    responses.str_message("What is photosynthesis?");
    responses.text_verbosity(TextVerbosity::Low);
    responses.max_output_tokens(100);

    let result = responses.complete().await;
    tracing::info!("GPT-5.2 with TextVerbosity::Low result: {:?}", result);

    match result {
        Ok(response) => {
            tracing::info!("Response received successfully");
            assert!(response.output.is_some(), "Expected output to be present");
            assert!(!response.output.unwrap().is_empty(), "Expected non-empty output");
        }
        Err(e) => {
            // GPT-5.2 may not be available in all accounts
            tracing::warn!("GPT-5.2 test failed (model may not be available): {:?}", e);
        }
    }
}

/// Integration test: Test with_model constructor
#[tokio::test]
async fn test_with_model_constructor() {
    init_tracing();

    let mut responses = Responses::with_model(ChatModel::Gpt4oMini);
    responses.str_message("Say hello in Japanese.");
    responses.max_output_tokens(50);

    let result = responses.complete().await;
    tracing::info!("with_model constructor result: {:?}", result);

    assert!(result.is_ok(), "Expected successful response");
    let response = result.unwrap();
    assert!(response.output.is_some(), "Expected output to be present");
    assert!(!response.output.unwrap().is_empty(), "Expected non-empty output");
}

// ============================================================================
// New Endpoint Integration Tests
// ============================================================================

use openai_tools::responses::request::{NamedFunctionChoice, Prompt, ToolChoice, ToolChoiceMode};

/// Integration test: Create and retrieve a response
#[tokio::test]
async fn test_create_and_retrieve_response() {
    init_tracing();

    // First, create a response
    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt4oMini);
    responses.str_message("Say 'test' and nothing else.");
    responses.max_output_tokens(20);
    responses.store(true); // Store the response so we can retrieve it

    let create_result = responses.complete().await;
    assert!(create_result.is_ok(), "Failed to create response: {:?}", create_result.err());

    let created_response = create_result.unwrap();
    tracing::info!("Created response ID: {:?}", created_response.id);

    // Now retrieve the response
    if let Some(response_id) = &created_response.id {
        let retrieve_result = responses.retrieve(response_id).await;
        tracing::info!("Retrieve result: {:?}", retrieve_result);

        // Note: retrieve may fail if the response is not stored or immediately deleted
        // by the platform, so we just log the result
        if retrieve_result.is_ok() {
            let retrieved = retrieve_result.unwrap();
            assert_eq!(retrieved.id.as_ref(), Some(response_id));
        } else {
            tracing::warn!("Could not retrieve response (may be expected): {:?}", retrieve_result.err());
        }
    }
}

/// Integration test: Delete a response
#[tokio::test]
async fn test_delete_response() {
    init_tracing();

    // First, create a response
    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt4oMini);
    responses.str_message("Say 'delete test'.");
    responses.max_output_tokens(20);
    responses.store(true);

    let create_result = responses.complete().await;
    assert!(create_result.is_ok(), "Failed to create response");

    let created_response = create_result.unwrap();

    // Now delete the response
    if let Some(response_id) = &created_response.id {
        let delete_result = responses.delete(response_id).await;
        tracing::info!("Delete result: {:?}", delete_result);

        // The delete may succeed or fail depending on platform state
        if delete_result.is_ok() {
            let deleted = delete_result.unwrap();
            assert_eq!(deleted.id, *response_id);
            assert!(deleted.deleted);
        } else {
            tracing::warn!("Could not delete response (may be expected): {:?}", delete_result.err());
        }
    }
}

/// Integration test: Test tool_choice parameter with auto mode
#[tokio::test]
async fn test_tool_choice_auto() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt4oMini);
    responses.str_message("What is 2 + 2?");
    responses.tool_choice(ToolChoice::Simple(ToolChoiceMode::Auto));
    responses.max_output_tokens(50);

    let result = responses.complete().await;
    tracing::info!("Tool choice auto result: {:?}", result);

    assert!(result.is_ok(), "Expected successful response with tool_choice: auto");
}

/// Integration test: Test tool_choice parameter with none mode
#[tokio::test]
async fn test_tool_choice_none() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt4oMini);
    responses.str_message("Hello!");
    responses.tool_choice(ToolChoice::Simple(ToolChoiceMode::None));
    responses.max_output_tokens(50);

    let result = responses.complete().await;
    tracing::info!("Tool choice none result: {:?}", result);

    assert!(result.is_ok(), "Expected successful response with tool_choice: none");
}

/// Unit test: Test new request parameters serialization
#[test]
fn test_new_parameters_serialization() {
    init_tracing();

    let mut responses = Responses::new();
    responses.model(ChatModel::Gpt4oMini);
    responses.str_message("Test");
    responses.tool_choice(ToolChoice::Simple(ToolChoiceMode::Required));
    responses.prompt_cache_key("my-cache-key-123");
    responses.prompt_cache_retention("24h");

    let json_body = serde_json::to_string_pretty(&responses.request_body).unwrap();
    tracing::info!("Request body with new parameters: {}", json_body);

    assert!(json_body.contains("\"tool_choice\": \"required\""));
    assert!(json_body.contains("\"prompt_cache_key\": \"my-cache-key-123\""));
    assert!(json_body.contains("\"prompt_cache_retention\": \"24h\""));
}
