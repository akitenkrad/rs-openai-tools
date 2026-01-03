//! Integration tests for Responses API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test responses_integration

use openai_tools::common::{
    message::{Content, Message},
    parameters::ParameterProperty,
    role::Role,
    structured_output::Schema,
    tool::Tool,
};
use openai_tools::responses::request::{Include, ReasoningEffort, ReasoningSummary, Responses, Truncation};
use serde::Deserialize;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .try_init();
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
    let messages = vec![Message::from_string(
        Role::User,
        "What's the weather like today?",
    )];
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
    let messages = vec![Message::from_string(
        Role::User,
        "Calculate 2 + 2 using a calculator tool.",
    )];
    responses.messages(messages);

    let tool = Tool::function(
        "calculator",
        "A simple calculator tool",
        vec![
            ("a", ParameterProperty::from_number("The first number")),
            ("b", ParameterProperty::from_number("The second number")),
        ],
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
                let function_call_output = res
                    .output
                    .as_ref()
                    .unwrap()
                    .iter()
                    .find(|output| output.type_name.as_ref().unwrap() == "function_call")
                    .unwrap();
                assert_eq!(
                    function_call_output.type_name.as_ref().unwrap(),
                    "function_call"
                );
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

    let messages = vec![Message::from_string(
        Role::User,
        "What is the capital of France? Also, list some countries with their population.",
    )];
    responses.messages(messages);

    let mut schema = Schema::responses_json_schema("capital");
    schema.add_property("capital", "string", "The capital city of France");
    schema.add_array(
        "countries",
        vec![
            ("name", "string"),
            ("population", "string"),
            ("big_or_small", "boolean"),
        ],
    );
    responses.structured_output(schema);

    let mut counter = 3;
    loop {
        match responses.complete().await {
            Ok(res) => {
                tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());

                // Find the message output in the response
                let res =
                    serde_json::from_str::<TestResponse>(&res.output_text().unwrap()).unwrap();
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
                        tracing::info!(
                            "Image URL: {}",
                            output.content.as_ref().unwrap()[0].text.as_ref().unwrap()
                        );
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
    responses.metadata(
        "test_type".to_string(),
        serde_json::Value::String("optional_params".to_string()),
    );
    responses.metadata(
        "version".to_string(),
        serde_json::Value::String("1".to_string()),
    );
    responses.metadata(
        "debug".to_string(),
        serde_json::Value::String("true".to_string()),
    );

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
    assert!(matches!(
        responses.request_body.truncation,
        Some(Truncation::Auto)
    ));
    assert_eq!(
        responses.request_body.conversation,
        Some("conv-test-conversation-123".to_string())
    );
    assert_eq!(
        responses.request_body.safety_identifier,
        Some("moderate".to_string())
    );
    assert_eq!(
        responses.request_body.service_tier,
        Some("default".to_string())
    );
    assert_eq!(responses.request_body.background, Some(false));
    assert!(responses.request_body.metadata.is_some());
    assert!(responses.request_body.reasoning.is_some());
    assert!(responses.request_body.include.is_some());

    // Verify metadata content
    let metadata = responses.request_body.metadata.as_ref().unwrap();
    assert_eq!(
        metadata.get("test_type"),
        Some(&serde_json::Value::String("optional_params".to_string()))
    );
    assert_eq!(
        metadata.get("version"),
        Some(&serde_json::Value::String("1".to_string()))
    );
    assert_eq!(
        metadata.get("debug"),
        Some(&serde_json::Value::String("true".to_string()))
    );

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
    assert!(
        json_result.is_ok(),
        "Failed to serialize request body to JSON: {:?}",
        json_result.err()
    );

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
