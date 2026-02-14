//! Integration tests for Chat Completions API
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test chat_integration

use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::{
    errors::OpenAIToolError, message::Message, parameters::ParameterProperty, role::Role, structured_output::Schema, tool::Tool,
};
use serde::{Deserialize, Serialize};
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
async fn test_chat_completion() {
    init_tracing();
    let mut chat = ChatCompletion::new();
    let messages = vec![Message::from_string(Role::User, "Hi there!")];

    chat.model_id("gpt-5-mini")
        .messages(messages)
        .temperature(1.0)
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match chat.chat().await {
            Ok(response) => {
                tracing::info!("{:?}", &response.choices[0].message.content.clone().expect("Response content should not be empty"));
                assert!(true);
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Chat completion failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Chat completion failed");
                }
            },
        };
    }
}

#[tokio::test]
async fn test_chat_completion_japanese() {
    init_tracing();
    let mut chat = ChatCompletion::new();
    let messages = vec![Message::from_string(Role::User, "トンネルを抜けると？")];

    chat.model_id("gpt-5-mini")
        .messages(messages)
        .temperature(1.5)
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match chat.chat().await {
            Ok(response) => {
                println!("{:?}", &response.choices[0].message.content.clone().expect("Response content should not be empty"));
                assert!(true);
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Chat completion failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Chat completion failed");
                }
            },
        };
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Weather {
    #[serde(default = "String::new")]
    location: String,
    #[serde(default = "String::new")]
    date: String,
    #[serde(default = "String::new")]
    weather: String,
    #[serde(default = "String::new")]
    error: String,
}

#[tokio::test]
async fn test_chat_completion_with_json_schema() {
    init_tracing();
    let mut openai = ChatCompletion::new();
    let messages = vec![Message::from_string(Role::User, "Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error.")];

    let mut json_schema = Schema::chat_json_schema("weather");
    json_schema.add_property("location", "string", "The location to check the weather for.");
    json_schema.add_property("date", "string", "The date to check the weather for.");
    json_schema.add_property("weather", "string", "The weather for the location and date.");
    json_schema.add_property("error", "string", "Error message. If there is no error, leave this field empty.");
    openai
        .model_id("gpt-5-mini")
        .messages(messages)
        .temperature(1.0)
        .json_schema(json_schema)
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match openai.chat().await {
            Ok(response) => {
                println!("{:#?}", response);
                match serde_json::from_str::<Weather>(
                    &response.choices[0]
                        .message
                        .content
                        .clone()
                        .expect("Response content should not be empty")
                        .text
                        .expect("Response content should not be empty"),
                ) {
                    Ok(weather) => {
                        println!("{:#?}", weather);
                        assert!(true);
                    }
                    Err(e) => {
                        println!("{:#?}", e);
                        assert!(false);
                    }
                }
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Chat completion failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Chat completion failed");
                }
            },
        };
    }
}

#[derive(Deserialize)]
struct Summary {
    pub is_survey: bool,
    pub research_question: String,
    pub contributions: String,
    pub dataset: String,
    pub proposed_method: String,
    pub experiment_results: String,
    pub comparison_with_related_works: String,
    pub future_works: String,
}

#[tokio::test]
async fn test_summarize() {
    init_tracing();
    let mut openai = ChatCompletion::new();
    let instruction = std::fs::read_to_string("src/test_rsc/sample_instruction.txt").unwrap();

    let messages = vec![Message::from_string(Role::User, instruction.clone())];

    let mut json_schema = Schema::chat_json_schema("summary");
    json_schema.add_property("is_survey", "boolean", "この論文がサーベイ論文かどうかをtrue/falseで判定．");
    json_schema.add_property(
        "research_question",
        "string",
        "この論文のリサーチクエスチョンの説明．この論文の背景や既存研究との関連も含めて記述する．",
    );
    json_schema.add_property("contributions", "string", "この論文のコントリビューションをリスト形式で記述する．");
    json_schema.add_property("dataset", "string", "この論文で使用されているデータセットをリストアップする．");
    json_schema.add_property("proposed_method", "string", "提案手法の詳細な説明．");
    json_schema.add_property("experiment_results", "string", "実験の結果の詳細な説明．");
    json_schema.add_property(
        "comparison_with_related_works",
        "string",
        "関連研究と比較した場合のこの論文の新規性についての説明．可能な限り既存研究を参照しながら記述すること．",
    );
    json_schema.add_property("future_works", "string", "未解決の課題および将来の研究の方向性について記述．");

    openai
        .model_id(String::from("gpt-5-mini"))
        .messages(messages)
        .temperature(1.0)
        .json_schema(json_schema)
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match openai.chat().await {
            Ok(response) => {
                println!("{:#?}", response);
                match serde_json::from_str::<Summary>(
                    &response.choices[0]
                        .message
                        .content
                        .clone()
                        .expect("Response content should not be empty")
                        .text
                        .expect("Response content should not be empty"),
                ) {
                    Ok(summary) => {
                        tracing::info!("Summary.is_survey: {}", summary.is_survey);
                        tracing::info!("Summary.research_question: {}", summary.research_question);
                        tracing::info!("Summary.contributions: {}", summary.contributions);
                        tracing::info!("Summary.dataset: {}", summary.dataset);
                        tracing::info!("Summary.proposed_method: {}", summary.proposed_method);
                        tracing::info!("Summary.experiment_results: {}", summary.experiment_results);
                        tracing::info!("Summary.comparison_with_related_works: {}", summary.comparison_with_related_works);
                        tracing::info!("Summary.future_works: {}", summary.future_works);
                        assert!(true);
                    }
                    Err(e) => {
                        tracing::error!("Error: {}", e);
                        assert!(false);
                    }
                }
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Chat completion failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Chat completion failed");
                }
            },
        };
    }
}

#[tokio::test]
async fn test_chat_completion_with_function_calling() {
    init_tracing();
    let mut chat = ChatCompletion::new();
    let messages = vec![Message::from_string(Role::User, "Please calculate 25 + 17 using the calculator tool.")];

    // Define a calculator function tool
    let calculator_tool = Tool::function(
        "calculator",
        "A calculator that can perform basic arithmetic operations",
        vec![
            ("operation", ParameterProperty::from_string("The operation to perform (add, subtract, multiply, divide)")),
            ("a", ParameterProperty::from_number("The first number")),
            ("b", ParameterProperty::from_number("The second number")),
        ],
        false, // strict mode
    );

    chat.model_id("gpt-5-mini")
        .messages(messages)
        .temperature(0.1)
        .tools(vec![calculator_tool])
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");
    // First call
    let mut counter = 3;
    loop {
        match chat.chat().await {
            Ok(response) => {
                tracing::info!("First Response: {:#?}", response);

                let message = response.choices[0].message.clone();
                chat.add_message(message.clone());

                // Check if the response contains tool calls
                if let Some(tool_calls) = &message.tool_calls {
                    assert!(!tool_calls.is_empty(), "Tool calls should not be empty");

                    for tool_call in tool_calls {
                        tracing::info!("Function called: {}", tool_call.function.name);
                        tracing::info!("Arguments: {:?}", tool_call.function.arguments);

                        // Verify that the calculator function was called
                        assert_eq!(tool_call.function.name, "calculator");

                        // Parse the arguments to verify they contain the expected operation
                        let args = tool_call.function.arguments_as_map().unwrap();
                        assert!(args.get("operation").is_some());
                        assert!(args.get("a").is_some());
                        assert!(args.get("b").is_some());

                        tracing::info!("Function call validation passed");

                        chat.add_message(Message::from_tool_call_response("42", &tool_call.id));
                    }
                    assert!(true);
                } else {
                    // If no tool calls, check if the content mentions function calling
                    tracing::info!(
                        "No tool calls found. Content: {}",
                        &response.choices[0]
                            .message
                            .content
                            .clone()
                            .expect("Response content should not be empty")
                            .text
                            .expect("Response content should not be empty")
                    );
                    // This might happen if the model decides not to use the tool
                    // We'll still consider this a valid response for testing purposes
                    assert!(false, "Expected tool calls but none found in response");
                }
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Function calling test failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Function calling test failed");
                }
            },
        };
    }

    // Second call to ensure the tool is still available
    let messages = chat.get_message_history();
    let mut chat = ChatCompletion::new();
    chat.model_id("gpt-5-mini")
        .messages(messages)
        .temperature(1.0)
        .safety_identifier("a5c75abeef286919b4bf3ae40bc74c2d9ba03ac1bde3759e470e0f2b7056a5a1");

    let mut counter = 3;
    loop {
        match chat.chat().await {
            Ok(response) => {
                tracing::info!("Second Response: {:#?}", response);
                assert!(!response.choices.is_empty(), "Response should contain at least one choice");
                let content = response.choices[0]
                    .message
                    .content
                    .clone()
                    .expect("Response content should not be empty")
                    .text
                    .expect("Response content should not be empty");
                tracing::info!("Content: {}", content);
                // Check if the content contains the expected result
                assert!(content.contains("42"), "Expected content to contain '42', found: {}", content);
                break;
            }
            Err(e) => match e {
                OpenAIToolError::RequestError(e) => {
                    tracing::warn!("Request error: {} (retrying... {})", e, counter);
                    counter -= 1;
                    if counter == 0 {
                        assert!(false, "Function calling test failed (retry limit reached)");
                    }
                    continue;
                }
                _ => {
                    tracing::error!("Error: {}", e);
                    assert!(false, "Function calling test failed");
                }
            },
        };
    }
}
