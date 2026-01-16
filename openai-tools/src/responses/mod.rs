//! OpenAI Responses API Module
//!
//! This module provides functionality for interacting with the OpenAI Responses API,
//! which is designed for creating AI assistants that can handle various types of input
//! including text, images, and structured data. The Responses API offers a more flexible
//! and powerful way to build conversational AI applications compared to the traditional
//! Chat Completions API.
//!
//! # Key Features
//!
//! - **Multi-modal Input**: Support for text, images, and other content types
//! - **Structured Output**: JSON schema-based response formatting
//! - **Tool Integration**: Function calling capabilities with custom tools
//! - **Flexible Instructions**: System-level instructions for AI behavior
//! - **Rich Content Handling**: Support for complex message structures
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the responses client
//!     let mut responses = Responses::new();
//!     
//!     // Configure basic parameters
//!     responses
//!         .model_id("gpt-5-mini")
//!         .instructions("You are a helpful assistant.");
//!     
//!     // Simple text input
//!     responses.str_message("Hello! How are you today?");
//!
//!     // Send the request
//!     let response = responses.complete().await?;
//!     
//!     println!("AI Response: {}", response.output_text().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! # Advanced Usage Examples
//!
//! ## Using Message-based Conversations
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses
//!         .model_id("gpt-5-mini")
//!         .instructions("You are a knowledgeable assistant.");
//!     
//!     // Create a conversation with multiple messages
//!     let messages = vec![
//!         Message::from_string(Role::User, "What is artificial intelligence?"),
//!         Message::from_string(Role::Assistant, "AI is a field of computer science..."),
//!         Message::from_string(Role::User, "Can you give me a simple example?"),
//!     ];
//!     
//!     responses.messages(messages);
//!     
//!     let response = responses.complete().await?;
//!     println!("Response: {}", response.output_text().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-modal Input with Images
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::message::{Message, Content};
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses
//!         .model_id("gpt-5-mini")
//!         .instructions("You are an image analysis assistant.");
//!     
//!     // Create a message with both text and image content
//!     let message = Message::from_message_array(
//!         Role::User,
//!         vec![
//!             Content::from_text("What do you see in this image?"),
//!             Content::from_image_file("path/to/image.jpg"),
//!         ],
//!     );
//!     
//!     responses.messages(vec![message]);
//!     
//!     let response = responses.complete().await?;
//!     println!("Image analysis: {}", response.output_text().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ## Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::structured_output::Schema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize)]
//! struct ProductInfo {
//!     name: String,
//!     price: f64,
//!     category: String,
//!     in_stock: bool,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses.model_id("gpt-5-mini");
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "Extract product information: 'MacBook Pro 16-inch, $2499, Electronics, Available'")
//!     ];
//!     responses.messages(messages);
//!     
//!     // Define JSON schema for structured output
//!     let mut schema = Schema::responses_json_schema("product_info");
//!     schema.add_property("name", "string", "Product name");
//!     schema.add_property("price", "number", "Product price");
//!     schema.add_property("category", "string", "Product category");
//!     schema.add_property("in_stock", "boolean", "Availability status");
//!     
//!     responses.structured_output(schema);
//!     
//!     let response = responses.complete().await?;
//!     
//!     // Parse structured response
//!     let product: ProductInfo = serde_json::from_str(&response.output_text().unwrap())?;
//!     
//!     println!("Product: {} - ${} ({})", product.name, product.price, product.category);
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::tool::Tool;
//! use openai_tools::common::parameters::{Parameters, ParameterProperty};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses
//!         .model_id("gpt-5-mini")
//!         .instructions("You are a helpful calculator assistant.");
//!     
//!     // Define a calculator tool
//!     let calculator_tool = Tool::function(
//!         "calculator",
//!         "Perform basic arithmetic operations",
//!         vec![
//!             ("operation", ParameterProperty::from_string("add, subtract, multiply, or divide")),
//!             ("a", ParameterProperty::from_number("First number")),
//!             ("b", ParameterProperty::from_number("Second number")),
//!         ],
//!         false,
//!     );
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User, "Calculate 15 * 7 using the calculator tool")
//!     ];
//!     
//!     responses
//!         .messages(messages)
//!         .tools(vec![calculator_tool]);
//!     
//!     let response = responses.complete().await?;
//!     
//!     // Check if the model made a function call
//!     println!("Response: {}", response.output_text().unwrap());
//!
//!     Ok(())
//! }
//! ```
//!
//! # API Differences from Chat Completions
//!
//! The Responses API differs from the Chat Completions API in several key ways:
//!
//! - **Input Format**: More flexible input handling with support for various content types
//! - **Output Structure**: Different response format optimized for assistant-style interactions
//! - **Instructions**: Dedicated field for system-level instructions
//! - **Multi-modal**: Native support for images and other media types
//! - **Tool Integration**: Enhanced function calling capabilities
//!
//! # Environment Setup
//!
//! Ensure your OpenAI API key is configured:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```
//!
//! Or in a `.env` file:
//!
//! ```text
//! OPENAI_API_KEY=your-api-key-here
//! ```
//!
//! # Error Handling
//!
//! All operations return `Result` types for proper error handling:
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::errors::OpenAIToolError;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut responses = Responses::new();
//!     
//!     match responses.model_id("gpt-5-mini").complete().await {
//!         Ok(response) => {
//!             println!("Success: {}", response.output_text().unwrap());
//!         }
//!         Err(OpenAIToolError::RequestError(e)) => {
//!             eprintln!("Network error: {}", e);
//!         }
//!         Err(OpenAIToolError::SerdeJsonError(e)) => {
//!             eprintln!("JSON parsing error: {}", e);
//!         }
//!         Err(e) => {
//!             eprintln!("Other error: {}", e);
//!         }
//!     }
//! }
//! ```

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::common::{
        message::{Content, Message},
        models::ChatModel,
        parameters::ParameterProperty,
        role::Role,
        structured_output::Schema,
        tool::Tool,
    };
    use crate::responses::request::{Include, ReasoningEffort, ReasoningSummary, Responses, TextConfig, TextVerbosity, Truncation};

    #[test]
    fn test_responses_builder_model() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4oMini);
        assert_eq!(responses.request_body.model, ChatModel::Gpt4oMini);
    }

    #[test]
    fn test_responses_builder_instructions() {
        let mut responses = Responses::new();
        responses.instructions("You are a helpful assistant.");
        assert_eq!(responses.request_body.instructions, Some("You are a helpful assistant.".to_string()));
    }

    #[test]
    fn test_responses_builder_str_message() {
        let mut responses = Responses::new();
        responses.str_message("Hello world!");
        assert_eq!(responses.request_body.plain_text_input, Some("Hello world!".to_string()));
    }

    #[test]
    fn test_responses_builder_messages() {
        let mut responses = Responses::new();
        let messages = vec![Message::from_string(Role::User, "Hello!")];
        responses.messages(messages);
        assert!(responses.request_body.messages_input.is_some());
        assert_eq!(responses.request_body.messages_input.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_responses_builder_tools() {
        let mut responses = Responses::new();
        let tool = Tool::function(
            "calculator",
            "A simple calculator",
            vec![("a", ParameterProperty::from_number("First number")), ("b", ParameterProperty::from_number("Second number"))],
            false,
        );
        responses.tools(vec![tool]);
        assert!(responses.request_body.tools.is_some());
        assert_eq!(responses.request_body.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_responses_builder_structured_output() {
        let mut responses = Responses::new();
        let mut schema = Schema::responses_json_schema("test");
        schema.add_property("name", "string", "A name");
        responses.structured_output(schema);
        assert!(responses.request_body.structured_output.is_some());
    }

    #[test]
    fn test_responses_builder_temperature() {
        let mut responses = Responses::new();
        responses.temperature(0.7);
        assert_eq!(responses.request_body.temperature, Some(0.7));
    }

    #[test]
    #[should_panic(expected = "Temperature must be between 0.0 and 2.0")]
    fn test_responses_builder_temperature_invalid() {
        let mut responses = Responses::new();
        responses.temperature(3.0); // Should panic
    }

    #[test]
    fn test_responses_builder_max_output_tokens() {
        let mut responses = Responses::new();
        responses.max_output_tokens(100);
        assert_eq!(responses.request_body.max_output_tokens, Some(100));
    }

    #[test]
    fn test_responses_builder_metadata() {
        let mut responses = Responses::new();
        responses.metadata("key1".to_string(), serde_json::Value::String("value1".to_string()));
        responses.metadata("key2".to_string(), serde_json::Value::Number(42.into()));

        let metadata = responses.request_body.metadata.as_ref().unwrap();
        assert_eq!(metadata.get("key1"), Some(&serde_json::Value::String("value1".to_string())));
        assert_eq!(metadata.get("key2"), Some(&serde_json::Value::Number(42.into())));
    }

    #[test]
    fn test_responses_builder_reasoning() {
        let mut responses = Responses::new();
        responses.reasoning(ReasoningEffort::High, ReasoningSummary::Detailed);

        let reasoning = responses.request_body.reasoning.as_ref().unwrap();
        assert!(matches!(reasoning.effort, Some(ReasoningEffort::High)));
        assert!(matches!(reasoning.summary, Some(ReasoningSummary::Detailed)));
    }

    #[test]
    fn test_responses_builder_include() {
        let mut responses = Responses::new();
        responses.include(vec![Include::WebSearchCall, Include::ReasoningEncryptedContent]);

        let includes = responses.request_body.include.as_ref().unwrap();
        assert!(includes.contains(&Include::WebSearchCall));
        assert!(includes.contains(&Include::ReasoningEncryptedContent));
    }

    #[test]
    fn test_responses_builder_truncation() {
        let mut responses = Responses::new();
        responses.truncation(Truncation::Auto);
        assert!(matches!(responses.request_body.truncation, Some(Truncation::Auto)));

        responses.truncation(Truncation::Disabled);
        assert!(matches!(responses.request_body.truncation, Some(Truncation::Disabled)));
    }

    #[test]
    fn test_content_from_text() {
        let content = Content::from_text("Hello!");
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_content_from_image_url() {
        let content = Content::from_image_url("https://example.com/image.png");
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("https://example.com/image.png"));
    }

    #[test]
    fn test_message_from_message_array() {
        let message = Message::from_message_array(
            Role::User,
            vec![Content::from_text("Look at this:"), Content::from_image_url("https://example.com/img.png")],
        );
        assert_eq!(message.role, Role::User);
    }

    #[test]
    fn test_request_body_serialization() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4oMini);
        responses.str_message("Test message");
        responses.temperature(0.5);
        responses.max_output_tokens(100);

        let json = serde_json::to_string(&responses.request_body).unwrap();
        assert!(json.contains("gpt-4o-mini"));
        assert!(json.contains("Test message"));
    }

    #[test]
    fn test_optional_parameters_serialization() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4oMini);
        responses.str_message("Test");
        responses.temperature(0.7);
        responses.max_output_tokens(100);
        responses.max_tool_calls(2);
        responses.parallel_tool_calls(true);
        responses.store(false);
        responses.stream(false);
        responses.top_logprobs(3);
        responses.top_p(0.9);
        responses.truncation(Truncation::Auto);
        responses.conversation("conv-test-123");
        responses.safety_identifier("moderate");
        responses.service_tier("default");
        responses.background(false);
        responses.reasoning(ReasoningEffort::Medium, ReasoningSummary::Concise);
        responses.include(vec![Include::WebSearchCall]);
        responses.metadata("key".to_string(), serde_json::Value::String("value".to_string()));

        // Verify all parameters are set
        assert_eq!(responses.request_body.temperature, Some(0.7));
        assert_eq!(responses.request_body.max_output_tokens, Some(100));
        assert_eq!(responses.request_body.max_tool_calls, Some(2));
        assert_eq!(responses.request_body.parallel_tool_calls, Some(true));
        assert_eq!(responses.request_body.store, Some(false));
        assert_eq!(responses.request_body.stream, Some(false));
        assert_eq!(responses.request_body.top_logprobs, Some(3));
        assert_eq!(responses.request_body.top_p, Some(0.9));
        assert!(matches!(responses.request_body.truncation, Some(Truncation::Auto)));
        assert_eq!(responses.request_body.conversation, Some("conv-test-123".to_string()));
        assert_eq!(responses.request_body.safety_identifier, Some("moderate".to_string()));
        assert_eq!(responses.request_body.service_tier, Some("default".to_string()));
        assert_eq!(responses.request_body.background, Some(false));

        // Verify serialization works
        let json_result = serde_json::to_string_pretty(&responses.request_body);
        assert!(json_result.is_ok());

        let json_body = json_result.unwrap();
        assert!(json_body.contains("\"temperature\": 0.7"));
        assert!(json_body.contains("\"max_output_tokens\": 100"));
        assert!(json_body.contains("\"reasoning\""));
        assert!(json_body.contains("\"include\""));
        assert!(json_body.contains("\"metadata\""));
    }

    #[test]
    fn test_reasoning_model_detection_o1() {
        // Test that o1 models are detected as reasoning models
        // and non-default temperature is ignored at setter time
        let mut responses = Responses::new();
        responses.model(ChatModel::O1);
        responses.str_message("Test");
        responses.temperature(0.5); // Should be ignored with warning

        // Validation now happens at setter time for reasoning models
        // Non-default temperature (0.5) should be ignored
        assert_eq!(responses.request_body.temperature, None);
        assert_eq!(responses.request_body.model, ChatModel::O1);
    }

    #[test]
    fn test_reasoning_model_detection_o3() {
        // Test that o3 models are detected as reasoning models
        // and non-default temperature is ignored at setter time
        let mut responses = Responses::new();
        responses.model(ChatModel::O3Mini);
        responses.str_message("Test");
        responses.temperature(0.3); // Should be ignored with warning

        // Validation now happens at setter time for reasoning models
        // Non-default temperature (0.3) should be ignored
        assert_eq!(responses.request_body.temperature, None);
        assert_eq!(responses.request_body.model, ChatModel::O3Mini);
    }

    #[test]
    fn test_non_reasoning_model() {
        // Test that regular models are not affected
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4o);
        responses.str_message("Test");
        responses.temperature(0.7);

        assert_eq!(responses.request_body.temperature, Some(0.7));
        assert_eq!(responses.request_body.model, ChatModel::Gpt4o);
    }

    #[test]
    fn test_reasoning_model_with_default_temperature() {
        // Test that default temperature (1.0) is allowed for reasoning models
        let mut responses = Responses::new();
        responses.model(ChatModel::O1);
        responses.str_message("Test");
        responses.temperature(1.0);

        assert_eq!(responses.request_body.temperature, Some(1.0));
    }

    #[test]
    fn test_reasoning_effort_none() {
        // Test ReasoningEffort::None serialization
        let effort = ReasoningEffort::None;
        let json = serde_json::to_string(&effort).unwrap();
        assert_eq!(json, "\"none\"");
    }

    #[test]
    fn test_reasoning_effort_xhigh() {
        // Test ReasoningEffort::Xhigh serialization
        let effort = ReasoningEffort::Xhigh;
        let json = serde_json::to_string(&effort).unwrap();
        assert_eq!(json, "\"xhigh\"");
    }

    #[test]
    fn test_reasoning_effort_all_variants() {
        // Test all ReasoningEffort variants serialize correctly
        assert_eq!(serde_json::to_string(&ReasoningEffort::None).unwrap(), "\"none\"");
        assert_eq!(serde_json::to_string(&ReasoningEffort::Minimal).unwrap(), "\"minimal\"");
        assert_eq!(serde_json::to_string(&ReasoningEffort::Low).unwrap(), "\"low\"");
        assert_eq!(serde_json::to_string(&ReasoningEffort::Medium).unwrap(), "\"medium\"");
        assert_eq!(serde_json::to_string(&ReasoningEffort::High).unwrap(), "\"high\"");
        assert_eq!(serde_json::to_string(&ReasoningEffort::Xhigh).unwrap(), "\"xhigh\"");
    }

    #[test]
    fn test_text_verbosity_low() {
        let verbosity = TextVerbosity::Low;
        let json = serde_json::to_string(&verbosity).unwrap();
        assert_eq!(json, "\"low\"");
    }

    #[test]
    fn test_text_verbosity_medium() {
        let verbosity = TextVerbosity::Medium;
        let json = serde_json::to_string(&verbosity).unwrap();
        assert_eq!(json, "\"medium\"");
    }

    #[test]
    fn test_text_verbosity_high() {
        let verbosity = TextVerbosity::High;
        let json = serde_json::to_string(&verbosity).unwrap();
        assert_eq!(json, "\"high\"");
    }

    #[test]
    fn test_text_config_serialization() {
        let config = TextConfig { verbosity: Some(TextVerbosity::High) };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"verbosity\":\"high\""));
    }

    #[test]
    fn test_responses_builder_text_verbosity() {
        let mut responses = Responses::new();
        responses.text_verbosity(TextVerbosity::Low);

        assert!(responses.request_body.text.is_some());
        let text = responses.request_body.text.as_ref().unwrap();
        assert_eq!(text.verbosity, Some(TextVerbosity::Low));
    }

    #[test]
    fn test_responses_builder_reasoning_with_none() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt5_2);
        responses.reasoning(ReasoningEffort::None, ReasoningSummary::Auto);

        assert!(responses.request_body.reasoning.is_some());
        let reasoning = responses.request_body.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.effort, Some(ReasoningEffort::None));
    }

    #[test]
    fn test_responses_builder_reasoning_with_xhigh() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt5_2);
        responses.reasoning(ReasoningEffort::Xhigh, ReasoningSummary::Detailed);

        assert!(responses.request_body.reasoning.is_some());
        let reasoning = responses.request_body.reasoning.as_ref().unwrap();
        assert_eq!(reasoning.effort, Some(ReasoningEffort::Xhigh));
    }

    #[test]
    fn test_request_body_with_text_serialization() {
        // Test that text config is properly serialized in the request body
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt5_2);
        responses.str_message("Test");
        responses.text_verbosity(TextVerbosity::High);

        let json_body = serde_json::to_string(&responses.request_body).unwrap();
        assert!(json_body.contains("\"text\""));
        assert!(json_body.contains("\"verbosity\":\"high\""));
    }

    // ================================================
    // Tests for new ToolChoice, Prompt, and endpoint types
    // ================================================

    use crate::responses::request::{NamedFunctionChoice, Prompt, ToolChoice, ToolChoiceMode};

    #[test]
    fn test_tool_choice_mode_auto_serialization() {
        let mode = ToolChoiceMode::Auto;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"auto\"");
    }

    #[test]
    fn test_tool_choice_mode_none_serialization() {
        let mode = ToolChoiceMode::None;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"none\"");
    }

    #[test]
    fn test_tool_choice_mode_required_serialization() {
        let mode = ToolChoiceMode::Required;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"required\"");
    }

    #[test]
    fn test_named_function_choice_new() {
        let choice = NamedFunctionChoice::new("get_weather");
        assert_eq!(choice.type_name, "function");
        assert_eq!(choice.name, "get_weather");
    }

    #[test]
    fn test_named_function_choice_serialization() {
        let choice = NamedFunctionChoice::new("calculate");
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"calculate\""));
    }

    #[test]
    fn test_tool_choice_simple_serialization() {
        let choice = ToolChoice::Simple(ToolChoiceMode::Auto);
        let json = serde_json::to_string(&choice).unwrap();
        assert_eq!(json, "\"auto\"");

        let choice = ToolChoice::Simple(ToolChoiceMode::Required);
        let json = serde_json::to_string(&choice).unwrap();
        assert_eq!(json, "\"required\"");
    }

    #[test]
    fn test_tool_choice_function_serialization() {
        let choice = ToolChoice::Function(NamedFunctionChoice::new("search"));
        let json = serde_json::to_string(&choice).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"search\""));
    }

    #[test]
    fn test_prompt_new() {
        let prompt = Prompt::new("prompt-abc123");
        assert_eq!(prompt.id, "prompt-abc123");
        assert!(prompt.variables.is_none());
    }

    #[test]
    fn test_prompt_with_variables() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("name".to_string(), "Alice".to_string());
        vars.insert("topic".to_string(), "AI".to_string());

        let prompt = Prompt::with_variables("prompt-xyz", vars);
        assert_eq!(prompt.id, "prompt-xyz");
        assert!(prompt.variables.is_some());

        let variables = prompt.variables.as_ref().unwrap();
        assert_eq!(variables.get("name"), Some(&"Alice".to_string()));
        assert_eq!(variables.get("topic"), Some(&"AI".to_string()));
    }

    #[test]
    fn test_prompt_serialization_without_variables() {
        let prompt = Prompt::new("prompt-123");
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("\"id\":\"prompt-123\""));
        // variables should be skipped when None
        assert!(!json.contains("variables"));
    }

    #[test]
    fn test_prompt_serialization_with_variables() {
        let mut vars = std::collections::HashMap::new();
        vars.insert("key".to_string(), "value".to_string());

        let prompt = Prompt::with_variables("prompt-456", vars);
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(json.contains("\"id\":\"prompt-456\""));
        assert!(json.contains("\"variables\""));
        assert!(json.contains("\"key\":\"value\""));
    }

    #[test]
    fn test_responses_builder_tool_choice_simple() {
        let mut responses = Responses::new();
        responses.tool_choice(ToolChoice::Simple(ToolChoiceMode::Required));

        assert!(responses.request_body.tool_choice.is_some());
    }

    #[test]
    fn test_responses_builder_tool_choice_function() {
        let mut responses = Responses::new();
        responses.tool_choice(ToolChoice::Function(NamedFunctionChoice::new("my_function")));

        assert!(responses.request_body.tool_choice.is_some());
    }

    #[test]
    fn test_responses_builder_prompt() {
        let mut responses = Responses::new();
        responses.prompt(Prompt::new("prompt-test"));

        assert!(responses.request_body.prompt.is_some());
        assert_eq!(responses.request_body.prompt.as_ref().unwrap().id, "prompt-test");
    }

    #[test]
    fn test_responses_builder_prompt_cache_key() {
        let mut responses = Responses::new();
        responses.prompt_cache_key("my-cache-key");

        assert_eq!(responses.request_body.prompt_cache_key, Some("my-cache-key".to_string()));
    }

    #[test]
    fn test_responses_builder_prompt_cache_retention() {
        let mut responses = Responses::new();
        responses.prompt_cache_retention("24h");

        assert_eq!(responses.request_body.prompt_cache_retention, Some("24h".to_string()));
    }

    #[test]
    fn test_request_body_with_tool_choice_serialization() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4oMini);
        responses.str_message("Test");
        responses.tool_choice(ToolChoice::Simple(ToolChoiceMode::Auto));

        let json_body = serde_json::to_string(&responses.request_body).unwrap();
        assert!(json_body.contains("\"tool_choice\":\"auto\""));
    }

    #[test]
    fn test_request_body_with_prompt_serialization() {
        let mut responses = Responses::new();
        responses.model(ChatModel::Gpt4oMini);
        responses.str_message("Test");
        responses.prompt(Prompt::new("prompt-id"));
        responses.prompt_cache_key("cache-key");
        responses.prompt_cache_retention("1h");

        let json_body = serde_json::to_string(&responses.request_body).unwrap();
        assert!(json_body.contains("\"prompt\""));
        assert!(json_body.contains("\"prompt_cache_key\":\"cache-key\""));
        assert!(json_body.contains("\"prompt_cache_retention\":\"1h\""));
    }
}
