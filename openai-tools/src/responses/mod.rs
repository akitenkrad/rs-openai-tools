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
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize the responses client
//!     let mut responses = Responses::new();
//!     
//!     // Configure basic parameters
//!     responses
//!         .model_id("gpt-4o-mini")
//!         .instructions("You are a helpful assistant.");
//!     
//!     // Simple text input
//!     responses.plain_text_input("Hello! How are you today?");
//!
//!     // Send the request
//!     let response = responses.complete().await?;
//!     
//!     println!("AI Response: {}",
//!              response.output[0].content.as_ref().unwrap()[0].text);
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
//!         .model_id("gpt-4o-mini")
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
//!     println!("Response: {}", response.output[0].content.as_ref().unwrap()[0].text);
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
//!         .model_id("gpt-4o-mini")
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
//!     println!("Image analysis: {}", response.output[0].content.as_ref().unwrap()[0].text);
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
//!     responses.model_id("gpt-4o-mini");
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
//!     responses.text(schema);
//!     
//!     let response = responses.complete().await?;
//!     
//!     // Parse structured response
//!     let product: ProductInfo = serde_json::from_str(
//!         &response.output[0].content.as_ref().unwrap()[0].text
//!     )?;
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
//! use openai_tools::common::parameters::{Parameters, ParameterProp};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses
//!         .model_id("gpt-4o-mini")
//!         .instructions("You are a helpful calculator assistant.");
//!     
//!     // Define a calculator tool
//!     let calculator_tool = Tool::function(
//!         "calculator",
//!         "Perform basic arithmetic operations",
//!         vec![
//!             ("operation", ParameterProp::string("add, subtract, multiply, or divide")),
//!             ("a", ParameterProp::number("First number")),
//!             ("b", ParameterProp::number("Second number")),
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
//!     if response.output[0].type_name == "function_call" {
//!         println!("Function called: {}", response.output[0].name.as_ref().unwrap());
//!         println!("Call ID: {}", response.output[0].call_id.as_ref().unwrap());
//!         // Handle the function call and continue the conversation...
//!     } else {
//!         println!("Text response: {}", response.output[0].content.as_ref().unwrap()[0].text);
//!     }
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
//!     match responses.model_id("gpt-4o-mini").complete().await {
//!         Ok(response) => {
//!             println!("Success: {}", response.output[0].content.as_ref().unwrap()[0].text);
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
        parameters::ParameterProperty,
        role::Role,
        structured_output::Schema,
        tool::Tool,
    };
    use crate::responses::request::Responses;

    use serde::Deserialize;
    use std::sync::Once;
    use tracing_subscriber::EnvFilter;

    static INIT: Once = Once::new();

    fn init_tracing() {
        INIT.call_once(|| {
            // `RUST_LOG` 環境変数があればそれを使い、なければ "info"
            let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_test_writer() // `cargo test` / nextest 用
                .init();
        });
    }

    #[tokio::test]
    async fn test_responses_with_plain_text() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");
        responses.plain_text_input("Hello world!");

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
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
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
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
                    assert_eq!(res.output[0].type_name, "function_call");
                    assert_eq!(res.output[0].name.as_ref().unwrap(), "calculator");
                    assert!(res.output[0].call_id.as_ref().unwrap().len() > 0);
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

    #[derive(Debug, Deserialize)]
    struct TestResponse {
        pub capital: String,
    }
    #[tokio::test]
    async fn test_responses_with_json_schema() {
        init_tracing();
        let mut responses = Responses::new();
        responses.model_id("gpt-4o-mini");

        let messages = vec![Message::from_string(Role::User, "What is the capital of France?")];
        responses.messages(messages);

        let mut schema = Schema::responses_json_schema("capital");
        schema.add_property("capital", "string", "The capital city of France");
        responses.text(schema);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    let res = serde_json::from_str::<TestResponse>(res.output[0].content.as_ref().unwrap()[0].text.as_str()).unwrap();
                    assert_eq!(res.capital, "Paris");
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
        responses.model_id("gpt-4o-mini");
        responses.instructions("test instructions");

        let message = Message::from_message_array(
            Role::User,
            vec![Content::from_text("Do you find a clock in this image?"), Content::from_image_file("src/test_rsc/sample_image.jpg")],
        );
        responses.messages(vec![message]);

        let body_json = serde_json::to_string_pretty(&responses.request_body).unwrap();
        tracing::info!("Request body: {}", body_json);

        let mut counter = 3;
        loop {
            match responses.complete().await {
                Ok(res) => {
                    tracing::info!("Response: {}", serde_json::to_string_pretty(&res).unwrap());
                    assert!(res.output[0].content.as_ref().unwrap()[0].text.len() > 0);
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
}
