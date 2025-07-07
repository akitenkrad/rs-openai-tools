//! # OpenAI Tools for Rust
//!
//! A comprehensive Rust library for interacting with OpenAI's APIs, providing easy-to-use
//! interfaces for chat completions, responses, and various AI-powered functionalities.
//! This crate offers both high-level convenience methods and low-level control for
//! advanced use cases.
//!
//! ## Features
//!
//! - **Chat Completions API**: Full support for OpenAI's Chat Completions with streaming, function calling, and structured output
//! - **Responses API**: Advanced assistant-style interactions with multi-modal input support
//!
//! ## Quick Start
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! openai-tools = "0.1.0"
//! tokio = { version = "1.0", features = ["full"] }
//! serde = { version = "1.0", features = ["derive"] }
//! ```
//!
//! Set up your API key:
//!
//! ```bash
//! export OPENAI_API_KEY="your-api-key-here"
//! ```
//!
//! ## Basic Chat Completion
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![
//!         Message::from_string(Role::User, "Hello! How are you?")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .temperature(0.7)
//!         .chat()
//!         .await?;
//!
//!     println!("AI: {}", response.choices[0].message.content.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ## Structured Output with JSON Schema
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::{message::Message, role::Role, structured_output::Schema};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize)]
//! struct PersonInfo {
//!     name: String,
//!     age: u32,
//!     occupation: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Create JSON schema
//!     let mut schema = Schema::chat_json_schema("person_info");
//!     schema.add_property("name", "string", "Person's full name");
//!     schema.add_property("age", "number", "Person's age");
//!     schema.add_property("occupation", "string", "Person's job");
//!     
//!     let messages = vec![
//!         Message::from_string(Role::User,
//!             "Extract info: John Smith, 30, Software Engineer")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .json_schema(schema)
//!         .chat()
//!         .await?;
//!
//!     let person: PersonInfo = serde_json::from_str(
//!         &response.choices[0].message.content.clone().unwrap()
//!     )?;
//!     
//!     println!("Extracted: {} ({}), {}", person.name, person.age, person.occupation);
//!     Ok(())
//! }
//! ```
//!
//! ## Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::{message::Message, role::Role, tool::{Tool, ParameterProp}};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     
//!     // Define a weather tool
//!     let weather_tool = Tool::function(
//!         "get_weather",
//!         "Get current weather for a location",
//!         vec![
//!             ("location", ParameterProp::string("City name")),
//!             ("unit", ParameterProp::string("Temperature unit (celsius/fahrenheit)")),
//!         ],
//!         false,
//!     );
//!
//!     let messages = vec![
//!         Message::from_string(Role::User, "What's the weather in Tokyo?")
//!     ];
//!
//!     let response = chat
//!         .model_id("gpt-4o-mini")
//!         .messages(messages)
//!         .tools(vec![weather_tool])
//!         .chat()
//!         .await?;
//!
//!     // Handle tool calls
//!     if let Some(tool_calls) = &response.choices[0].message.tool_calls {
//!         for call in tool_calls {
//!             println!("Tool: {}", call.function.name);
//!             println!("Args: {}", call.function.arguments);
//!             // Execute the function and continue conversation...
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-modal Input with Responses API
//!
//! ```rust,no_run
//! use openai_tools::responses::request::Responses;
//! use openai_tools::common::{message::{Message, Content}, role::Role};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut responses = Responses::new();
//!     
//!     responses
//!         .model_id("gpt-4o-mini")
//!         .instructions("You are an image analysis assistant.");
//!     
//!     // Multi-modal message with text and image
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
//!     println!("Analysis: {}", response.output[0].content.as_ref().unwrap()[0].text);
//!     Ok(())
//! }
//! ```
//!
//! ## Module Structure
//!
//! This crate is organized into three main modules:
//!
//! - [`chat`] - OpenAI Chat Completions API interface
//!   - [`chat::request`] - Request building and sending
//!   - [`chat::response`] - Response data structures
//!
//! - [`responses`] - OpenAI Responses API interface (assistant-style interactions)
//!   - [`responses::request`] - Advanced request handling with multi-modal support
//!   - [`responses::response`] - Response structures for assistant interactions
//!
//! - [`common`] - Shared utilities and data types
//!   - [`common::message`] - Message and content structures
//!   - [`common::role`] - User roles (User, Assistant, System, etc.)
//!   - [`common::tool`] - Function calling and tool definitions
//!   - [`common::structured_output`] - JSON schema utilities
//!   - [`common::errors`] - Error types and handling
//!   - [`common::usage`] - Token usage tracking
//!
//! ## Error Handling
//!
//! All operations return `Result` types with detailed error information:
//!
//! ```rust,no_run
//! use openai_tools::common::errors::OpenAIToolError;
//! # use openai_tools::chat::request::ChatCompletion;
//!
//! # #[tokio::main]
//! # async fn main() {
//! # let mut chat = ChatCompletion::new();
//! match chat.chat().await {
//!     Ok(response) => println!("Success: {}", response.choices[0].message.content.clone().unwrap()),
//!     Err(OpenAIToolError::RequestError(e)) => eprintln!("Network error: {}", e),
//!     Err(OpenAIToolError::SerdeJsonError(e)) => eprintln!("JSON error: {}", e),
//!     Err(e) => eprintln!("Other error: {}", e),
//! }
//! # }
//! ```
//!
//! ## Environment Configuration
//!
//! The library automatically loads configuration from environment variables and `.env` files:
//!
//! ```bash
//! # Required
//! OPENAI_API_KEY=your-api-key-here
//!
//! # Optional
//! RUST_LOG=info  # For enabling debug logging
//! ```
//!

pub mod chat;
pub mod common;
pub mod responses;
