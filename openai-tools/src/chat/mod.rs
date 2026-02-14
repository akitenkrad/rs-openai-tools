//! # Chat Module
//!
//! This module provides functionality for interacting with the OpenAI Chat Completions API.
//! It includes tools for building requests, sending them to OpenAI's chat completion endpoint,
//! and processing the responses.
//!
//! ## Key Features
//!
//! - Chat completion request building and sending
//! - Structured output support with JSON schema
//! - Response parsing and processing
//! - Support for various OpenAI models and parameters
//!
//! ## Usage Examples
//!
//! ### Basic Chat Completion
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![Message::from_string(Role::User, "Hello!")];
//!     
//!     let response = chat
//!         .model_id("gpt-5-mini")
//!         .messages(messages)
//!         .temperature(1.0)
//!         .chat()
//!         .await?;
//!         
//!     println!("{}", response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap());
//!     Ok(())
//! }
//! ```
//!
//! ### Using JSON Schema for Structured Output
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::structured_output::Schema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Serialize, Deserialize)]
//! struct WeatherInfo {
//!     location: String,
//!     date: String,
//!     weather: String,
//!     temperature: String,
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![Message::from_string(
//!         Role::User,
//!         "What's the weather like tomorrow in Tokyo?"
//!     )];
//!     
//!     // Create JSON schema for structured output
//!     let mut json_schema = Schema::chat_json_schema("weather");
//!     json_schema.add_property("location", "string", "The location for weather check");
//!     json_schema.add_property("date", "string", "The date for weather forecast");
//!     json_schema.add_property("weather", "string", "Weather condition description");
//!     json_schema.add_property("temperature", "string", "Temperature information");
//!     
//!     let response = chat
//!         .model_id("gpt-5-mini")
//!         .messages(messages)
//!         .temperature(0.7)
//!         .json_schema(json_schema)
//!         .chat()
//!         .await?;
//!         
//!     // Parse structured response
//!     let weather: WeatherInfo = serde_json::from_str(
//!         response.choices[0].message.content.as_ref().unwrap().text.as_ref().unwrap()
//!     )?;
//!     println!("Weather in {}: {} on {}, Temperature: {}",
//!              weather.location, weather.weather, weather.date, weather.temperature);
//!     Ok(())
//! }
//! ```
//!
//! ### Using Function Calling with Tools
//!
//! ```rust,no_run
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::common::message::Message;
//! use openai_tools::common::role::Role;
//! use openai_tools::common::tool::Tool;
//! use openai_tools::common::parameters::ParameterProperty;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![Message::from_string(
//!         Role::User,
//!         "Please calculate 25 + 17 using the calculator tool"
//!     )];
//!     
//!     // Define a calculator function tool
//!     let calculator_tool = Tool::function(
//!         "calculator",
//!         "A calculator that can perform basic arithmetic operations",
//!         vec![
//!             ("operation", ParameterProperty::from_string("The operation to perform (add, subtract, multiply, divide)")),
//!             ("a", ParameterProperty::from_number("The first number")),
//!             ("b", ParameterProperty::from_number("The second number")),
//!         ],
//!         false, // strict mode
//!     );
//!     
//!     let response = chat
//!         .model_id("gpt-5-mini")
//!         .messages(messages)
//!         .temperature(0.1)
//!         .tools(vec![calculator_tool])
//!         .chat()
//!         .await?;
//!         
//!     // Handle function calls in the response
//!     if let Some(tool_calls) = &response.choices[0].message.tool_calls {
//!         // Add the assistant's message with tool calls to conversation history
//!         chat.add_message(response.choices[0].message.clone());
//!         
//!         for tool_call in tool_calls {
//!             println!("Function called: {}", tool_call.function.name);
//!             if let Ok(args) = tool_call.function.arguments_as_map() {
//!                 println!("Arguments: {:?}", args);
//!             }
//!             
//!             // Execute the function (in this example, we simulate the calculation)
//!             let result = "42"; // This would be the actual calculation result
//!             
//!             // Add the tool call response to continue the conversation
//!             chat.add_message(Message::from_tool_call_response(result, &tool_call.id));
//!         }
//!         
//!         // Get the final response after tool execution
//!         let final_response = chat.chat().await?;
//!         if let Some(content) = &final_response.choices[0].message.content {
//!             if let Some(text) = &content.text {
//!                 println!("Final answer: {}", text);
//!             }
//!         }
//!     } else if let Some(content) = &response.choices[0].message.content {
//!         if let Some(text) = &content.text {
//!             println!("{}", text);
//!         }
//!     }
//!     Ok(())
//! }
//! ```

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::common::{message::Message, parameters::ParameterProperty, role::Role, structured_output::Schema, tool::Tool};

    #[test]
    fn test_message_creation() {
        let message = Message::from_string(Role::User, "Hello!");
        assert_eq!(message.role, Role::User);
    }

    #[test]
    fn test_tool_creation() {
        let tool = Tool::function(
            "calculator",
            "A simple calculator",
            vec![("a", ParameterProperty::from_number("First number")), ("b", ParameterProperty::from_number("Second number"))],
            false,
        );

        // Verify tool can be serialized
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("calculator"));
        assert!(json.contains("A simple calculator"));
    }

    #[test]
    fn test_schema_creation() {
        let mut schema = Schema::chat_json_schema("test_schema");
        schema.add_property("name", "string", "A name field");
        schema.add_property("age", "number", "An age field");

        // Verify schema can be serialized
        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("test_schema"));
        assert!(json.contains("name"));
        assert!(json.contains("age"));
    }

    #[test]
    fn test_schema_with_array() {
        let mut schema = Schema::chat_json_schema("array_schema");
        schema.add_array("items", vec![("id", "number"), ("value", "string")]);

        let json = serde_json::to_string(&schema).unwrap();
        assert!(json.contains("items"));
        assert!(json.contains("array"));
    }
}
