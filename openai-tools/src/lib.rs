//! # openai-tools
//! This crate provides a simple interface to interact with OpenAI's Chat Completion API.
//! # Example Usage
//!
//! ```rust
//! # use openai_tools::common::{Message, Role};
//! # use openai_tools::chat::ChatCompletionResponse;
//! # use openai_tools::chat::ChatCompletion;
//! # #[tokio::main]
//! # async fn main() {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![
//!         Message::from_string(Role::User, String::from("Hi there!"))
//!     ];
//!
//!     chat
//!         .model_id(String::from("gpt-4o-mini"))
//!         .messages(messages)
//!         .temperature(1.0);
//!
//!     let response: ChatCompletionResponse = chat.chat().await.unwrap();
//!     println!("{}", &response.choices[0].message.content);
//!     // Hello! How can I assist you today?
//! # }
//! ```
//!
//! ### Simple Chat Completion
//!
//! ```rust
//! # use openai_tools::common::{Message, Role};
//! # use openai_tools::chat::{ChatCompletion, ChatCompletionResponse};
//! # #[tokio::main]
//! # async fn main() {
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![
//!         Message::from_string(Role::User, String::from("Hi there!"))
//!     ];
//!
//!     chat
//!         .model_id(String::from("gpt-4o-mini"))
//!         .messages(messages)
//!         .temperature(1.0);
//!
//!     let response: ChatCompletionResponse = chat.chat().await.unwrap();
//!     println!("{}", &response.choices[0].message.content);
//!     // Hello! How can I assist you today?
//! # }
//! ```
//
//! ### Chat with Json Schema
//
//! ```rust
//! # use openai_tools::structured_output::Schema;
//! # use openai_tools::common::{Message, Role};
//! # use openai_tools::chat::{
//! #   ChatCompletion, ChatCompletionResponse, ChatCompletionResponseFormat
//! # };
//! # use serde::{Deserialize, Serialize};
//! # use serde_json;
//! # use std::env;
//! # #[tokio::main]
//! # async fn main() {
//!     #[derive(Debug, Serialize, Deserialize)]
//!     struct Weather {
//!         location: String,
//!         date: String,
//!         weather: String,
//!         error: String,
//!     }
//
//!     let mut chat = ChatCompletion::new();
//!     let messages = vec![Message::from_string(
//!         Role::User,
//!         String::from("Hi there! How's the weather tomorrow in Tokyo? If you can't answer, report error."),
//!     )];
//
//!     // build json schema
//!     let mut json_schema = Schema::chat_json_schema("weather".to_string());
//!     json_schema.add_property(
//!         String::from("location"),
//!         String::from("string"),
//!         Option::from(String::from("The location to check the weather for.")),
//!     );
//!     json_schema.add_property(
//!         String::from("date"),
//!         String::from("string"),
//!         Option::from(String::from("The date to check the weather for.")),
//!     );
//!     json_schema.add_property(
//!         String::from("weather"),
//!         String::from("string"),
//!         Option::from(String::from("The weather for the location and date.")),
//!     );
//!     json_schema.add_property(
//!         String::from("error"),
//!         String::from("string"),
//!         Option::from(String::from("Error message. If there is no error, leave this field empty.")),
//!     );
//
//!     // configure chat completion model
//!     chat
//!         .model_id(String::from("gpt-4o-mini"))
//!         .messages(messages)
//!         .temperature(1.0)
//!         .response_format(ChatCompletionResponseFormat::new(String::from("json_schema"), json_schema));
//!
//!     // execute chat
//!     let response = chat.chat().await.unwrap();
//
//!     let answer: Weather = serde_json::from_str::<Weather>(&response.choices[0].message.content).unwrap();
//!     println!("{:?}", answer)
//!     // Weather {
//!     //     location: "Tokyo",
//!     //     date: "2023-10-01",
//!     //     weather: "Temperatures around 25°C with partly cloudy skies and a slight chance of rain.",
//!     //     error: "",
//!     // }
//! # }
//! ```
//!
//! ### Details
//! - `chat` -> [`ChatCompletion`](chat::ChatCompletion)
//! - `responses` -> [`Response`](responses::Responses)
//!
pub mod chat;
pub mod common;
pub mod errors;
pub mod responses;
pub mod structured_output;
