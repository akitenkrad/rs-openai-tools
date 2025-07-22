use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[derive(Display, Debug, Clone, Deserialize, Serialize)]
pub enum Role {
    #[serde(rename = "system")]
    #[strum(to_string = "system")]
    System,
    #[serde(rename = "user")]
    #[strum(to_string = "user")]
    User,
    #[serde(rename = "assistant")]
    #[strum(to_string = "assistant")]
    Assistant,
    #[serde(rename = "function")]
    #[strum(to_string = "function")]
    Function,
    #[serde(rename = "tool")]
    #[strum(to_string = "tool")]
    Tool,
}

impl From<&str> for Role {
    fn from(role: &str) -> Self {
        let role = role.to_lowercase();
        match role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "function" => Role::Function,
            "tool" => Role::Tool,
            _ => panic!("Unknown role: {}", role),
        }
    }
}

impl Role {
    pub fn as_str(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Function => "function",
            Role::Tool => "tool",
        }
    }
}
