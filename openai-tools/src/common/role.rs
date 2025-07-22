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

impl TryFrom<&str> for Role {
    type Error = &'static str;

    fn try_from(role: &str) -> Result<Self, Self::Error> {
        let role = role.to_lowercase();
        match role.as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "function" => Ok(Role::Function),
            "tool" => Ok(Role::Tool),
            _ => Err("Unknown role"),
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
