use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[derive(Display, Debug, Clone, Deserialize, Serialize, PartialEq)]
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

impl TryFrom<String> for Role {
    type Error = &'static str;

    fn try_from(role: String) -> Result<Self, Self::Error> {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_role_conversion() {
        assert_eq!(Role::try_from("system".to_string()).unwrap(), Role::System);
        assert_eq!(Role::try_from("user".to_string()).unwrap(), Role::User);
        assert_eq!(Role::try_from("assistant".to_string()).unwrap(), Role::Assistant);
        assert_eq!(Role::try_from("function".to_string()).unwrap(), Role::Function);
        assert_eq!(Role::try_from("tool".to_string()).unwrap(), Role::Tool);
        assert!(Role::try_from("unknown".to_string()).is_err());
    }

    #[test]
    fn test_role_as_str() {
        assert_eq!(Role::System.as_str(), "system");
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
        assert_eq!(Role::Function.as_str(), "function");
        assert_eq!(Role::Tool.as_str(), "tool");
    }
}
