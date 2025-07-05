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
