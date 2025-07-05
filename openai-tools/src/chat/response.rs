use crate::common::usage::Usage;
use core::str;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub content: String,
    pub role: String,
    pub refusal: Option<String>,
    pub annotations: Option<Vec<String>>,
}

pub struct TopLogProbItem {
    pub token: String,
    pub logprob: f32,
}

pub struct LogProbItem {
    pub token: String,
    pub logprob: f32,
    pub top_logprobs: Option<Vec<TopLogProbItem>>,
}

pub struct LogProbs {
    pub content: Vec<LogProbItem>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub logprobs: Option<Vec<String>>,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}
