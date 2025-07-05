use crate::chat::response::Response;
use crate::common::{
    errors::{OpenAIToolError, Result},
    message::Message,
    structured_output::Schema,
};
use core::str;
use dotenvy::dotenv;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Format {
    #[serde(rename = "type")]
    type_name: String,
    json_schema: Schema,
}

impl Format {
    pub fn new<T: AsRef<str>>(type_name: T, json_schema: Schema) -> Self {
        Self { type_name: type_name.as_ref().to_string(), json_schema }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
struct Body {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Format>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ChatCompletion {
    api_key: String,
    request_body: Body,
}

impl ChatCompletion {
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        Self { api_key, request_body: Body::default() }
    }

    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.request_body.model = model_id.as_ref().to_string();
        self
    }

    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages = messages;
        self
    }

    pub fn store(&mut self, store: bool) -> &mut Self {
        self.request_body.store = Option::from(store);
        self
    }

    pub fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.request_body.frequency_penalty = Option::from(frequency_penalty);
        self
    }

    pub fn logit_bias<T: AsRef<str>>(&mut self, logit_bias: HashMap<T, i32>) -> &mut Self {
        self.request_body.logit_bias = Option::from(logit_bias.into_iter().map(|(k, v)| (k.as_ref().to_string(), v)).collect::<HashMap<String, i32>>());
        self
    }

    pub fn logprobs(&mut self, logprobs: bool) -> &mut Self {
        self.request_body.logprobs = Option::from(logprobs);
        self
    }

    pub fn top_logprobs(&mut self, top_logprobs: u8) -> &mut Self {
        self.request_body.top_logprobs = Option::from(top_logprobs);
        self
    }

    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.request_body.max_completion_tokens = Option::from(max_completion_tokens);
        self
    }

    pub fn n(&mut self, n: u32) -> &mut Self {
        self.request_body.n = Option::from(n);
        self
    }

    pub fn modalities<T: AsRef<str>>(&mut self, modalities: Vec<T>) -> &mut Self {
        self.request_body.modalities = Option::from(modalities.into_iter().map(|m| m.as_ref().to_string()).collect::<Vec<String>>());
        self
    }

    pub fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        self.request_body.presence_penalty = Option::from(presence_penalty);
        self
    }

    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        self.request_body.temperature = Option::from(temperature);
        self
    }

    pub fn json_schema(&mut self, json_schema: Schema) -> &mut Self {
        self.request_body.response_format = Option::from(Format::new(String::from("json_schema"), json_schema));
        self
    }

    pub async fn chat(&mut self) -> Result<Response> {
        // Check if the API key is set & body is built.
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        if self.request_body.model.is_empty() {
            return Err(OpenAIToolError::Error("Model ID is not set.".into()));
        }
        if self.request_body.messages.is_empty() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        }

        let body = serde_json::to_string(&self.request_body)?;
        let url = "https://api.openai.com/v1/chat/completions";

        let client = request::Client::new();
        let mut header = request::header::HeaderMap::new();
        header.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        header.insert("Authorization", request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap());
        header.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust/0.1.0"));

        if cfg!(debug_assertions) {
            // Replace API key with a placeholder in debug mode
            let body_for_debug = serde_json::to_string_pretty(&self.request_body).unwrap().replace(&self.api_key, "*************");
            tracing::info!("Request body: {}", body_for_debug);
        }

        let response = client.post(url).headers(header).body(body).send().await.map_err(OpenAIToolError::RequestError)?;
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(debug_assertions) {
            tracing::info!("Response content: {}", content);
        }

        serde_json::from_str::<Response>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
