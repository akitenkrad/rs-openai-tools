use crate::{
    common::{
        errors::{OpenAIToolError, Result},
        message::Message,
        structured_output::Schema,
        tool::Tool,
    },
    responses::response::Response,
};
use derive_new::new;
use dotenvy::dotenv;
use request;
use serde::{ser::SerializeStruct, Serialize};
use std::env;

#[derive(Debug, Clone, Default, Serialize, new)]
pub struct Format {
    pub format: Schema,
}

#[derive(Debug, Clone, Default, new)]
pub struct Body {
    pub model: String,
    pub instructions: Option<String>,
    pub plain_text_input: Option<String>,
    pub messages_input: Option<Vec<Message>>,
    pub tools: Option<Vec<Tool>>,
    pub text: Option<Format>,
}

impl Serialize for Body {
    fn serialize<S>(&self, serializer: S) -> anyhow::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let input = if self.plain_text_input.is_some() {
            self.plain_text_input.clone().unwrap()
        } else if self.messages_input.is_some() {
            serde_json::to_string(&self.messages_input).unwrap()
        } else {
            return Err(serde::ser::Error::custom("Either plain_text_input or messages_input must be set."));
        };
        let mut state = serializer.serialize_struct("ResponsesBody", 4)?;
        state.serialize_field("model", &self.model)?;
        state.serialize_field("instructions", &self.instructions)?;
        state.serialize_field("input", &input)?;
        if self.tools.is_some() {
            state.serialize_field("tools", &self.tools)?;
        }
        if self.text.is_some() {
            state.serialize_field("text", &self.text)?;
        }
        state.end()
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Responses {
    api_key: String,
    pub request_body: Body,
}

impl Responses {
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        Self { api_key, request_body: Body::default() }
    }

    pub fn model_id<T: AsRef<str>>(&mut self, model_id: T) -> &mut Self {
        self.request_body.model = model_id.as_ref().to_string();
        self
    }

    pub fn instructions<T: AsRef<str>>(&mut self, instructions: T) -> &mut Self {
        self.request_body.instructions = Some(instructions.as_ref().to_string());
        self
    }

    pub fn plain_text_input<T: AsRef<str>>(&mut self, input: T) -> &mut Self {
        self.request_body.plain_text_input = Some(input.as_ref().to_string());
        self
    }

    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.request_body.messages_input = Some(messages);
        self
    }

    pub fn tools(&mut self, tools: Vec<Tool>) -> &mut Self {
        self.request_body.tools = Some(tools);
        self
    }

    pub fn text(&mut self, text_format: Schema) -> &mut Self {
        self.request_body.text = Option::from(Format::new(text_format));
        self
    }

    pub async fn complete(&self) -> Result<Response> {
        if self.api_key.is_empty() {
            return Err(OpenAIToolError::Error("API key is not set.".into()));
        }
        if self.request_body.model.is_empty() {
            return Err(OpenAIToolError::Error("Model ID is not set.".into()));
        }
        if self.request_body.messages_input.is_none() && self.request_body.plain_text_input.is_none() {
            return Err(OpenAIToolError::Error("Messages are not set.".into()));
        } else if self.request_body.plain_text_input.is_none() && self.request_body.messages_input.is_none() {
            return Err(OpenAIToolError::Error("Both plain text input and messages are set. Please use one of them.".into()));
        }

        let body = serde_json::to_string(&self.request_body)?;
        let url = "https://api.openai.com/v1/responses".to_string();

        let client = request::Client::new();
        let mut header = request::header::HeaderMap::new();
        header.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));
        header.insert("Authorization", request::header::HeaderValue::from_str(&format!("Bearer {}", self.api_key)).unwrap());
        header.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust/0.1.0"));

        if cfg!(debug_assertions) {
            // Replace API key with a placeholder for security
            let body_for_debug = serde_json::to_string_pretty(&self.request_body).unwrap().replace(&self.api_key, "*************");
            // Log the request body for debugging purposes
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
