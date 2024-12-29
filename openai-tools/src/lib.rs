pub mod json_schema;

use anyhow::Result;
use dotenvy::dotenv;
use fxhash::FxHashMap;
use json_schema::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;
use std::process::Command;

#[derive(Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

impl Message {
    pub fn new(role: &str, message: &str) -> Self {
        Self {
            role: String::from(role),
            content: String::from(message),
            refusal: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub type_name: String,
    pub json_schema: JsonSchema,
}

impl ResponseFormat {
    pub fn new(type_name: &str, json_schema: JsonSchema) -> Self {
        Self {
            type_name: String::from(type_name),
            json_schema,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestBody {
    // ID of the model to use. (https://platform.openai.com/docs/models#model-endpoint-compatibility)
    pub model: String,
    // A list of messages comprising teh conversation so far.
    pub messages: Vec<Message>,
    // Whether or not to store the output of this chat completion request for user. false by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    // -2.0 ~ 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    // Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens to an associated bias value from 100 to 100.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<FxHashMap<String, i32>>,
    // Whether to return log probabilities of the output tokens or not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    // 0 ~ 20. Specify the number of most likely tokens to return at each token position, each with an associated log probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    // An upper bound for the number of tokens that can be generated for a completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u64>,
    // How many chat completion choices to generate for each input message. 1 by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    // Output types that you would like the model to generate for this request. ["text"] for most models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    // -2.0 ~ 2.0. Positive values penalize new tokens based on whether they apper in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    // 0 ~ 2. What sampling temperature to use. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    // An object specifying the format that the model must output. (https://platform.openai.com/docs/guides/structured-outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

impl ChatCompletionRequestBody {
    pub fn new(
        model_id: String,
        messages: Vec<Message>,
        store: Option<bool>,
        frequency_penalty: Option<f32>,
        logit_bias: Option<FxHashMap<String, i32>>,
        logprobs: Option<bool>,
        top_logprobs: Option<u8>,
        max_completion_tokens: Option<u64>,
        n: Option<u32>,
        modalities: Option<Vec<String>>,
        presence_penalty: Option<f32>,
        temperature: Option<f32>,
        response_format: Option<ResponseFormat>,
    ) -> Self {
        Self {
            model: model_id,
            messages,
            store: if let Some(value) = store {
                Option::from(value)
            } else {
                None
            },
            frequency_penalty: if let Some(value) = frequency_penalty {
                Option::from(value)
            } else {
                None
            },
            logit_bias: if let Some(value) = logit_bias {
                Option::from(value)
            } else {
                None
            },
            logprobs: if let Some(value) = logprobs {
                Option::from(value)
            } else {
                None
            },
            top_logprobs: if let Some(value) = top_logprobs {
                Option::from(value)
            } else {
                None
            },
            max_completion_tokens: if let Some(value) = max_completion_tokens {
                Option::from(value)
            } else {
                None
            },
            n: if let Some(value) = n {
                Option::from(value)
            } else {
                None
            },
            modalities: if let Some(value) = modalities {
                Option::from(value)
            } else {
                None
            },
            presence_penalty: if let Some(value) = presence_penalty {
                Option::from(value)
            } else {
                None
            },
            temperature: if let Some(value) = temperature {
                Option::from(value)
            } else {
                None
            },
            response_format: if let Some(value) = response_format {
                Option::from(value)
            } else {
                None
            },
        }
    }

    pub fn default() -> Self {
        Self {
            model: String::default(),
            messages: Vec::new(),
            store: None,
            frequency_penalty: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            max_completion_tokens: None,
            n: None,
            modalities: None,
            presence_penalty: None,
            temperature: None,
            response_format: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub completion_tokens_details: FxHashMap<String, u64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Response {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

pub struct OpenAI {
    api_key: String,
    pub completion_body: ChatCompletionRequestBody,
}

impl OpenAI {
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is not set.");
        return Self {
            api_key,
            completion_body: ChatCompletionRequestBody::default(),
        };
    }

    pub fn model_id(&mut self, model_id: &str) -> &mut Self {
        self.completion_body.model = String::from(model_id);
        return self;
    }

    pub fn messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.completion_body.messages = messages;
        return self;
    }

    pub fn store(&mut self, store: bool) -> &mut Self {
        self.completion_body.store = Option::from(store);
        return self;
    }

    pub fn frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.completion_body.frequency_penalty = Option::from(frequency_penalty);
        return self;
    }

    pub fn logit_bias(&mut self, logit_bias: FxHashMap<String, i32>) -> &mut Self {
        self.completion_body.logit_bias = Option::from(logit_bias);
        return self;
    }

    pub fn logprobs(&mut self, logprobs: bool) -> &mut Self {
        self.completion_body.logprobs = Option::from(logprobs);
        return self;
    }

    pub fn top_logprobs(&mut self, top_logprobs: u8) -> &mut Self {
        self.completion_body.top_logprobs = Option::from(top_logprobs);
        return self;
    }

    pub fn max_completion_tokens(&mut self, max_completion_tokens: u64) -> &mut Self {
        self.completion_body.max_completion_tokens = Option::from(max_completion_tokens);
        return self;
    }

    pub fn n(&mut self, n: u32) -> &mut Self {
        self.completion_body.n = Option::from(n);
        return self;
    }

    pub fn modalities(&mut self, modalities: Vec<String>) -> &mut Self {
        self.completion_body.modalities = Option::from(modalities);
        return self;
    }

    pub fn presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        self.completion_body.presence_penalty = Option::from(presence_penalty);
        return self;
    }

    pub fn temperature(&mut self, temperature: f32) -> &mut Self {
        self.completion_body.temperature = Option::from(temperature);
        return self;
    }

    pub fn response_format(&mut self, response_format: ResponseFormat) -> &mut Self {
        self.completion_body.response_format = Option::from(response_format);
        return self;
    }

    pub fn chat(&mut self) -> Result<Response> {
        // Check if the API key is set & body is built.
        if self.api_key.is_empty() {
            return Err(anyhow::Error::msg("API key is not set."));
        }
        if self.completion_body.model.is_empty() {
            return Err(anyhow::Error::msg("Model ID is not set."));
        }
        if self.completion_body.messages.is_empty() {
            return Err(anyhow::Error::msg("Messages are not set."));
        }

        let body = serde_json::to_string(&self.completion_body)?;
        let url = "https://api.openai.com/v1/chat/completions";
        let cmd = Command::new("curl")
            .arg(url)
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-H")
            .arg(format!("Authorization: Bearer {}", self.api_key))
            .arg("-d")
            .arg(body)
            .output()
            .expect("Failed to execute command");

        let content = String::from_utf8_lossy(&cmd.stdout).to_string();

        match serde_json::from_str::<Response>(&content) {
            Ok(response) => return Ok(response),
            Err(e) => {
                let e_msg = format!("Failed to parse JSON: {} CONTENT: {}", e, content);
                return Err(anyhow::Error::msg(e_msg));
            }
        }
    }
}

#[cfg(test)]
mod tests;
