use crate::common::role::Role;
use base64::prelude::*;
use serde::{ser::SerializeStruct, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Function {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

impl Serialize for Function {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Function", 2)?;
        state.serialize_field("name", &self.name)?;
        let args_json = serde_json::to_string(&self.arguments).map_err(serde::ser::Error::custom)?;
        state.serialize_field("arguments", &args_json)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Function {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(serde::de::Error::custom("Invalid function format"));
        }
        let arguments: serde_json::Value = serde_json::from_str(parts[1]).map_err(serde::de::Error::custom)?;
        let mut args_map = HashMap::new();
        if let serde_json::Value::Object(obj) = arguments {
            for (key, value) in obj {
                args_map.insert(key, value);
            }
        } else {
            return Err(serde::de::Error::custom("Function arguments must be a JSON object"));
        }
        Ok(Function { name: parts[0].to_string(), arguments: args_map })
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_name: String,
    pub function: Function,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Content {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl Content {
    pub fn from_text<T: AsRef<str>>(text: T) -> Self {
        Self { type_name: "input_text".to_string(), text: Some(text.as_ref().to_string()), image_url: None }
    }

    pub fn from_image_url<T: AsRef<str>>(image_url: T) -> Self {
        Self { type_name: "input_image".to_string(), text: None, image_url: Some(image_url.as_ref().to_string()) }
    }

    pub fn from_image_file<T: AsRef<str>>(file_path: T) -> Self {
        let ext = file_path.as_ref();
        let ext = std::path::Path::new(&ext).extension().and_then(|s| s.to_str()).unwrap();
        let img = image::ImageReader::open(file_path.as_ref()).expect("Failed to open image file").decode().expect("Failed to decode image");
        let img_fmt = match ext {
            "png" => image::ImageFormat::Png,
            "jpg" | "jpeg" => image::ImageFormat::Jpeg,
            "gif" => image::ImageFormat::Gif,
            _ => panic!("Unsupported image format"),
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, img_fmt).expect("Failed to write image to buffer");
        let base64_string = BASE64_STANDARD.encode(buf.into_inner());
        let image_url = format!("data:image/{ext};base64,{base64_string}");
        Self { type_name: "input_image".to_string(), text: None, image_url: Some(image_url) }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Option<Content>,
    pub content_list: Option<Vec<Content>>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
    pub annotations: Option<Vec<String>>,
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Message", 3)?;
        state.serialize_field("role", &self.role)?;

        // Ensure that either content or contents is present, but not both
        if (self.content.is_none() && self.content_list.is_none()) || (self.content.is_some() && self.content_list.is_some()) {
            return Err(serde::ser::Error::custom("Message must have either content or contents"));
        }

        // Serialize content or contents based on which one is present
        if let Some(content) = &self.content {
            state.serialize_field("content", &content.text)?;
        }
        if let Some(contents) = &self.content_list {
            state.serialize_field("content", contents)?;
        }
        state.end()
    }
}

impl Message {
    pub fn from_string<T: AsRef<str>>(role: Role, message: T) -> Self {
        Self { role, content: Some(Content::from_text(message.as_ref())), content_list: None, tool_calls: None, refusal: None, annotations: None }
    }

    pub fn from_message_array(role: Role, contents: Vec<Content>) -> Self {
        Self { role, content: None, content_list: Some(contents), tool_calls: None, refusal: None, annotations: None }
    }

    pub fn get_input_token_count(&self) -> usize {
        let bpe = tiktoken_rs::o200k_base().unwrap();
        if let Some(content) = &self.content {
            bpe.encode_with_special_tokens(&content.clone().text.unwrap()).len()
        } else if let Some(contents) = &self.content_list {
            let mut total_tokens = 0;
            for content in contents {
                if let Some(text) = &content.text {
                    total_tokens += bpe.encode_with_special_tokens(text).len();
                }
            }
            total_tokens
        } else {
            0 // No content to count tokens for
        }
    }
}
