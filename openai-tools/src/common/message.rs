use crate::common::role::Role;
use base64::prelude::*;
use serde::{ser::SerializeStruct, Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MessageContent {
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

impl MessageContent {
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
    role: Role,
    content: Option<MessageContent>,
    contents: Option<Vec<MessageContent>>,
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Message", 3)?;
        state.serialize_field("role", &self.role)?;

        // Ensure that either content or contents is present, but not both
        if (self.content.is_none() && self.contents.is_none()) || (self.content.is_some() && self.contents.is_some()) {
            return Err(serde::ser::Error::custom("Message must have either content or contents"));
        }

        // Serialize content or contents based on which one is present
        if let Some(content) = &self.content {
            state.serialize_field("content", &content.text)?;
        }
        if let Some(contents) = &self.contents {
            state.serialize_field("content", contents)?;
        }
        state.end()
    }
}

impl Message {
    pub fn from_string<T: AsRef<str>>(role: Role, message: T) -> Self {
        Self { role, content: Some(MessageContent::from_text(message.as_ref())), contents: None }
    }

    pub fn from_message_array(role: Role, contents: Vec<MessageContent>) -> Self {
        Self { role, content: None, contents: Some(contents) }
    }

    pub fn get_input_token_count(&self) -> usize {
        let bpe = tiktoken_rs::o200k_base().unwrap();
        if let Some(content) = &self.content {
            bpe.encode_with_special_tokens(&content.clone().text.unwrap()).len()
        } else if let Some(contents) = &self.contents {
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
