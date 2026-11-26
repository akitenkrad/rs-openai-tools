//! OpenAI Images API Response Types
//!
//! This module defines the response structures for the OpenAI Images API.

use serde::{Deserialize, Serialize};

/// Response structure from image generation/edit/variation endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageResponse {
    /// Unix timestamp when the image was created
    pub created: i64,
    /// Array of generated images
    pub data: Vec<ImageData>,
    /// Output format used for the image data.
    #[serde(default)]
    pub output_format: Option<String>,
    /// Quality used for the image data.
    #[serde(default)]
    pub quality: Option<String>,
    /// Size used for the image data.
    #[serde(default)]
    pub size: Option<String>,
    /// Background used for the image data.
    #[serde(default)]
    pub background: Option<String>,
    /// Usage details for SVG output.
    #[serde(default)]
    pub svg_usage: Option<SvgUsage>,
}

/// Usage details returned for SVG output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SvgUsage {
    /// Input token count.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Output token count.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Reasoning token count.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    /// Total token count.
    #[serde(default)]
    pub total_tokens: Option<u64>,
    /// Model used for the SVG output.
    #[serde(default)]
    pub model: Option<String>,
}

/// Individual image data from generation response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// URL of the generated image (if response_format = url)
    /// Note: URLs are only valid for 60 minutes after generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// Base64-encoded image data (if response_format = b64_json)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,

    /// The revised prompt used by the model (DALL-E 3 only)
    /// DALL-E 3 may revise the original prompt for safety or quality
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

impl ImageData {
    /// Decodes base64 image data to bytes.
    ///
    /// Returns `None` if no b64_json data is present.
    /// Returns `Some(Err(...))` if base64 decoding fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::images::response::ImageData;
    /// use std::fs;
    ///
    /// # fn example(image_data: &ImageData) -> Result<(), Box<dyn std::error::Error>> {
    /// if let Some(bytes_result) = image_data.as_bytes() {
    ///     let bytes = bytes_result?;
    ///     fs::write("output.png", bytes)?;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn as_bytes(&self) -> Option<std::result::Result<Vec<u8>, base64::DecodeError>> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        self.b64_json.as_ref().map(|b64| STANDARD.decode(b64))
    }

    /// Returns true if this image data contains a URL.
    pub fn has_url(&self) -> bool {
        self.url.is_some()
    }

    /// Returns true if this image data contains base64 data.
    pub fn has_b64(&self) -> bool {
        self.b64_json.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::ImageResponse;

    #[test]
    fn deserializes_new_image_response_metadata() {
        let json = r#"{
            "created": 123,
            "data": [{"b64_json": "PHN2Zz4="}],
            "output_format": "svg",
            "quality": "high",
            "size": "1024x1024",
            "background": "transparent",
            "svg_usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "reasoning_tokens": 30,
                "total_tokens": 60,
                "model": "gpt-image-2.6-sunburst"
            }
        }"#;

        let response: ImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.output_format.as_deref(), Some("svg"));
        let usage = response.svg_usage.unwrap();
        assert_eq!(usage.input_tokens, Some(10));
        assert_eq!(usage.output_tokens, Some(20));
        assert_eq!(usage.reasoning_tokens, Some(30));
        assert_eq!(usage.total_tokens, Some(60));
        assert_eq!(usage.model.as_deref(), Some("gpt-image-2.6-sunburst"));
    }

    #[test]
    fn deserializes_legacy_image_response_without_metadata() {
        let json = r#"{"created":123,"data":[{"url":"https://example.com/image.png"}]}"#;

        let response: ImageResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.created, 123);
        assert_eq!(response.data.len(), 1);
        assert!(response.output_format.is_none());
        assert!(response.quality.is_none());
        assert!(response.size.is_none());
        assert!(response.background.is_none());
        assert!(response.svg_usage.is_none());
    }
}
