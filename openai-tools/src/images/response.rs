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
