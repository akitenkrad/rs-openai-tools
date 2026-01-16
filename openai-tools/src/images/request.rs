//! OpenAI Images API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Images API.
//! It allows you to generate, edit, and create variations of images using DALL-E models.
//!
//! # Key Features
//!
//! - **Generate**: Create images from text prompts
//! - **Edit**: Modify existing images with new prompts and masks
//! - **Variations**: Create variations of existing images (DALL-E 2 only)
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::images::request::{Images, GenerateOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let images = Images::new()?;
//!
//!     // Generate an image
//!     let response = images.generate("A white cat", GenerateOptions::default()).await?;
//!     println!("Image URL: {:?}", response.data[0].url);
//!
//!     Ok(())
//! }
//! ```

use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use crate::images::response::ImageResponse;
use request::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Default API path for Images
const IMAGES_PATH: &str = "images";

/// Image generation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageModel {
    /// DALL-E 2 model - supports variations, smaller sizes
    #[serde(rename = "dall-e-2")]
    DallE2,
    /// DALL-E 3 model - higher quality, HD support, style options
    #[serde(rename = "dall-e-3")]
    DallE3,
    /// GPT Image model - latest generation
    #[serde(rename = "gpt-image-1")]
    GptImage1,
}

impl Default for ImageModel {
    fn default() -> Self {
        Self::DallE3
    }
}

impl ImageModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DallE2 => "dall-e-2",
            Self::DallE3 => "dall-e-3",
            Self::GptImage1 => "gpt-image-1",
        }
    }
}

impl std::fmt::Display for ImageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Image sizes for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageSize {
    /// 256x256 pixels (DALL-E 2 only)
    #[serde(rename = "256x256")]
    Size256x256,
    /// 512x512 pixels (DALL-E 2 only)
    #[serde(rename = "512x512")]
    Size512x512,
    /// 1024x1024 pixels (all models)
    #[serde(rename = "1024x1024")]
    Size1024x1024,
    /// 1792x1024 pixels - landscape (DALL-E 3 only)
    #[serde(rename = "1792x1024")]
    Size1792x1024,
    /// 1024x1792 pixels - portrait (DALL-E 3 only)
    #[serde(rename = "1024x1792")]
    Size1024x1792,
}

impl Default for ImageSize {
    fn default() -> Self {
        Self::Size1024x1024
    }
}

impl ImageSize {
    /// Returns the size string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Size256x256 => "256x256",
            Self::Size512x512 => "512x512",
            Self::Size1024x1024 => "1024x1024",
            Self::Size1792x1024 => "1792x1024",
            Self::Size1024x1792 => "1024x1792",
        }
    }
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Image quality options (DALL-E 3 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageQuality {
    /// Standard quality
    Standard,
    /// High definition quality
    Hd,
}

impl Default for ImageQuality {
    fn default() -> Self {
        Self::Standard
    }
}

impl ImageQuality {
    /// Returns the quality string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Hd => "hd",
        }
    }
}

/// Image style options (DALL-E 3 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageStyle {
    /// Vivid - hyper-real and dramatic
    Vivid,
    /// Natural - more natural, less hyper-real
    Natural,
}

impl Default for ImageStyle {
    fn default() -> Self {
        Self::Vivid
    }
}

impl ImageStyle {
    /// Returns the style string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vivid => "vivid",
            Self::Natural => "natural",
        }
    }
}

/// Response format for images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Return URLs to the generated images (valid for 60 minutes)
    Url,
    /// Return base64-encoded image data
    B64Json,
}

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Url
    }
}

impl ResponseFormat {
    /// Returns the format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Url => "url",
            Self::B64Json => "b64_json",
        }
    }
}

/// Options for image generation.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// The model to use (defaults to DALL-E 3)
    pub model: Option<ImageModel>,
    /// Number of images to generate (1-10, DALL-E 3 only supports 1)
    pub n: Option<u32>,
    /// Image quality (DALL-E 3 only)
    pub quality: Option<ImageQuality>,
    /// Response format (URL or base64)
    pub response_format: Option<ResponseFormat>,
    /// Image size
    pub size: Option<ImageSize>,
    /// Image style (DALL-E 3 only)
    pub style: Option<ImageStyle>,
    /// User identifier for abuse monitoring
    pub user: Option<String>,
}

/// Options for image editing.
#[derive(Debug, Clone, Default)]
pub struct EditOptions {
    /// Path to the mask image (transparent areas will be edited)
    pub mask: Option<String>,
    /// The model to use (only DALL-E 2 supports editing)
    pub model: Option<ImageModel>,
    /// Number of images to generate (1-10)
    pub n: Option<u32>,
    /// Image size
    pub size: Option<ImageSize>,
    /// Response format
    pub response_format: Option<ResponseFormat>,
    /// User identifier for abuse monitoring
    pub user: Option<String>,
}

/// Options for image variations.
#[derive(Debug, Clone, Default)]
pub struct VariationOptions {
    /// The model to use (only DALL-E 2 supports variations)
    pub model: Option<ImageModel>,
    /// Number of variations to generate (1-10)
    pub n: Option<u32>,
    /// Response format
    pub response_format: Option<ResponseFormat>,
    /// Image size
    pub size: Option<ImageSize>,
    /// User identifier for abuse monitoring
    pub user: Option<String>,
}

/// Request payload for image generation.
#[derive(Debug, Clone, Serialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

/// Client for interacting with the OpenAI Images API.
///
/// This struct provides methods to generate, edit, and create variations of images.
/// Use [`Images::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::images::request::{Images, GenerateOptions, ImageModel, ImageSize};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let images = Images::new()?;
///
///     let options = GenerateOptions {
///         model: Some(ImageModel::DallE3),
///         size: Some(ImageSize::Size1024x1024),
///         ..Default::default()
///     };
///
///     let response = images.generate("A sunset over mountains", options).await?;
///     println!("Generated image: {:?}", response.data[0].url);
///
///     Ok(())
/// }
/// ```
pub struct Images {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Images {
    /// Creates a new Images client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Images)` - A new Images client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::images::request::Images;
    ///
    /// let images = Images::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Images client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Images client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Images client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Images client with URL-based provider detection
    pub fn with_url<S: Into<String>>(base_url: S, api_key: S) -> Self {
        let auth = AuthProvider::from_url_with_key(base_url, api_key);
        Self { auth, timeout: None }
    }

    /// Creates a new Images client from URL using environment variables
    pub fn from_url<S: Into<String>>(url: S) -> Result<Self> {
        let auth = AuthProvider::from_url(url)?;
        Ok(Self { auth, timeout: None })
    }

    /// Returns the authentication provider
    pub fn auth(&self) -> &AuthProvider {
        &self.auth
    }

    /// Sets the request timeout duration.
    ///
    /// # Arguments
    ///
    /// * `timeout` - The maximum time to wait for a response
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn timeout(&mut self, timeout: Duration) -> &mut Self {
        self.timeout = Some(timeout);
        self
    }

    /// Creates the HTTP client with default headers.
    fn create_client(&self) -> Result<(request::Client, request::header::HeaderMap)> {
        let client = create_http_client(self.timeout)?;
        let mut headers = request::header::HeaderMap::new();
        self.auth.apply_headers(&mut headers)?;
        headers.insert(
            "User-Agent",
            request::header::HeaderValue::from_static("openai-tools-rust"),
        );
        Ok((client, headers))
    }

    /// Generates images from a text prompt.
    ///
    /// Creates one or more images based on the provided text description.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Text description of the desired image(s)
    /// * `options` - Generation options (model, size, quality, etc.)
    ///
    /// # Returns
    ///
    /// * `Ok(ImageResponse)` - The generated image(s)
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::images::request::{Images, GenerateOptions, ImageQuality, ImageStyle};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let images = Images::new()?;
    ///
    ///     let options = GenerateOptions {
    ///         quality: Some(ImageQuality::Hd),
    ///         style: Some(ImageStyle::Natural),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = images.generate("A serene lake at dawn", options).await?;
    ///
    ///     if let Some(url) = &response.data[0].url {
    ///         println!("Image URL: {}", url);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn generate(&self, prompt: &str, options: GenerateOptions) -> Result<ImageResponse> {
        let (client, mut headers) = self.create_client()?;
        headers.insert(
            "Content-Type",
            request::header::HeaderValue::from_static("application/json"),
        );

        let request_body = GenerateRequest {
            prompt: prompt.to_string(),
            model: options.model.map(|m| m.as_str().to_string()),
            n: options.n,
            quality: options.quality.map(|q| q.as_str().to_string()),
            response_format: options.response_format.map(|f| f.as_str().to_string()),
            size: options.size.map(|s| s.as_str().to_string()),
            style: options.style.map(|s| s.as_str().to_string()),
            user: options.user,
        };

        let body =
            serde_json::to_string(&request_body).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = format!("{}/generations", self.auth.endpoint(IMAGES_PATH));

        let response = client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<ImageResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Edits an existing image based on a prompt.
    ///
    /// Creates edited versions of an image by replacing areas indicated by
    /// a transparent mask. Only available with DALL-E 2.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image to edit (PNG, max 4MB, square)
    /// * `prompt` - Text description of the desired edit
    /// * `options` - Edit options (mask, size, etc.)
    ///
    /// # Returns
    ///
    /// * `Ok(ImageResponse)` - The edited image(s)
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::images::request::{Images, EditOptions};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let images = Images::new()?;
    ///
    ///     let options = EditOptions {
    ///         mask: Some("mask.png".to_string()),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = images.edit("original.png", "Add a red hat", options).await?;
    ///     println!("Edited image: {:?}", response.data[0].url);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn edit(
        &self,
        image_path: &str,
        prompt: &str,
        options: EditOptions,
    ) -> Result<ImageResponse> {
        let (client, headers) = self.create_client()?;

        // Read the image file
        let image_content = tokio::fs::read(image_path)
            .await
            .map_err(|e| OpenAIToolError::Error(format!("Failed to read image: {}", e)))?;

        let image_filename = Path::new(image_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image.png")
            .to_string();

        let image_part = Part::bytes(image_content)
            .file_name(image_filename)
            .mime_str("image/png")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new()
            .part("image", image_part)
            .text("prompt", prompt.to_string());

        // Add mask if provided
        if let Some(mask_path) = options.mask {
            let mask_content = tokio::fs::read(&mask_path)
                .await
                .map_err(|e| OpenAIToolError::Error(format!("Failed to read mask: {}", e)))?;

            let mask_filename = Path::new(&mask_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("mask.png")
                .to_string();

            let mask_part = Part::bytes(mask_content)
                .file_name(mask_filename)
                .mime_str("image/png")
                .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

            form = form.part("mask", mask_part);
        }

        // Add optional parameters
        if let Some(model) = options.model {
            form = form.text("model", model.as_str().to_string());
        }
        if let Some(n) = options.n {
            form = form.text("n", n.to_string());
        }
        if let Some(size) = options.size {
            form = form.text("size", size.as_str().to_string());
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(user) = options.user {
            form = form.text("user", user);
        }

        let url = format!("{}/edits", self.auth.endpoint(IMAGES_PATH));

        let response = client
            .post(&url)
            .headers(headers)
            .multipart(form)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<ImageResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Creates variations of an existing image.
    ///
    /// Only available with DALL-E 2.
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to the image to create variations of (PNG, max 4MB, square)
    /// * `options` - Variation options (n, size, etc.)
    ///
    /// # Returns
    ///
    /// * `Ok(ImageResponse)` - The image variation(s)
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::images::request::{Images, VariationOptions, ImageModel};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let images = Images::new()?;
    ///
    ///     let options = VariationOptions {
    ///         model: Some(ImageModel::DallE2),
    ///         n: Some(3),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = images.variation("original.png", options).await?;
    ///
    ///     for (i, image) in response.data.iter().enumerate() {
    ///         println!("Variation {}: {:?}", i + 1, image.url);
    ///     }
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn variation(
        &self,
        image_path: &str,
        options: VariationOptions,
    ) -> Result<ImageResponse> {
        let (client, headers) = self.create_client()?;

        // Read the image file
        let image_content = tokio::fs::read(image_path)
            .await
            .map_err(|e| OpenAIToolError::Error(format!("Failed to read image: {}", e)))?;

        let image_filename = Path::new(image_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("image.png")
            .to_string();

        let image_part = Part::bytes(image_content)
            .file_name(image_filename)
            .mime_str("image/png")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("image", image_part);

        // Add optional parameters
        if let Some(model) = options.model {
            form = form.text("model", model.as_str().to_string());
        }
        if let Some(n) = options.n {
            form = form.text("n", n.to_string());
        }
        if let Some(size) = options.size {
            form = form.text("size", size.as_str().to_string());
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(user) = options.user {
            form = form.text("user", user);
        }

        let url = format!("{}/variations", self.auth.endpoint(IMAGES_PATH));

        let response = client
            .post(&url)
            .headers(headers)
            .multipart(form)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let status = response.status();
        let content = response.text().await.map_err(OpenAIToolError::RequestError)?;

        if cfg!(test) {
            tracing::info!("Response content: {}", content);
        }

        if !status.is_success() {
            if let Ok(error_resp) = serde_json::from_str::<ErrorResponse>(&content) {
                return Err(OpenAIToolError::Error(error_resp.error.message.unwrap_or_default()));
            }
            return Err(OpenAIToolError::Error(format!("API error ({}): {}", status, content)));
        }

        serde_json::from_str::<ImageResponse>(&content).map_err(OpenAIToolError::SerdeJsonError)
    }
}
