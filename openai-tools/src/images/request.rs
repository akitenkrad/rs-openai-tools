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
///
/// # DALL-E retirement
///
/// OpenAI retired `dall-e-2` and `dall-e-3` on `api.openai.com` - requests
/// naming them fail with "The model 'dall-e-3' does not exist." The variants
/// are kept because Azure OpenAI deployments can still serve DALL-E, but new
/// OpenAI code should use one of the GPT Image models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ImageModel {
    /// DALL-E 2 model - supports variations, smaller sizes
    ///
    /// Retired on OpenAI; may still exist as an Azure deployment.
    #[serde(rename = "dall-e-2")]
    DallE2,
    /// DALL-E 3 model - higher quality, HD support, style options
    ///
    /// Retired on OpenAI; may still exist as an Azure deployment.
    #[serde(rename = "dall-e-3")]
    DallE3,
    /// GPT Image model - latest generation (default)
    #[serde(rename = "gpt-image-1")]
    #[default]
    GptImage1,
    /// GPT Image 1 Mini - cheaper gpt-image-1 for high-volume generation
    #[serde(rename = "gpt-image-1-mini")]
    GptImage1Mini,
    /// GPT Image 1.5 - refreshed gpt-image-1
    #[serde(rename = "gpt-image-1.5")]
    GptImage1_5,
    /// GPT Image 2 - state-of-the-art generation and editing
    #[serde(rename = "gpt-image-2")]
    GptImage2,
    /// GPT Image 2.6 Flare
    #[serde(rename = "gpt-image-2.6-flare")]
    GptImage2_6Flare,
    /// GPT Image 2.6 Sunburst
    #[serde(rename = "gpt-image-2.6-sunburst")]
    GptImage2_6Sunburst,
    /// GPT Image 2.5 Flare - fast, everyday image generation and editing
    ///
    /// Has approximately 50% lower latency than GPT Image 2. It has the same
    /// pricing as GPT Image 2.5 Sunburst.
    ///
    /// Accepts `low`, `medium`, `high`, `auto`, `xhigh`, and `max` quality, but
    /// rejects DALL-E's `standard` and `hd` values. Although `max` is accepted,
    /// API validation errors only list `low`, `medium`, `high`, and `auto`.
    /// `response_format` and `style` are unknown parameters and responses are
    /// always base64. The API accepts `output_format: webp` and
    /// `background: transparent`.
    #[serde(rename = "gpt-image-2.5-flare")]
    GptImage2_5Flare,
    /// GPT Image 2.5 Sunburst - generation and finely controlled editing
    ///
    /// Has the same pricing as GPT Image 2.5 Flare. At `max` quality, one
    /// generation can take more than 60 seconds; configure a longer timeout
    /// when using request timeouts.
    ///
    /// Accepts `low`, `medium`, `high`, `auto`, `xhigh`, and `max` quality, but
    /// rejects DALL-E's `standard` and `hd` values. Although `max` is accepted,
    /// API validation errors only list `low`, `medium`, `high`, and `auto`.
    /// `response_format` and `style` are unknown parameters and responses are
    /// always base64. The API accepts `output_format: webp` and
    /// `background: transparent`.
    #[serde(rename = "gpt-image-2.5-sunburst")]
    GptImage2_5Sunburst,
    /// ChatGPT Image Latest - image model currently used in ChatGPT
    ///
    /// Requires a verified organization.
    #[serde(rename = "chatgpt-image-latest")]
    ChatGptImageLatest,
}

impl ImageModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DallE2 => "dall-e-2",
            Self::DallE3 => "dall-e-3",
            Self::GptImage1 => "gpt-image-1",
            Self::GptImage1Mini => "gpt-image-1-mini",
            Self::GptImage1_5 => "gpt-image-1.5",
            Self::GptImage2 => "gpt-image-2",
            Self::GptImage2_6Flare => "gpt-image-2.6-flare",
            Self::GptImage2_6Sunburst => "gpt-image-2.6-sunburst",
            Self::GptImage2_5Flare => "gpt-image-2.5-flare",
            Self::GptImage2_5Sunburst => "gpt-image-2.5-sunburst",
            Self::ChatGptImageLatest => "chatgpt-image-latest",
        }
    }
}

impl std::fmt::Display for ImageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Image sizes for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum ImageSize {
    /// 256x256 pixels (DALL-E 2 only)
    #[serde(rename = "256x256")]
    Size256x256,
    /// 512x512 pixels (DALL-E 2 only)
    #[serde(rename = "512x512")]
    Size512x512,
    /// 1024x1024 pixels (all models)
    #[serde(rename = "1024x1024")]
    #[default]
    Size1024x1024,
    /// 1792x1024 pixels - landscape (DALL-E 3 only)
    #[serde(rename = "1792x1024")]
    Size1792x1024,
    /// 1024x1792 pixels - portrait (DALL-E 3 only)
    #[serde(rename = "1024x1792")]
    Size1024x1792,
    /// 1024x1536 pixels - portrait (GPT Image models)
    #[serde(rename = "1024x1536")]
    Size1024x1536,
    /// 1536x1024 pixels - landscape (GPT Image models)
    #[serde(rename = "1536x1024")]
    Size1536x1024,
    /// Let the model choose the size (GPT Image models)
    #[serde(rename = "auto")]
    Auto,
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
            Self::Size1024x1536 => "1024x1536",
            Self::Size1536x1024 => "1536x1024",
            Self::Auto => "auto",
        }
    }
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Image quality options.
///
/// `Standard` and `Hd` apply to DALL-E 3. The GPT Image models use
/// `Low`/`Medium`/`High`/`Auto` instead. GPT Image 2.5/2.6 Sunburst and Flare
/// also accept `Xhigh` and `Max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageQuality {
    /// Standard quality (DALL-E 3)
    #[default]
    Standard,
    /// High definition quality (DALL-E 3)
    Hd,
    /// Low quality - cheapest and fastest (GPT Image models)
    Low,
    /// Medium quality (GPT Image models)
    Medium,
    /// High quality (GPT Image models)
    High,
    /// Extra-high quality (GPT Image 2.5/2.6 Sunburst and Flare)
    Xhigh,
    /// Maximum quality (GPT Image 2.5/2.6 Sunburst and Flare)
    Max,
    /// Let the model pick the quality (GPT Image models)
    Auto,
}

impl ImageQuality {
    /// Returns the quality string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Hd => "hd",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
            Self::Auto => "auto",
        }
    }
}

/// Image style options (DALL-E 3 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageStyle {
    /// Vivid - hyper-real and dramatic
    #[default]
    Vivid,
    /// Natural - more natural, less hyper-real
    Natural,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Return URLs to the generated images (valid for 60 minutes)
    #[default]
    Url,
    /// Return base64-encoded image data
    B64Json,
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

/// Output format for generated or edited images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageOutputFormat {
    /// PNG output.
    Png,
    /// JPEG output.
    Jpeg,
    /// WebP output.
    Webp,
    /// GIF output.
    Gif,
    /// SVG output.
    Svg,
}

impl ImageOutputFormat {
    /// Returns the output format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Gif => "gif",
            Self::Svg => "svg",
        }
    }
}

/// Planning complexity for image generation or editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ImageComplexity {
    /// Low complexity.
    Low,
    /// Medium complexity.
    Medium,
    /// High complexity.
    High,
}

impl ImageComplexity {
    /// Returns the complexity string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Options for image generation.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    /// The model to use; when `None` the API picks its own default
    pub model: Option<ImageModel>,
    /// Number of images to generate (1-10, DALL-E 3 only supports 1)
    pub n: Option<u32>,
    /// Image quality
    ///
    /// DALL-E 3 takes `standard`/`hd`; the GPT Image models take
    /// `low`/`medium`/`high`/`auto` and reject the DALL-E values. GPT Image
    /// 2.5/2.6 Sunburst and Flare additionally accept `xhigh` and `max`.
    pub quality: Option<ImageQuality>,
    /// Response format (URL or base64)
    ///
    /// **DALL-E only.** The GPT Image models reject this parameter with
    /// `unknown_parameter` and always return base64 data.
    pub response_format: Option<ResponseFormat>,
    /// Image size
    pub size: Option<ImageSize>,
    /// Image style
    ///
    /// **DALL-E 3 only.** The GPT Image models reject this parameter with
    /// `unknown_parameter`.
    pub style: Option<ImageStyle>,
    /// User identifier for abuse monitoring
    pub user: Option<String>,
    /// Output format. The default is PNG.
    ///
    /// SVG requires `n=1` and cannot be combined with streaming, partial
    /// images, or `output_compression`. It requires access to SVG model
    /// selection. The result is base64 in `data[].b64_json`; decoding it yields
    /// SVG as UTF-8 XML.
    ///
    /// GIF cannot be combined with streaming, partial images, masks, or output
    /// compression. It requires access to the GIF preview. The result is in
    /// `b64_json`.
    ///
    /// Use PNG or WebP with a `transparent` background.
    pub output_format: Option<ImageOutputFormat>,
    /// Controls how much the image model plans before rendering. GPT Image 2.6
    /// Sunburst and Flare support this option. When omitted, the model default
    /// is used. This is separate from `quality` and has no additional charge.
    pub complexity: Option<ImageComplexity>,
    /// Used only when `output_format=gif`. The range is 2 to 16. The default is
    /// 9 when omitted.
    pub frame_count: Option<u32>,
    /// Used only when `output_format=gif`. The minimum is 20. The default is
    /// 200 when omitted, and the value is rounded to 10 ms increments.
    pub frame_duration_ms: Option<u32>,
}

/// Options for image editing.
#[derive(Debug, Clone, Default)]
pub struct EditOptions {
    /// Path to the mask image (transparent areas will be edited)
    pub mask: Option<String>,
    /// The model to use (GPT Image models and DALL-E 2 support editing)
    pub model: Option<ImageModel>,
    /// Number of images to generate (1-10)
    pub n: Option<u32>,
    /// Image size
    pub size: Option<ImageSize>,
    /// Response format
    pub response_format: Option<ResponseFormat>,
    /// User identifier for abuse monitoring
    pub user: Option<String>,
    /// Output format. The default is PNG.
    ///
    /// SVG requires `n=1` and cannot be combined with streaming, partial
    /// images, or `output_compression`. It requires access to SVG model
    /// selection. The result is base64 in `data[].b64_json`; decoding it yields
    /// SVG as UTF-8 XML.
    ///
    /// GIF cannot be combined with streaming, partial images, masks, or output
    /// compression. It requires access to the GIF preview. The result is in
    /// `b64_json`.
    ///
    /// Use PNG or WebP with a `transparent` background.
    pub output_format: Option<ImageOutputFormat>,
    /// Controls how much the image model plans before rendering. GPT Image 2.6
    /// Sunburst and Flare support this option. When omitted, the model default
    /// is used. This is separate from `quality` and has no additional charge.
    pub complexity: Option<ImageComplexity>,
    /// Used only when `output_format=gif`. The range is 2 to 16. The default is
    /// 9 when omitted.
    pub frame_count: Option<u32>,
    /// Used only when `output_format=gif`. The minimum is 20. The default is
    /// 200 when omitted, and the value is rounded to 10 ms increments.
    pub frame_duration_ms: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    complexity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frame_duration_ms: Option<u32>,
}

impl GenerateRequest {
    /// Builds the JSON payload for `POST /images/generations`.
    fn new(prompt: &str, options: GenerateOptions) -> Self {
        Self {
            prompt: prompt.to_string(),
            model: options.model.map(|m| m.as_str().to_string()),
            n: options.n,
            quality: options.quality.map(|q| q.as_str().to_string()),
            response_format: options.response_format.map(|f| f.as_str().to_string()),
            size: options.size.map(|s| s.as_str().to_string()),
            style: options.style.map(|s| s.as_str().to_string()),
            user: options.user,
            output_format: options.output_format.map(|value| value.as_str().to_string()),
            complexity: options.complexity.map(|value| value.as_str().to_string()),
            frame_count: options.frame_count,
            frame_duration_ms: options.frame_duration_ms,
        }
    }
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
        headers.insert("User-Agent", request::header::HeaderValue::from_static("openai-tools-rust"));
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
        headers.insert("Content-Type", request::header::HeaderValue::from_static("application/json"));

        let request_body = GenerateRequest::new(prompt, options);

        let body = serde_json::to_string(&request_body).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = format!("{}/generations", self.auth.endpoint(IMAGES_PATH));

        let response = client.post(&url).headers(headers).body(body).send().await.map_err(OpenAIToolError::RequestError)?;

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
    pub async fn edit(&self, image_path: &str, prompt: &str, options: EditOptions) -> Result<ImageResponse> {
        let (client, headers) = self.create_client()?;

        // Read the image file
        let image_content = tokio::fs::read(image_path).await.map_err(|e| OpenAIToolError::Error(format!("Failed to read image: {}", e)))?;

        let image_filename = Path::new(image_path).file_name().and_then(|n| n.to_str()).unwrap_or("image.png").to_string();

        let image_part = Part::bytes(image_content)
            .file_name(image_filename)
            .mime_str("image/png")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("image", image_part).text("prompt", prompt.to_string());

        // Add mask if provided
        if let Some(mask_path) = options.mask {
            let mask_content = tokio::fs::read(&mask_path).await.map_err(|e| OpenAIToolError::Error(format!("Failed to read mask: {}", e)))?;

            let mask_filename = Path::new(&mask_path).file_name().and_then(|n| n.to_str()).unwrap_or("mask.png").to_string();

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
        if let Some(output_format) = options.output_format {
            form = form.text("output_format", output_format.as_str().to_string());
        }
        if let Some(complexity) = options.complexity {
            form = form.text("complexity", complexity.as_str().to_string());
        }
        if let Some(frame_count) = options.frame_count {
            form = form.text("frame_count", frame_count.to_string());
        }
        if let Some(frame_duration_ms) = options.frame_duration_ms {
            form = form.text("frame_duration_ms", frame_duration_ms.to_string());
        }

        let url = format!("{}/edits", self.auth.endpoint(IMAGES_PATH));

        let response = client.post(&url).headers(headers).multipart(form).send().await.map_err(OpenAIToolError::RequestError)?;

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
    pub async fn variation(&self, image_path: &str, options: VariationOptions) -> Result<ImageResponse> {
        let (client, headers) = self.create_client()?;

        // Read the image file
        let image_content = tokio::fs::read(image_path).await.map_err(|e| OpenAIToolError::Error(format!("Failed to read image: {}", e)))?;

        let image_filename = Path::new(image_path).file_name().and_then(|n| n.to_str()).unwrap_or("image.png").to_string();

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

        let response = client.post(&url).headers(headers).multipart(form).send().await.map_err(OpenAIToolError::RequestError)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_request(options: GenerateOptions) -> GenerateRequest {
        GenerateRequest::new("draw a fox", options)
    }

    // =========================================================================
    // Image models added in 2026.
    //
    // Model IDs, sizes and quality values verified against the OpenAI API
    // reference (https://developers.openai.com/api/docs/models), August 2026.
    // =========================================================================

    #[test]
    fn test_new_image_models_as_str() {
        assert_eq!(ImageModel::GptImage2.as_str(), "gpt-image-2");
        assert_eq!(ImageModel::GptImage2_6Flare.as_str(), "gpt-image-2.6-flare");
        assert_eq!(ImageModel::GptImage2_6Sunburst.as_str(), "gpt-image-2.6-sunburst");
        assert_eq!(ImageModel::GptImage2_5Flare.as_str(), "gpt-image-2.5-flare");
        assert_eq!(ImageModel::GptImage2_5Sunburst.as_str(), "gpt-image-2.5-sunburst");
        assert_eq!(ImageModel::GptImage1Mini.as_str(), "gpt-image-1-mini");
    }

    #[test]
    fn test_new_image_models_serialization() {
        for (model, expected) in [
            (ImageModel::GptImage2, "gpt-image-2"),
            (ImageModel::GptImage2_6Flare, "gpt-image-2.6-flare"),
            (ImageModel::GptImage2_6Sunburst, "gpt-image-2.6-sunburst"),
            (ImageModel::GptImage2_5Flare, "gpt-image-2.5-flare"),
            (ImageModel::GptImage2_5Sunburst, "gpt-image-2.5-sunburst"),
            (ImageModel::GptImage1Mini, "gpt-image-1-mini"),
        ] {
            let json = serde_json::to_string(&model).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", model);
            let deserialized: ImageModel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, model, "Serialization roundtrip failed for {:?}", model);
        }
    }

    /// The GPT Image models use portrait/landscape sizes that differ from the
    /// DALL-E 3 set.
    #[test]
    fn test_gpt_image_sizes() {
        assert_eq!(ImageSize::Size1024x1536.as_str(), "1024x1536");
        assert_eq!(ImageSize::Size1536x1024.as_str(), "1536x1024");

        for (size, expected) in [(ImageSize::Size1024x1536, "1024x1536"), (ImageSize::Size1536x1024, "1536x1024")] {
            let json = serde_json::to_string(&size).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", size);
        }
    }

    /// GPT Image models take `low`/`medium`/`high`/`auto` instead of the
    /// DALL-E 3 `standard`/`hd` pair; GPT Image 2.5/2.6 Sunburst and Flare also
    /// accept `xhigh` and `max`.
    #[test]
    fn test_gpt_image_quality_values() {
        assert_eq!(ImageQuality::Low.as_str(), "low");
        assert_eq!(ImageQuality::Medium.as_str(), "medium");
        assert_eq!(ImageQuality::High.as_str(), "high");
        assert_eq!(ImageQuality::Xhigh.as_str(), "xhigh");
        assert_eq!(ImageQuality::Max.as_str(), "max");
        assert_eq!(ImageQuality::Auto.as_str(), "auto");

        // The DALL-E 3 values must keep working.
        assert_eq!(ImageQuality::Standard.as_str(), "standard");
        assert_eq!(ImageQuality::Hd.as_str(), "hd");
    }

    #[test]
    fn test_gpt_image_quality_serialization() {
        for (quality, expected) in [
            (ImageQuality::Low, "low"),
            (ImageQuality::Medium, "medium"),
            (ImageQuality::High, "high"),
            (ImageQuality::Xhigh, "xhigh"),
            (ImageQuality::Max, "max"),
            (ImageQuality::Auto, "auto"),
            (ImageQuality::Standard, "standard"),
            (ImageQuality::Hd, "hd"),
        ] {
            let json = serde_json::to_string(&quality).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", quality);
            let deserialized: ImageQuality = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, quality, "Serialization roundtrip failed for {:?}", quality);
        }
    }

    /// Image models present in the live /v1/models listing but previously
    /// missing from the enum. Verified live against the API (August 2026).
    #[test]
    fn test_previously_missing_image_models() {
        for (model, expected) in [(ImageModel::GptImage1_5, "gpt-image-1.5"), (ImageModel::ChatGptImageLatest, "chatgpt-image-latest")] {
            assert_eq!(model.as_str(), expected, "Wrong model ID for {:?}", model);
            let json = serde_json::to_string(&model).unwrap();
            assert_eq!(json, format!("\"{}\"", expected), "Wrong serialization for {:?}", model);
            let deserialized: ImageModel = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, model, "Serialization roundtrip failed for {:?}", model);
        }
    }

    /// OpenAI retired `dall-e-2`/`dall-e-3` on api.openai.com, so the default
    /// must name a model that still resolves there. The variants stay because
    /// Azure OpenAI deployments can still serve DALL-E.
    #[test]
    fn test_default_image_model_is_a_live_openai_model() {
        assert_eq!(ImageModel::default(), ImageModel::GptImage1);
        assert_ne!(ImageModel::default(), ImageModel::DallE3, "the default must not be a retired model");
    }

    /// The GPT Image models accept `auto` in place of an explicit size.
    #[test]
    fn test_auto_image_size() {
        assert_eq!(ImageSize::Auto.as_str(), "auto");
        assert_eq!(serde_json::to_string(&ImageSize::Auto).unwrap(), "\"auto\"");
    }

    #[test]
    fn test_new_generation_fields_serialize() {
        let request = generate_request(GenerateOptions {
            output_format: Some(ImageOutputFormat::Gif),
            complexity: Some(ImageComplexity::High),
            frame_count: Some(16),
            frame_duration_ms: Some(20),
            ..Default::default()
        });

        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["output_format"], "gif");
        assert_eq!(json["complexity"], "high");
        assert_eq!(json["frame_count"], 16);
        assert_eq!(json["frame_duration_ms"], 20);
    }

    #[test]
    fn test_new_generation_fields_are_omitted_when_unset() {
        let json = serde_json::to_value(generate_request(GenerateOptions::default())).unwrap();

        assert!(json.get("output_format").is_none());
        assert!(json.get("complexity").is_none());
        assert!(json.get("frame_count").is_none());
        assert!(json.get("frame_duration_ms").is_none());
    }

    #[test]
    fn test_new_image_option_enums_as_str_and_serde_roundtrip() {
        for (value, expected) in [
            (ImageOutputFormat::Png, "png"),
            (ImageOutputFormat::Jpeg, "jpeg"),
            (ImageOutputFormat::Webp, "webp"),
            (ImageOutputFormat::Gif, "gif"),
            (ImageOutputFormat::Svg, "svg"),
        ] {
            assert_eq!(value.as_str(), expected);
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(serde_json::from_str::<ImageOutputFormat>(&json).unwrap(), value);
        }

        for (value, expected) in [(ImageComplexity::Low, "low"), (ImageComplexity::Medium, "medium"), (ImageComplexity::High, "high")] {
            assert_eq!(value.as_str(), expected);
            let json = serde_json::to_string(&value).unwrap();
            assert_eq!(serde_json::from_str::<ImageComplexity>(&json).unwrap(), value);
        }
    }

    #[test]
    fn test_generation_frame_values_are_not_validated_client_side() {
        let request = generate_request(GenerateOptions {
            output_format: Some(ImageOutputFormat::Gif),
            frame_count: Some(1),
            frame_duration_ms: Some(19),
            ..Default::default()
        });

        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["frame_count"], 1);
        assert_eq!(json["frame_duration_ms"], 19);
    }
}
