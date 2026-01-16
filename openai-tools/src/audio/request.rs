//! OpenAI Audio API Request Module
//!
//! This module provides the functionality to interact with the OpenAI Audio API.
//! It supports text-to-speech (TTS), transcription, and translation.
//!
//! # Key Features
//!
//! - **Text-to-Speech**: Convert text to natural-sounding audio
//! - **Transcription**: Convert audio to text (speech-to-text)
//! - **Translation**: Translate audio to English text
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TtsOptions, Voice};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     // Generate speech from text
//!     let options = TtsOptions::default();
//!     let audio_bytes = audio.text_to_speech("Hello, world!", options).await?;
//!     std::fs::write("output.mp3", audio_bytes)?;
//!
//!     Ok(())
//! }
//! ```

use crate::audio::response::TranscriptionResponse;
use crate::common::auth::AuthProvider;
use crate::common::client::create_http_client;
use crate::common::errors::{ErrorResponse, OpenAIToolError, Result};
use request::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Default API path for Audio
const AUDIO_PATH: &str = "audio";

/// Text-to-speech models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsModel {
    /// Standard quality TTS model
    #[serde(rename = "tts-1")]
    Tts1,
    /// High definition TTS model
    #[serde(rename = "tts-1-hd")]
    Tts1Hd,
    /// GPT-4o Mini TTS model
    #[serde(rename = "gpt-4o-mini-tts")]
    Gpt4oMiniTts,
}

impl Default for TtsModel {
    fn default() -> Self {
        Self::Tts1
    }
}

impl TtsModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tts1 => "tts-1",
            Self::Tts1Hd => "tts-1-hd",
            Self::Gpt4oMiniTts => "gpt-4o-mini-tts",
        }
    }
}

impl std::fmt::Display for TtsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Voice options for text-to-speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Voice {
    /// Alloy voice
    Alloy,
    /// Ash voice
    Ash,
    /// Coral voice
    Coral,
    /// Echo voice
    Echo,
    /// Fable voice
    Fable,
    /// Onyx voice
    Onyx,
    /// Nova voice
    Nova,
    /// Sage voice
    Sage,
    /// Shimmer voice
    Shimmer,
}

impl Default for Voice {
    fn default() -> Self {
        Self::Alloy
    }
}

impl Voice {
    /// Returns the voice identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Alloy => "alloy",
            Self::Ash => "ash",
            Self::Coral => "coral",
            Self::Echo => "echo",
            Self::Fable => "fable",
            Self::Onyx => "onyx",
            Self::Nova => "nova",
            Self::Sage => "sage",
            Self::Shimmer => "shimmer",
        }
    }
}

impl std::fmt::Display for Voice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Audio output formats for TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MP3 format (default)
    Mp3,
    /// Opus format
    Opus,
    /// AAC format
    Aac,
    /// FLAC format
    Flac,
    /// WAV format
    Wav,
    /// PCM format
    Pcm,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self::Mp3
    }
}

impl AudioFormat {
    /// Returns the format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }

    /// Returns the file extension for this format.
    pub fn file_extension(&self) -> &'static str {
        self.as_str()
    }
}

impl std::fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Speech-to-text models for transcription and translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SttModel {
    /// Whisper v1 model
    #[serde(rename = "whisper-1")]
    Whisper1,
    /// GPT-4o Transcribe model
    #[serde(rename = "gpt-4o-transcribe")]
    Gpt4oTranscribe,
}

impl Default for SttModel {
    fn default() -> Self {
        Self::Whisper1
    }
}

impl SttModel {
    /// Returns the model identifier string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Whisper1 => "whisper-1",
            Self::Gpt4oTranscribe => "gpt-4o-transcribe",
        }
    }
}

impl std::fmt::Display for SttModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Transcription response formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionFormat {
    /// JSON format
    Json,
    /// Plain text format
    Text,
    /// SRT subtitle format
    Srt,
    /// Verbose JSON with timestamps
    VerboseJson,
    /// VTT subtitle format
    Vtt,
}

impl Default for TranscriptionFormat {
    fn default() -> Self {
        Self::Json
    }
}

impl TranscriptionFormat {
    /// Returns the format string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Text => "text",
            Self::Srt => "srt",
            Self::VerboseJson => "verbose_json",
            Self::Vtt => "vtt",
        }
    }
}

/// Timestamp granularity options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimestampGranularity {
    /// Word-level timestamps
    Word,
    /// Segment-level timestamps
    Segment,
}

impl TimestampGranularity {
    /// Returns the granularity string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Segment => "segment",
        }
    }
}

/// Options for text-to-speech generation.
#[derive(Debug, Clone)]
pub struct TtsOptions {
    /// The model to use (defaults to tts-1)
    pub model: TtsModel,
    /// The voice to use (defaults to alloy)
    pub voice: Voice,
    /// The output audio format (defaults to mp3)
    pub response_format: AudioFormat,
    /// Speech speed (0.25 to 4.0, defaults to 1.0)
    pub speed: Option<f32>,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            model: TtsModel::default(),
            voice: Voice::default(),
            response_format: AudioFormat::default(),
            speed: None,
        }
    }
}

/// Options for audio transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// The model to use (defaults to whisper-1)
    pub model: Option<SttModel>,
    /// The language of the input audio (ISO-639-1 code)
    pub language: Option<String>,
    /// Optional prompt to guide the model's style
    pub prompt: Option<String>,
    /// Response format (defaults to json)
    pub response_format: Option<TranscriptionFormat>,
    /// Temperature for sampling (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Timestamp granularities to include
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

/// Options for audio translation.
#[derive(Debug, Clone, Default)]
pub struct TranslateOptions {
    /// The model to use (only whisper-1 is supported)
    pub model: Option<SttModel>,
    /// Optional prompt to guide the model's style
    pub prompt: Option<String>,
    /// Response format (defaults to json)
    pub response_format: Option<TranscriptionFormat>,
    /// Temperature for sampling (0.0 to 1.0)
    pub temperature: Option<f32>,
}

/// Request payload for TTS.
#[derive(Debug, Clone, Serialize)]
struct TtsRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
}

/// Client for interacting with the OpenAI Audio API.
///
/// This struct provides methods for text-to-speech, transcription, and translation.
/// Use [`Audio::new()`] to create a new instance.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::audio::request::{Audio, TtsOptions, Voice, AudioFormat};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let audio = Audio::new()?;
///
///     let options = TtsOptions {
///         voice: Voice::Nova,
///         response_format: AudioFormat::Mp3,
///         ..Default::default()
///     };
///
///     let bytes = audio.text_to_speech("Welcome to our app!", options).await?;
///     std::fs::write("welcome.mp3", bytes)?;
///
///     Ok(())
/// }
/// ```
pub struct Audio {
    /// Authentication provider (OpenAI or Azure)
    auth: AuthProvider,
    /// Optional request timeout duration
    timeout: Option<Duration>,
}

impl Audio {
    /// Creates a new Audio client for OpenAI API.
    ///
    /// Initializes the client by loading the OpenAI API key from
    /// the environment variable `OPENAI_API_KEY`. Supports `.env` file loading
    /// via dotenvy.
    ///
    /// # Returns
    ///
    /// * `Ok(Audio)` - A new Audio client ready for use
    /// * `Err(OpenAIToolError)` - If the API key is not found in the environment
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::Audio;
    ///
    /// let audio = Audio::new().expect("API key should be set");
    /// ```
    pub fn new() -> Result<Self> {
        let auth = AuthProvider::openai_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client with a custom authentication provider
    pub fn with_auth(auth: AuthProvider) -> Self {
        Self { auth, timeout: None }
    }

    /// Creates a new Audio client for Azure OpenAI API
    pub fn azure() -> Result<Self> {
        let auth = AuthProvider::azure_from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client by auto-detecting the provider
    pub fn detect_provider() -> Result<Self> {
        let auth = AuthProvider::from_env()?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client with URL-based provider detection
    pub fn with_url<S: Into<String>>(
        url: S,
        api_key: S,
        deployment_name: Option<S>,
    ) -> Result<Self> {
        let auth = AuthProvider::from_url_with_hint(url, api_key, deployment_name)?;
        Ok(Self { auth, timeout: None })
    }

    /// Creates a new Audio client from URL using environment variables
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

    /// Converts text to speech.
    ///
    /// Returns audio bytes in the specified format.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to convert to speech (max 4096 characters)
    /// * `options` - TTS options (model, voice, format, speed)
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - The audio data as bytes
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TtsOptions, TtsModel, Voice};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TtsOptions {
    ///         model: TtsModel::Tts1Hd,
    ///         voice: Voice::Shimmer,
    ///         speed: Some(1.2),
    ///         ..Default::default()
    ///     };
    ///
    ///     let bytes = audio.text_to_speech("Hello, this is a test.", options).await?;
    ///     std::fs::write("speech.mp3", bytes)?;
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn text_to_speech(&self, text: &str, options: TtsOptions) -> Result<Vec<u8>> {
        let (client, mut headers) = self.create_client()?;
        headers.insert(
            "Content-Type",
            request::header::HeaderValue::from_static("application/json"),
        );

        let request_body = TtsRequest {
            model: options.model.as_str().to_string(),
            input: text.to_string(),
            voice: options.voice.as_str().to_string(),
            response_format: Some(options.response_format.as_str().to_string()),
            speed: options.speed,
        };

        let body =
            serde_json::to_string(&request_body).map_err(OpenAIToolError::SerdeJsonError)?;

        let url = format!("{}/speech", self.auth.endpoint(AUDIO_PATH));

        let response = client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        let bytes = response
            .bytes()
            .await
            .map_err(OpenAIToolError::RequestError)?;

        Ok(bytes.to_vec())
    }

    /// Transcribes audio from a file path.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file
    /// * `options` - Transcription options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The transcription result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranscribeOptions};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TranscribeOptions {
    ///         language: Some("en".to_string()),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = audio.transcribe("audio.mp3", options).await?;
    ///     println!("Transcription: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn transcribe(
        &self,
        audio_path: &str,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResponse> {
        let audio_content = tokio::fs::read(audio_path)
            .await
            .map_err(|e| OpenAIToolError::Error(format!("Failed to read audio file: {}", e)))?;

        let filename = Path::new(audio_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.mp3")
            .to_string();

        self.transcribe_bytes(&audio_content, &filename, options)
            .await
    }

    /// Transcribes audio from bytes.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio data as bytes
    /// * `filename` - The filename with extension (e.g., "audio.mp3")
    /// * `options` - Transcription options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The transcription result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranscribeOptions, SttModel};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let audio_data = std::fs::read("recording.mp3")?;
    ///     let options = TranscribeOptions {
    ///         model: Some(SttModel::Whisper1),
    ///         ..Default::default()
    ///     };
    ///
    ///     let response = audio.transcribe_bytes(&audio_data, "recording.mp3", options).await?;
    ///     println!("Transcription: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn transcribe_bytes(
        &self,
        audio_data: &[u8],
        filename: &str,
        options: TranscribeOptions,
    ) -> Result<TranscriptionResponse> {
        let (client, headers) = self.create_client()?;

        let audio_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("file", audio_part);

        // Add model
        let model = options.model.unwrap_or_default();
        form = form.text("model", model.as_str().to_string());

        // Add optional parameters
        if let Some(language) = options.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = options.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(temperature) = options.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(granularities) = options.timestamp_granularities {
            for g in granularities {
                form = form.text("timestamp_granularities[]", g.as_str().to_string());
            }
        }

        let url = format!("{}/transcriptions", self.auth.endpoint(AUDIO_PATH));

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

        serde_json::from_str::<TranscriptionResponse>(&content)
            .map_err(OpenAIToolError::SerdeJsonError)
    }

    /// Translates audio to English text.
    ///
    /// Only supports translation to English using the whisper-1 model.
    ///
    /// # Arguments
    ///
    /// * `audio_path` - Path to the audio file
    /// * `options` - Translation options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The translation result
    /// * `Err(OpenAIToolError)` - If the request fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use openai_tools::audio::request::{Audio, TranslateOptions};
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let audio = Audio::new()?;
    ///
    ///     let options = TranslateOptions::default();
    ///     let response = audio.translate("french_audio.mp3", options).await?;
    ///     println!("English translation: {}", response.text);
    ///
    ///     Ok(())
    /// }
    /// ```
    pub async fn translate(
        &self,
        audio_path: &str,
        options: TranslateOptions,
    ) -> Result<TranscriptionResponse> {
        let audio_content = tokio::fs::read(audio_path)
            .await
            .map_err(|e| OpenAIToolError::Error(format!("Failed to read audio file: {}", e)))?;

        let filename = Path::new(audio_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.mp3")
            .to_string();

        self.translate_bytes(&audio_content, &filename, options)
            .await
    }

    /// Translates audio from bytes to English text.
    ///
    /// # Arguments
    ///
    /// * `audio_data` - The audio data as bytes
    /// * `filename` - The filename with extension (e.g., "audio.mp3")
    /// * `options` - Translation options
    ///
    /// # Returns
    ///
    /// * `Ok(TranscriptionResponse)` - The translation result
    /// * `Err(OpenAIToolError)` - If the request fails
    pub async fn translate_bytes(
        &self,
        audio_data: &[u8],
        filename: &str,
        options: TranslateOptions,
    ) -> Result<TranscriptionResponse> {
        let (client, headers) = self.create_client()?;

        let audio_part = Part::bytes(audio_data.to_vec())
            .file_name(filename.to_string())
            .mime_str("audio/mpeg")
            .map_err(|e| OpenAIToolError::Error(format!("Failed to set MIME type: {}", e)))?;

        let mut form = Form::new().part("file", audio_part);

        // Add model (whisper-1 is the only supported model for translation)
        let model = options.model.unwrap_or(SttModel::Whisper1);
        form = form.text("model", model.as_str().to_string());

        // Add optional parameters
        if let Some(prompt) = options.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(response_format) = options.response_format {
            form = form.text("response_format", response_format.as_str().to_string());
        }
        if let Some(temperature) = options.temperature {
            form = form.text("temperature", temperature.to_string());
        }

        let url = format!("{}/translations", self.auth.endpoint(AUDIO_PATH));

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

        serde_json::from_str::<TranscriptionResponse>(&content)
            .map_err(OpenAIToolError::SerdeJsonError)
    }
}
