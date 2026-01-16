//! OpenAI Model Types
//!
//! This module provides strongly-typed enums for specifying OpenAI models
//! across different APIs. Using enums instead of strings provides:
//!
//! - Compile-time validation of model names
//! - IDE autocompletion support
//! - Prevention of typos in model names
//! - Clear documentation of available models
//!
//! # Model Categories
//!
//! - [`ChatModel`]: Models for Chat Completions and Responses APIs
//! - [`EmbeddingModel`]: Models for text embeddings
//! - [`RealtimeModel`]: Models for real-time audio/text interactions
//! - [`FineTuningModel`]: Base models that can be fine-tuned
//!
//! # Example
//!
//! ```rust,no_run
//! use openai_tools::common::models::{ChatModel, EmbeddingModel};
//! use openai_tools::chat::request::ChatCompletion;
//! use openai_tools::embedding::request::Embedding;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Using ChatModel enum
//! let mut chat = ChatCompletion::new();
//! chat.model(ChatModel::Gpt4oMini);
//!
//! // Using EmbeddingModel enum
//! let mut embedding = Embedding::new()?;
//! embedding.model(EmbeddingModel::TextEmbedding3Small);
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - [OpenAI Models Documentation](https://platform.openai.com/docs/models)
//! - [Model Deprecations](https://platform.openai.com/docs/deprecations)

use serde::{Deserialize, Serialize};

// ============================================================================
// Parameter Restriction Types
// ============================================================================

/// Defines how a parameter is restricted for a model.
///
/// This enum is used to specify whether a parameter can accept any value,
/// only a fixed value, or is not supported at all.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterRestriction {
    /// Parameter accepts any value within its valid range
    Any,
    /// Parameter only supports a specific fixed value
    FixedValue(f64),
    /// Parameter is not supported by this model
    NotSupported,
}

/// Parameter support information for a model.
///
/// This struct provides detailed information about which parameters are
/// supported by a model and any restrictions that apply.
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::{ChatModel, ParameterRestriction};
///
/// let model = ChatModel::O3Mini;
/// let support = model.parameter_support();
///
/// // Reasoning models only support temperature = 1.0
/// assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
///
/// // Reasoning models don't support logprobs
/// assert!(!support.logprobs);
/// ```
#[derive(Debug, Clone)]
pub struct ParameterSupport {
    /// Temperature parameter restriction (Chat & Responses API)
    pub temperature: ParameterRestriction,
    /// Frequency penalty parameter restriction (Chat API only)
    pub frequency_penalty: ParameterRestriction,
    /// Presence penalty parameter restriction (Chat API only)
    pub presence_penalty: ParameterRestriction,
    /// Whether logprobs parameter is supported (Chat API only)
    pub logprobs: bool,
    /// Whether top_logprobs parameter is supported (Chat & Responses API)
    pub top_logprobs: bool,
    /// Whether logit_bias parameter is supported (Chat API only)
    pub logit_bias: bool,
    /// Whether n > 1 (multiple completions) is supported (Chat API only)
    pub n_multiple: bool,
    /// Top P parameter restriction (Responses API only)
    pub top_p: ParameterRestriction,
    /// Whether reasoning parameter is supported (Responses API only, reasoning models)
    pub reasoning: bool,
}

impl ParameterSupport {
    /// Creates parameter support info for standard (non-reasoning) models.
    ///
    /// Standard models support all parameters with full range.
    pub fn standard_model() -> Self {
        Self {
            temperature: ParameterRestriction::Any,
            frequency_penalty: ParameterRestriction::Any,
            presence_penalty: ParameterRestriction::Any,
            logprobs: true,
            top_logprobs: true,
            logit_bias: true,
            n_multiple: true,
            top_p: ParameterRestriction::Any,
            reasoning: false,
        }
    }

    /// Creates parameter support info for reasoning models (GPT-5, o-series).
    ///
    /// Reasoning models have restricted parameter support:
    /// - temperature: only 1.0
    /// - top_p: only 1.0
    /// - frequency_penalty: only 0
    /// - presence_penalty: only 0
    /// - logprobs, top_logprobs, logit_bias: not supported
    /// - n: only 1
    /// - reasoning: supported
    pub fn reasoning_model() -> Self {
        Self {
            temperature: ParameterRestriction::FixedValue(1.0),
            frequency_penalty: ParameterRestriction::FixedValue(0.0),
            presence_penalty: ParameterRestriction::FixedValue(0.0),
            logprobs: false,
            top_logprobs: false,
            logit_bias: false,
            n_multiple: false,
            top_p: ParameterRestriction::FixedValue(1.0),
            reasoning: true,
        }
    }
}

/// Models available for Chat Completions and Responses APIs.
///
/// This enum covers all models that can be used with the Chat Completions API
/// (`/v1/chat/completions`) and the Responses API (`/v1/responses`).
///
/// # Model Categories
///
/// ## GPT-5 Series (Latest Flagship)
/// - [`Gpt52`]: GPT-5.2 Thinking - flagship model for coding and agentic tasks
/// - [`Gpt52ChatLatest`]: GPT-5.2 Instant - fast workhorse for everyday work
/// - [`Gpt52Pro`]: GPT-5.2 Pro - smartest for difficult questions (Responses API only)
/// - [`Gpt51`]: GPT-5.1 - configurable reasoning and non-reasoning
/// - [`Gpt51CodexMax`]: GPT-5.1 Codex Max - powers Codex CLI
/// - [`Gpt5Mini`]: GPT-5 Mini - smaller, faster variant
///
/// ## GPT-4.1 Series
/// - [`Gpt41`]: 1M context window flagship
/// - [`Gpt41Mini`]: Balanced performance and cost
/// - [`Gpt41Nano`]: Fastest and most cost-efficient
///
/// ## GPT-4o Series
/// - [`Gpt4o`]: High-intelligence flagship model
/// - [`Gpt4oMini`]: Cost-effective GPT-4o variant
/// - [`Gpt4oAudioPreview`]: Audio-capable GPT-4o
///
/// ## Reasoning Models (o-series)
/// - [`O1`], [`O1Pro`]: Full reasoning models
/// - [`O3`], [`O3Mini`]: Latest reasoning models
/// - [`O4Mini`]: Fast, cost-efficient reasoning
///
/// # Reasoning Model Restrictions
///
/// Reasoning models (GPT-5 series, o1, o3, o4 series) have parameter restrictions:
/// - `temperature`: Only 1.0 supported
/// - `top_p`: Only 1.0 supported
/// - `frequency_penalty`: Only 0 supported
/// - `presence_penalty`: Only 0 supported
///
/// GPT-5 models support `reasoning.effort` parameter:
/// - `none`: No reasoning (GPT-5.1 default)
/// - `minimal`: Very few reasoning tokens
/// - `low`, `medium`, `high`: Increasing reasoning depth
/// - `xhigh`: Maximum reasoning (GPT-5.2 Pro, GPT-5.1 Codex Max)
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::ChatModel;
///
/// // Check if a model is a reasoning model
/// let model = ChatModel::O3Mini;
/// assert!(model.is_reasoning_model());
///
/// // GPT-5 models are also reasoning models
/// let gpt5 = ChatModel::Gpt52;
/// assert!(gpt5.is_reasoning_model());
///
/// // Get the API model ID string
/// assert_eq!(model.as_str(), "o3-mini");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatModel {
    // === GPT-5 Series (Latest Flagship) ===
    /// GPT-5.2 Thinking - Flagship model for coding and agentic tasks
    ///
    /// - Context: 128K tokens (256K with thinking)
    /// - Supports: reasoning.effort (none, minimal, low, medium, high, xhigh)
    /// - Supports: verbosity parameter (low, medium, high)
    #[serde(rename = "gpt-5.2")]
    Gpt52,

    /// GPT-5.2 Instant - Fast workhorse for everyday work
    ///
    /// Points to the GPT-5.2 snapshot used in ChatGPT
    #[serde(rename = "gpt-5.2-chat-latest")]
    Gpt52ChatLatest,

    /// GPT-5.2 Pro - Smartest for difficult questions
    ///
    /// - Available in Responses API only
    /// - Supports: xhigh reasoning effort
    #[serde(rename = "gpt-5.2-pro")]
    Gpt52Pro,

    /// GPT-5.1 - Configurable reasoning and non-reasoning
    ///
    /// - Defaults to no reasoning (effort: none)
    /// - Supports: reasoning.effort (none, low, medium, high)
    #[serde(rename = "gpt-5.1")]
    Gpt51,

    /// GPT-5.1 Chat Latest - Chat-optimized GPT-5.1
    #[serde(rename = "gpt-5.1-chat-latest")]
    Gpt51ChatLatest,

    /// GPT-5.1 Codex Max - Powers Codex and Codex CLI
    ///
    /// - Available in Responses API only
    /// - Supports: reasoning.effort (none, medium, high, xhigh)
    #[serde(rename = "gpt-5.1-codex-max")]
    Gpt51CodexMax,

    /// GPT-5 Mini - Smaller, faster GPT-5 variant
    #[serde(rename = "gpt-5-mini")]
    Gpt5Mini,

    // === GPT-4.1 Series ===
    /// GPT-4.1 - Smartest non-reasoning model with 1M token context
    #[serde(rename = "gpt-4.1")]
    Gpt41,

    /// GPT-4.1 Mini - Balanced performance and cost
    #[serde(rename = "gpt-4.1-mini")]
    Gpt41Mini,

    /// GPT-4.1 Nano - Fastest and most cost-efficient
    #[serde(rename = "gpt-4.1-nano")]
    Gpt41Nano,

    // === GPT-4o Series ===
    /// GPT-4o - High-intelligence flagship model (multimodal)
    #[serde(rename = "gpt-4o")]
    Gpt4o,

    /// GPT-4o Mini - Cost-effective GPT-4o variant
    #[serde(rename = "gpt-4o-mini")]
    Gpt4oMini,

    /// GPT-4o Audio Preview - Audio-capable GPT-4o
    #[serde(rename = "gpt-4o-audio-preview")]
    Gpt4oAudioPreview,

    // === GPT-4 Series ===
    /// GPT-4 Turbo - High capability with faster responses
    #[serde(rename = "gpt-4-turbo")]
    Gpt4Turbo,

    /// GPT-4 - Original GPT-4 model
    #[serde(rename = "gpt-4")]
    Gpt4,

    // === GPT-3.5 Series ===
    /// GPT-3.5 Turbo - Fast and cost-effective
    #[serde(rename = "gpt-3.5-turbo")]
    Gpt35Turbo,

    // === Reasoning Models (o-series) ===
    /// O1 - Full reasoning model for complex tasks
    #[serde(rename = "o1")]
    O1,

    /// O1 Pro - O1 with more compute for complex problems
    #[serde(rename = "o1-pro")]
    O1Pro,

    /// O3 - Latest full reasoning model
    #[serde(rename = "o3")]
    O3,

    /// O3 Mini - Smaller, faster reasoning model
    #[serde(rename = "o3-mini")]
    O3Mini,

    /// O4 Mini - Fast, cost-efficient reasoning model
    #[serde(rename = "o4-mini")]
    O4Mini,

    // === Custom Model ===
    /// Custom model ID for fine-tuned models or new models not yet in enum
    #[serde(untagged)]
    Custom(String),
}

impl ChatModel {
    /// Returns the model identifier string for API requests.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// assert_eq!(ChatModel::Gpt4oMini.as_str(), "gpt-4o-mini");
    /// assert_eq!(ChatModel::O3Mini.as_str(), "o3-mini");
    /// assert_eq!(ChatModel::Gpt52.as_str(), "gpt-5.2");
    /// ```
    pub fn as_str(&self) -> &str {
        match self {
            // GPT-5 Series
            Self::Gpt52 => "gpt-5.2",
            Self::Gpt52ChatLatest => "gpt-5.2-chat-latest",
            Self::Gpt52Pro => "gpt-5.2-pro",
            Self::Gpt51 => "gpt-5.1",
            Self::Gpt51ChatLatest => "gpt-5.1-chat-latest",
            Self::Gpt51CodexMax => "gpt-5.1-codex-max",
            Self::Gpt5Mini => "gpt-5-mini",
            // GPT-4.1 Series
            Self::Gpt41 => "gpt-4.1",
            Self::Gpt41Mini => "gpt-4.1-mini",
            Self::Gpt41Nano => "gpt-4.1-nano",
            // GPT-4o Series
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4oAudioPreview => "gpt-4o-audio-preview",
            // GPT-4 Series
            Self::Gpt4Turbo => "gpt-4-turbo",
            Self::Gpt4 => "gpt-4",
            // GPT-3.5 Series
            Self::Gpt35Turbo => "gpt-3.5-turbo",
            // Reasoning Models
            Self::O1 => "o1",
            Self::O1Pro => "o1-pro",
            Self::O3 => "o3",
            Self::O3Mini => "o3-mini",
            Self::O4Mini => "o4-mini",
            // Custom
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Checks if this is a reasoning model with parameter restrictions.
    ///
    /// Reasoning models (GPT-5 series, o1, o3, o4 series) only support:
    /// - `temperature = 1.0`
    /// - `top_p = 1.0`
    /// - `frequency_penalty = 0`
    /// - `presence_penalty = 0`
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// assert!(ChatModel::O3Mini.is_reasoning_model());
    /// assert!(ChatModel::Gpt52.is_reasoning_model());
    /// assert!(!ChatModel::Gpt4oMini.is_reasoning_model());
    /// assert!(!ChatModel::Gpt41.is_reasoning_model());
    /// ```
    pub fn is_reasoning_model(&self) -> bool {
        matches!(
            self,
            // GPT-5 series are reasoning models
            Self::Gpt52 | Self::Gpt52ChatLatest | Self::Gpt52Pro |
            Self::Gpt51 | Self::Gpt51ChatLatest | Self::Gpt51CodexMax |
            Self::Gpt5Mini |
            // O-series reasoning models
            Self::O1 | Self::O1Pro | Self::O3 | Self::O3Mini | Self::O4Mini
        ) || matches!(
            self,
            Self::Custom(s) if s.starts_with("gpt-5") || s.starts_with("o1") || s.starts_with("o3") || s.starts_with("o4")
        )
    }

    /// Returns parameter support information for this model.
    ///
    /// This method provides detailed information about which parameters
    /// are supported by the model and any restrictions that apply.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::{ChatModel, ParameterRestriction};
    ///
    /// // Standard model supports all parameters
    /// let standard = ChatModel::Gpt4oMini;
    /// let support = standard.parameter_support();
    /// assert_eq!(support.temperature, ParameterRestriction::Any);
    /// assert!(support.logprobs);
    ///
    /// // Reasoning model has restrictions
    /// let reasoning = ChatModel::O3Mini;
    /// let support = reasoning.parameter_support();
    /// assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
    /// assert!(!support.logprobs);
    /// assert!(support.reasoning);
    /// ```
    pub fn parameter_support(&self) -> ParameterSupport {
        if self.is_reasoning_model() {
            ParameterSupport::reasoning_model()
        } else {
            ParameterSupport::standard_model()
        }
    }

    /// Creates a custom model from a string.
    ///
    /// Use this for fine-tuned models or new models not yet in the enum.
    ///
    /// # Example
    ///
    /// ```rust
    /// use openai_tools::common::models::ChatModel;
    ///
    /// let model = ChatModel::custom("ft:gpt-4o-mini:my-org::abc123");
    /// assert_eq!(model.as_str(), "ft:gpt-4o-mini:my-org::abc123");
    /// ```
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self::Custom(model_id.into())
    }
}

impl Default for ChatModel {
    fn default() -> Self {
        Self::Gpt4oMini
    }
}

impl std::fmt::Display for ChatModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for ChatModel {
    fn from(s: &str) -> Self {
        match s {
            // GPT-5 Series
            "gpt-5.2" => Self::Gpt52,
            "gpt-5.2-chat-latest" => Self::Gpt52ChatLatest,
            "gpt-5.2-pro" => Self::Gpt52Pro,
            "gpt-5.1" => Self::Gpt51,
            "gpt-5.1-chat-latest" => Self::Gpt51ChatLatest,
            "gpt-5.1-codex-max" => Self::Gpt51CodexMax,
            "gpt-5-mini" => Self::Gpt5Mini,
            // GPT-4.1 Series
            "gpt-4.1" => Self::Gpt41,
            "gpt-4.1-mini" => Self::Gpt41Mini,
            "gpt-4.1-nano" => Self::Gpt41Nano,
            // GPT-4o Series
            "gpt-4o" => Self::Gpt4o,
            "gpt-4o-mini" => Self::Gpt4oMini,
            "gpt-4o-audio-preview" => Self::Gpt4oAudioPreview,
            // GPT-4 Series
            "gpt-4-turbo" => Self::Gpt4Turbo,
            "gpt-4" => Self::Gpt4,
            // GPT-3.5 Series
            "gpt-3.5-turbo" => Self::Gpt35Turbo,
            // Reasoning Models
            "o1" => Self::O1,
            "o1-pro" => Self::O1Pro,
            "o3" => Self::O3,
            "o3-mini" => Self::O3Mini,
            "o4-mini" => Self::O4Mini,
            // Custom
            other => Self::Custom(other.to_string()),
        }
    }
}

impl From<String> for ChatModel {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

// ============================================================================
// Embedding Models
// ============================================================================

/// Models available for the Embeddings API.
///
/// This enum covers all models that can be used with the Embeddings API
/// (`/v1/embeddings`) for converting text into vector representations.
///
/// # Available Models
///
/// - [`TextEmbedding3Small`]: Improved, performant model (default)
/// - [`TextEmbedding3Large`]: Most capable model for English and non-English
/// - [`TextEmbeddingAda002`]: Legacy model (not recommended for new projects)
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::EmbeddingModel;
///
/// let model = EmbeddingModel::TextEmbedding3Small;
/// assert_eq!(model.as_str(), "text-embedding-3-small");
/// assert_eq!(model.dimensions(), 1536);
/// ```
///
/// # Reference
///
/// See [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingModel {
    /// text-embedding-3-small - Improved, more performant embedding model
    ///
    /// - Dimensions: 1536
    /// - Max input: 8191 tokens
    /// - Recommended for most use cases
    #[serde(rename = "text-embedding-3-small")]
    TextEmbedding3Small,

    /// text-embedding-3-large - Most capable embedding model
    ///
    /// - Dimensions: 3072
    /// - Max input: 8191 tokens
    /// - Best for high-accuracy tasks
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,

    /// text-embedding-ada-002 - Legacy embedding model
    ///
    /// - Dimensions: 1536
    /// - Max input: 8191 tokens
    /// - Not recommended for new projects
    #[serde(rename = "text-embedding-ada-002")]
    TextEmbeddingAda002,
}

impl EmbeddingModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }

    /// Returns the default output dimensions for this model.
    ///
    /// Note: For `text-embedding-3-*` models, you can request fewer dimensions
    /// via the API's `dimensions` parameter. This returns the default/maximum.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::TextEmbedding3Small => 1536,
            Self::TextEmbedding3Large => 3072,
            Self::TextEmbeddingAda002 => 1536,
        }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        Self::TextEmbedding3Small
    }
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for EmbeddingModel {
    fn from(s: &str) -> Self {
        match s {
            "text-embedding-3-small" => Self::TextEmbedding3Small,
            "text-embedding-3-large" => Self::TextEmbedding3Large,
            "text-embedding-ada-002" => Self::TextEmbeddingAda002,
            _ => Self::TextEmbedding3Small, // Default fallback
        }
    }
}

// ============================================================================
// Realtime Models
// ============================================================================

/// Models available for the Realtime API.
///
/// This enum covers all models that can be used with the Realtime API
/// for real-time audio and text interactions via WebSocket.
///
/// # Available Models
///
/// - [`Gpt4oRealtimePreview`]: Full-featured realtime preview (default)
/// - [`Gpt4oMiniRealtimePreview`]: Cost-effective realtime preview
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::RealtimeModel;
///
/// let model = RealtimeModel::Gpt4oRealtimePreview;
/// assert_eq!(model.as_str(), "gpt-4o-realtime-preview");
/// ```
///
/// # Reference
///
/// See [OpenAI Realtime API Documentation](https://platform.openai.com/docs/guides/realtime)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RealtimeModel {
    /// gpt-4o-realtime-preview - Full-featured realtime model
    #[serde(rename = "gpt-4o-realtime-preview")]
    Gpt4oRealtimePreview,

    /// gpt-4o-mini-realtime-preview - Cost-effective realtime model
    #[serde(rename = "gpt-4o-mini-realtime-preview")]
    Gpt4oMiniRealtimePreview,

    /// Custom model ID for new models not yet in enum
    #[serde(untagged)]
    Custom(String),
}

impl RealtimeModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gpt4oRealtimePreview => "gpt-4o-realtime-preview",
            Self::Gpt4oMiniRealtimePreview => "gpt-4o-mini-realtime-preview",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Creates a custom model from a string.
    pub fn custom(model_id: impl Into<String>) -> Self {
        Self::Custom(model_id.into())
    }
}

impl Default for RealtimeModel {
    fn default() -> Self {
        Self::Gpt4oRealtimePreview
    }
}

impl std::fmt::Display for RealtimeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for RealtimeModel {
    fn from(s: &str) -> Self {
        match s {
            "gpt-4o-realtime-preview" => Self::Gpt4oRealtimePreview,
            "gpt-4o-mini-realtime-preview" => Self::Gpt4oMiniRealtimePreview,
            other => Self::Custom(other.to_string()),
        }
    }
}

// ============================================================================
// Fine-tuning Models
// ============================================================================

/// Base models that can be used for fine-tuning.
///
/// This enum covers all models that can be fine-tuned via the Fine-tuning API
/// (`/v1/fine_tuning/jobs`). Note that fine-tuning requires specific dated
/// model versions.
///
/// # Available Models
///
/// ## GPT-4.1 Series (Latest)
/// - [`Gpt41_2025_04_14`]: GPT-4.1 for fine-tuning
/// - [`Gpt41Mini_2025_04_14`]: GPT-4.1 Mini for fine-tuning
/// - [`Gpt41Nano_2025_04_14`]: GPT-4.1 Nano for fine-tuning
///
/// ## GPT-4o Series
/// - [`Gpt4oMini_2024_07_18`]: GPT-4o Mini for fine-tuning
/// - [`Gpt4o_2024_08_06`]: GPT-4o for fine-tuning
///
/// ## GPT-4 Series
/// - [`Gpt4_0613`]: GPT-4 for fine-tuning
///
/// ## GPT-3.5 Series
/// - [`Gpt35Turbo_0125`]: GPT-3.5 Turbo for fine-tuning
///
/// # Example
///
/// ```rust
/// use openai_tools::common::models::FineTuningModel;
///
/// let model = FineTuningModel::Gpt4oMini_2024_07_18;
/// assert_eq!(model.as_str(), "gpt-4o-mini-2024-07-18");
/// ```
///
/// # Reference
///
/// See [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FineTuningModel {
    // === GPT-4.1 Series ===
    /// gpt-4.1-2025-04-14 - GPT-4.1 for fine-tuning
    #[serde(rename = "gpt-4.1-2025-04-14")]
    Gpt41_2025_04_14,

    /// gpt-4.1-mini-2025-04-14 - GPT-4.1 Mini for fine-tuning
    #[serde(rename = "gpt-4.1-mini-2025-04-14")]
    Gpt41Mini_2025_04_14,

    /// gpt-4.1-nano-2025-04-14 - GPT-4.1 Nano for fine-tuning
    #[serde(rename = "gpt-4.1-nano-2025-04-14")]
    Gpt41Nano_2025_04_14,

    // === GPT-4o Series ===
    /// gpt-4o-mini-2024-07-18 - GPT-4o Mini for fine-tuning
    #[serde(rename = "gpt-4o-mini-2024-07-18")]
    Gpt4oMini_2024_07_18,

    /// gpt-4o-2024-08-06 - GPT-4o for fine-tuning
    #[serde(rename = "gpt-4o-2024-08-06")]
    Gpt4o_2024_08_06,

    // === GPT-4 Series ===
    /// gpt-4-0613 - GPT-4 for fine-tuning
    #[serde(rename = "gpt-4-0613")]
    Gpt4_0613,

    // === GPT-3.5 Series ===
    /// gpt-3.5-turbo-0125 - GPT-3.5 Turbo for fine-tuning
    #[serde(rename = "gpt-3.5-turbo-0125")]
    Gpt35Turbo_0125,

    /// gpt-3.5-turbo-1106 - GPT-3.5 Turbo (older version)
    #[serde(rename = "gpt-3.5-turbo-1106")]
    Gpt35Turbo_1106,

    /// gpt-3.5-turbo-0613 - GPT-3.5 Turbo (legacy)
    #[serde(rename = "gpt-3.5-turbo-0613")]
    Gpt35Turbo_0613,
}

impl FineTuningModel {
    /// Returns the model identifier string for API requests.
    pub fn as_str(&self) -> &str {
        match self {
            // GPT-4.1 Series
            Self::Gpt41_2025_04_14 => "gpt-4.1-2025-04-14",
            Self::Gpt41Mini_2025_04_14 => "gpt-4.1-mini-2025-04-14",
            Self::Gpt41Nano_2025_04_14 => "gpt-4.1-nano-2025-04-14",
            // GPT-4o Series
            Self::Gpt4oMini_2024_07_18 => "gpt-4o-mini-2024-07-18",
            Self::Gpt4o_2024_08_06 => "gpt-4o-2024-08-06",
            // GPT-4 Series
            Self::Gpt4_0613 => "gpt-4-0613",
            // GPT-3.5 Series
            Self::Gpt35Turbo_0125 => "gpt-3.5-turbo-0125",
            Self::Gpt35Turbo_1106 => "gpt-3.5-turbo-1106",
            Self::Gpt35Turbo_0613 => "gpt-3.5-turbo-0613",
        }
    }
}

impl Default for FineTuningModel {
    fn default() -> Self {
        Self::Gpt4oMini_2024_07_18
    }
}

impl std::fmt::Display for FineTuningModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_model_as_str() {
        assert_eq!(ChatModel::Gpt4oMini.as_str(), "gpt-4o-mini");
        assert_eq!(ChatModel::O3Mini.as_str(), "o3-mini");
        assert_eq!(ChatModel::Gpt41.as_str(), "gpt-4.1");
        // GPT-5 models
        assert_eq!(ChatModel::Gpt52.as_str(), "gpt-5.2");
        assert_eq!(ChatModel::Gpt52ChatLatest.as_str(), "gpt-5.2-chat-latest");
        assert_eq!(ChatModel::Gpt52Pro.as_str(), "gpt-5.2-pro");
        assert_eq!(ChatModel::Gpt51.as_str(), "gpt-5.1");
        assert_eq!(ChatModel::Gpt51CodexMax.as_str(), "gpt-5.1-codex-max");
        assert_eq!(ChatModel::Gpt5Mini.as_str(), "gpt-5-mini");
    }

    #[test]
    fn test_chat_model_is_reasoning() {
        // O-series reasoning models
        assert!(ChatModel::O1.is_reasoning_model());
        assert!(ChatModel::O3.is_reasoning_model());
        assert!(ChatModel::O3Mini.is_reasoning_model());
        assert!(ChatModel::O4Mini.is_reasoning_model());
        // GPT-5 series are also reasoning models
        assert!(ChatModel::Gpt52.is_reasoning_model());
        assert!(ChatModel::Gpt52ChatLatest.is_reasoning_model());
        assert!(ChatModel::Gpt52Pro.is_reasoning_model());
        assert!(ChatModel::Gpt51.is_reasoning_model());
        assert!(ChatModel::Gpt51CodexMax.is_reasoning_model());
        assert!(ChatModel::Gpt5Mini.is_reasoning_model());
        // Non-reasoning models
        assert!(!ChatModel::Gpt4oMini.is_reasoning_model());
        assert!(!ChatModel::Gpt41.is_reasoning_model());
    }

    #[test]
    fn test_chat_model_from_str() {
        assert_eq!(ChatModel::from("gpt-4o-mini"), ChatModel::Gpt4oMini);
        assert_eq!(ChatModel::from("o3-mini"), ChatModel::O3Mini);
        // GPT-5 models
        assert_eq!(ChatModel::from("gpt-5.2"), ChatModel::Gpt52);
        assert_eq!(ChatModel::from("gpt-5.2-chat-latest"), ChatModel::Gpt52ChatLatest);
        assert_eq!(ChatModel::from("gpt-5.2-pro"), ChatModel::Gpt52Pro);
        assert_eq!(ChatModel::from("gpt-5.1"), ChatModel::Gpt51);
        assert_eq!(ChatModel::from("gpt-5.1-codex-max"), ChatModel::Gpt51CodexMax);
        assert_eq!(ChatModel::from("gpt-5-mini"), ChatModel::Gpt5Mini);
        // Unknown models become Custom
        assert!(matches!(ChatModel::from("unknown-model"), ChatModel::Custom(_)));
    }

    #[test]
    fn test_chat_model_custom() {
        let custom = ChatModel::custom("ft:gpt-4o-mini:org::123");
        assert_eq!(custom.as_str(), "ft:gpt-4o-mini:org::123");
    }

    #[test]
    fn test_chat_model_custom_gpt5_is_reasoning() {
        // Custom GPT-5 models should also be detected as reasoning models
        let custom_gpt5 = ChatModel::custom("gpt-5.3-preview");
        assert!(custom_gpt5.is_reasoning_model());
    }

    #[test]
    fn test_embedding_model_dimensions() {
        assert_eq!(EmbeddingModel::TextEmbedding3Small.dimensions(), 1536);
        assert_eq!(EmbeddingModel::TextEmbedding3Large.dimensions(), 3072);
    }

    #[test]
    fn test_realtime_model_as_str() {
        assert_eq!(RealtimeModel::Gpt4oRealtimePreview.as_str(), "gpt-4o-realtime-preview");
    }

    #[test]
    fn test_fine_tuning_model_as_str() {
        assert_eq!(FineTuningModel::Gpt4oMini_2024_07_18.as_str(), "gpt-4o-mini-2024-07-18");
        assert_eq!(FineTuningModel::Gpt41_2025_04_14.as_str(), "gpt-4.1-2025-04-14");
    }

    #[test]
    fn test_chat_model_serialization() {
        let model = ChatModel::Gpt4oMini;
        let json = serde_json::to_string(&model).unwrap();
        assert_eq!(json, "\"gpt-4o-mini\"");
        // GPT-5 serialization
        let gpt52 = ChatModel::Gpt52;
        let json = serde_json::to_string(&gpt52).unwrap();
        assert_eq!(json, "\"gpt-5.2\"");
    }

    #[test]
    fn test_chat_model_deserialization() {
        let model: ChatModel = serde_json::from_str("\"gpt-4o-mini\"").unwrap();
        assert_eq!(model, ChatModel::Gpt4oMini);
        // GPT-5 deserialization
        let gpt52: ChatModel = serde_json::from_str("\"gpt-5.2\"").unwrap();
        assert_eq!(gpt52, ChatModel::Gpt52);
    }

    #[test]
    fn test_parameter_support_standard_model() {
        let model = ChatModel::Gpt4oMini;
        let support = model.parameter_support();

        // Standard models support all parameters
        assert_eq!(support.temperature, ParameterRestriction::Any);
        assert_eq!(support.frequency_penalty, ParameterRestriction::Any);
        assert_eq!(support.presence_penalty, ParameterRestriction::Any);
        assert_eq!(support.top_p, ParameterRestriction::Any);
        assert!(support.logprobs);
        assert!(support.top_logprobs);
        assert!(support.logit_bias);
        assert!(support.n_multiple);
        assert!(!support.reasoning); // Standard models don't support reasoning
    }

    #[test]
    fn test_parameter_support_reasoning_model() {
        let model = ChatModel::O3Mini;
        let support = model.parameter_support();

        // Reasoning models have restrictions
        assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
        assert_eq!(support.frequency_penalty, ParameterRestriction::FixedValue(0.0));
        assert_eq!(support.presence_penalty, ParameterRestriction::FixedValue(0.0));
        assert_eq!(support.top_p, ParameterRestriction::FixedValue(1.0));
        assert!(!support.logprobs);
        assert!(!support.top_logprobs);
        assert!(!support.logit_bias);
        assert!(!support.n_multiple);
        assert!(support.reasoning); // Reasoning models support reasoning
    }

    #[test]
    fn test_parameter_support_gpt5_model() {
        // GPT-5 models are also reasoning models
        let model = ChatModel::Gpt52;
        let support = model.parameter_support();

        assert_eq!(support.temperature, ParameterRestriction::FixedValue(1.0));
        assert!(!support.logprobs);
        assert!(support.reasoning);
    }
}
