//! OpenAI Moderations API Response Types
//!
//! This module defines the response structures for the OpenAI Moderations API.

use serde::{Deserialize, Serialize};

/// Response structure from the moderation endpoint.
///
/// Contains the results of content moderation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResponse {
    /// Unique identifier for the moderation request
    pub id: String,
    /// The model used for moderation
    pub model: String,
    /// Array of moderation results (one per input)
    pub results: Vec<ModerationResult>,
}

/// Individual moderation result for a single input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationResult {
    /// Whether any category was flagged
    pub flagged: bool,
    /// Category flags indicating which types of content were detected
    pub categories: ModerationCategories,
    /// Confidence scores for each category (0.0 to 1.0)
    pub category_scores: ModerationCategoryScores,
}

/// Category flags for content moderation.
///
/// Each field indicates whether that category of content was detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategories {
    /// Content that expresses, incites, or promotes hate based on identity
    pub hate: bool,
    /// Hateful content that also includes violence or threat
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    /// Content that expresses, incites, or promotes harassing language
    pub harassment: bool,
    /// Harassment content that also includes violence or threat
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    /// Content that promotes, encourages, or depicts self-harm
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    /// Content indicating intent to commit self-harm
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    /// Content that provides instructions for self-harm
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    /// Sexual content
    pub sexual: bool,
    /// Sexual content involving minors
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    /// Content that depicts violence
    pub violence: bool,
    /// Violent content that is graphic or gory
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
    /// Content that is illicit (newer models only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub illicit: Option<bool>,
    /// Illicit content that includes violence (newer models only)
    #[serde(rename = "illicit/violent", skip_serializing_if = "Option::is_none")]
    pub illicit_violent: Option<bool>,
}

/// Confidence scores for each moderation category.
///
/// Values range from 0.0 to 1.0, where higher values indicate
/// higher confidence that the content belongs to that category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModerationCategoryScores {
    /// Score for hate content
    pub hate: f64,
    /// Score for hate/threatening content
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,
    /// Score for harassment content
    pub harassment: f64,
    /// Score for harassment/threatening content
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f64,
    /// Score for self-harm content
    #[serde(rename = "self-harm")]
    pub self_harm: f64,
    /// Score for self-harm/intent content
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f64,
    /// Score for self-harm/instructions content
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f64,
    /// Score for sexual content
    pub sexual: f64,
    /// Score for sexual/minors content
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,
    /// Score for violence content
    pub violence: f64,
    /// Score for violence/graphic content
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
    /// Score for illicit content (newer models only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub illicit: Option<f64>,
    /// Score for illicit/violent content (newer models only)
    #[serde(rename = "illicit/violent", skip_serializing_if = "Option::is_none")]
    pub illicit_violent: Option<f64>,
}
