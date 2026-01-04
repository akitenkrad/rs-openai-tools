//! OpenAI Audio API Response Types
//!
//! This module defines the response structures for the OpenAI Audio API.

use serde::{Deserialize, Serialize};

/// Response structure from transcription/translation endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed or translated text
    pub text: String,

    /// The language of the input audio (only in verbose JSON response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// The duration of the audio in seconds (only in verbose JSON response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,

    /// Word-level timestamps (only when timestamp_granularities includes "word")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<Word>>,

    /// Segment-level timestamps (only when timestamp_granularities includes "segment")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<Segment>>,
}

/// Word-level timestamp information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

/// Segment-level timestamp information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Segment ID
    pub id: i32,
    /// Seek position
    pub seek: i32,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Segment text
    pub text: String,
    /// Token IDs
    pub tokens: Vec<i32>,
    /// Temperature used for generation
    pub temperature: f64,
    /// Average log probability
    pub avg_logprob: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// No speech probability
    pub no_speech_prob: f64,
}
