//! # Audio Module
//!
//! This module provides functionality for interacting with the OpenAI Audio API.
//! It supports text-to-speech (TTS), transcription (speech-to-text), and translation.
//!
//! ## Key Features
//!
//! - **Text-to-Speech (TTS)**: Convert text to natural-sounding audio
//! - **Transcription**: Convert audio files to text
//! - **Translation**: Translate audio to English text
//! - **Multiple Voices**: Choose from various voice options
//! - **Multiple Formats**: Support for MP3, WAV, FLAC, and more
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TtsOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an Audio client
//!     let audio = Audio::new()?;
//!
//!     // Generate speech
//!     let bytes = audio.text_to_speech("Hello!", TtsOptions::default()).await?;
//!     std::fs::write("hello.mp3", bytes)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Text-to-Speech with Custom Voice
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TtsOptions, TtsModel, Voice, AudioFormat};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     let options = TtsOptions {
//!         model: TtsModel::Tts1Hd,
//!         voice: Voice::Nova,
//!         response_format: AudioFormat::Mp3,
//!         speed: Some(1.1),
//!         ..Default::default()
//!     };
//!
//!     let bytes = audio.text_to_speech("Welcome to the podcast!", options).await?;
//!     std::fs::write("intro.mp3", bytes)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Transcribe Audio File
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TranscribeOptions, SttModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     let options = TranscribeOptions {
//!         model: Some(SttModel::Whisper1),
//!         language: Some("en".to_string()),
//!         ..Default::default()
//!     };
//!
//!     let response = audio.transcribe("meeting.mp3", options).await?;
//!     println!("Transcript: {}", response.text);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Transcribe with Timestamps
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{
//!     Audio, TranscribeOptions, TranscriptionFormat, TimestampGranularity
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     let options = TranscribeOptions {
//!         response_format: Some(TranscriptionFormat::VerboseJson),
//!         timestamp_granularities: Some(vec![
//!             TimestampGranularity::Word,
//!             TimestampGranularity::Segment,
//!         ]),
//!         ..Default::default()
//!     };
//!
//!     let response = audio.transcribe("audio.mp3", options).await?;
//!
//!     if let Some(words) = &response.words {
//!         for word in words {
//!             println!("[{:.2}s - {:.2}s] {}", word.start, word.end, word.word);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Translate Audio to English
//!
//! ```rust,no_run
//! use openai_tools::audio::request::{Audio, TranslateOptions};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let audio = Audio::new()?;
//!
//!     // Translate Spanish audio to English
//!     let options = TranslateOptions::default();
//!     let response = audio.translate("spanish_audio.mp3", options).await?;
//!
//!     println!("English: {}", response.text);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Voice Options
//!
//! Available voices for TTS:
//! - `Alloy` - Neutral, balanced voice
//! - `Ash` - Warm, conversational
//! - `Coral` - Clear, articulate
//! - `Echo` - Smooth, calming
//! - `Fable` - Storytelling, expressive
//! - `Onyx` - Deep, authoritative
//! - `Nova` - Bright, energetic
//! - `Sage` - Wise, measured
//! - `Shimmer` - Soft, gentle
//!
//! ## Audio Formats
//!
//! Supported output formats for TTS:
//! - `Mp3` - Standard compressed audio (default)
//! - `Opus` - Low-latency streaming
//! - `Aac` - Apple-compatible
//! - `Flac` - Lossless compression
//! - `Wav` - Uncompressed
//! - `Pcm` - Raw audio samples
//!
//! ## Supported Input Formats
//!
//! For transcription and translation:
//! - MP3, MP4, MPEG, MPGA
//! - M4A, OGG, WAV, WEBM
//! - FLAC
//! - Maximum file size: 25 MB

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::audio::request::{AudioFormat, SttModel, TimestampGranularity, TranscriptionFormat, TtsModel, Voice};
    use crate::audio::response::TranscriptionResponse;

    #[test]
    fn test_transcription_response_deserialization() {
        let json = r#"{
            "text": "Hello, world!"
        }"#;

        let response: TranscriptionResponse = serde_json::from_str(json).expect("Should deserialize TranscriptionResponse");
        assert_eq!(response.text, "Hello, world!");
        assert!(response.language.is_none());
        assert!(response.duration.is_none());
    }

    #[test]
    fn test_transcription_response_verbose() {
        let json = r#"{
            "text": "Hello, world!",
            "language": "en",
            "duration": 1.5
        }"#;

        let response: TranscriptionResponse = serde_json::from_str(json).expect("Should deserialize verbose response");
        assert_eq!(response.text, "Hello, world!");
        assert_eq!(response.language, Some("en".to_string()));
        assert_eq!(response.duration, Some(1.5));
    }

    #[test]
    fn test_transcription_response_with_words() {
        let json = r#"{
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.6, "end": 1.0}
            ]
        }"#;

        let response: TranscriptionResponse = serde_json::from_str(json).expect("Should deserialize with words");
        let words = response.words.expect("Should have words");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert_eq!(words[0].start, 0.0);
        assert_eq!(words[1].word, "world");
    }

    #[test]
    fn test_tts_model_serialization() {
        assert_eq!(serde_json::to_string(&TtsModel::Tts1).unwrap(), "\"tts-1\"");
        assert_eq!(serde_json::to_string(&TtsModel::Tts1Hd).unwrap(), "\"tts-1-hd\"");
    }

    #[test]
    fn test_voice_serialization() {
        assert_eq!(serde_json::to_string(&Voice::Alloy).unwrap(), "\"alloy\"");
        assert_eq!(serde_json::to_string(&Voice::Nova).unwrap(), "\"nova\"");
        assert_eq!(serde_json::to_string(&Voice::Shimmer).unwrap(), "\"shimmer\"");
    }

    #[test]
    fn test_audio_format_serialization() {
        assert_eq!(serde_json::to_string(&AudioFormat::Mp3).unwrap(), "\"mp3\"");
        assert_eq!(serde_json::to_string(&AudioFormat::Wav).unwrap(), "\"wav\"");
        assert_eq!(serde_json::to_string(&AudioFormat::Flac).unwrap(), "\"flac\"");
    }

    #[test]
    fn test_stt_model_serialization() {
        assert_eq!(serde_json::to_string(&SttModel::Whisper1).unwrap(), "\"whisper-1\"");
    }

    #[test]
    fn test_transcription_format_serialization() {
        assert_eq!(serde_json::to_string(&TranscriptionFormat::Json).unwrap(), "\"json\"");
        assert_eq!(serde_json::to_string(&TranscriptionFormat::VerboseJson).unwrap(), "\"verbose_json\"");
        assert_eq!(serde_json::to_string(&TranscriptionFormat::Srt).unwrap(), "\"srt\"");
    }

    #[test]
    fn test_defaults() {
        assert_eq!(TtsModel::default(), TtsModel::Tts1);
        assert_eq!(Voice::default(), Voice::Alloy);
        assert_eq!(AudioFormat::default(), AudioFormat::Mp3);
        assert_eq!(SttModel::default(), SttModel::Whisper1);
        assert_eq!(TranscriptionFormat::default(), TranscriptionFormat::Json);
    }

    #[test]
    fn test_audio_format_file_extension() {
        assert_eq!(AudioFormat::Mp3.file_extension(), "mp3");
        assert_eq!(AudioFormat::Wav.file_extension(), "wav");
        assert_eq!(AudioFormat::Flac.file_extension(), "flac");
    }

    #[test]
    fn test_timestamp_granularity() {
        assert_eq!(TimestampGranularity::Word.as_str(), "word");
        assert_eq!(TimestampGranularity::Segment.as_str(), "segment");
    }
}
