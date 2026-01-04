//! Integration tests for the OpenAI Audio API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test audio_integration
//!
//! Note: Audio API tests incur API costs. These tests use minimal text
//! to reduce costs while validating the API integration.

use openai_tools::audio::request::{Audio, AudioFormat, TtsModel, TtsOptions, Voice};

/// Test basic text-to-speech generation.
#[tokio::test]
async fn test_text_to_speech_basic() {
    let audio = Audio::new().expect("Should create Audio client");

    let options = TtsOptions::default();
    let bytes = audio
        .text_to_speech("Hello", options)
        .await
        .expect("Should generate speech");

    // Verify we got audio data
    assert!(!bytes.is_empty(), "Audio bytes should not be empty");

    // MP3 files start with specific bytes (ID3 tag or MPEG sync)
    // ID3 starts with "ID3" (0x49, 0x44, 0x33)
    // MPEG sync starts with 0xFF 0xFB or similar
    let is_valid_mp3 = (bytes.len() > 3 && bytes[0] == 0x49 && bytes[1] == 0x44 && bytes[2] == 0x33)
        || (bytes.len() > 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0);

    assert!(is_valid_mp3, "Should be valid MP3 data");

    println!("Generated {} bytes of audio", bytes.len());
}

/// Test text-to-speech with different voices.
#[tokio::test]
async fn test_text_to_speech_voices() {
    let audio = Audio::new().expect("Should create Audio client");

    // Test a few different voices
    let voices = vec![Voice::Alloy, Voice::Nova, Voice::Shimmer];

    for voice in voices {
        let options = TtsOptions {
            voice,
            ..Default::default()
        };

        let bytes = audio
            .text_to_speech("Test", options)
            .await
            .expect(&format!("Should generate speech with {:?} voice", voice));

        assert!(!bytes.is_empty(), "{:?} voice should produce audio", voice);
        println!("{:?} voice: {} bytes", voice, bytes.len());
    }
}

/// Test text-to-speech with HD model.
#[tokio::test]
async fn test_text_to_speech_hd() {
    let audio = Audio::new().expect("Should create Audio client");

    let options = TtsOptions {
        model: TtsModel::Tts1Hd,
        voice: Voice::Onyx,
        ..Default::default()
    };

    let bytes = audio
        .text_to_speech("Hi", options)
        .await
        .expect("Should generate HD speech");

    assert!(!bytes.is_empty(), "HD audio should not be empty");
    println!("HD audio: {} bytes", bytes.len());
}

/// Test text-to-speech with different audio formats.
#[tokio::test]
async fn test_text_to_speech_formats() {
    let audio = Audio::new().expect("Should create Audio client");

    // Test a few formats
    let formats = vec![
        (AudioFormat::Mp3, vec![0x49, 0x44, 0x33]), // ID3 tag
        (AudioFormat::Wav, vec![0x52, 0x49, 0x46, 0x46]), // RIFF header
    ];

    for (format, expected_header) in formats {
        let options = TtsOptions {
            response_format: format,
            ..Default::default()
        };

        let bytes = audio
            .text_to_speech("A", options)
            .await
            .expect(&format!("Should generate {:?} audio", format));

        assert!(!bytes.is_empty(), "{:?} format should produce audio", format);

        // For WAV, check RIFF header
        if format == AudioFormat::Wav {
            assert!(
                bytes.len() > 4 && bytes[0..4] == expected_header[..],
                "WAV should have RIFF header"
            );
        }

        println!("{:?} format: {} bytes", format, bytes.len());
    }
}

/// Test text-to-speech with speed adjustment.
#[tokio::test]
async fn test_text_to_speech_speed() {
    let audio = Audio::new().expect("Should create Audio client");

    let options = TtsOptions {
        speed: Some(1.5), // 1.5x speed
        ..Default::default()
    };

    let bytes = audio
        .text_to_speech("Testing speed", options)
        .await
        .expect("Should generate speech with custom speed");

    assert!(!bytes.is_empty(), "Should produce audio");
    println!("1.5x speed audio: {} bytes", bytes.len());
}

/// Test TtsModel enum functionality.
#[test]
fn test_tts_model_enum() {
    assert_eq!(TtsModel::Tts1.as_str(), "tts-1");
    assert_eq!(TtsModel::Tts1Hd.as_str(), "tts-1-hd");
    assert_eq!(TtsModel::Gpt4oMiniTts.as_str(), "gpt-4o-mini-tts");

    assert_eq!(format!("{}", TtsModel::Tts1), "tts-1");
    assert_eq!(TtsModel::default(), TtsModel::Tts1);
}

/// Test Voice enum functionality.
#[test]
fn test_voice_enum() {
    assert_eq!(Voice::Alloy.as_str(), "alloy");
    assert_eq!(Voice::Nova.as_str(), "nova");
    assert_eq!(Voice::Shimmer.as_str(), "shimmer");
    assert_eq!(Voice::Onyx.as_str(), "onyx");

    assert_eq!(format!("{}", Voice::Echo), "echo");
    assert_eq!(Voice::default(), Voice::Alloy);
}

/// Test AudioFormat enum functionality.
#[test]
fn test_audio_format_enum() {
    assert_eq!(AudioFormat::Mp3.as_str(), "mp3");
    assert_eq!(AudioFormat::Wav.as_str(), "wav");
    assert_eq!(AudioFormat::Flac.as_str(), "flac");
    assert_eq!(AudioFormat::Opus.as_str(), "opus");

    assert_eq!(AudioFormat::Mp3.file_extension(), "mp3");
    assert_eq!(AudioFormat::default(), AudioFormat::Mp3);
}

/// Test TtsOptions default values.
#[test]
fn test_tts_options_default() {
    let options = TtsOptions::default();

    assert_eq!(options.model, TtsModel::Tts1);
    assert_eq!(options.voice, Voice::Alloy);
    assert_eq!(options.response_format, AudioFormat::Mp3);
    assert!(options.speed.is_none());
}
