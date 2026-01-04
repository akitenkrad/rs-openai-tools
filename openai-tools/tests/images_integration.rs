//! Integration tests for the OpenAI Images API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test images_integration
//!
//! Note: Image generation tests incur API costs. These tests generate minimal
//! images to reduce costs while still validating the API integration.

use openai_tools::images::request::{
    GenerateOptions, ImageModel, ImageQuality, ImageSize, ImageStyle, Images, ResponseFormat,
};

/// Test basic image generation with DALL-E 3.
#[tokio::test]
async fn test_generate_image_dall_e_3() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::DallE3),
        size: Some(ImageSize::Size1024x1024),
        ..Default::default()
    };

    let response = images
        .generate("A simple red circle on white background", options)
        .await
        .expect("Should generate image");

    // Verify response structure
    assert!(response.created > 0, "Created timestamp should be positive");
    assert_eq!(response.data.len(), 1, "DALL-E 3 generates one image");

    // Verify image data
    let image = &response.data[0];
    assert!(image.url.is_some(), "Should have URL");

    // DALL-E 3 may revise the prompt
    // revised_prompt may or may not be present
    println!("Image URL: {:?}", image.url);
    if let Some(revised) = &image.revised_prompt {
        println!("Revised prompt: {}", revised);
    }
}

/// Test image generation with different options.
#[tokio::test]
async fn test_generate_image_with_options() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::DallE3),
        size: Some(ImageSize::Size1024x1024),
        quality: Some(ImageQuality::Standard),
        style: Some(ImageStyle::Natural),
        ..Default::default()
    };

    let response = images
        .generate("A blue square", options)
        .await
        .expect("Should generate image with options");

    assert_eq!(response.data.len(), 1);
    assert!(response.data[0].url.is_some());
}

/// Test image generation with base64 response format.
/// Note: This test generates a small image to reduce data transfer.
#[tokio::test]
async fn test_generate_image_b64_json() {
    let images = Images::new().expect("Should create Images client");

    let options = GenerateOptions {
        model: Some(ImageModel::DallE3),
        response_format: Some(ResponseFormat::B64Json),
        size: Some(ImageSize::Size1024x1024),
        ..Default::default()
    };

    let response = images
        .generate("A small green dot", options)
        .await
        .expect("Should generate image with base64");

    assert_eq!(response.data.len(), 1);
    let image = &response.data[0];

    // Should have base64 data, not URL
    assert!(image.b64_json.is_some(), "Should have b64_json");
    assert!(image.url.is_none(), "Should not have URL");

    // Verify we can decode the base64
    let bytes = image.as_bytes().expect("Should have bytes");
    assert!(bytes.is_ok(), "Should decode base64 successfully");

    let decoded = bytes.unwrap();
    assert!(!decoded.is_empty(), "Decoded bytes should not be empty");

    println!("Decoded image size: {} bytes", decoded.len());
}

/// Test ImageModel enum functionality.
#[test]
fn test_image_model_enum() {
    assert_eq!(ImageModel::DallE2.as_str(), "dall-e-2");
    assert_eq!(ImageModel::DallE3.as_str(), "dall-e-3");
    assert_eq!(ImageModel::GptImage1.as_str(), "gpt-image-1");

    assert_eq!(format!("{}", ImageModel::DallE3), "dall-e-3");
    assert_eq!(ImageModel::default(), ImageModel::DallE3);
}

/// Test ImageSize enum functionality.
#[test]
fn test_image_size_enum() {
    assert_eq!(ImageSize::Size256x256.as_str(), "256x256");
    assert_eq!(ImageSize::Size512x512.as_str(), "512x512");
    assert_eq!(ImageSize::Size1024x1024.as_str(), "1024x1024");
    assert_eq!(ImageSize::Size1792x1024.as_str(), "1792x1024");
    assert_eq!(ImageSize::Size1024x1792.as_str(), "1024x1792");

    assert_eq!(ImageSize::default(), ImageSize::Size1024x1024);
}

/// Test ImageQuality and ImageStyle enums.
#[test]
fn test_image_quality_and_style_enums() {
    assert_eq!(ImageQuality::Standard.as_str(), "standard");
    assert_eq!(ImageQuality::Hd.as_str(), "hd");
    assert_eq!(ImageQuality::default(), ImageQuality::Standard);

    assert_eq!(ImageStyle::Vivid.as_str(), "vivid");
    assert_eq!(ImageStyle::Natural.as_str(), "natural");
    assert_eq!(ImageStyle::default(), ImageStyle::Vivid);
}

/// Test ResponseFormat enum.
#[test]
fn test_response_format_enum() {
    assert_eq!(ResponseFormat::Url.as_str(), "url");
    assert_eq!(ResponseFormat::B64Json.as_str(), "b64_json");
    assert_eq!(ResponseFormat::default(), ResponseFormat::Url);
}
