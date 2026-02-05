//! Integration tests for the OpenAI Moderations API.
//!
//! These tests require a valid OPENAI_API_KEY environment variable.
//! Run with: cargo test --test moderations_integration

use openai_tools::moderations::request::{ModerationModel, Moderations};

/// Test moderating a single safe text.
#[tokio::test]
async fn test_moderate_single_safe_text() {
    let moderations = Moderations::new().expect("Should create Moderations client");

    let response = moderations.moderate_text("Hello, how are you today?", None).await.expect("Should moderate text");

    // Verify response structure
    assert!(!response.id.is_empty(), "ID should not be empty");
    assert!(!response.model.is_empty(), "Model should not be empty");
    assert_eq!(response.results.len(), 1, "Should have one result");

    // Safe text should not be flagged
    let result = &response.results[0];
    assert!(!result.flagged, "Safe text should not be flagged");

    // Verify category scores are low for safe content
    assert!(result.category_scores.hate < 0.5, "Hate score should be low for safe text");
    assert!(result.category_scores.violence < 0.5, "Violence score should be low for safe text");

    println!("Moderation ID: {}", response.id);
    println!("Model: {}", response.model);
    println!("Flagged: {}", result.flagged);
}

/// Test moderating multiple texts at once.
#[tokio::test]
async fn test_moderate_multiple_texts() {
    let moderations = Moderations::new().expect("Should create Moderations client");

    let texts = vec!["Good morning!".to_string(), "Have a nice day!".to_string(), "Thank you for your help.".to_string()];

    let response = moderations.moderate_texts(texts.clone(), None).await.expect("Should moderate multiple texts");

    // Should have one result per input
    assert_eq!(response.results.len(), texts.len(), "Should have one result per input");

    // All safe texts should not be flagged
    for (i, result) in response.results.iter().enumerate() {
        assert!(!result.flagged, "Text {} should not be flagged: '{}'", i + 1, texts[i]);
    }

    println!("Moderated {} texts", response.results.len());
}

/// Test using a specific moderation model.
#[tokio::test]
async fn test_moderate_with_specific_model() {
    let moderations = Moderations::new().expect("Should create Moderations client");

    // Test with omni-moderation model
    let response = moderations
        .moderate_text("This is a test message.", Some(ModerationModel::OmniModerationLatest))
        .await
        .expect("Should moderate with omni-moderation model");

    assert!(response.model.contains("omni-moderation"), "Should use omni-moderation model");

    println!("Used model: {}", response.model);
}

/// Test that category scores are returned.
#[tokio::test]
async fn test_category_scores_returned() {
    let moderations = Moderations::new().expect("Should create Moderations client");

    let response = moderations.moderate_text("A friendly greeting to everyone.", None).await.expect("Should moderate text");

    let result = &response.results[0];

    // Verify all basic category scores are present (non-negative values)
    assert!(result.category_scores.hate >= 0.0);
    assert!(result.category_scores.hate_threatening >= 0.0);
    assert!(result.category_scores.harassment >= 0.0);
    assert!(result.category_scores.harassment_threatening >= 0.0);
    assert!(result.category_scores.self_harm >= 0.0);
    assert!(result.category_scores.self_harm_intent >= 0.0);
    assert!(result.category_scores.self_harm_instructions >= 0.0);
    assert!(result.category_scores.sexual >= 0.0);
    assert!(result.category_scores.sexual_minors >= 0.0);
    assert!(result.category_scores.violence >= 0.0);
    assert!(result.category_scores.violence_graphic >= 0.0);

    println!("All category scores are valid");
}

/// Test ModerationModel enum functionality.
#[test]
fn test_moderation_model_enum() {
    // Test as_str
    assert_eq!(ModerationModel::OmniModerationLatest.as_str(), "omni-moderation-latest");
    assert_eq!(ModerationModel::TextModerationLatest.as_str(), "text-moderation-latest");

    // Test Display
    assert_eq!(format!("{}", ModerationModel::OmniModerationLatest), "omni-moderation-latest");

    // Test Default
    assert_eq!(ModerationModel::default(), ModerationModel::OmniModerationLatest);
}
