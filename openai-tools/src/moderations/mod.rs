//! # Moderations Module
//!
//! This module provides functionality for interacting with the OpenAI Moderations API.
//! It allows you to classify text content to determine if it violates OpenAI's content policy.
//!
//! ## Key Features
//!
//! - **Text Classification**: Analyze text for policy violations
//! - **Multiple Inputs**: Process multiple texts in a single request
//! - **Detailed Categories**: Get granular results across 11+ content categories
//! - **Confidence Scores**: Access probability scores for each category
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use openai_tools::moderations::request::Moderations;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a Moderations client
//!     let moderations = Moderations::new()?;
//!
//!     // Check text for policy violations
//!     let response = moderations.moderate_text("Hello, world!", None).await?;
//!
//!     if response.results[0].flagged {
//!         println!("Content was flagged!");
//!     } else {
//!         println!("Content is safe");
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Usage Examples
//!
//! ### Single Text Moderation
//!
//! ```rust,no_run
//! use openai_tools::moderations::request::Moderations;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let moderations = Moderations::new()?;
//!     let response = moderations.moderate_text("Sample text to check", None).await?;
//!
//!     let result = &response.results[0];
//!     println!("Flagged: {}", result.flagged);
//!     println!("Violence score: {}", result.category_scores.violence);
//!     println!("Hate score: {}", result.category_scores.hate);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Batch Moderation
//!
//! ```rust,no_run
//! use openai_tools::moderations::request::Moderations;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let moderations = Moderations::new()?;
//!
//!     let texts = vec![
//!         "First message".to_string(),
//!         "Second message".to_string(),
//!         "Third message".to_string(),
//!     ];
//!
//!     let response = moderations.moderate_texts(texts, None).await?;
//!
//!     for (i, result) in response.results.iter().enumerate() {
//!         println!("Text {}: flagged = {}", i + 1, result.flagged);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Using a Specific Model
//!
//! ```rust,no_run
//! use openai_tools::moderations::request::{Moderations, ModerationModel};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let moderations = Moderations::new()?;
//!
//!     // Use the omni-moderation model for best results
//!     let response = moderations
//!         .moderate_text("Text to check", Some(ModerationModel::OmniModerationLatest))
//!         .await?;
//!
//!     println!("Model used: {}", response.model);
//!     Ok(())
//! }
//! ```
//!
//! ## Response Structure
//!
//! ### Categories
//!
//! The moderation API classifies content into these categories:
//! - `hate`: Content promoting hate based on identity
//! - `hate/threatening`: Hateful content with violence/threats
//! - `harassment`: Harassing language
//! - `harassment/threatening`: Harassment with violence/threats
//! - `self-harm`: Content about self-harm
//! - `self-harm/intent`: Intent to commit self-harm
//! - `self-harm/instructions`: Instructions for self-harm
//! - `sexual`: Sexual content
//! - `sexual/minors`: Sexual content involving minors
//! - `violence`: Violent content
//! - `violence/graphic`: Graphic violence
//!
//! ### Model Options
//!
//! - `omni-moderation-latest`: Latest model with multi-modal support
//! - `text-moderation-latest`: Legacy text-only model

pub mod request;
pub mod response;

#[cfg(test)]
mod tests {
    use crate::moderations::request::ModerationModel;
    use crate::moderations::response::ModerationResponse;

    #[test]
    fn test_moderation_response_deserialization() {
        let json = r#"{
            "id": "modr-XXXXX",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": false,
                    "categories": {
                        "hate": false,
                        "hate/threatening": false,
                        "harassment": false,
                        "harassment/threatening": false,
                        "self-harm": false,
                        "self-harm/intent": false,
                        "self-harm/instructions": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": false,
                        "violence/graphic": false
                    },
                    "category_scores": {
                        "hate": 0.00001,
                        "hate/threatening": 0.00001,
                        "harassment": 0.00001,
                        "harassment/threatening": 0.00001,
                        "self-harm": 0.00001,
                        "self-harm/intent": 0.00001,
                        "self-harm/instructions": 0.00001,
                        "sexual": 0.00001,
                        "sexual/minors": 0.00001,
                        "violence": 0.00001,
                        "violence/graphic": 0.00001
                    }
                }
            ]
        }"#;

        let response: ModerationResponse =
            serde_json::from_str(json).expect("Should deserialize ModerationResponse");
        assert_eq!(response.id, "modr-XXXXX");
        assert_eq!(response.model, "omni-moderation-latest");
        assert_eq!(response.results.len(), 1);
        assert!(!response.results[0].flagged);
    }

    #[test]
    fn test_moderation_response_with_flagged_content() {
        let json = r#"{
            "id": "modr-YYYYY",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": true,
                    "categories": {
                        "hate": true,
                        "hate/threatening": false,
                        "harassment": true,
                        "harassment/threatening": false,
                        "self-harm": false,
                        "self-harm/intent": false,
                        "self-harm/instructions": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": false,
                        "violence/graphic": false
                    },
                    "category_scores": {
                        "hate": 0.85,
                        "hate/threatening": 0.02,
                        "harassment": 0.75,
                        "harassment/threatening": 0.01,
                        "self-harm": 0.001,
                        "self-harm/intent": 0.001,
                        "self-harm/instructions": 0.001,
                        "sexual": 0.001,
                        "sexual/minors": 0.001,
                        "violence": 0.05,
                        "violence/graphic": 0.01
                    }
                }
            ]
        }"#;

        let response: ModerationResponse =
            serde_json::from_str(json).expect("Should deserialize flagged ModerationResponse");
        assert!(response.results[0].flagged);
        assert!(response.results[0].categories.hate);
        assert!(response.results[0].categories.harassment);
        assert!(response.results[0].category_scores.hate > 0.8);
    }

    #[test]
    fn test_moderation_response_with_illicit_fields() {
        let json = r#"{
            "id": "modr-ZZZZZ",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": false,
                    "categories": {
                        "hate": false,
                        "hate/threatening": false,
                        "harassment": false,
                        "harassment/threatening": false,
                        "self-harm": false,
                        "self-harm/intent": false,
                        "self-harm/instructions": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": false,
                        "violence/graphic": false,
                        "illicit": false,
                        "illicit/violent": false
                    },
                    "category_scores": {
                        "hate": 0.00001,
                        "hate/threatening": 0.00001,
                        "harassment": 0.00001,
                        "harassment/threatening": 0.00001,
                        "self-harm": 0.00001,
                        "self-harm/intent": 0.00001,
                        "self-harm/instructions": 0.00001,
                        "sexual": 0.00001,
                        "sexual/minors": 0.00001,
                        "violence": 0.00001,
                        "violence/graphic": 0.00001,
                        "illicit": 0.00001,
                        "illicit/violent": 0.00001
                    }
                }
            ]
        }"#;

        let response: ModerationResponse = serde_json::from_str(json)
            .expect("Should deserialize ModerationResponse with illicit fields");
        assert_eq!(response.results[0].categories.illicit, Some(false));
        assert_eq!(response.results[0].categories.illicit_violent, Some(false));
        assert!(response.results[0].category_scores.illicit.is_some());
    }

    #[test]
    fn test_moderation_model_serialization() {
        let model = ModerationModel::OmniModerationLatest;
        let json = serde_json::to_string(&model).expect("Should serialize ModerationModel");
        assert_eq!(json, "\"omni-moderation-latest\"");

        let model = ModerationModel::TextModerationLatest;
        let json = serde_json::to_string(&model).expect("Should serialize ModerationModel");
        assert_eq!(json, "\"text-moderation-latest\"");
    }

    #[test]
    fn test_moderation_model_as_str() {
        assert_eq!(
            ModerationModel::OmniModerationLatest.as_str(),
            "omni-moderation-latest"
        );
        assert_eq!(
            ModerationModel::TextModerationLatest.as_str(),
            "text-moderation-latest"
        );
    }

    #[test]
    fn test_moderation_model_default() {
        let default_model = ModerationModel::default();
        assert_eq!(default_model, ModerationModel::OmniModerationLatest);
    }

    #[test]
    fn test_multiple_results_deserialization() {
        let json = r#"{
            "id": "modr-MULTI",
            "model": "omni-moderation-latest",
            "results": [
                {
                    "flagged": false,
                    "categories": {
                        "hate": false,
                        "hate/threatening": false,
                        "harassment": false,
                        "harassment/threatening": false,
                        "self-harm": false,
                        "self-harm/intent": false,
                        "self-harm/instructions": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": false,
                        "violence/graphic": false
                    },
                    "category_scores": {
                        "hate": 0.001,
                        "hate/threatening": 0.001,
                        "harassment": 0.001,
                        "harassment/threatening": 0.001,
                        "self-harm": 0.001,
                        "self-harm/intent": 0.001,
                        "self-harm/instructions": 0.001,
                        "sexual": 0.001,
                        "sexual/minors": 0.001,
                        "violence": 0.001,
                        "violence/graphic": 0.001
                    }
                },
                {
                    "flagged": true,
                    "categories": {
                        "hate": false,
                        "hate/threatening": false,
                        "harassment": false,
                        "harassment/threatening": false,
                        "self-harm": false,
                        "self-harm/intent": false,
                        "self-harm/instructions": false,
                        "sexual": false,
                        "sexual/minors": false,
                        "violence": true,
                        "violence/graphic": false
                    },
                    "category_scores": {
                        "hate": 0.001,
                        "hate/threatening": 0.001,
                        "harassment": 0.001,
                        "harassment/threatening": 0.001,
                        "self-harm": 0.001,
                        "self-harm/intent": 0.001,
                        "self-harm/instructions": 0.001,
                        "sexual": 0.001,
                        "sexual/minors": 0.001,
                        "violence": 0.9,
                        "violence/graphic": 0.1
                    }
                }
            ]
        }"#;

        let response: ModerationResponse =
            serde_json::from_str(json).expect("Should deserialize multiple results");
        assert_eq!(response.results.len(), 2);
        assert!(!response.results[0].flagged);
        assert!(response.results[1].flagged);
        assert!(response.results[1].categories.violence);
    }
}
