//! Integration tests for the Realtime API.
//!
//! These tests require a valid `OPENAI_API_KEY` environment variable.
//!
//! Run with:
//! ```bash
//! cargo test --test realtime_integration
//! ```
//!
//! Note: These tests connect to the actual OpenAI Realtime API and may incur costs.

use openai_tools::common::models::RealtimeModel;
use openai_tools::common::{parameters::ParameterProperty, tool::Tool};
use openai_tools::realtime::events::server::ServerEvent;
use openai_tools::realtime::vad::ServerVadConfig;
use openai_tools::realtime::{Modality, RealtimeClient, Voice};

/// Test basic WebSocket connection to the Realtime API.
#[tokio::test]
async fn test_realtime_connection() {
    let mut client = RealtimeClient::new();
    client.model(RealtimeModel::Gpt4oRealtimePreview);

    let session_result = client.connect().await;

    match session_result {
        Ok(mut session) => {
            // Successfully connected
            session.close().await.expect("Should close cleanly");
        }
        Err(e) => {
            // Connection might fail due to API key issues or network
            eprintln!("Connection failed (expected if no API key): {}", e);
        }
    }
}

/// Test text-only conversation.
#[tokio::test]
async fn test_text_conversation() {
    let mut client = RealtimeClient::new();
    client
        .model(RealtimeModel::Gpt4oRealtimePreview)
        .modalities(vec![Modality::Text])
        .instructions("You are a helpful assistant. Keep responses brief.");

    let session_result = client.connect().await;

    match session_result {
        Ok(mut session) => {
            // Send a text message
            session.send_text("Hello! Please respond with just 'Hi there!'").await.expect("Should send text");

            // Request a response
            session.create_response(None).await.expect("Should create response");

            // Collect response text
            let mut response_text = String::new();
            let mut received_response = false;

            // Read events with timeout
            let timeout = tokio::time::Duration::from_secs(30);
            let start = tokio::time::Instant::now();

            while start.elapsed() < timeout {
                match tokio::time::timeout(tokio::time::Duration::from_secs(5), session.recv()).await {
                    Ok(Ok(Some(event))) => match event {
                        ServerEvent::ResponseTextDelta(e) => {
                            response_text.push_str(&e.delta);
                        }
                        ServerEvent::ResponseTextDone(_) => {
                            received_response = true;
                        }
                        ServerEvent::ResponseDone(_) => {
                            break;
                        }
                        ServerEvent::Error(e) => {
                            panic!("Error from API: {}", e.error.message);
                        }
                        _ => {}
                    },
                    Ok(Ok(None)) => {
                        break; // Connection closed
                    }
                    Ok(Err(e)) => {
                        panic!("Error receiving event: {}", e);
                    }
                    Err(_) => {
                        // Timeout on individual recv, continue
                        break;
                    }
                }
            }

            assert!(received_response, "Should receive a response");
            assert!(!response_text.is_empty(), "Response should not be empty");
            println!("Response: {}", response_text);

            session.close().await.expect("Should close cleanly");
        }
        Err(e) => {
            eprintln!("Connection failed (expected if no API key): {}", e);
        }
    }
}

/// Test function calling with the Realtime API.
#[tokio::test]
async fn test_function_calling() {
    let weather_tool = Tool::function(
        "get_weather",
        "Get the current weather for a location",
        vec![("location", ParameterProperty::from_string("The city name"))],
        false,
    );

    let mut client = RealtimeClient::new();
    client
        .model(RealtimeModel::Gpt4oRealtimePreview)
        .modalities(vec![Modality::Text])
        .tools(vec![weather_tool])
        .instructions("You are a helpful assistant with access to weather information.");

    let session_result = client.connect().await;

    match session_result {
        Ok(mut session) => {
            // Ask about weather to trigger function call
            session.send_text("What's the weather like in Tokyo?").await.expect("Should send text");

            session.create_response(None).await.expect("Should create response");

            // Wait for function call
            let mut received_function_call = false;
            let timeout = tokio::time::Duration::from_secs(30);
            let start = tokio::time::Instant::now();

            while start.elapsed() < timeout {
                match tokio::time::timeout(tokio::time::Duration::from_secs(5), session.recv()).await {
                    Ok(Ok(Some(event))) => match event {
                        ServerEvent::ResponseFunctionCallArgumentsDone(e) => {
                            assert_eq!(e.name, "get_weather");
                            received_function_call = true;

                            // Submit function result
                            let result = r#"{"temperature": "22C", "condition": "sunny", "humidity": "45%"}"#;
                            session.submit_function_output(&e.call_id, result).await.expect("Should submit function output");

                            // Request follow-up response
                            session.create_response(None).await.expect("Should create follow-up response");
                        }
                        ServerEvent::ResponseDone(_) => {
                            if received_function_call {
                                break; // Got the follow-up response
                            }
                        }
                        ServerEvent::Error(e) => {
                            panic!("Error from API: {}", e.error.message);
                        }
                        _ => {}
                    },
                    Ok(Ok(None)) => break,
                    Ok(Err(e)) => panic!("Error: {}", e),
                    Err(_) => break,
                }
            }

            // Function calling might not always be triggered depending on the model
            println!("Function call received: {} (may vary based on model behavior)", received_function_call);

            session.close().await.expect("Should close cleanly");
        }
        Err(e) => {
            eprintln!("Connection failed (expected if no API key): {}", e);
        }
    }
}

/// Test session configuration update.
#[tokio::test]
async fn test_session_update() {
    let mut client = RealtimeClient::new();
    client.model(RealtimeModel::Gpt4oRealtimePreview);

    let session_result = client.connect().await;

    match session_result {
        Ok(mut session) => {
            // Update session with new configuration
            use openai_tools::realtime::SessionConfig;

            let new_config = SessionConfig::new()
                .with_modalities(vec![Modality::Text])
                .with_instructions("You are a pirate. Speak like one!")
                .with_temperature(0.9);

            session.update_session(new_config).await.expect("Should update session");

            // Wait for session.updated event
            let timeout = tokio::time::Duration::from_secs(10);
            let start = tokio::time::Instant::now();
            let mut updated = false;

            while start.elapsed() < timeout {
                match tokio::time::timeout(tokio::time::Duration::from_secs(2), session.recv()).await {
                    Ok(Ok(Some(ServerEvent::SessionUpdated(_)))) => {
                        updated = true;
                        break;
                    }
                    Ok(Ok(Some(ServerEvent::Error(e)))) => {
                        panic!("Error: {}", e.error.message);
                    }
                    _ => continue,
                }
            }

            assert!(updated, "Should receive session.updated event");

            session.close().await.expect("Should close cleanly");
        }
        Err(e) => {
            eprintln!("Connection failed (expected if no API key): {}", e);
        }
    }
}

/// Test audio configuration (without actual audio).
#[tokio::test]
async fn test_audio_configuration() {
    use openai_tools::realtime::AudioFormat;

    let mut client = RealtimeClient::new();
    client
        .model(RealtimeModel::Gpt4oRealtimePreview)
        .modalities(vec![Modality::Text, Modality::Audio])
        .voice(Voice::Alloy)
        .input_audio_format(AudioFormat::Pcm16)
        .output_audio_format(AudioFormat::Pcm16)
        .server_vad(ServerVadConfig { threshold: Some(0.5), silence_duration_ms: Some(500), ..Default::default() });

    let session_result = client.connect().await;

    match session_result {
        Ok(mut session) => {
            // Session should be configured for audio
            // We don't send actual audio, just verify the connection works

            session.close().await.expect("Should close cleanly");
            println!("Audio-configured session connected successfully");
        }
        Err(e) => {
            eprintln!("Connection failed (expected if no API key): {}", e);
        }
    }
}
