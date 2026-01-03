![Crates.io Version](https://img.shields.io/crates/v/openai-tools?style=flat-square&color=blue)

# OpenAI Tools

API Wrapper for OpenAI API.

<img src="../LOGO.png" alt="LOGO" width="150" height="150"/>

## Installation

To start using the `openai-tools`, add it to your projects's dependencies in the `Cargo.toml' file:

```bash
cargo add openai-tools
```

API key is necessary to access OpenAI API.
Set it in the `.env` file:

```text
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Then, import the necesarry modules in your code:

```rust
use openai_tools::chat::ChatCompletion;
use openai_tools::responses::Responses;
use openai_tools::embedding::Embedding;
use openai_tools::realtime::RealtimeClient;
```

# Features

| Feature Name                   | [Chat Completion](src/chat/mod.rs) | [Responses](src/responses/mod.rs) | [Embedding](src/embedding/mod.rs) | [Realtime](src/realtime/mod.rs) | Images | Audio | Eval |
|--------------------------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Basic Features                 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Structured Output              | ✅ | ✅ | - | - | - | - | - |
| Function Calling / MCP Tools   | ✅ | ✅ | - | ✅ | - | - | - |
| Image Input                    | ✅ | ✅ | - | - | - | - | - |
| Audio Input/Output             | - | - | - | ✅ | - | - | - |
| Voice Activity Detection (VAD) | - | - | - | ✅ | - | - | - |
| WebSocket Streaming            | - | - | - | ✅ | - | - | - |

✅: Implemented
🔧: In Progress
❌: Not yet

## Realtime API

The Realtime API enables real-time audio and text communication with GPT-4o models through WebSocket connections.

### Basic Usage

```rust
use openai_tools::realtime::{RealtimeClient, Modality, Voice};
use openai_tools::realtime::events::server::ServerEvent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = RealtimeClient::new();
    client
        .model("gpt-4o-realtime-preview")
        .modalities(vec![Modality::Text, Modality::Audio])
        .voice(Voice::Alloy)
        .instructions("You are a helpful assistant.");

    let mut session = client.connect().await?;

    // Send a text message
    session.send_text("Hello!").await?;
    session.create_response(None).await?;

    // Process events
    while let Some(event) = session.recv().await? {
        match event {
            ServerEvent::ResponseTextDelta(e) => print!("{}", e.delta),
            ServerEvent::ResponseDone(_) => break,
            _ => {}
        }
    }

    session.close().await?;
    Ok(())
}
```

### Function Calling

```rust
use openai_tools::realtime::{RealtimeClient, Modality, RealtimeTool};
use openai_tools::realtime::events::server::ServerEvent;
use openai_tools::common::parameters::ParameterProperty;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = RealtimeClient::new();

    // Use RealtimeTool for native Realtime API format
    let weather_tool = RealtimeTool::function(
        "get_weather",
        "Get weather for a location",
        vec![("location", ParameterProperty::from_string("City name"))],
    );

    client
        .modalities(vec![Modality::Text])
        .realtime_tools(vec![weather_tool]);

    let mut session = client.connect().await?;

    session.send_text("What's the weather in Tokyo?").await?;
    session.create_response(None).await?;

    while let Some(event) = session.recv().await? {
        match event {
            ServerEvent::ResponseFunctionCallArgumentsDone(e) => {
                let result = r#"{"temp": "22C", "condition": "sunny"}"#;
                session.submit_function_output(&e.call_id, result).await?;
                session.create_response(None).await?;
            }
            ServerEvent::ResponseDone(_) => break,
            _ => {}
        }
    }

    session.close().await?;
    Ok(())
}
```

> **Note**: You can also use `Tool::function()` from the common module - it will be automatically converted to `RealtimeTool` format.
