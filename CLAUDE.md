# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Rust library (`openai-tools`) providing a type-safe API wrapper for OpenAI's APIs. Published on crates.io.

## Build & Test Commands

```bash
# Build
cargo build
cargo make build-tools  # release build

# Run unit tests only (no API key required)
cargo test --lib
cargo make test-unit

# Run integration tests (requires OPENAI_API_KEY)
cargo make test-integration
# or individual tests:
cargo test --test chat_integration
cargo test --test responses_integration
cargo test --test embedding_integration
cargo test --test realtime_integration

# Run all tests (unit + integration)
cargo make test

# Run a specific test
cargo test test_chat_completion

# Run tests with logging output
RUST_LOG=info cargo test -- --nocapture

# Check without building
cargo check

# Format code
cargo fmt
cargo make format-all  # includes clippy and taplo

# Lint
cargo clippy
```

## Test Structure

- **Unit tests** (`src/**/mod.rs`): Test builders, serialization, and validation without API calls
- **Integration tests** (`tests/*_integration.rs`): Test actual API calls, require `OPENAI_API_KEY`

## Architecture

### Workspace Structure

```
rs-openai-tools/
├── Cargo.toml          # Workspace root
└── openai-tools/       # Main library crate
    └── src/
        ├── lib.rs      # Crate entry point
        ├── chat/       # Chat Completions API
        ├── responses/  # Responses API (assistant-style)
        ├── embedding/  # Embeddings API
        ├── realtime/   # Realtime API (WebSocket)
        └── common/     # Shared types
```

### Module Responsibilities

- **`chat/`**: Chat Completions API (`/v1/chat/completions`)
  - `request.rs`: `ChatCompletion` builder for requests
  - `response.rs`: Response types

- **`responses/`**: Responses API (`/v1/responses`) - newer assistant-style API
  - `request.rs`: `Responses` builder with multi-modal support
  - `response.rs`: Response types

- **`embedding/`**: Embeddings API (`/v1/embeddings`)
  - `request.rs`: `Embedding` builder
  - `response.rs`: Vector response types (1D/2D/3D)

- **`realtime/`**: Realtime API (WebSocket, `wss://api.openai.com/v1/realtime`)
  - `client.rs`: `RealtimeClient` builder and `RealtimeSession` handle
  - `session.rs`: `SessionConfig`, `Modality`, `ToolChoice`
  - `audio.rs`: `AudioFormat`, `Voice`, `TranscriptionModel`
  - `vad.rs`: `TurnDetection`, `ServerVadConfig`, `SemanticVadConfig`
  - `conversation.rs`: `ConversationItem`, `ContentPart`
  - `events/client.rs`: Client-to-server events (9 types)
  - `events/server.rs`: Server-to-client events (28 types)
  - `stream.rs`: `EventHandler` for callback-based processing

- **`common/`**: Shared types across all APIs
  - `message.rs`: `Message`, `Content`, `ToolCall`
  - `role.rs`: `Role` enum (User, Assistant, System, Tool)
  - `tool.rs`: `Tool` definition for function calling
  - `parameters.rs`: `ParameterProperty` for tool parameters
  - `structured_output.rs`: `Schema` for JSON schema responses
  - `errors.rs`: `OpenAIToolError` error type
  - `usage.rs`: Token usage tracking

### Key Patterns

**Builder Pattern**: All API clients use builder-style configuration:
```rust
let mut chat = ChatCompletion::new();
chat.model_id("gpt-4o-mini")
    .messages(messages)
    .temperature(0.7)
    .chat()
    .await?;
```

**Schema for Structured Output**: Two factory methods depending on API:
- `Schema::chat_json_schema("name")` - for Chat Completions
- `Schema::responses_json_schema("name")` - for Responses API

**Tool Definition**: Function tools follow this pattern:
```rust
Tool::function(
    "name",
    "description",
    vec![("param", ParameterProperty::from_string("desc"))],
    false, // strict mode
)
```

**Realtime API**: WebSocket-based real-time communication:
```rust
let mut client = RealtimeClient::new();
client
    .model("gpt-4o-realtime-preview")
    .modalities(vec![Modality::Text, Modality::Audio])
    .voice(Voice::Alloy)
    .server_vad(ServerVadConfig::default());

let mut session = client.connect().await?;
session.send_text("Hello!").await?;
session.create_response(None).await?;

while let Some(event) = session.recv().await? {
    match event {
        ServerEvent::ResponseTextDelta(e) => print!("{}", e.delta),
        ServerEvent::ResponseDone(_) => break,
        _ => {}
    }
}
session.close().await?;
```

## Environment Setup

Requires `OPENAI_API_KEY` environment variable or in `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Feature Status

| Feature | Chat | Responses | Embedding | Realtime |
|---------|:----:|:---------:|:---------:|:--------:|
| Basic   | ✅   | ✅        | ✅        | ✅       |
| Structured Output | ✅ | ✅ | - | - |
| Function Calling  | ✅ | ✅ | - | ✅ |
| Image Input       | ✅ | ✅ | - | - |
| Audio Input/Output | - | - | - | ✅ |
| VAD (Voice Activity Detection) | - | - | - | ✅ |
| WebSocket Streaming | - | - | - | ✅ |
