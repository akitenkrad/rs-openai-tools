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
cargo test --test models_integration
cargo test --test models_by_version_integration
cargo test --test files_integration
cargo test --test moderations_integration
cargo test --test images_integration
cargo test --test audio_integration
cargo test --test batch_integration
cargo test --test fine_tuning_integration
cargo test --test conversations_integration

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
        ├── conversations/ # Conversations API
        ├── embedding/  # Embeddings API
        ├── realtime/   # Realtime API (WebSocket)
        ├── models/     # Models API
        ├── files/      # Files API
        ├── moderations/ # Moderations API
        ├── images/     # Images API (DALL-E)
        ├── audio/      # Audio API (TTS, STT)
        ├── batch/      # Batch API
        ├── fine_tuning/ # Fine-tuning API
        └── common/     # Shared types
```

### Module Responsibilities

- **`chat/`**: Chat Completions API (`/v1/chat/completions`)
  - `request.rs`: `ChatCompletion` builder for requests
  - `response.rs`: Response types

- **`responses/`**: Responses API (`/v1/responses`) - newer assistant-style API
  - `request.rs`: `Responses` builder with multi-modal support, `Include`, `ReasoningEffort`, `ReasoningSummary`, `Reasoning`, `Truncation`
  - `response.rs`: Response types

- **`conversations/`**: Conversations API (`/v1/conversations`) - long-running conversation management
  - `request.rs`: `Conversations` client for create, retrieve, update, delete, items
  - `response.rs`: `Conversation`, `ConversationItem`, `InputItem`

- **`embedding/`**: Embeddings API (`/v1/embeddings`)
  - `request.rs`: `Embedding` builder
  - `response.rs`: Vector response types (1D/2D/3D)

- **`realtime/`**: Realtime API (WebSocket, `wss://api.openai.com/v1/realtime`)
  - `client.rs`: `RealtimeClient` builder and `RealtimeSession` handle
  - `session.rs`: `SessionConfig`, `Modality`, `ToolChoice`
  - `audio.rs`: `AudioFormat`, `Voice` (Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse), `TranscriptionModel` (Whisper1, Gpt4oTranscribe, Gpt4oMiniTranscribe, Gpt4oTranscribeDiarize), `InputAudioTranscription`, `InputAudioNoiseReduction`
  - `vad.rs`: `TurnDetection`, `ServerVadConfig`, `SemanticVadConfig`
  - `conversation.rs`: `ConversationItem`, `ContentPart`
  - `events/client.rs`: Client-to-server events (9 types)
  - `events/server.rs`: Server-to-client events (28 types)
  - `stream.rs`: `EventHandler` for callback-based processing

- **`models/`**: Models API (`/v1/models`)
  - `request.rs`: `Models` client for list, retrieve, delete
  - `response.rs`: `Model`, `ModelsListResponse`, `DeleteResponse`

- **`files/`**: Files API (`/v1/files`)
  - `request.rs`: `Files` client with multipart upload support
  - `response.rs`: `File`, `FileListResponse`, `DeleteResponse`

- **`moderations/`**: Moderations API (`/v1/moderations`)
  - `request.rs`: `Moderations` client for content policy classification
  - `response.rs`: `ModerationResponse`, `ModerationCategories`, `ModerationCategoryScores`

- **`images/`**: Images API (`/v1/images`)
  - `request.rs`: `Images` client for generate, edit, variation
  - `response.rs`: `ImageResponse`, `ImageData`

- **`audio/`**: Audio API (`/v1/audio`)
  - `request.rs`: `Audio` client for TTS, transcription, translation; `TtsModel` (Tts1, Tts1Hd, Gpt4oMiniTts), `Voice` (Alloy, Ash, Coral, Echo, Fable, Onyx, Nova, Sage, Shimmer), `AudioFormat`
  - `response.rs`: `TranscriptionResponse`, `Word`, `Segment`

- **`batch/`**: Batch API (`/v1/batches`)
  - `request.rs`: `Batches` client for create, list, retrieve, cancel
  - `response.rs`: `BatchObject`, `BatchStatus`, `RequestCounts`

- **`fine_tuning/`**: Fine-tuning API (`/v1/fine_tuning/jobs`)
  - `request.rs`: `FineTuning` client for jobs, events, checkpoints
  - `response.rs`: `FineTuningJob`, `FineTuningEvent`, `FineTuningCheckpoint`

- **`common/`**: Shared types across all APIs
  - `client.rs`: HTTP client utilities with timeout configuration (`create_http_client`)
  - `message.rs`: `Message`, `Content`, `ToolCall`
  - `role.rs`: `Role` enum (User, Assistant, System, Tool)
  - `models.rs`: Type-safe model enums (`ChatModel`, `EmbeddingModel`, `RealtimeModel`, `FineTuningModel`) and `ParameterSupport`/`ParameterRestriction` for model parameter validation
  - `tool.rs`: `Tool` definition for function calling
  - `function.rs`: `Function` struct (internal function metadata used by `Tool`)
  - `parameters.rs`: `ParameterProperty` for tool parameters
  - `structured_output.rs`: `Schema` for JSON schema responses
  - `errors.rs`: `OpenAIToolError` error type
  - `usage.rs`: Token usage tracking

### Key Patterns

**Builder Pattern**: All API clients use builder-style configuration:
```rust
use openai_tools::common::models::ChatModel;

let mut chat = ChatCompletion::new();
chat.model(ChatModel::Gpt4oMini)  // Type-safe model selection
    .messages(messages)
    .temperature(0.7)
    .chat()
    .await?;
```

**Timeout Configuration**: All HTTP-based API clients support optional request timeouts:
```rust
use std::time::Duration;
use openai_tools::common::models::ChatModel;

// Builder pattern clients
let mut chat = ChatCompletion::new();
chat.model(ChatModel::Gpt4oMini)
    .timeout(Duration::from_secs(30))  // Set 30 second timeout
    .messages(messages)
    .chat()
    .await?;

// Service pattern clients
let mut files = Files::new()?;
files.timeout(Duration::from_secs(120));  // Longer timeout for file uploads
let file = files.upload_path("data.jsonl", FilePurpose::FineTune).await?;
```

Timeout is optional - if not set, requests have no timeout limit (default reqwest behavior).

**Type-Safe Model Selection**: All APIs use enum-based model selection for compile-time validation:

| Enum | API | Available Variants |
|------|-----|-------------------|
| `ChatModel` | Chat, Responses | **GPT-5**: `Gpt5_2`, `Gpt5_2ChatLatest`, `Gpt5_2Pro`, `Gpt5_1`, `Gpt5_1ChatLatest`, `Gpt5_1CodexMax`, `Gpt5Mini` / **GPT-4.1**: `Gpt4_1`, `Gpt4_1Mini`, `Gpt4_1Nano` / **GPT-4o**: `Gpt4o`, `Gpt4oMini`, `Gpt4oAudioPreview` / **GPT-4/3.5**: `Gpt4Turbo`, `Gpt4`, `Gpt3_5Turbo` / **Reasoning**: `O1`, `O1Pro`, `O3`, `O3Mini`, `O4Mini` / `Custom(String)` |
| `EmbeddingModel` | Embedding | `TextEmbedding3Small`, `TextEmbedding3Large`, `TextEmbeddingAda002` |
| `RealtimeModel` | Realtime | `Gpt4oRealtimePreview`, `Gpt4oMiniRealtimePreview`, `Custom(String)` |
| `FineTuningModel` | Fine-tuning | `Gpt41_2025_04_14`, `Gpt41Mini_2025_04_14`, `Gpt41Nano_2025_04_14`, `Gpt4oMini_2024_07_18`, `Gpt4o_2024_08_06`, `Gpt4_0613`, `Gpt35Turbo_0125`, etc. |

```rust
use openai_tools::common::models::{ChatModel, EmbeddingModel, RealtimeModel, FineTuningModel};

// Chat/Responses API - GPT-5 series (latest flagship)
chat.model(ChatModel::Gpt5_2);          // GPT-5.2 Thinking - coding & agentic tasks
responses.model(ChatModel::Gpt5_2Pro);  // GPT-5.2 Pro - most capable (Responses API only)
chat.model(ChatModel::Gpt5_1);          // GPT-5.1 - configurable reasoning

// Chat/Responses API - Other models
chat.model(ChatModel::Gpt4oMini);       // Cost-effective
responses.model(ChatModel::O3Mini);     // Reasoning model

// Embedding API
embedding.model(EmbeddingModel::TextEmbedding3Small);

// Realtime API
client.model(RealtimeModel::Gpt4oRealtimePreview);

// Fine-tuning API
CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-id");

// Custom models (for fine-tuned models or new models)
chat.model(ChatModel::custom("ft:gpt-4o-mini:my-org::abc123"));

// Check if model is a reasoning model (GPT-5 series and o-series)
if ChatModel::Gpt5_2.is_reasoning_model() {
    // Handle reasoning model restrictions
}
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
use openai_tools::common::models::RealtimeModel;

let mut client = RealtimeClient::new();
client
    .model(RealtimeModel::Gpt4oRealtimePreview)  // Type-safe model selection
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

**Conversations API**: Manage long-running conversations with the Responses API:
```rust
use openai_tools::conversations::request::Conversations;
use openai_tools::conversations::response::InputItem;
use openai_tools::common::models::ChatModel;
use std::collections::HashMap;

let conversations = Conversations::new()?;

// Create a conversation with metadata
let mut metadata = HashMap::new();
metadata.insert("user_id".to_string(), "user123".to_string());

let conv = conversations.create(Some(metadata), None).await?;
println!("Created conversation: {}", conv.id);

// Add items to the conversation
let items = vec![InputItem::user_message("Hello!")];
conversations.create_items(&conv.id, items).await?;

// List conversation items
let items = conversations.list_items(&conv.id, Some(10), None, None, None).await?;
for item in &items.data {
    println!("Item: {} ({})", item.id, item.item_type);
}

// Use with Responses API
let mut client = Responses::new();
client.model(ChatModel::Gpt4o).conversation(&conv.id).str_message("How are you?").complete().await?;

// Delete conversation when done
conversations.delete(&conv.id).await?;
```

**Models API**: List and retrieve available models:
```rust
let models = Models::new()?;

// List all models
let response = models.list().await?;
for model in &response.data {
    println!("{}: owned by {}", model.id, model.owned_by);
}

// Retrieve a specific model
let model = models.retrieve("gpt-4o-mini").await?;
```

**Files API**: Upload, manage, and retrieve files:
```rust
let files = Files::new()?;

// Upload a file for fine-tuning
let file = files.upload_path("training.jsonl", FilePurpose::FineTune).await?;

// Or upload from bytes
let content = b"jsonl content here";
let file = files.upload_bytes(content, "data.jsonl", FilePurpose::Batch).await?;

// List files
let response = files.list(Some(FilePurpose::FineTune)).await?;

// Get file content
let content = files.content(&file.id).await?;

// Delete file
files.delete(&file.id).await?;
```

**Moderations API**: Check content for policy violations:
```rust
let moderations = Moderations::new()?;

// Check a single text
let response = moderations.moderate_text("Hello, world!", None).await?;
if response.results[0].flagged {
    println!("Content was flagged!");
}

// Check multiple texts at once
let texts = vec!["Text 1".to_string(), "Text 2".to_string()];
let response = moderations.moderate_texts(texts, None).await?;
```

**Images API**: Generate images with DALL-E:
```rust
let images = Images::new()?;

// Generate an image
let options = GenerateOptions {
    model: Some(ImageModel::DallE3),
    size: Some(ImageSize::Size1024x1024),
    quality: Some(ImageQuality::Hd),
    ..Default::default()
};
let response = images.generate("A sunset over mountains", options).await?;
println!("Image URL: {:?}", response.data[0].url);

// Get base64-encoded image
let options = GenerateOptions {
    response_format: Some(ResponseFormat::B64Json),
    ..Default::default()
};
let response = images.generate("A cute robot", options).await?;
let bytes = response.data[0].as_bytes().unwrap()?;
std::fs::write("robot.png", bytes)?;
```

**Audio API**: Text-to-speech and transcription:
```rust
let audio = Audio::new()?;

// Text-to-speech
let options = TtsOptions {
    model: TtsModel::Tts1Hd,
    voice: Voice::Nova,
    ..Default::default()
};
let bytes = audio.text_to_speech("Hello!", options).await?;
std::fs::write("hello.mp3", bytes)?;

// Transcribe audio
let options = TranscribeOptions {
    language: Some("en".to_string()),
    ..Default::default()
};
let response = audio.transcribe("audio.mp3", options).await?;
println!("Transcript: {}", response.text);

// Translate audio to English
let response = audio.translate("foreign_audio.mp3", TranslateOptions::default()).await?;
```

**Batch API**: Process large volumes of requests asynchronously with 50% cost savings:
```rust
let batches = Batches::new()?;

// Create a batch job (input file must be uploaded via Files API with purpose "batch")
let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
let batch = batches.create(request).await?;

// Check batch status
let batch = batches.retrieve(&batch.id).await?;
match batch.status {
    BatchStatus::Completed => {
        println!("Output file: {:?}", batch.output_file_id);
    }
    _ => println!("Status: {:?}", batch.status),
}

// List all batches
let response = batches.list(Some(20), None).await?;
```

**Fine-tuning API**: Customize models with your training data:
```rust
use openai_tools::common::models::FineTuningModel;

let fine_tuning = FineTuning::new()?;

// Create a fine-tuning job
let hyperparams = Hyperparameters {
    n_epochs: Some(3),
    ..Default::default()
};
let request = CreateFineTuningJobRequest::new(FineTuningModel::Gpt4oMini_2024_07_18, "file-abc123")
    .with_suffix("my-model")
    .with_supervised_method(Some(hyperparams));

let job = fine_tuning.create(request).await?;

// Check job status
let job = fine_tuning.retrieve(&job.id).await?;
if job.status == FineTuningJobStatus::Succeeded {
    println!("Model: {:?}", job.fine_tuned_model);
}

// List training events
let events = fine_tuning.list_events(&job.id, Some(10), None).await?;
for event in &events.data {
    println!("[{}] {}", event.level, event.message);
}
```

## Environment Setup

Requires `OPENAI_API_KEY` environment variable or in `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Feature Status

| Feature | Chat | Responses | Conversations | Embedding | Realtime | Models | Files | Moderations | Images | Audio | Batch | Fine-tuning |
|---------|:----:|:---------:|:-------------:|:---------:|:--------:|:------:|:-----:|:-----------:|:------:|:-----:|:-----:|:-----------:|
| Basic   | ✅   | ✅        | ✅            | ✅        | ✅       | ✅     | ✅    | ✅          | ✅     | ✅    | ✅    | ✅          |
| Structured Output | ✅ | ✅ | - | - | - | - | - | - | - | - | - | - |
| Function Calling  | ✅ | ✅ | - | - | ✅ | - | - | - | - | - | - | - |
| Image Input       | ✅ | ✅ | - | - | - | - | - | - | - | - | - | - |
| Reasoning Config  | - | ✅ | - | - | - | - | - | - | - | - | - | - |
| Audio Input/Output | - | - | - | - | ✅ | - | - | - | - | ✅ | - | - |
| VAD (Voice Activity Detection) | - | - | - | - | ✅ | - | - | - | - | - | - | - |
| WebSocket Streaming | - | - | - | - | ✅ | - | - | - | - | - | - | - |
| Multipart Upload | - | - | - | - | - | - | ✅ | - | ✅ | ✅ | - | - |

**Responses API Features:**
- Multi-modal input support (text, images)
- Configurable reasoning with `ReasoningEffort` (none, minimal, low, medium, high, xhigh) and `ReasoningSummary` (auto, concise, detailed)
- Input truncation behavior with `Truncation` (Auto, Disabled)
- Response data inclusion with `Include`:
  - `WebSearchCall` - web search results
  - `CodeInterpreterCall` - code execution outputs
  - `FileSearchCall` - file search results
  - `ImageUrlInInputMessages` - image URLs from input
  - `ImageUrlInComputerCallOutput` - computer call output images
  - `LogprobsInOutput` - token log probabilities
  - `ReasoningEncryptedContent` - encrypted reasoning content
- Parallel tool calls support
- Conversation integration
- Metadata tracking

**Conversations API Features:**
- Create conversations with optional metadata and initial items
- Retrieve, update, and delete conversations
- Add items (messages, tool calls, etc.) to conversations
- List conversation items with pagination
- Integration with Responses API for multi-turn interactions
- Metadata support (key-value pairs for tracking)

**Models API Features:**
- List all available models
- Retrieve specific model details
- Delete fine-tuned models

**Files API Features:**
- Upload files (from path or bytes)
- List files (with purpose filter)
- Retrieve file details
- Delete files
- Get file content

**Moderations API Features:**
- Single text moderation
- Batch text moderation
- Multiple models (omni-moderation, text-moderation)
- Detailed category scores

**Images API Features:**
- Image generation (DALL-E 2, DALL-E 3, GPT Image)
- Image editing with masks
- Image variations (DALL-E 2 only)
- URL or base64 response format
- Quality and style options

**Realtime API Features:**
- WebSocket-based real-time communication
- Text and audio modalities
- Voice options: Alloy, Ash, Ballad, Coral, Echo, Sage, Shimmer, Verse
- Transcription models: Whisper1, Gpt4oTranscribe, Gpt4oMiniTranscribe, Gpt4oTranscribeDiarize
- Voice Activity Detection (VAD): Server VAD and Semantic VAD
- Audio formats: PCM16, G711Ulaw, G711Alaw
- Input audio transcription with language hints
- Noise reduction (near-field, far-field)
- Function calling support

**Audio API Features:**
- Text-to-speech (TTS) with multiple voices: Alloy, Ash, Coral, Echo, Fable, Onyx, Nova, Sage, Shimmer
- TTS models: Tts1, Tts1Hd, Gpt4oMiniTts
- Audio transcription (Whisper, GPT-4o)
- Audio translation to English
- Multiple audio formats (MP3, WAV, FLAC, Opus, AAC, PCM)
- Word and segment timestamps

**Batch API Features:**
- Create async batch jobs with 50% cost savings
- Support for Chat Completions, Embeddings, Responses, Moderations endpoints
- Retrieve batch status and progress
- List all batches with pagination
- Cancel in-progress batches
- 24-hour completion window

**Fine-tuning API Features:**
- Create fine-tuning jobs (Supervised, DPO methods)
- Configurable hyperparameters (epochs, batch size, learning rate)
- Monitor training with events
- Access training checkpoints
- List and retrieve job details
- Cancel in-progress jobs
- Support for GPT-4o-mini, GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo

## Model-Specific Parameter Restrictions

### Reasoning Models (GPT-5 Series, o1, o3, o4 Series)

OpenAI's reasoning models (GPT-5 series: `gpt-5.2`, `gpt-5.1`, `gpt-5-mini`, and o-series: `o1`, `o3`, `o3-mini`, `o4-mini`, etc.) have specific parameter restrictions. This library automatically handles these restrictions by ignoring unsupported parameters and logging warnings.

#### GPT-5 Series Reasoning Configuration

GPT-5 models support configurable reasoning effort via the `reasoning` parameter (Responses API only):

```rust
use openai_tools::responses::request::{Responses, Reasoning, ReasoningEffort, ReasoningSummary};
use openai_tools::common::models::ChatModel;

let mut responses = Responses::new();
responses.model(ChatModel::Gpt5_2)
    .reasoning(Reasoning {
        effort: Some(ReasoningEffort::High),    // none, minimal, low, medium, high, xhigh
        summary: Some(ReasoningSummary::Auto),  // auto, concise, detailed
    })
    .str_message("Solve this complex problem...")
    .complete()
    .await?;
```

**ReasoningEffort levels:**
| Level | Description | Supported Models |
|-------|-------------|------------------|
| `None` | No reasoning tokens (fastest) | GPT-5.1, GPT-5.2 |
| `Minimal` | Very few reasoning tokens | GPT-5 Mini |
| `Low` | Light reasoning | GPT-5.1, GPT-5.2, o-series |
| `Medium` | Balanced reasoning | All reasoning models |
| `High` | Deep reasoning | All reasoning models |
| `Xhigh` | Maximum reasoning | GPT-5.2 Pro, GPT-5.1 Codex Max |

#### Chat Completions API Parameter Restrictions

| Parameter | Restriction | Library Behavior |
|-----------|-------------|------------------|
| `temperature` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_p` | Only `1.0` supported | Not implemented in Chat API |
| `frequency_penalty` | Only `0` supported | Values ≠ 0 are ignored with warning |
| `presence_penalty` | Only `0` supported | Values ≠ 0 are ignored with warning |
| `logprobs` | Not supported | Ignored with warning if set |
| `top_logprobs` | Not supported | Ignored with warning if set |
| `logit_bias` | Not supported | Ignored with warning if set |
| `n` | Only `1` supported | Values ≠ 1 are ignored with warning |

#### Responses API Parameter Restrictions

| Parameter | Restriction | Library Behavior |
|-----------|-------------|------------------|
| `temperature` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_p` | Only `1.0` supported | Values ≠ 1.0 are ignored with warning |
| `top_logprobs` | Not supported | Ignored with warning if set |

#### Example Usage with Reasoning Models

```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::responses::request::Responses;
use openai_tools::common::models::ChatModel;

// Chat API with reasoning model
let mut chat = ChatCompletion::new();
chat.model(ChatModel::O1)           // Type-safe reasoning model selection
    .temperature(0.3)               // Warning: ignored, using default 1.0
    .frequency_penalty(0.5)         // Warning: ignored, using default 0
    .messages(messages)
    .chat()
    .await?;  // Request succeeds without API error

// Responses API with reasoning model
let mut responses = Responses::new();
responses.model(ChatModel::O3Mini)  // Type-safe reasoning model selection
    .temperature(0.7)               // Warning: ignored, using default 1.0
    .top_p(0.9)                     // Warning: ignored, using default 1.0
    .str_message("Hello!")
    .complete()
    .await?;  // Request succeeds without API error
```

#### Warning Log Examples

When using unsupported parameters with reasoning models, warnings are logged via `tracing::warn!`:

```
WARN: Reasoning model 'o1-preview' does not support custom temperature. Ignoring temperature=0.3 and using default (1.0).
WARN: Reasoning model 'o1-preview' does not support frequency_penalty. Ignoring frequency_penalty=0.5 and using default (0).
WARN: Reasoning model 'o3-mini' does not support top_p. Ignoring top_p=0.9 and using default (1.0).
```

### Standard Models (GPT-4o, GPT-4, GPT-3.5, etc.)

Standard models support all available parameters without restrictions:

| API | Supported Parameters |
|-----|---------------------|
| Chat Completions | `temperature`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `logit_bias`, `n`, `max_completion_tokens`, `modalities`, `store`, `tools`, `response_format` |
| Responses | `temperature`, `top_p`, `top_logprobs`, `max_output_tokens`, `max_tool_calls`, `parallel_tool_calls`, `truncation`, `reasoning`, `tools`, `structured_output`, `metadata`, `include`, `background`, `conversation`, `store`, `stream` |

### API Parameter Comparison

| Parameter | Chat API | Responses API | Notes |
|-----------|:--------:|:-------------:|-------|
| `temperature` | ✅ | ✅ | Controls randomness (0.0-2.0) |
| `top_p` | ❌ | ✅ | Nucleus sampling |
| `frequency_penalty` | ✅ | ❌ | Reduces repetition |
| `presence_penalty` | ✅ | ❌ | Encourages new topics |
| `logprobs` | ✅ | ❌ | Token probabilities |
| `top_logprobs` | ✅ | ✅ | Top N probabilities |
| `logit_bias` | ✅ | ❌ | Token probability adjustment |
| `n` | ✅ | ❌ | Number of completions |
| `max_tokens` | `max_completion_tokens` | `max_output_tokens` | Different parameter names |
| `tools` | ✅ | ✅ | Function calling |
| `structured_output` | `response_format` | ✅ | JSON schema output |
| `reasoning` | ❌ | ✅ | Reasoning effort/summary |
| `truncation` | ❌ | ✅ | Input truncation behavior |
| `parallel_tool_calls` | ❌ | ✅ | Concurrent tool execution |
| `metadata` | ❌ | ✅ | Request tracking |

### References

- [OpenAI Reasoning Models Guide](https://platform.openai.com/docs/guides/reasoning)
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
