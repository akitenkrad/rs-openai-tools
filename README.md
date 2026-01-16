![Crates.io Version](https://img.shields.io/crates/v/openai-tools?style=flat-square&color=blue)

# OpenAI Tools

API Wrapper for OpenAI API.

<img src="LOGO.png" alt="LOGO" width="150" height="150"/>

## Installation

To start using the `openai-tools`, add it to your projects's dependencies in the `Cargo.toml' file:

```bash
cargo add openai-tools
```

## Environment Setup

### OpenAI API

Set the API key in the `.env` file:

```text
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Azure OpenAI API

Set Azure-specific environment variables:

```text
AZURE_OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
AZURE_OPENAI_RESOURCE_NAME = "my-resource"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-deployment"
```

### Provider Detection

All API clients support multiple ways to configure authentication:

```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::auth::{AuthProvider, AzureAuth};

// OpenAI (default)
let chat = ChatCompletion::new();

// Azure
let chat = ChatCompletion::azure()?;

// Auto-detect provider from environment variables
let chat = ChatCompletion::detect_provider()?;

// URL-based detection (auto-detects provider from URL pattern)
let chat = ChatCompletion::with_url(
    "https://my-resource.openai.azure.com",
    "azure-key",
    Some("gpt-4o-deployment")
)?;

// OpenAI-compatible APIs (Ollama, vLLM, LocalAI, etc.)
let chat = ChatCompletion::with_url(
    "http://localhost:11434/v1",
    "ollama",
    None
)?;
```

## Modules

Import the necessary modules in your code:

```rust
use openai_tools::chat::ChatCompletion;
use openai_tools::responses::Responses;
use openai_tools::embedding::Embedding;
use openai_tools::realtime::RealtimeClient;
use openai_tools::conversations::Conversations;
use openai_tools::models::Models;
use openai_tools::files::Files;
use openai_tools::moderations::Moderations;
use openai_tools::images::Images;
use openai_tools::audio::Audio;
use openai_tools::batch::Batches;
use openai_tools::fine_tuning::FineTuning;
```

# Features

| Feature | Chat | Responses | Conversations | Embedding | Realtime | Models | Files | Moderations | Images | Audio | Batch | Fine-tuning |
|---------|:----:|:---------:|:-------------:|:---------:|:--------:|:------:|:-----:|:-----------:|:------:|:-----:|:-----:|:-----------:|
| Basic   | ✅   | ✅        | ✅            | ✅        | ✅       | ✅     | ✅    | ✅          | ✅     | ✅    | ✅    | ✅          |
| Structured Output | ✅ | ✅ | - | - | - | - | - | - | - | - | - | - |
| Function Calling  | ✅ | ✅ | - | - | ✅ | - | - | - | - | - | - | - |
| Image Input       | ✅ | ✅ | - | - | - | - | - | - | - | - | - | - |
| Audio Input/Output | - | - | - | - | ✅ | - | - | - | - | ✅ | - | - |
| VAD | - | - | - | - | ✅ | - | - | - | - | - | - | - |
| WebSocket | - | - | - | - | ✅ | - | - | - | - | - | - | - |
| Multipart Upload | - | - | - | - | - | - | ✅ | - | ✅ | ✅ | - | - |

## Chat Completions API

```rust
use openai_tools::chat::request::ChatCompletion;
use openai_tools::common::message::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let messages = vec![Message::from_string("user", "Hello!")];

    let mut chat = ChatCompletion::new();
    let response = chat
        .model_id("gpt-4o-mini")
        .messages(messages)
        .temperature(0.7)
        .chat()
        .await?;

    println!("{}", response.choices[0].message.content);
    Ok(())
}
```

## Responses API

```rust
use openai_tools::responses::request::Responses;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = Responses::new();
    let response = client
        .model_id("gpt-4o")
        .str_message("What is the capital of France?")
        .complete()
        .await?;

    println!("{}", response.output_text());
    Ok(())
}
```

## Conversations API

Manage long-running conversations with the Responses API:

```rust
use openai_tools::conversations::request::Conversations;
use openai_tools::conversations::response::InputItem;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Delete conversation when done
    conversations.delete(&conv.id).await?;

    Ok(())
}
```

## Realtime API

Real-time audio and text communication with GPT-4o models through WebSocket:

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

## Models API

List and retrieve available models:

```rust
use openai_tools::models::request::Models;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = Models::new()?;

    // List all models
    let response = models.list().await?;
    for model in &response.data {
        println!("{}: owned by {}", model.id, model.owned_by);
    }

    // Retrieve a specific model
    let model = models.retrieve("gpt-4o-mini").await?;
    println!("Model: {}", model.id);

    Ok(())
}
```

## Files API

Upload, manage, and retrieve files:

```rust
use openai_tools::files::request::Files;
use openai_tools::files::response::FilePurpose;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let files = Files::new()?;

    // Upload a file for fine-tuning
    let file = files.upload_path("training.jsonl", FilePurpose::FineTune).await?;
    println!("Uploaded: {}", file.id);

    // List files
    let response = files.list(None).await?;
    for file in &response.data {
        println!("{}: {} bytes", file.filename, file.bytes);
    }

    // Delete file
    files.delete(&file.id).await?;

    Ok(())
}
```

## Moderations API

Check content for policy violations:

```rust
use openai_tools::moderations::request::Moderations;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let moderations = Moderations::new()?;

    // Check a single text
    let response = moderations.moderate_text("Hello, world!", None).await?;
    if response.results[0].flagged {
        println!("Content was flagged!");
    } else {
        println!("Content is safe.");
    }

    // Check multiple texts at once
    let texts = vec!["Text 1".to_string(), "Text 2".to_string()];
    let response = moderations.moderate_texts(texts, None).await?;

    Ok(())
}
```

## Images API (DALL-E)

Generate images with DALL-E:

```rust
use openai_tools::images::request::{Images, GenerateOptions, ImageModel, ImageSize, ImageQuality};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    Ok(())
}
```

## Audio API

Text-to-speech and transcription:

```rust
use openai_tools::audio::request::{Audio, TtsOptions, TranscribeOptions};
use openai_tools::audio::response::{TtsModel, Voice};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    Ok(())
}
```

## Batch API

Process large volumes of requests asynchronously with 50% cost savings:

```rust
use openai_tools::batch::request::{Batches, CreateBatchRequest, BatchEndpoint};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batches = Batches::new()?;

    // List all batches
    let response = batches.list(Some(20), None).await?;
    for batch in &response.data {
        println!("Batch: {} - {:?}", batch.id, batch.status);
    }

    // Create a batch job (input file must be uploaded via Files API with purpose "batch")
    let request = CreateBatchRequest::new("file-abc123", BatchEndpoint::ChatCompletions);
    let batch = batches.create(request).await?;
    println!("Created batch: {}", batch.id);

    Ok(())
}
```

## Fine-tuning API

Customize models with your training data:

```rust
use openai_tools::fine_tuning::request::{FineTuning, CreateFineTuningJobRequest};
use openai_tools::fine_tuning::response::Hyperparameters;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fine_tuning = FineTuning::new()?;

    // List fine-tuning jobs
    let response = fine_tuning.list(Some(10), None).await?;
    for job in &response.data {
        println!("Job: {} - {:?}", job.id, job.status);
    }

    // Create a fine-tuning job
    let hyperparams = Hyperparameters {
        n_epochs: Some(3),
        ..Default::default()
    };
    let request = CreateFineTuningJobRequest::new("gpt-4o-mini-2024-07-18", "file-abc123")
        .with_suffix("my-model")
        .with_supervised_method(Some(hyperparams));

    let job = fine_tuning.create(request).await?;
    println!("Created job: {}", job.id);

    Ok(())
}
```

## Embedding API

```rust
use openai_tools::embedding::request::Embedding;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut embedding = Embedding::new();
    let response = embedding
        .model("text-embedding-3-small")
        .input_text("Hello, world!")
        .embed()
        .await?;

    println!("Embedding dimensions: {}", response.data[0].embedding.as_1d().unwrap().len());
    Ok(())
}
```

## Update History

<details>
<summary>v1.0.2</summary>

- Added URL-based provider detection for all API clients
  - `with_url(url, api_key, deployment_name)` - auto-detect provider from URL pattern
  - `from_url(url)` - auto-detect with env var credentials
  - `*.openai.azure.com` → Azure, all other URLs → OpenAI-compatible
- Support for OpenAI-compatible APIs (Ollama, vLLM, LocalAI, etc.)
- Added Azure OpenAI support with `azure()` and environment variable configuration
- Added `AuthProvider` abstraction for unified authentication handling

</details>

<details>
<summary>v1.0.1</summary>

- Added automatic handling for reasoning model (o1, o3 series) parameter restrictions
  - Chat API: temperature, frequency_penalty, presence_penalty, logprobs, top_logprobs, logit_bias, n
  - Responses API: temperature, top_p, top_logprobs
- Unsupported parameters are automatically ignored with `tracing::warn!` warnings
- Added "Model-Specific Parameter Restrictions" documentation section

</details>

<details>
<summary>v1.0.0</summary>

- Initial release with all OpenAI APIs:
  - Chat Completions API
  - Responses API
  - Conversations API
  - Embedding API
  - Realtime API (WebSocket)
  - Models API
  - Files API
  - Moderations API
  - Images API (DALL-E)
  - Audio API (TTS, STT)
  - Batch API
  - Fine-tuning API

</details>

## License

MIT License
