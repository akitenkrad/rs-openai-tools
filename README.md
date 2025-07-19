![Crates.io Version](https://img.shields.io/crates/v/openai-tools?style=flat-square&color=blue)

# OpenAI Tools

API Wrapper for OpenAI API.

<img src="/LOGO.png" alt="LOGO" width="150" height="150"/>

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
use openai_tools::OpenAI;
```

# Features

| Feature Name                   | [Chat Completion](/openai-tools/src/chat/mod.rs) | [Responses](/openai-tools/src/responses/mod.rs) | Embedding | Realtime | Images | Audio | Eval |
|--------------------------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Basic Features                 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Structured Output              | ✅ | ✅ | - | - | - | - | - |
| Function Calling / MCP Tools   | ✅ | ✅ | - | - | - | - | - |
| Image Input                    | ✅ | ✅ | - | - | - | - | - |

✅: Implemented  
🔧: In Progress  
❌: Not yet  
