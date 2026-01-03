//! Client-to-server events for the Realtime API.

use serde::Serialize;

use crate::realtime::conversation::ConversationItem;
use crate::realtime::session::{ResponseCreateConfig, SessionConfig};

/// Client events sent to the OpenAI Realtime API server.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    // ==================== Session Events ====================
    /// Update the session configuration.
    #[serde(rename = "session.update")]
    SessionUpdate {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Session configuration to update.
        session: SessionConfig,
    },

    // ==================== Input Audio Buffer Events ====================
    /// Append audio data to the input buffer.
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Base64-encoded audio data.
        audio: String,
    },

    /// Clear the input audio buffer.
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    /// Commit the input audio buffer.
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    // ==================== Output Audio Buffer Events (WebRTC only) ====================
    /// Clear the output audio buffer (WebRTC only).
    #[serde(rename = "output_audio_buffer.clear")]
    OutputAudioBufferClear {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },

    // ==================== Conversation Item Events ====================
    /// Create a new conversation item.
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to insert after (omit to append).
        #[serde(skip_serializing_if = "Option::is_none")]
        previous_item_id: Option<String>,
        /// The conversation item to create.
        item: ConversationItem,
    },

    /// Delete a conversation item.
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to delete.
        item_id: String,
    },

    /// Retrieve a conversation item.
    #[serde(rename = "conversation.item.retrieve")]
    ConversationItemRetrieve {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to retrieve.
        item_id: String,
    },

    /// Truncate a conversation item's audio.
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// ID of the item to truncate.
        item_id: String,
        /// Index of the content part to truncate.
        content_index: u32,
        /// Audio end position in milliseconds.
        audio_end_ms: u32,
    },

    // ==================== Response Events ====================
    /// Create a new response.
    #[serde(rename = "response.create")]
    ResponseCreate {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Response configuration.
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseCreateConfig>,
    },

    /// Cancel the current response.
    #[serde(rename = "response.cancel")]
    ResponseCancel {
        /// Optional client-generated event ID.
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
}

impl ClientEvent {
    /// Create a session update event.
    pub fn session_update(config: SessionConfig) -> Self {
        Self::SessionUpdate {
            event_id: None,
            session: config,
        }
    }

    /// Create an input audio buffer append event.
    pub fn append_audio(audio_base64: impl Into<String>) -> Self {
        Self::InputAudioBufferAppend {
            event_id: None,
            audio: audio_base64.into(),
        }
    }

    /// Create an input audio buffer clear event.
    pub fn clear_audio() -> Self {
        Self::InputAudioBufferClear { event_id: None }
    }

    /// Create an input audio buffer commit event.
    pub fn commit_audio() -> Self {
        Self::InputAudioBufferCommit { event_id: None }
    }

    /// Create a conversation item create event.
    pub fn create_item(item: ConversationItem) -> Self {
        Self::ConversationItemCreate {
            event_id: None,
            previous_item_id: None,
            item,
        }
    }

    /// Create a conversation item create event with a specific position.
    pub fn create_item_after(item: ConversationItem, previous_item_id: impl Into<String>) -> Self {
        Self::ConversationItemCreate {
            event_id: None,
            previous_item_id: Some(previous_item_id.into()),
            item,
        }
    }

    /// Create a conversation item delete event.
    pub fn delete_item(item_id: impl Into<String>) -> Self {
        Self::ConversationItemDelete {
            event_id: None,
            item_id: item_id.into(),
        }
    }

    /// Create a response create event.
    pub fn create_response(config: Option<ResponseCreateConfig>) -> Self {
        Self::ResponseCreate {
            event_id: None,
            response: config,
        }
    }

    /// Create a response cancel event.
    pub fn cancel_response() -> Self {
        Self::ResponseCancel { event_id: None }
    }

    /// Set a custom event ID.
    pub fn with_event_id(mut self, id: impl Into<String>) -> Self {
        let id = Some(id.into());
        match &mut self {
            Self::SessionUpdate { event_id, .. } => *event_id = id,
            Self::InputAudioBufferAppend { event_id, .. } => *event_id = id,
            Self::InputAudioBufferClear { event_id } => *event_id = id,
            Self::InputAudioBufferCommit { event_id } => *event_id = id,
            Self::OutputAudioBufferClear { event_id } => *event_id = id,
            Self::ConversationItemCreate { event_id, .. } => *event_id = id,
            Self::ConversationItemDelete { event_id, .. } => *event_id = id,
            Self::ConversationItemRetrieve { event_id, .. } => *event_id = id,
            Self::ConversationItemTruncate { event_id, .. } => *event_id = id,
            Self::ResponseCreate { event_id, .. } => *event_id = id,
            Self::ResponseCancel { event_id } => *event_id = id,
        }
        self
    }
}
