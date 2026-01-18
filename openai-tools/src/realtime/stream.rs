//! Event handler utilities for the Realtime API.

use super::events::server::*;

/// Callback-based event handler for processing server events.
///
/// # Example
///
/// ```rust,no_run
/// use openai_tools::realtime::EventHandler;
/// use openai_tools::realtime::events::server::ServerEvent;
///
/// let mut handler = EventHandler::new();
/// handler
///     .on_text_delta(|e| {
///         print!("{}", e.delta);
///     })
///     .on_audio_delta(|e| {
///         // Play audio: base64::decode(&e.delta)
///     })
///     .on_error(|e| {
///         eprintln!("Error: {}", e.error.message);
///     });
///
/// // In your event loop:
/// // handler.handle(&event);
/// ```
#[allow(clippy::type_complexity)]
pub struct EventHandler {
    on_session_created: Option<Box<dyn Fn(&SessionCreatedEvent) + Send + Sync>>,
    on_session_updated: Option<Box<dyn Fn(&SessionUpdatedEvent) + Send + Sync>>,
    on_conversation_item_created: Option<Box<dyn Fn(&ConversationItemCreatedEvent) + Send + Sync>>,
    on_input_audio_transcription_completed:
        Option<Box<dyn Fn(&InputAudioTranscriptionCompletedEvent) + Send + Sync>>,
    on_speech_started: Option<Box<dyn Fn(&SpeechStartedEvent) + Send + Sync>>,
    on_speech_stopped: Option<Box<dyn Fn(&SpeechStoppedEvent) + Send + Sync>>,
    on_response_created: Option<Box<dyn Fn(&ResponseCreatedEvent) + Send + Sync>>,
    on_response_done: Option<Box<dyn Fn(&ResponseDoneEvent) + Send + Sync>>,
    on_text_delta: Option<Box<dyn Fn(&ResponseTextDeltaEvent) + Send + Sync>>,
    on_text_done: Option<Box<dyn Fn(&ResponseTextDoneEvent) + Send + Sync>>,
    on_audio_delta: Option<Box<dyn Fn(&ResponseAudioDeltaEvent) + Send + Sync>>,
    on_audio_done: Option<Box<dyn Fn(&ResponseAudioDoneEvent) + Send + Sync>>,
    on_audio_transcript_delta: Option<Box<dyn Fn(&ResponseAudioTranscriptDeltaEvent) + Send + Sync>>,
    on_audio_transcript_done: Option<Box<dyn Fn(&ResponseAudioTranscriptDoneEvent) + Send + Sync>>,
    on_function_call_arguments_delta:
        Option<Box<dyn Fn(&ResponseFunctionCallArgumentsDeltaEvent) + Send + Sync>>,
    on_function_call_arguments_done:
        Option<Box<dyn Fn(&ResponseFunctionCallArgumentsDoneEvent) + Send + Sync>>,
    on_rate_limits_updated: Option<Box<dyn Fn(&RateLimitsUpdatedEvent) + Send + Sync>>,
    on_error: Option<Box<dyn Fn(&ErrorEvent) + Send + Sync>>,
}

impl EventHandler {
    /// Create a new event handler with no callbacks set.
    pub fn new() -> Self {
        Self {
            on_session_created: None,
            on_session_updated: None,
            on_conversation_item_created: None,
            on_input_audio_transcription_completed: None,
            on_speech_started: None,
            on_speech_stopped: None,
            on_response_created: None,
            on_response_done: None,
            on_text_delta: None,
            on_text_done: None,
            on_audio_delta: None,
            on_audio_done: None,
            on_audio_transcript_delta: None,
            on_audio_transcript_done: None,
            on_function_call_arguments_delta: None,
            on_function_call_arguments_done: None,
            on_rate_limits_updated: None,
            on_error: None,
        }
    }

    /// Set callback for session.created events.
    pub fn on_session_created<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&SessionCreatedEvent) + Send + Sync + 'static,
    {
        self.on_session_created = Some(Box::new(f));
        self
    }

    /// Set callback for session.updated events.
    pub fn on_session_updated<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&SessionUpdatedEvent) + Send + Sync + 'static,
    {
        self.on_session_updated = Some(Box::new(f));
        self
    }

    /// Set callback for conversation.item.created events.
    pub fn on_conversation_item_created<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ConversationItemCreatedEvent) + Send + Sync + 'static,
    {
        self.on_conversation_item_created = Some(Box::new(f));
        self
    }

    /// Set callback for input audio transcription completed events.
    pub fn on_input_audio_transcription_completed<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&InputAudioTranscriptionCompletedEvent) + Send + Sync + 'static,
    {
        self.on_input_audio_transcription_completed = Some(Box::new(f));
        self
    }

    /// Set callback for speech started events.
    pub fn on_speech_started<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&SpeechStartedEvent) + Send + Sync + 'static,
    {
        self.on_speech_started = Some(Box::new(f));
        self
    }

    /// Set callback for speech stopped events.
    pub fn on_speech_stopped<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&SpeechStoppedEvent) + Send + Sync + 'static,
    {
        self.on_speech_stopped = Some(Box::new(f));
        self
    }

    /// Set callback for response.created events.
    pub fn on_response_created<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseCreatedEvent) + Send + Sync + 'static,
    {
        self.on_response_created = Some(Box::new(f));
        self
    }

    /// Set callback for response.done events.
    pub fn on_response_done<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseDoneEvent) + Send + Sync + 'static,
    {
        self.on_response_done = Some(Box::new(f));
        self
    }

    /// Set callback for response.text.delta events.
    pub fn on_text_delta<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseTextDeltaEvent) + Send + Sync + 'static,
    {
        self.on_text_delta = Some(Box::new(f));
        self
    }

    /// Set callback for response.text.done events.
    pub fn on_text_done<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseTextDoneEvent) + Send + Sync + 'static,
    {
        self.on_text_done = Some(Box::new(f));
        self
    }

    /// Set callback for response.audio.delta events.
    pub fn on_audio_delta<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseAudioDeltaEvent) + Send + Sync + 'static,
    {
        self.on_audio_delta = Some(Box::new(f));
        self
    }

    /// Set callback for response.audio.done events.
    pub fn on_audio_done<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseAudioDoneEvent) + Send + Sync + 'static,
    {
        self.on_audio_done = Some(Box::new(f));
        self
    }

    /// Set callback for response.audio_transcript.delta events.
    pub fn on_audio_transcript_delta<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseAudioTranscriptDeltaEvent) + Send + Sync + 'static,
    {
        self.on_audio_transcript_delta = Some(Box::new(f));
        self
    }

    /// Set callback for response.audio_transcript.done events.
    pub fn on_audio_transcript_done<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseAudioTranscriptDoneEvent) + Send + Sync + 'static,
    {
        self.on_audio_transcript_done = Some(Box::new(f));
        self
    }

    /// Set callback for function call arguments delta events.
    pub fn on_function_call_arguments_delta<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseFunctionCallArgumentsDeltaEvent) + Send + Sync + 'static,
    {
        self.on_function_call_arguments_delta = Some(Box::new(f));
        self
    }

    /// Set callback for function call arguments done events.
    pub fn on_function_call_arguments_done<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ResponseFunctionCallArgumentsDoneEvent) + Send + Sync + 'static,
    {
        self.on_function_call_arguments_done = Some(Box::new(f));
        self
    }

    /// Set callback for rate limits updated events.
    pub fn on_rate_limits_updated<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&RateLimitsUpdatedEvent) + Send + Sync + 'static,
    {
        self.on_rate_limits_updated = Some(Box::new(f));
        self
    }

    /// Set callback for error events.
    pub fn on_error<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(&ErrorEvent) + Send + Sync + 'static,
    {
        self.on_error = Some(Box::new(f));
        self
    }

    /// Process a server event, calling the appropriate callback.
    pub fn handle(&self, event: &ServerEvent) {
        match event {
            ServerEvent::SessionCreated(e) => {
                if let Some(f) = &self.on_session_created {
                    f(e);
                }
            }
            ServerEvent::SessionUpdated(e) => {
                if let Some(f) = &self.on_session_updated {
                    f(e);
                }
            }
            ServerEvent::ConversationItemCreated(e) => {
                if let Some(f) = &self.on_conversation_item_created {
                    f(e);
                }
            }
            ServerEvent::InputAudioTranscriptionCompleted(e) => {
                if let Some(f) = &self.on_input_audio_transcription_completed {
                    f(e);
                }
            }
            ServerEvent::InputAudioBufferSpeechStarted(e) => {
                if let Some(f) = &self.on_speech_started {
                    f(e);
                }
            }
            ServerEvent::InputAudioBufferSpeechStopped(e) => {
                if let Some(f) = &self.on_speech_stopped {
                    f(e);
                }
            }
            ServerEvent::ResponseCreated(e) => {
                if let Some(f) = &self.on_response_created {
                    f(e);
                }
            }
            ServerEvent::ResponseDone(e) => {
                if let Some(f) = &self.on_response_done {
                    f(e);
                }
            }
            ServerEvent::ResponseTextDelta(e) => {
                if let Some(f) = &self.on_text_delta {
                    f(e);
                }
            }
            ServerEvent::ResponseTextDone(e) => {
                if let Some(f) = &self.on_text_done {
                    f(e);
                }
            }
            ServerEvent::ResponseAudioDelta(e) => {
                if let Some(f) = &self.on_audio_delta {
                    f(e);
                }
            }
            ServerEvent::ResponseAudioDone(e) => {
                if let Some(f) = &self.on_audio_done {
                    f(e);
                }
            }
            ServerEvent::ResponseAudioTranscriptDelta(e) => {
                if let Some(f) = &self.on_audio_transcript_delta {
                    f(e);
                }
            }
            ServerEvent::ResponseAudioTranscriptDone(e) => {
                if let Some(f) = &self.on_audio_transcript_done {
                    f(e);
                }
            }
            ServerEvent::ResponseFunctionCallArgumentsDelta(e) => {
                if let Some(f) = &self.on_function_call_arguments_delta {
                    f(e);
                }
            }
            ServerEvent::ResponseFunctionCallArgumentsDone(e) => {
                if let Some(f) = &self.on_function_call_arguments_done {
                    f(e);
                }
            }
            ServerEvent::RateLimitsUpdated(e) => {
                if let Some(f) = &self.on_rate_limits_updated {
                    f(e);
                }
            }
            ServerEvent::Error(e) => {
                if let Some(f) = &self.on_error {
                    f(e);
                }
            }
            // Events without specific handlers
            _ => {}
        }
    }
}

impl Default for EventHandler {
    fn default() -> Self {
        Self::new()
    }
}
