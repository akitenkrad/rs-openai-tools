//! Event types for the Realtime API.
//!
//! This module contains all client-to-server and server-to-client events
//! used in the Realtime API WebSocket communication.

pub mod client;
pub mod server;

pub use client::ClientEvent;
pub use server::ServerEvent;
