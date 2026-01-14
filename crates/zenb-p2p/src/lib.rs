//! AGOLOS P2P Infrastructure
//!
//! Peer-to-peer networking layer for distributed AGOLOS agents.
//!
//! # VAJRA V5: Digital Lifeform Communication
//!
//! This crate provides the networking foundation for AGOLOS agents to:
//! - Discover and connect to peer agents
//! - Share learned patterns and insights
//! - Collaboratively refine safety constraints
//! - Maintain resilient distributed operation
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐
//! │   Agent A    │────▶│   Agent B    │
//! │  (Local AI)  │◀────│  (Peer AI)   │
//! └──────────────┘     └──────────────┘
//!        │                    │
//!        └───────┐    ┌───────┘
//!                ▼    ▼
//!         ┌──────────────┐
//!         │  P2P Meshᵓ   │
//!         │  (libp2p)    │
//!         └──────────────┘
//! ```

pub mod message;
pub mod peer;
pub mod node;

pub use message::{P2PMessage, MessageType};
pub use peer::{PeerId, PeerInfo, PeerStatus, PeerScore, PeerRegistry};
pub use node::P2PNode;

/// Result type for P2P operations
pub type P2PResult<T> = Result<T, P2PError>;

/// Errors that can occur during P2P operations
#[derive(Debug, thiserror::Error)]
pub enum P2PError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Peer not found: {0}")]
    PeerNotFound(String),
    
    #[error("Message encoding error: {0}")]
    EncodingError(String),
    
    #[error("Invalid payload: {0}")]
    InvalidPayload(String),
    
    #[error("Channel closed")]
    ChannelClosed,
    
    #[error("Timeout: {0}")]
    Timeout(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}
