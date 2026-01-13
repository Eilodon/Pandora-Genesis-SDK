//! P2P Node Implementation
//!
//! Main entry point for P2P networking.

use crate::{P2PError, P2PResult};
use crate::message::{P2PMessage, MessageType};
use crate::peer::{PeerId, PeerInfo, PeerRegistry, PeerStatus};

use tokio::sync::mpsc;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Configuration for P2P node
#[derive(Debug, Clone)]
pub struct P2PConfig {
    /// Local peer ID
    pub peer_id: PeerId,
    /// Maximum connected peers
    pub max_peers: usize,
    /// Ping interval in milliseconds
    pub ping_interval_ms: u64,
    /// Stale peer timeout in milliseconds
    pub stale_timeout_ms: u64,
    /// Enable pattern sharing
    pub enable_pattern_share: bool,
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            peer_id: PeerId::generate(),
            max_peers: 16,
            ping_interval_ms: 30_000,
            stale_timeout_ms: 120_000,
            enable_pattern_share: true,
        }
    }
}

/// P2P Node for agent networking
///
/// This is the main entry point for P2P communication.
/// Create a node, start it, and use it to send/receive messages.
pub struct P2PNode {
    config: P2PConfig,
    registry: Arc<Mutex<PeerRegistry>>,
    outbox_tx: mpsc::Sender<P2PMessage>,
    #[allow(dead_code)] // Will be used in full network loop implementation
    outbox_rx: Arc<Mutex<mpsc::Receiver<P2PMessage>>>,
    inbox_tx: mpsc::Sender<P2PMessage>,
    inbox_rx: Arc<Mutex<mpsc::Receiver<P2PMessage>>>,
    running: Arc<Mutex<bool>>,
}

impl P2PNode {
    /// Create a new P2P node
    pub fn new(config: P2PConfig) -> Self {
        let (outbox_tx, outbox_rx) = mpsc::channel(1024);
        let (inbox_tx, inbox_rx) = mpsc::channel(1024);
        
        Self {
            registry: Arc::new(Mutex::new(PeerRegistry::new(
                config.max_peers,
                config.stale_timeout_ms,
            ))),
            outbox_tx,
            outbox_rx: Arc::new(Mutex::new(outbox_rx)),
            inbox_tx,
            inbox_rx: Arc::new(Mutex::new(inbox_rx)),
            running: Arc::new(Mutex::new(false)),
            config,
        }
    }

    /// Create with default config
    pub fn default_node() -> Self {
        Self::new(P2PConfig::default())
    }

    /// Get local peer ID
    pub fn peer_id(&self) -> &PeerId {
        &self.config.peer_id
    }

    /// Send a message to outbox
    pub async fn send(&self, msg: P2PMessage) -> P2PResult<()> {
        self.outbox_tx
            .send(msg)
            .await
            .map_err(|_| P2PError::ChannelClosed)
    }

    /// Receive next message from inbox
    pub async fn receive(&self) -> Option<P2PMessage> {
        let mut rx = self.inbox_rx.lock().await;
        rx.recv().await
    }

    /// Broadcast a message to all connected peers
    pub async fn broadcast(&self, msg_type: MessageType, payload: Vec<u8>) -> P2PResult<usize> {
        let registry = self.registry.lock().await;
        let peer_count = registry.connected_count();
        
        let msg = P2PMessage::new(msg_type, self.config.peer_id.as_str(), payload);
        self.send(msg).await?;
        
        Ok(peer_count)
    }

    /// Add a peer to the registry
    pub async fn add_peer(&self, info: PeerInfo) {
        let mut registry = self.registry.lock().await;
        registry.upsert(info);
    }

    /// Get peer count
    pub async fn peer_count(&self) -> usize {
        let registry = self.registry.lock().await;
        registry.connected_count()
    }

    /// Check if node is running
    pub async fn is_running(&self) -> bool {
        *self.running.lock().await
    }

    /// Start the node (placeholder - full implementation needs async runtime)
    pub async fn start(&self) -> P2PResult<()> {
        let mut running = self.running.lock().await;
        *running = true;
        log::info!("P2P Node {} started", self.config.peer_id);
        Ok(())
    }

    /// Stop the node
    pub async fn stop(&self) -> P2PResult<()> {
        let mut running = self.running.lock().await;
        *running = false;
        log::info!("P2P Node {} stopped", self.config.peer_id);
        Ok(())
    }

    /// Process an incoming message (for testing/simulation)
    pub async fn process_message(&self, msg: P2PMessage) -> P2PResult<Option<P2PMessage>> {
        // Update peer last-seen
        {
            let mut registry = self.registry.lock().await;
            if let Some(peer) = registry.get_mut(&PeerId::new_from_str(&msg.sender)) {
                peer.touch();
            }
        }

        // Handle message based on type
        match msg.msg_type {
            MessageType::Ping => {
                // Respond with pong
                let pong = P2PMessage::new(
                    MessageType::Pong,
                    self.config.peer_id.as_str(),
                    msg.id.as_bytes().to_vec(), // Echo message ID as payload
                ).with_target(msg.sender);
                
                Ok(Some(pong))
            }
            MessageType::Pong => {
                // Update RTT (could compute from timestamp)
                Ok(None)
            }
            MessageType::Discovery => {
                // Add to registry if we have capacity
                let mut registry = self.registry.lock().await;
                if registry.can_accept() {
                    let mut info = PeerInfo::new(PeerId::new_from_str(&msg.sender));
                    info.status = PeerStatus::Connected;
                    info.touch();
                    registry.upsert(info);
                }
                Ok(None)
            }
            MessageType::Goodbye => {
                // Remove peer
                let mut registry = self.registry.lock().await;
                registry.remove(&PeerId::new_from_str(&msg.sender));
                Ok(None)
            }
            _ => {
                // Forward to inbox for application processing
                self.inbox_tx
                    .send(msg)
                    .await
                    .map_err(|_| P2PError::ChannelClosed)?;
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_node_creation() {
        let node = P2PNode::default_node();
        assert!(!node.is_running().await);
        assert!(node.peer_id().0.starts_with("agolos-"));
    }

    #[tokio::test]
    async fn test_node_start_stop() {
        let node = P2PNode::default_node();
        
        node.start().await.unwrap();
        assert!(node.is_running().await);
        
        node.stop().await.unwrap();
        assert!(!node.is_running().await);
    }

    #[tokio::test]
    async fn test_ping_pong() {
        let node = P2PNode::default_node();
        node.start().await.unwrap();
        
        let ping = P2PMessage::new(
            MessageType::Ping,
            "remote-peer",
            b"ping".to_vec(),
        );
        
        let response = node.process_message(ping).await.unwrap();
        assert!(response.is_some());
        
        let pong = response.unwrap();
        assert_eq!(pong.msg_type, MessageType::Pong);
        assert_eq!(pong.target, Some("remote-peer".to_string()));
    }

    #[tokio::test]
    async fn test_peer_discovery() {
        let node = P2PNode::default_node();
        node.start().await.unwrap();
        
        let discovery = P2PMessage::new(
            MessageType::Discovery,
            "new-peer-123",
            b"hello".to_vec(),
        );
        
        node.process_message(discovery).await.unwrap();
        
        // Should have added the peer
        assert_eq!(node.peer_count().await, 1);
    }
}
