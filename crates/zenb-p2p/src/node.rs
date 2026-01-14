//! P2P Node Implementation
//!
//! Main entry point for P2P networking.

use crate::{P2PError, P2PResult};
use crate::message::{P2PMessage, MessageType, PatternPayload, TraumaPayload, BeliefPayload};
use crate::peer::{PeerId, PeerInfo, PeerRegistry, PeerStatus};
use crate::identity::PeerIdentity;

use tokio::sync::mpsc;
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
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
    /// Enable pattern sharing (opt-in: requires explicit consent)
    pub enable_pattern_share: bool,
    /// Lightcone radius in microseconds (only process temporally relevant messages)
    /// Default: 60 seconds = 60_000_000 us
    pub lightcone_radius_us: i64,
    /// Enable trauma sync (bi-directional safety sharing)
    pub enable_trauma_sync: bool,
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            peer_id: PeerId::generate(),
            max_peers: 16,
            ping_interval_ms: 30_000,
            stale_timeout_ms: 120_000,
            enable_pattern_share: false, // Opt-in by default
            lightcone_radius_us: 60_000_000, // 60 seconds
            enable_trauma_sync: true, // Bi-directional enabled
        }
    }
}

/// Trauma sync entry with receive timestamp
#[derive(Debug, Clone)]
struct TraumaSyncEntry {
    payload: TraumaPayload,
    received_at_us: i64,
}

/// P2P Node for agent networking
///
/// This is the main entry point for P2P communication.
/// Create a node, start it, and use it to send/receive messages.
pub struct P2PNode {
    config: P2PConfig,
    /// Cryptographic identity for message signing
    identity: PeerIdentity,
    registry: Arc<Mutex<PeerRegistry>>,
    outbox_tx: mpsc::Sender<P2PMessage>,
    #[allow(dead_code)] // Will be used in full network loop implementation
    outbox_rx: Arc<Mutex<mpsc::Receiver<P2PMessage>>>,
    inbox_tx: mpsc::Sender<P2PMessage>,
    inbox_rx: Arc<Mutex<mpsc::Receiver<P2PMessage>>>,
    running: Arc<Mutex<bool>>,
    /// Pattern storage for sharing (opt-in)
    patterns: Arc<Mutex<Vec<PatternPayload>>>,
    /// Trauma cache for bi-directional sync
    trauma_cache: Arc<Mutex<HashMap<[u8; 32], TraumaSyncEntry>>>,
    /// Belief history for sync
    belief_history: Arc<Mutex<VecDeque<BeliefPayload>>>,
}

impl P2PNode {
    /// Create a new P2P node
    pub fn new(config: P2PConfig) -> Self {
        let (outbox_tx, outbox_rx) = mpsc::channel(1024);
        let (inbox_tx, inbox_rx) = mpsc::channel(1024);
        let identity = PeerIdentity::generate();
        
        Self {
            registry: Arc::new(Mutex::new(PeerRegistry::new(
                config.max_peers,
                config.stale_timeout_ms,
            ))),
            identity,
            outbox_tx,
            outbox_rx: Arc::new(Mutex::new(outbox_rx)),
            inbox_tx,
            inbox_rx: Arc::new(Mutex::new(inbox_rx)),
            running: Arc::new(Mutex::new(false)),
            patterns: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            trauma_cache: Arc::new(Mutex::new(HashMap::new())),
            belief_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
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
        
        let msg = P2PMessage::new_unsigned(msg_type, self.config.peer_id.as_str(), payload)
            .sign_with_identity(&self.identity);
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
    ///
    /// Implements full message handling with:
    /// - Signature verification
    /// - Lightcone filtering (temporal relevance)
    /// - Pattern, Trauma, Belief sync handlers
    pub async fn process_message(&self, msg: P2PMessage) -> P2PResult<Option<P2PMessage>> {
        // Verify signature
        if !msg.verify() {
            log::warn!("Invalid signature from {}", msg.sender);
            return Err(P2PError::InvalidPayload("Invalid signature".to_string()));
        }
        
        // Lightcone Filter: Only process temporally relevant messages
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        
        let age_us = (now_us - msg.timestamp_us).abs();
        if age_us > self.config.lightcone_radius_us {
            log::debug!(
                "Message {} outside lightcone (age={}us > {}us)",
                msg.id,
                age_us,
                self.config.lightcone_radius_us
            );
            return Ok(None);
        }

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
                let pong = P2PMessage::new_unsigned(
                    MessageType::Pong,
                    self.config.peer_id.as_str(),
                    msg.id.as_bytes().to_vec(), // Echo message ID as payload
                )
                .sign_with_identity(&self.identity)
                .with_target(msg.sender.clone());
                
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
            MessageType::PatternShare => {
                self.handle_pattern_share(&msg).await?;
                Ok(None)
            }
            MessageType::TraumaSync => {
                self.handle_trauma_sync(&msg).await?;
                Ok(None)
            }
            MessageType::BeliefSync => {
                self.handle_belief_sync(&msg).await?;
                Ok(None)
            }
            MessageType::Request => {
                self.handle_request(&msg).await
            }
            MessageType::Response => {
                // Forward to inbox for application processing
                self.inbox_tx
                    .send(msg)
                    .await
                    .map_err(|_| P2PError::ChannelClosed)?;
                Ok(None)
            }
        }
    }

    // =========================================================================
    // Message Handlers
    // =========================================================================

    /// Handle pattern share (opt-in)
    async fn handle_pattern_share(&self, msg: &P2PMessage) -> P2PResult<()> {
        // Check if pattern sharing is enabled (opt-in)
        if !self.config.enable_pattern_share {
            log::debug!("Pattern sharing disabled, ignoring from {}", msg.sender);
            return Ok(());
        }

        let payload: PatternPayload = msg.decode_payload()
            .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

        // Validate confidence threshold
        if payload.confidence < 0.5 {
            log::debug!("Ignoring low-confidence pattern from {} (conf={:.2})", 
                msg.sender, payload.confidence);
            return Ok(());
        }

        let mut patterns = self.patterns.lock().await;

        // Deduplicate by context_hash
        if patterns.iter().any(|p| p.context_hash == payload.context_hash) {
            log::debug!("Pattern already known, ignoring duplicate");
            return Ok(());
        }

        // Limit storage (LRU-style)
        if patterns.len() >= 1000 {
            patterns.remove(0);
        }
        
        log::info!(
            "PatternShare: received from {} mode={} conf={:.2}",
            msg.sender, payload.mode, payload.confidence
        );
        patterns.push(payload);

        Ok(())
    }

    /// Handle trauma sync (bi-directional)
    async fn handle_trauma_sync(&self, msg: &P2PMessage) -> P2PResult<()> {
        // Check if trauma sync is enabled
        if !self.config.enable_trauma_sync {
            log::debug!("Trauma sync disabled, ignoring from {}", msg.sender);
            return Ok(());
        }

        let payload: TraumaPayload = msg.decode_payload()
            .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        let mut cache = self.trauma_cache.lock().await;

        // Bi-directional merge: Keep entry with higher hit count or severity
        if let Some(existing) = cache.get(&payload.context_sig) {
            if existing.payload.hit_count >= payload.hit_count
                && existing.payload.severity >= payload.severity
            {
                log::debug!("Trauma already known with higher count, ignoring");
                return Ok(()); // Our version is better
            }
        }

        log::info!(
            "TraumaSync: received from {} sig={:02x}{:02x}.. count={} sev={:.2}",
            msg.sender,
            payload.context_sig[0], payload.context_sig[1],
            payload.hit_count, payload.severity
        );

        cache.insert(payload.context_sig, TraumaSyncEntry {
            payload,
            received_at_us: now_us,
        });

        Ok(())
    }

    /// Handle belief sync
    async fn handle_belief_sync(&self, msg: &P2PMessage) -> P2PResult<()> {
        let payload: BeliefPayload = msg.decode_payload()
            .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

        // Store in history
        let mut history = self.belief_history.lock().await;
        if history.len() >= 100 {
            history.pop_front();
        }
        history.push_back(payload.clone());

        // Forward to inbox for Engine integration
        self.inbox_tx
            .send(msg.clone())
            .await
            .map_err(|_| P2PError::ChannelClosed)?;

        log::debug!(
            "BeliefSync: received from {} mode={} conf={:.2}",
            msg.sender, payload.mode, payload.confidence
        );

        Ok(())
    }

    /// Handle data requests
    async fn handle_request(&self, msg: &P2PMessage) -> P2PResult<Option<P2PMessage>> {
        // Decode request type from payload
        let request: serde_json::Value = msg.decode_payload()
            .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

        let request_type = request.get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match request_type {
            "patterns" => {
                // Only share if opt-in enabled
                if !self.config.enable_pattern_share {
                    return Ok(None);
                }
                
                let patterns = self.patterns.lock().await;
                let payload = serde_json::to_vec(&*patterns)
                    .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

                let response = P2PMessage::new_unsigned(
                    MessageType::Response,
                    self.config.peer_id.as_str(),
                    payload,
                )
                .sign_with_identity(&self.identity)
                .with_target(&msg.sender);

                Ok(Some(response))
            }
            "trauma" => {
                // Only share if trauma sync enabled
                if !self.config.enable_trauma_sync {
                    return Ok(None);
                }
                
                let cache = self.trauma_cache.lock().await;
                let entries: Vec<_> = cache.values().map(|e| &e.payload).collect();
                let payload = serde_json::to_vec(&entries)
                    .map_err(|e| P2PError::InvalidPayload(e.to_string()))?;

                let response = P2PMessage::new_unsigned(
                    MessageType::Response,
                    self.config.peer_id.as_str(),
                    payload,
                )
                .sign_with_identity(&self.identity)
                .with_target(&msg.sender);

                Ok(Some(response))
            }
            _ => {
                log::debug!("Unknown request type: {}", request_type);
                Ok(None)
            }
        }
    }

    /// Get stored patterns (for local engine integration)
    pub async fn get_patterns(&self) -> Vec<PatternPayload> {
        self.patterns.lock().await.clone()
    }

    /// Get trauma cache (for local safety integration)
    pub async fn get_trauma_cache(&self) -> Vec<TraumaPayload> {
        self.trauma_cache.lock().await
            .values()
            .map(|e| e.payload.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::PeerIdentity;

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
        
        // Create signed ping from remote peer
        let remote_identity = PeerIdentity::generate();
        let ping = P2PMessage::new_unsigned(
            MessageType::Ping,
            "remote-peer",
            b"ping".to_vec(),
        ).sign_with_identity(&remote_identity);
        
        let response = node.process_message(ping).await.unwrap();
        assert!(response.is_some());
        
        let pong = response.unwrap();
        assert_eq!(pong.msg_type, MessageType::Pong);
        assert_eq!(pong.target, Some("remote-peer".to_string()));
        assert!(pong.verify(), "Response pong must be signed");
    }

    #[tokio::test]
    async fn test_peer_discovery() {
        let node = P2PNode::default_node();
        node.start().await.unwrap();
        
        // Create signed discovery from new peer
        let remote_identity = PeerIdentity::generate();
        let discovery = P2PMessage::new_unsigned(
            MessageType::Discovery,
            "new-peer-123",
            b"hello".to_vec(),
        ).sign_with_identity(&remote_identity);
        
        node.process_message(discovery).await.unwrap();
        
        // Should have added the peer
        assert_eq!(node.peer_count().await, 1);
    }
    
    #[tokio::test]
    async fn test_unsigned_message_rejected() {
        let node = P2PNode::default_node();
        node.start().await.unwrap();
        
        // Create UNSIGNED message (attack simulation)
        let unsigned = P2PMessage::new_unsigned(
            MessageType::Ping,
            "attacker",
            b"malicious".to_vec(),
        ); // Note: NOT signed!
        
        // Should be rejected
        let result = node.process_message(unsigned).await;
        assert!(result.is_err(), "Unsigned messages MUST be rejected");
    }
}
