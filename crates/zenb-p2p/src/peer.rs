//! Peer Management
//!
//! Types and utilities for managing P2P peer connections.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique peer identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PeerId(pub String);

impl PeerId {
    /// Generate a new random peer ID
    pub fn generate() -> Self {
        let random: u128 = rand::random();
        Self(format!("agolos-{:032x}", random))
    }

    /// Create from string
    pub fn new_from_str(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get as str
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PeerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Peer connection status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerStatus {
    /// Discovered but not connected
    Discovered,
    /// Connection in progress
    Connecting,
    /// Fully connected and healthy
    Connected,
    /// Connection lost, attempting reconnect
    Reconnecting,
    /// Peer is unreachable
    Disconnected,
}

/// Information about a peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Peer ID
    pub id: PeerId,
    /// Connection status
    pub status: PeerStatus,
    /// Address (IP:port or multiaddr)
    pub address: Option<String>,
    /// Agent version
    pub version: String,
    /// Capabilities supported by this peer
    pub capabilities: Vec<String>,
    /// Last seen timestamp (us)
    pub last_seen_us: i64,
    /// Round-trip time in milliseconds
    pub rtt_ms: Option<f32>,
    /// Trust score (0-1, based on history)
    pub trust: f32,
}

impl PeerInfo {
    /// Create new peer info
    pub fn new(id: PeerId) -> Self {
        Self {
            id,
            status: PeerStatus::Discovered,
            address: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
            capabilities: vec!["pattern-share".to_string(), "belief-sync".to_string()],
            last_seen_us: 0,
            rtt_ms: None,
            trust: 0.5, // Neutral initial trust
        }
    }

    /// Update last seen timestamp
    pub fn touch(&mut self) {
        self.last_seen_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
    }

    /// Check if peer is stale (not seen in timeout_ms)
    pub fn is_stale(&self, timeout_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);
        
        let age_ms = (now - self.last_seen_us) / 1000;
        age_ms > timeout_ms as i64
    }
}

/// Peer registry for managing known peers
#[derive(Debug)]
pub struct PeerRegistry {
    /// Known peers by ID
    peers: HashMap<PeerId, PeerInfo>,
    /// Maximum number of connected peers
    max_peers: usize,
    /// Stale timeout in milliseconds
    stale_timeout_ms: u64,
}

impl Default for PeerRegistry {
    fn default() -> Self {
        Self::new(32, 60_000) // 32 peers, 60s timeout
    }
}

impl PeerRegistry {
    /// Create new registry
    pub fn new(max_peers: usize, stale_timeout_ms: u64) -> Self {
        Self {
            peers: HashMap::new(),
            max_peers,
            stale_timeout_ms,
        }
    }

    /// Add or update a peer
    pub fn upsert(&mut self, info: PeerInfo) {
        self.peers.insert(info.id.clone(), info);
    }

    /// Get peer by ID
    pub fn get(&self, id: &PeerId) -> Option<&PeerInfo> {
        self.peers.get(id)
    }

    /// Get mutable peer by ID
    pub fn get_mut(&mut self, id: &PeerId) -> Option<&mut PeerInfo> {
        self.peers.get_mut(id)
    }

    /// Remove a peer
    pub fn remove(&mut self, id: &PeerId) -> Option<PeerInfo> {
        self.peers.remove(id)
    }

    /// Get all connected peers
    pub fn connected(&self) -> impl Iterator<Item = &PeerInfo> {
        self.peers.values().filter(|p| p.status == PeerStatus::Connected)
    }

    /// Count connected peers
    pub fn connected_count(&self) -> usize {
        self.connected().count()
    }

    /// Check if we can accept more peers
    pub fn can_accept(&self) -> bool {
        self.connected_count() < self.max_peers
    }

    /// Remove stale peers
    pub fn prune_stale(&mut self) -> Vec<PeerId> {
        let stale: Vec<PeerId> = self.peers
            .iter()
            .filter(|(_, p)| p.is_stale(self.stale_timeout_ms))
            .map(|(id, _)| id.clone())
            .collect();

        for id in &stale {
            self.peers.remove(id);
        }

        stale
    }

    /// Get all peer IDs
    pub fn peer_ids(&self) -> impl Iterator<Item = &PeerId> {
        self.peers.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_id_generation() {
        let id1 = PeerId::generate();
        let id2 = PeerId::generate();
        
        assert!(id1.0.starts_with("agolos-"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_peer_registry() {
        let mut registry = PeerRegistry::new(10, 1000);
        
        let id = PeerId::new_from_str("test-peer");
        let mut info = PeerInfo::new(id.clone());
        info.status = PeerStatus::Connected;
        info.touch();
        
        registry.upsert(info);
        
        assert_eq!(registry.connected_count(), 1);
        assert!(registry.can_accept());
        
        let retrieved = registry.get(&id).unwrap();
        assert_eq!(retrieved.status, PeerStatus::Connected);
    }
}
