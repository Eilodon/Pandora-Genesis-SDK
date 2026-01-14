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
    /// Trust score (0-1, based on history) - deprecated, use score_tracker.trust()
    pub trust: f32,
    /// AETHER V29: Elo-like peer scoring for Byzantine resistance
    pub score_tracker: PeerScore,
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
            trust: 0.5, // Neutral initial trust (deprecated)
            score_tracker: PeerScore::default(), // AETHER V29: Initialize scoring
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

    /// Get score for peer selection
    pub fn score(&self) -> &PeerScore {
        &self.score_tracker
    }

    /// Get mutable score for updates
    pub fn score_mut(&mut self) -> &mut PeerScore {
        &mut self.score_tracker
    }
}

// ============================================================================
// AETHER V29 TRANSPLANT: Elo-like Peer Scoring
// ============================================================================
// Adapted from Aether's ProvScore system for Byzantine-resistant peer selection.
// Uses Elo rating to dynamically rank peers based on reliability.
// ============================================================================

/// Elo-like peer scoring for Byzantine resistance.
/// 
/// # Aether V29 Transplant
/// From Aether's ProvScore: R_new = R_old + K * (S_actual - E_expected)
/// 
/// # Key Properties
/// - **Adaptive K-factor**: Reduces as more games played (like chess)
/// - **Strike system**: 3 strikes and peer is flagged as unreliable
/// - **Streak bonus**: Consistent success increases rating faster
/// - **Decay**: Old ratings decay toward neutral over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerScore {
    /// Elo-style rating (baseline 1500, like chess)
    pub rating: f32,
    /// Total successes
    pub successes: u32,
    /// Total failures  
    pub failures: u32,
    /// Consecutive failures (strike counter)
    pub strike_count: u32,
    /// Consecutive successes (streak counter)
    pub success_streak: u32,
    /// Last update timestamp (us)
    pub last_update_us: i64,
    /// Is this peer flagged as unreliable?
    pub flagged: bool,
}

impl Default for PeerScore {
    fn default() -> Self {
        Self {
            rating: 1500.0, // Neutral starting rating (chess baseline)
            successes: 0,
            failures: 0,
            strike_count: 0,
            success_streak: 0,
            last_update_us: 0,
            flagged: false,
        }
    }
}

impl PeerScore {
    /// Maximum strikes before flagging peer as unreliable
    const MAX_STRIKES: u32 = 3;
    
    /// Base K-factor for Elo updates
    const K_BASE: f32 = 32.0;
    
    /// Minimum K-factor (for established peers)
    const K_MIN: f32 = 8.0;
    
    /// Calculate adaptive K-factor
    /// 
    /// K decreases as peer has more history, making ratings more stable.
    fn k_factor(&self) -> f32 {
        let games = self.successes + self.failures;
        if games < 10 {
            Self::K_BASE // New peer: high volatility
        } else if games < 50 {
            Self::K_BASE * 0.75
        } else {
            Self::K_MIN // Established peer: stable rating
        }
    }
    
    /// Calculate expected score based on rating
    /// 
    /// E(A) = 1 / (1 + 10^((R_baseline - R_A) / 400))
    fn expected_score(&self) -> f32 {
        // Compare against baseline (1500)
        let rating_diff = 1500.0 - self.rating;
        1.0 / (1.0 + 10f32.powf(rating_diff / 400.0))
    }
    
    /// Update score based on outcome
    /// 
    /// # Aether V29 Elo Formula
    /// R_new = R_old + K * (S_actual - E_expected)
    /// 
    /// # Arguments
    /// * `success` - Whether this interaction was successful
    /// * `now_us` - Current timestamp in microseconds
    pub fn update(&mut self, success: bool, now_us: i64) {
        let expected = self.expected_score();
        let actual = if success { 1.0 } else { 0.0 };
        let k = self.k_factor();
        
        // Apply Elo update
        self.rating += k * (actual - expected);
        
        // Clamp rating to reasonable bounds
        self.rating = self.rating.clamp(500.0, 2500.0);
        
        // Update counters
        if success {
            self.successes += 1;
            self.success_streak += 1;
            self.strike_count = 0; // Reset strikes on success
            
            // Streak bonus: extra rating for consistency
            if self.success_streak >= 5 {
                self.rating += 2.0;
            }
        } else {
            self.failures += 1;
            self.success_streak = 0;
            self.strike_count += 1;
            
            // Check for flag threshold
            if self.strike_count >= Self::MAX_STRIKES {
                self.flagged = true;
            }
        }
        
        self.last_update_us = now_us;
    }
    
    /// Get selection probability (higher rating = higher probability)
    /// 
    /// Uses softmax-like scaling for peer selection.
    pub fn selection_probability(&self) -> f32 {
        if self.flagged {
            return 0.0; // Never select flagged peers
        }
        
        // Normalize rating to 0-1 range
        // Rating 1000 → 0.0, Rating 2000 → 1.0
        let normalized = ((self.rating - 1000.0) / 1000.0).clamp(0.0, 1.0);
        
        // Apply softmax-like transformation
        let temp = 0.5; // Temperature parameter
        (normalized / temp).exp() / (1.0 / temp).exp()
    }
    
    /// Decay rating toward neutral over time
    /// 
    /// # Arguments
    /// * `now_us` - Current timestamp
    /// * `half_life_days` - Days for rating to decay halfway to 1500
    pub fn decay_toward_neutral(&mut self, now_us: i64, half_life_days: f64) {
        let elapsed_us = (now_us - self.last_update_us).max(0) as f64;
        let elapsed_days = elapsed_us / 86_400_000_000.0;
        
        let decay_factor = (-elapsed_days / half_life_days * std::f64::consts::LN_2).exp() as f32;
        
        // Decay toward 1500 (neutral)
        self.rating = 1500.0 + (self.rating - 1500.0) * decay_factor;
        
        // Unflag after decay period if rating is reasonable
        if self.flagged && self.rating >= 1200.0 {
            self.flagged = false;
            self.strike_count = 0;
        }
    }
    
    /// Is this peer reliable? (above threshold rating, not flagged)
    pub fn is_reliable(&self) -> bool {
        !self.flagged && self.rating >= 1200.0
    }
    
    /// Trust level as 0-1 float (for backward compatibility)
    pub fn trust(&self) -> f32 {
        if self.flagged {
            0.0
        } else {
            ((self.rating - 500.0) / 2000.0).clamp(0.0, 1.0)
        }
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

    // ========================================================================
    // AETHER V29 TRANSPLANT: Score-Based Peer Selection
    // ========================================================================

    /// Select best N peers by Elo rating for a task
    /// 
    /// # Aether V29 Transplant
    /// Uses Elo-based selection with overfetch for Byzantine resistance.
    /// Prefers reliable, high-rated peers while avoiding flagged ones.
    /// 
    /// # Arguments
    /// * `n` - Number of peers to select
    /// * `overfetch` - Extra candidates to consider (default 0)
    /// 
    /// # Returns
    /// Vec of (PeerId, rating) sorted by rating descending
    pub fn select_best_peers(&self, n: usize, overfetch: usize) -> Vec<(PeerId, f32)> {
        let mut candidates: Vec<_> = self.peers
            .iter()
            .filter(|(_, p)| p.status == PeerStatus::Connected)
            .filter(|(_, p)| p.score_tracker.is_reliable())
            .map(|(id, p)| (id.clone(), p.score_tracker.rating))
            .collect();

        // Sort by rating descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top n + overfetch, then trim to n
        candidates.truncate(n + overfetch);
        candidates.truncate(n);

        candidates
    }

    /// Update peer score based on interaction outcome
    /// 
    /// # Aether V29 Transplant
    /// Call this after every peer interaction to update Elo rating.
    pub fn update_peer_score(&mut self, id: &PeerId, success: bool) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        if let Some(peer) = self.peers.get_mut(id) {
            peer.score_tracker.update(success, now_us);
            // Keep trust field in sync for backward compatibility
            peer.trust = peer.score_tracker.trust();
        }
    }

    /// Decay all peer scores toward neutral
    /// 
    /// Call periodically (e.g., once per hour) to prevent rating stagnation.
    pub fn decay_all_scores(&mut self, half_life_days: f64) {
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0);

        for peer in self.peers.values_mut() {
            peer.score_tracker.decay_toward_neutral(now_us, half_life_days);
            peer.trust = peer.score_tracker.trust();
        }
    }

    /// Get statistics about peer scores
    pub fn score_stats(&self) -> (f32, f32, usize, usize) {
        let connected: Vec<_> = self.peers
            .values()
            .filter(|p| p.status == PeerStatus::Connected)
            .collect();

        if connected.is_empty() {
            return (0.0, 0.0, 0, 0);
        }

        let sum: f32 = connected.iter().map(|p| p.score_tracker.rating).sum();
        let avg = sum / connected.len() as f32;
        
        let reliable = connected.iter().filter(|p| p.score_tracker.is_reliable()).count();
        let flagged = connected.iter().filter(|p| p.score_tracker.flagged).count();

        (avg, sum, reliable, flagged)
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
