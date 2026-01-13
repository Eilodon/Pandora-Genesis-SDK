//! P2P Message Types
//!
//! Defines the message protocol for inter-agent communication.

use serde::{Deserialize, Serialize};

/// Types of P2P messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Peer discovery announcement
    Discovery,
    /// Ping for liveness check
    Ping,
    /// Pong response
    Pong,
    /// Pattern sharing (learned knowledge)
    PatternShare,
    /// Trauma/safety constraint update
    TraumaSync,
    /// Belief state synchronization
    BeliefSync,
    /// Request for specific data
    Request,
    /// Response to a request
    Response,
    /// Goodbye (graceful disconnect)
    Goodbye,
}

/// P2P Message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PMessage {
    /// Unique message ID
    pub id: String,
    /// Message type
    pub msg_type: MessageType,
    /// Sender peer ID
    pub sender: String,
    /// Target peer ID (empty for broadcast)
    pub target: Option<String>,
    /// Timestamp in microseconds
    pub timestamp_us: i64,
    /// Message payload (JSON encoded)
    pub payload: Vec<u8>,
    /// BLAKE3 signature of payload
    pub signature: [u8; 32],
}

impl P2PMessage {
    /// Create a new message
    pub fn new(msg_type: MessageType, sender: impl Into<String>, payload: Vec<u8>) -> Self {
        let sender_str = sender.into();
        let id = format!("{}-{}", &sender_str, Self::generate_id());
        let signature = blake3::hash(&payload).into();
        
        Self {
            id,
            msg_type,
            sender: sender_str,
            target: None,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0),
            payload,
            signature,
        }
    }

    /// Create a targeted message
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Verify message signature
    pub fn verify(&self) -> bool {
        let computed: [u8; 32] = blake3::hash(&self.payload).into();
        computed == self.signature
    }

    /// Decode payload as JSON
    pub fn decode_payload<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_slice(&self.payload)
    }

    fn generate_id() -> String {
        let random: u64 = rand::random();
        format!("{:016x}", random)
    }
}

// ============================================================================
// Common Payload Types
// ============================================================================

/// Pattern share payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPayload {
    /// Context hash
    pub context_hash: [u8; 32],
    /// State mode
    pub mode: String,
    /// Learned weights (compressed)
    pub weights: Vec<f32>,
    /// Confidence in this pattern
    pub confidence: f32,
}

/// Trauma sync payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraumaPayload {
    /// Context signature
    pub context_sig: [u8; 32],
    /// Hit count
    pub hit_count: u32,
    /// Last action that caused trauma
    pub action_type: String,
    /// Severity (0-1)
    pub severity: f32,
}

/// Belief sync payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefPayload {
    /// 5-mode probability distribution
    pub probabilities: [f32; 5],
    /// Current mode name
    pub mode: String,
    /// Confidence
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = P2PMessage::new(
            MessageType::Ping,
            "peer-123",
            b"hello".to_vec(),
        );
        
        assert_eq!(msg.msg_type, MessageType::Ping);
        assert!(msg.id.starts_with("peer-123-"));
        assert!(msg.verify());
    }

    #[test]
    fn test_message_verification() {
        let mut msg = P2PMessage::new(
            MessageType::Discovery,
            "peer-abc",
            b"test payload".to_vec(),
        );
        
        assert!(msg.verify());
        
        // Tamper with payload
        msg.payload.push(0);
        assert!(!msg.verify());
    }

    #[test]
    fn test_payload_decode() {
        let payload = PatternPayload {
            context_hash: [0u8; 32],
            mode: "Calm".to_string(),
            weights: vec![0.1, 0.2, 0.3],
            confidence: 0.85,
        };
        
        let encoded = serde_json::to_vec(&payload).unwrap();
        let msg = P2PMessage::new(MessageType::PatternShare, "peer", encoded);
        
        let decoded: PatternPayload = msg.decode_payload().unwrap();
        assert_eq!(decoded.mode, "Calm");
        assert_eq!(decoded.confidence, 0.85);
    }
}
