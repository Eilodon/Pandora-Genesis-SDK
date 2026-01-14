//! P2P Message Types
//!
//! Defines the message protocol for inter-agent communication.

use serde::{Deserialize, Serialize};
use serde_with::serde_as;

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
/// 
/// # EIDOLON FIX: Trust Placebo → Real Authentication
/// Signature is now 64-byte Ed25519 (was 32-byte BLAKE3 hash).
/// Messages MUST be signed with `sign_with_identity()` before sending.
#[serde_as]
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
    /// Ed25519 signature of (sender || timestamp || payload)
    #[serde_as(as = "[_; 64]")]
    pub signature: [u8; 64],
    /// Sender's public key for verification
    pub sender_pubkey: [u8; 32],
}

impl P2PMessage {
    /// Create a new UNSIGNED message.
    /// 
    /// # IMPORTANT
    /// This message is NOT authenticated until `sign_with_identity()` is called.
    /// Sending unsigned messages will fail verification on receiver.
    pub fn new_unsigned(msg_type: MessageType, sender: impl Into<String>, payload: Vec<u8>) -> Self {
        let sender_str = sender.into();
        let id = format!("{}-{}", &sender_str, Self::generate_id());
        
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
            signature: [0u8; 64],  // UNSIGNED - must call sign_with_identity()
            sender_pubkey: [0u8; 32],
        }
    }
    
    /// Sign message with peer identity.
    /// 
    /// # EIDOLON FIX
    /// Signature covers: sender + timestamp + payload
    /// This prevents tampering with any of these fields.
    pub fn sign_with_identity(mut self, identity: &crate::identity::PeerIdentity) -> Self {
        let sign_data = self.signable_data();
        self.signature = identity.sign(&sign_data);
        self.sender_pubkey = identity.public_key_bytes();
        self
    }

    /// Create a targeted message
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Verify message signature using embedded public key.
    /// 
    /// # EIDOLON FIX
    /// Now uses Ed25519 verification instead of BLAKE3 hash comparison.
    pub fn verify(&self) -> bool {
        // All-zero signature = unsigned message
        if self.signature == [0u8; 64] {
            return false;
        }
        
        let sign_data = self.signable_data();
        crate::identity::verify_signature(&self.sender_pubkey, &sign_data, &self.signature)
    }
    
    /// Get data that is signed (for both signing and verification).
    fn signable_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(self.sender.as_bytes());
        data.extend_from_slice(&self.timestamp_us.to_le_bytes());
        data.extend_from_slice(&self.payload);
        data
    }

    /// Decode payload as JSON
    pub fn decode_payload<T: for<'de> Deserialize<'de>>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_slice(&self.payload)
    }

    fn generate_id() -> String {
        let random: u64 = rand::random();
        format!("{:016x}", random)
    }
    
    /// Check if message is properly authenticated.
    /// 
    /// Verifies both signature validity AND that public key matches expected sender.
    pub fn is_authenticated_from(&self, expected_pubkey: &[u8; 32]) -> bool {
        self.verify() && &self.sender_pubkey == expected_pubkey
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
    use crate::identity::PeerIdentity;

    #[test]
    fn test_message_creation_and_signing() {
        let identity = PeerIdentity::generate();
        
        let msg = P2PMessage::new_unsigned(
            MessageType::Ping,
            "peer-123",
            b"hello".to_vec(),
        ).sign_with_identity(&identity);
        
        assert_eq!(msg.msg_type, MessageType::Ping);
        assert!(msg.id.starts_with("peer-123-"));
        assert!(msg.verify());
    }

    #[test]
    fn test_unsigned_message_fails_verification() {
        let msg = P2PMessage::new_unsigned(
            MessageType::Ping,
            "peer-123",
            b"hello".to_vec(),
        );
        
        // Unsigned message should NOT verify
        assert!(!msg.verify());
    }

    #[test]
    fn test_tampered_payload_fails_verification() {
        let identity = PeerIdentity::generate();
        
        let mut msg = P2PMessage::new_unsigned(
            MessageType::Discovery,
            "peer-abc",
            b"test payload".to_vec(),
        ).sign_with_identity(&identity);
        
        assert!(msg.verify());
        
        // Tamper with payload
        msg.payload.push(0);
        assert!(!msg.verify(), "Tampered message MUST fail verification");
    }

    #[test]
    fn test_forged_message_rejected() {
        let alice = PeerIdentity::generate();
        let mallory = PeerIdentity::generate();
        
        // Alice creates and signs message
        let msg = P2PMessage::new_unsigned(
            MessageType::Ping,
            "alice",
            b"from alice".to_vec(),
        ).sign_with_identity(&alice);
        
        // Valid message verifies
        assert!(msg.verify());
        assert!(msg.is_authenticated_from(&alice.public_key_bytes()));
        
        // Mallory tries to forge by replacing payload (keeps Alice's signature)
        let mut forged = msg.clone();
        forged.payload = b"ATTACK from mallory".to_vec();
        assert!(!forged.verify(), "Forged message MUST fail verification");
        
        // Mallory tries to re-sign with her key but claim Alice's sender name
        let forged2 = P2PMessage::new_unsigned(
            MessageType::Ping,
            "alice",  // Claims to be alice
            b"hello".to_vec(),
        ).sign_with_identity(&mallory);  // But signed by mallory
        
        // Crypto validates (mallory's sig is valid)
        assert!(forged2.verify());
        // But identity check FAILS (pubkey doesn't match Alice)
        assert!(!forged2.is_authenticated_from(&alice.public_key_bytes()), 
            "Impersonation attempt must be detectable");
    }

    #[test]
    fn test_payload_decode() {
        let identity = PeerIdentity::generate();
        
        let payload = PatternPayload {
            context_hash: [0u8; 32],
            mode: "Calm".to_string(),
            weights: vec![0.1, 0.2, 0.3],
            confidence: 0.85,
        };
        
        let encoded = serde_json::to_vec(&payload).unwrap();
        let msg = P2PMessage::new_unsigned(MessageType::PatternShare, "peer", encoded)
            .sign_with_identity(&identity);
        
        let decoded: PatternPayload = msg.decode_payload().unwrap();
        assert_eq!(decoded.mode, "Calm");
        assert_eq!(decoded.confidence, 0.85);
    }
}
