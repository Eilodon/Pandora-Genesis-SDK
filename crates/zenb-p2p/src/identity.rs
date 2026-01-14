//! Peer Identity and Cryptographic Authentication
//!
//! # EIDOLON FIX: Trust Placebo → Real Authentication
//! 
//! Replaces BLAKE3 hash "signatures" with proper Ed25519 cryptographic signatures.
//! This ensures message authenticity - peers cannot forge messages from others.
//!
//! # Security Model
//! - Each peer generates a unique Ed25519 keypair on first run
//! - Messages are signed with private key, verified with public key
//! - Public keys should be exchanged through trusted channel or certificate

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// Cryptographic identity for a P2P peer.
/// 
/// # Thread Safety
/// SigningKey operations are NOT thread-safe. Wrap in Mutex for concurrent signing.
pub struct PeerIdentity {
    signing_key: SigningKey,
    /// Public verifying key (can be shared freely)
    pub verifying_key: VerifyingKey,
}

impl PeerIdentity {
    /// Generate new random identity using OS entropy.
    /// 
    /// # Security
    /// Uses OsRng for cryptographically secure random generation.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key }
    }
    
    /// Create identity from existing signing key bytes.
    /// 
    /// # Returns
    /// None if key bytes are invalid.
    pub fn from_bytes(key_bytes: &[u8; 32]) -> Option<Self> {
        let signing_key = SigningKey::from_bytes(key_bytes);
        let verifying_key = signing_key.verifying_key();
        Some(Self { signing_key, verifying_key })
    }
    
    /// Sign a message with this identity.
    /// 
    /// # Returns
    /// 64-byte Ed25519 signature
    pub fn sign(&self, message: &[u8]) -> [u8; 64] {
        self.signing_key.sign(message).to_bytes()
    }
    
    /// Get signing key bytes (SECRET - do not expose!)
    pub fn private_key_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
    
    /// Get public key bytes for sharing with peers.
    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }
    
    /// Get peer ID derived from public key (first 16 bytes of blake3 hash).
    pub fn peer_id(&self) -> String {
        let hash = blake3::hash(&self.public_key_bytes());
        hex::encode(&hash.as_bytes()[..16])
    }
}

impl std::fmt::Debug for PeerIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerIdentity")
            .field("peer_id", &self.peer_id())
            .field("public_key", &hex::encode(self.public_key_bytes()))
            .finish_non_exhaustive()
    }
}

/// Serializable public key for sharing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicKeyInfo {
    /// Ed25519 public key bytes
    pub key_bytes: [u8; 32],
    /// Human-readable peer ID
    pub peer_id: String,
}

impl From<&PeerIdentity> for PublicKeyInfo {
    fn from(identity: &PeerIdentity) -> Self {
        Self {
            key_bytes: identity.public_key_bytes(),
            peer_id: identity.peer_id(),
        }
    }
}

/// Verify signature with public key.
/// 
/// # Arguments
/// * `public_key` - 32-byte Ed25519 public key
/// * `message` - Original message bytes
/// * `signature` - 64-byte Ed25519 signature
/// 
/// # Returns
/// true if signature is valid, false otherwise
pub fn verify_signature(
    public_key: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> bool {
    let Ok(vk) = VerifyingKey::from_bytes(public_key) else { return false };
    let Ok(sig) = Signature::from_slice(signature) else { return false };
    vk.verify(message, &sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_identity_generation() {
        let id = PeerIdentity::generate();
        assert_eq!(id.public_key_bytes().len(), 32);
        assert_eq!(id.peer_id().len(), 32); // 16 bytes = 32 hex chars
    }
    
    #[test]
    fn test_sign_verify() {
        let id = PeerIdentity::generate();
        let message = b"hello world";
        let sig = id.sign(message);
        
        assert!(verify_signature(&id.public_key_bytes(), message, &sig));
    }
    
    #[test]
    fn test_tampered_message_fails() {
        let id = PeerIdentity::generate();
        let message = b"hello world";
        let sig = id.sign(message);
        
        let tampered = b"hello ATTACK";
        assert!(!verify_signature(&id.public_key_bytes(), tampered, &sig));
    }
    
    #[test]
    fn test_wrong_key_fails() {
        let alice = PeerIdentity::generate();
        let mallory = PeerIdentity::generate();
        
        let message = b"from alice";
        let sig = alice.sign(message);
        
        // Mallory's key cannot verify Alice's signature
        assert!(!verify_signature(&mallory.public_key_bytes(), message, &sig));
    }
    
    #[test]
    fn test_identity_roundtrip() {
        let id1 = PeerIdentity::generate();
        let bytes = id1.private_key_bytes();
        let id2 = PeerIdentity::from_bytes(&bytes).unwrap();
        
        assert_eq!(id1.public_key_bytes(), id2.public_key_bytes());
    }
}
