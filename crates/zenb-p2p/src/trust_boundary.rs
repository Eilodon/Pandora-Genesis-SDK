//! Trust Boundary Middleware
//!
//! # EIDOLON FIX: Trust Boundary Enforcement
//!
//! This module provides types and utilities for enforcing message authentication
//! at ingress points. The `AuthenticatedMessage` wrapper type ensures that only
//! verified messages can enter the trusted zone of the application.
//!
//! # Security Model
//! - Raw `P2PMessage` = untrusted, from network
//! - `AuthenticatedMessage` = verified, safe to process
//! - Functions requiring authentication take `&AuthenticatedMessage`
//!
//! # Example
//! ```ignore
//! // At network boundary
//! let raw_msg = receive_from_network();
//! let verified = AuthenticatedMessage::verify(raw_msg)?;
//!
//! // In trusted zone - type system prevents unverified access
//! process_message(&verified);
//! ```

use crate::message::P2PMessage;

/// A verified P2P message that has passed cryptographic authentication.
///
/// # EIDOLON FIX: Trust Boundary
/// This wrapper type ensures that only messages that have been cryptographically
/// verified can be processed by internal application logic. The private `inner`
/// field prevents construction without going through `verify()`.
///
/// # Type-Level Security
/// Functions that require authentication should take `&AuthenticatedMessage`
/// instead of `&P2PMessage`. This makes it impossible to accidentally process
/// unverified messages.
#[derive(Debug, Clone)]
pub struct AuthenticatedMessage {
    inner: P2PMessage,
    /// Public key of the verified sender
    sender_pubkey: [u8; 32],
}

/// Error returned when message authentication fails.
#[derive(Debug, Clone)]
pub enum AuthError {
    /// Signature verification failed
    InvalidSignature,
    /// Message format is invalid
    MalformedMessage(String),
    /// Sender is not in trusted peers list
    UntrustedSender { pubkey: [u8; 32] },
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthError::InvalidSignature => write!(f, "message signature verification failed"),
            AuthError::MalformedMessage(msg) => write!(f, "malformed message: {}", msg),
            AuthError::UntrustedSender { pubkey } => {
                write!(f, "untrusted sender: {:02x}{:02x}...", pubkey[0], pubkey[1])
            }
        }
    }
}

impl std::error::Error for AuthError {}

impl AuthenticatedMessage {
    /// Verify a raw P2P message and wrap it in an authenticated container.
    ///
    /// # Arguments
    /// * `msg` - Raw message from network
    ///
    /// # Returns
    /// * `Ok(AuthenticatedMessage)` - Message is cryptographically valid
    /// * `Err(AuthError)` - Verification failed
    ///
    /// # Security
    /// This is the ONLY way to create an `AuthenticatedMessage`.
    /// The type system guarantees that any `AuthenticatedMessage` has been verified.
    pub fn verify(msg: P2PMessage) -> Result<Self, AuthError> {
        if !msg.verify() {
            return Err(AuthError::InvalidSignature);
        }
        
        Ok(Self {
            sender_pubkey: msg.sender_pubkey,
            inner: msg,
        })
    }
    
    /// Verify a message and additionally check that sender is in trusted list.
    ///
    /// # Arguments
    /// * `msg` - Raw message from network
    /// * `trusted_keys` - List of trusted public keys
    ///
    /// # Returns
    /// * `Ok(AuthenticatedMessage)` - Message is valid AND sender is trusted
    /// * `Err(AuthError)` - Verification failed or sender not trusted
    pub fn verify_with_trust_list(
        msg: P2PMessage,
        trusted_keys: &[[u8; 32]],
    ) -> Result<Self, AuthError> {
        let verified = Self::verify(msg)?;
        
        if !trusted_keys.contains(&verified.sender_pubkey) {
            return Err(AuthError::UntrustedSender {
                pubkey: verified.sender_pubkey,
            });
        }
        
        Ok(verified)
    }
    
    /// Get the verified message.
    #[inline]
    pub fn message(&self) -> &P2PMessage {
        &self.inner
    }
    
    /// Consume and return the inner message.
    #[inline]
    pub fn into_inner(self) -> P2PMessage {
        self.inner
    }
    
    /// Get the verified sender's public key.
    #[inline]
    pub fn sender_pubkey(&self) -> &[u8; 32] {
        &self.sender_pubkey
    }
    
    /// Get the message payload.
    #[inline]
    pub fn payload(&self) -> &[u8] {
        &self.inner.payload
    }
    
    /// Get the message type.
    #[inline]
    pub fn msg_type(&self) -> &crate::message::MessageType {
        &self.inner.msg_type
    }
}

/// Extension trait for batch message verification.
pub trait BatchVerify {
    /// Verify all messages, returning only those that pass.
    /// Failed messages are silently dropped.
    fn verify_all(self) -> Vec<AuthenticatedMessage>;
    
    /// Verify all messages, returning results for each.
    fn verify_all_results(self) -> Vec<Result<AuthenticatedMessage, AuthError>>;
}

impl BatchVerify for Vec<P2PMessage> {
    fn verify_all(self) -> Vec<AuthenticatedMessage> {
        self.into_iter()
            .filter_map(|msg| AuthenticatedMessage::verify(msg).ok())
            .collect()
    }
    
    fn verify_all_results(self) -> Vec<Result<AuthenticatedMessage, AuthError>> {
        self.into_iter()
            .map(AuthenticatedMessage::verify)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::PeerIdentity;
    use crate::message::MessageType;
    
    #[test]
    fn test_verified_message() {
        let identity = PeerIdentity::generate();
        let msg = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"hello".to_vec())
            .sign_with_identity(&identity);
        
        let verified = AuthenticatedMessage::verify(msg).unwrap();
        assert_eq!(verified.payload(), b"hello");
        assert_eq!(verified.sender_pubkey(), &identity.public_key_bytes());
    }
    
    #[test]
    fn test_tampered_message_rejected() {
        let identity = PeerIdentity::generate();
        let mut msg = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"hello".to_vec())
            .sign_with_identity(&identity);
        
        // Tamper with payload
        msg.payload = b"ATTACK".to_vec();
        
        let result = AuthenticatedMessage::verify(msg);
        assert!(matches!(result, Err(AuthError::InvalidSignature)));
    }
    
    #[test]
    fn test_trust_list_accept() {
        let identity = PeerIdentity::generate();
        let trusted = [identity.public_key_bytes()];
        
        let msg = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"hello".to_vec())
            .sign_with_identity(&identity);
        
        let result = AuthenticatedMessage::verify_with_trust_list(msg, &trusted);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_trust_list_reject() {
        let identity = PeerIdentity::generate();
        let other = PeerIdentity::generate();
        let trusted = [other.public_key_bytes()]; // identity NOT in list
        
        let msg = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"hello".to_vec())
            .sign_with_identity(&identity);
        
        let result = AuthenticatedMessage::verify_with_trust_list(msg, &trusted);
        assert!(matches!(result, Err(AuthError::UntrustedSender { .. })));
    }
    
    #[test]
    fn test_batch_verify() {
        let identity = PeerIdentity::generate();
        
        let good = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"good".to_vec())
            .sign_with_identity(&identity);
        
        let mut bad = P2PMessage::new_unsigned(MessageType::PatternShare, "test", b"bad".to_vec())
            .sign_with_identity(&identity);
        bad.payload = b"TAMPERED".to_vec(); // Tamper
        
        let batch = vec![good, bad];
        let verified = batch.verify_all();
        
        assert_eq!(verified.len(), 1);
        assert_eq!(verified[0].payload(), b"good");
    }
}
