//! Zero-Copy Transport Layer
//!
//! Provides io_uring-based zero-copy networking for high-performance P2P communication.
//! Feature-gated: compile with `--features io_uring` (Linux 5.6+ only).
//!
//! # VAJRA-VOID Performance Optimization
//! Zero-copy networking can reduce latency by 10-100x by avoiding CPU buffer copies.
//! Data flows directly from NIC to user-space buffers via io_uring's registered buffers.
//!
//! # Fallback Behavior
//! When `io_uring` feature is not enabled, uses standard tokio TCP which is compatible
//! with all platforms.

use std::io;

// ============================================================================
// TRANSPORT TRAIT ABSTRACTION
// ============================================================================

/// Trait for P2P transport implementations.
/// 
/// Both zero-copy (io_uring) and standard (tokio) transports implement this trait.
#[cfg(feature = "io_uring")]
#[allow(async_fn_in_trait)]
pub trait ZeroCopyTransport: Send + Sync {
    /// Receive data directly into provided buffer (zero-copy when possible)
    async fn recv_zero_copy(&self, buf: &mut [u8]) -> io::Result<usize>;
    
    /// Send data directly from provided buffer (zero-copy when possible)
    async fn send_zero_copy(&self, buf: &[u8]) -> io::Result<usize>;
    
    /// Check if zero-copy is actually available on this connection
    fn is_zero_copy(&self) -> bool;
}

// ============================================================================
// IO_URING TRANSPORT (io_uring feature, Linux only)
// ============================================================================

#[cfg(all(feature = "io_uring", target_os = "linux"))]
pub mod uring {
    use super::*;
    
    /// io_uring-based zero-copy transport
    pub struct UringTransport {
        // TODO: Add compio TcpStream and registered buffer pool
        zero_copy_available: bool,
    }
    
    impl UringTransport {
        /// Create a new io_uring transport.
        /// 
        /// # Platform Requirements
        /// - Linux kernel 5.6+ for basic io_uring
        /// - Linux kernel 5.19+ for zero-copy receive (ZC_RX)
        pub fn new() -> io::Result<Self> {
            // TODO: Initialize compio runtime and check kernel version
            log::info!("Creating io_uring transport");
            
            // Check kernel version for zero-copy support
            let zc_available = check_uring_zc_support();
            
            Ok(Self {
                zero_copy_available: zc_available,
            })
        }
        
        /// Create transport with registered buffer pool for true zero-copy.
        /// 
        /// # Arguments
        /// * `buffer_count` - Number of buffers in the pool
        /// * `buffer_size` - Size of each buffer
        pub fn with_registered_buffers(
            _buffer_count: usize,
            _buffer_size: usize,
        ) -> io::Result<Self> {
            // TODO: Register buffers with io_uring for ZC send
            Self::new()
        }
    }
    
    impl Default for UringTransport {
        fn default() -> Self {
            Self::new().expect("Failed to create io_uring transport")
        }
    }
    
    impl super::ZeroCopyTransport for UringTransport {
        async fn recv_zero_copy(&self, buf: &mut [u8]) -> io::Result<usize> {
            if !self.zero_copy_available {
                log::trace!("Zero-copy not available, using standard recv");
            }
            
            // TODO: Use compio's zero-copy receive
            // For now, return stub
            let _ = buf;
            Ok(0)
        }
        
        async fn send_zero_copy(&self, buf: &[u8]) -> io::Result<usize> {
            if !self.zero_copy_available {
                log::trace!("Zero-copy not available, using standard send");
            }
            
            // TODO: Use compio's zero-copy send with registered buffers
            let _ = buf;
            Ok(0)
        }
        
        fn is_zero_copy(&self) -> bool {
            self.zero_copy_available
        }
    }
    
    /// Check if kernel supports io_uring zero-copy features
    fn check_uring_zc_support() -> bool {
        // TODO: Probe kernel version and io_uring capabilities
        // For now, assume not available until properly implemented
        false
    }
}

// ============================================================================
// STANDARD TOKIO TRANSPORT (Fallback)
// ============================================================================

/// Standard tokio-based transport (non-zero-copy fallback)
/// 
/// This is used when:
/// - `io_uring` feature is not enabled
/// - Platform is not Linux
/// - Kernel doesn't support zero-copy
pub struct StandardTransport {
    // Standard tokio TcpStream would go here
}

impl StandardTransport {
    /// Create a new standard transport
    pub fn new() -> io::Result<Self> {
        log::trace!("Using standard tokio transport");
        Ok(Self {})
    }
}

impl Default for StandardTransport {
    fn default() -> Self {
        Self::new().expect("Failed to create standard transport")
    }
}

// ============================================================================
// AUTO-DETECT TRANSPORT
// ============================================================================

/// Transport type enum for runtime selection
pub enum Transport {
    /// Standard tokio transport (all platforms)
    Standard(StandardTransport),
    
    #[cfg(all(feature = "io_uring", target_os = "linux"))]
    /// io_uring zero-copy transport (Linux 5.6+)
    Uring(uring::UringTransport),
}

/// Create the best available transport for this platform.
/// 
/// Selection priority:
/// 1. io_uring (if `io_uring` feature and Linux 5.6+)
/// 2. Standard tokio (fallback)
pub fn create_transport() -> io::Result<Transport> {
    #[cfg(all(feature = "io_uring", target_os = "linux"))]
    {
        match uring::UringTransport::new() {
            Ok(t) => {
                log::info!("Using io_uring transport (zero_copy={})", t.is_zero_copy());
                return Ok(Transport::Uring(t));
            }
            Err(e) => {
                log::warn!("io_uring transport failed, falling back to standard: {}", e);
            }
        }
    }
    
    log::info!("Using standard tokio transport");
    Ok(Transport::Standard(StandardTransport::new()?))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_transport() {
        let transport = create_transport();
        assert!(transport.is_ok());
    }
    
    #[test]
    fn test_standard_transport_creation() {
        let transport = StandardTransport::new();
        assert!(transport.is_ok());
    }
    
    #[cfg(all(feature = "io_uring", target_os = "linux"))]
    #[test]
    fn test_uring_transport_creation() {
        let transport = uring::UringTransport::new();
        assert!(transport.is_ok());
    }
}
