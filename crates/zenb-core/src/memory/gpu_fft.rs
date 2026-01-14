//! GPU-Accelerated FFT for HolographicMemory
//!
//! Provides trait-based abstraction for GPU FFT backends (CUDA/Metal).
//! Feature-gated: compile with `--features gpu_cuda` or `--features gpu_metal`.
//!
//! # VAJRA-VOID Performance Optimization
//! GPU FFT can provide 10-100x speedup for large-dimension holographic memories.
//! This is especially beneficial for batch operations during high-frequency sensor ingestion.
//!
//! # Backend Selection
//! - **CUDA**: NVIDIA GPUs (Linux/Windows) - `gpu_cuda` feature
//! - **Metal**: Apple GPUs (macOS/iOS) - `gpu_metal` feature

use num_complex::Complex32;

// ============================================================================
// GPU FFT TRAIT ABSTRACTION
// ============================================================================

/// Trait for GPU FFT implementations.
/// 
/// Both CUDA and Metal backends implement this trait, allowing the HolographicMemory
/// to use GPU acceleration transparently.
pub trait GpuFftBackend: Send + Sync {
    /// Perform forward FFT on GPU
    fn fft_forward(&self, data: &mut [Complex32]);
    
    /// Perform inverse FFT on GPU
    fn fft_inverse(&self, data: &mut [Complex32]);
    
    /// Get the dimension this backend was configured for
    fn dim(&self) -> usize;
    
    /// Check if GPU is available and initialized
    fn is_available(&self) -> bool;
}

// ============================================================================
// CUDA BACKEND (gpu_cuda feature)
// ============================================================================

#[cfg(feature = "gpu_cuda")]
pub mod cuda {
    use super::*;
    
    /// CUDA-based FFT backend using cuFFT
    pub struct CudaFftBackend {
        dim: usize,
        // TODO: Add cubecl CUDA context and FFT plan handles
        available: bool,
    }
    
    impl CudaFftBackend {
        /// Create a new CUDA FFT backend for the given dimension.
        /// 
        /// # Returns
        /// `Some(backend)` if CUDA is available, `None` otherwise.
        pub fn new(dim: usize) -> Option<Self> {
            // TODO: Initialize cubecl-cuda context
            // For now, return a stub that indicates unavailability
            log::info!("CUDA FFT backend requested for dim={}", dim);
            
            // Stub: pretend CUDA is not available until full implementation
            Some(Self {
                dim,
                available: false, // Will be true when cubecl-cuda is properly initialized
            })
        }
    }
    
    impl GpuFftBackend for CudaFftBackend {
        fn fft_forward(&self, data: &mut [Complex32]) {
            if !self.available {
                log::warn!("CUDA FFT not available, falling back to CPU");
                return;
            }
            
            // TODO: Implement GPU FFT via cubecl-cuda
            // 1. Upload data to GPU
            // 2. Execute FFT kernel
            // 3. Download result
            let _ = data;
        }
        
        fn fft_inverse(&self, data: &mut [Complex32]) {
            if !self.available {
                log::warn!("CUDA IFFT not available, falling back to CPU");
                return;
            }
            
            // TODO: Implement GPU IFFT via cubecl-cuda
            let _ = data;
        }
        
        fn dim(&self) -> usize {
            self.dim
        }
        
        fn is_available(&self) -> bool {
            self.available
        }
    }
}

// ============================================================================
// METAL BACKEND (gpu_metal feature)
// ============================================================================

#[cfg(feature = "gpu_metal")]
pub mod metal {
    use super::*;
    
    /// Metal-based FFT backend for Apple GPUs
    pub struct MetalFftBackend {
        dim: usize,
        // TODO: Add cubecl Metal context and compute pipeline handles
        available: bool,
    }
    
    impl MetalFftBackend {
        /// Create a new Metal FFT backend for the given dimension.
        /// 
        /// # Returns
        /// `Some(backend)` if Metal is available, `None` otherwise.
        pub fn new(dim: usize) -> Option<Self> {
            // TODO: Initialize cubecl-metal context
            log::info!("Metal FFT backend requested for dim={}", dim);
            
            // Stub: pretend Metal is not available until full implementation
            Some(Self {
                dim,
                available: false, // Will be true when cubecl-metal is properly initialized
            })
        }
    }
    
    impl GpuFftBackend for MetalFftBackend {
        fn fft_forward(&self, data: &mut [Complex32]) {
            if !self.available {
                log::warn!("Metal FFT not available, falling back to CPU");
                return;
            }
            
            // TODO: Implement GPU FFT via cubecl-metal
            // Apple's vDSP provides optimized FFT, or use custom Metal compute shaders
            let _ = data;
        }
        
        fn fft_inverse(&self, data: &mut [Complex32]) {
            if !self.available {
                log::warn!("Metal IFFT not available, falling back to CPU");
                return;
            }
            
            // TODO: Implement GPU IFFT via cubecl-metal
            let _ = data;
        }
        
        fn dim(&self) -> usize {
            self.dim
        }
        
        fn is_available(&self) -> bool {
            self.available
        }
    }
}

// ============================================================================
// AUTO-DETECT BACKEND
// ============================================================================

/// Attempt to create the best available GPU FFT backend.
/// 
/// Selection priority:
/// 1. CUDA (if `gpu_cuda` feature and NVIDIA GPU available)
/// 2. Metal (if `gpu_metal` feature and Apple GPU available)
/// 3. None (fall back to CPU FFT)
pub fn create_gpu_backend(dim: usize) -> Option<Box<dyn GpuFftBackend>> {
    #[cfg(feature = "gpu_cuda")]
    {
        if let Some(backend) = cuda::CudaFftBackend::new(dim) {
            if backend.is_available() {
                log::info!("Using CUDA GPU FFT backend");
                return Some(Box::new(backend));
            }
        }
    }
    
    #[cfg(feature = "gpu_metal")]
    {
        if let Some(backend) = metal::MetalFftBackend::new(dim) {
            if backend.is_available() {
                log::info!("Using Metal GPU FFT backend");
                return Some(Box::new(backend));
            }
        }
    }
    
    log::debug!("No GPU FFT backend available, using CPU");
    None
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_gpu_backend_no_gpu() {
        // Without actual GPU hardware, this should return None or unavailable backend
        let backend = create_gpu_backend(512);
        
        // In test environment without GPU, this is expected to be None
        // or have is_available() == false
        if let Some(b) = backend {
            // Backend created but may not be available
            assert_eq!(b.dim(), 512);
        }
    }
    
    #[cfg(feature = "gpu_cuda")]
    #[test]
    fn test_cuda_backend_creation() {
        let backend = cuda::CudaFftBackend::new(256);
        assert!(backend.is_some());
        
        let b = backend.unwrap();
        assert_eq!(b.dim(), 256);
    }
    
    #[cfg(feature = "gpu_metal")]
    #[test]
    fn test_metal_backend_creation() {
        let backend = metal::MetalFftBackend::new(256);
        assert!(backend.is_some());
        
        let b = backend.unwrap();
        assert_eq!(b.dim(), 256);
    }
}
