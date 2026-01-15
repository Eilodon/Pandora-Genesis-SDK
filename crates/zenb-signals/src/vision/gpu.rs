//! GPU Acceleration Support
//!
//! Provides device detection and GPU-accelerated operations via Candle.
//! Supports CUDA (NVIDIA), Metal (Apple Silicon), and CPU fallback.
//!
//! # Features
//!
//! - Device auto-detection with priority: CUDA > Metal > CPU
//! - Accelerated tensor operations for image processing
//! - Memory-efficient batch processing
//! - Async compute support (when available)
//!
//! # Example
//!
//! ```ignore
//! use zenb_signals::vision::gpu::{GpuDevice, get_best_device};
//!
//! let device = get_best_device();
//! println!("Using: {}", device.name());
//! ```

#[cfg(feature = "candle-detection")]
use candle_core::{Device, Tensor, DType, Result as CandleResult};

/// GPU acceleration capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal (macOS/iOS)
    Metal,
    /// CPU fallback
    Cpu,
}

impl GpuBackend {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA (NVIDIA GPU)",
            Self::Metal => "Metal (Apple GPU)",
            Self::Cpu => "CPU",
        }
    }
    
    /// Check if this is a GPU backend
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Backend type
    pub backend: GpuBackend,
    /// Device name (e.g., "NVIDIA RTX 4090")
    pub name: String,
    /// Total memory in bytes (if available)
    pub memory_total: Option<u64>,
    /// Free memory in bytes (if available)
    pub memory_free: Option<u64>,
    /// Compute capability (CUDA) or version (Metal)
    pub compute_version: Option<String>,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            backend: GpuBackend::Cpu,
            name: "CPU".to_string(),
            memory_total: None,
            memory_free: None,
            compute_version: None,
        }
    }
}

/// GPU device wrapper for Candle integration
#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub info: GpuInfo,
    #[cfg(feature = "candle-detection")]
    pub device: Device,
}

impl GpuDevice {
    /// Create CPU device
    pub fn cpu() -> Self {
        Self {
            info: GpuInfo::default(),
            #[cfg(feature = "candle-detection")]
            device: Device::Cpu,
        }
    }
    
    /// Try to create CUDA device
    #[cfg(feature = "candle-detection")]
    pub fn cuda(ordinal: usize) -> Result<Self, String> {
        match Device::cuda_if_available(ordinal) {
            Ok(device) if device.is_cuda() => {
                Ok(Self {
                    info: GpuInfo {
                        backend: GpuBackend::Cuda,
                        name: format!("CUDA Device {}", ordinal),
                        memory_total: None, // Would need cudarc for this
                        memory_free: None,
                        compute_version: None,
                    },
                    device,
                })
            }
            Ok(_) => Err("CUDA not available".to_string()),
            Err(e) => Err(format!("CUDA init failed: {:?}", e)),
        }
    }
    
    /// Try to create Metal device (macOS/iOS)
    #[cfg(feature = "candle-detection")]
    pub fn metal() -> Result<Self, String> {
        match Device::new_metal(0) {
            Ok(device) => Ok(Self {
                info: GpuInfo {
                    backend: GpuBackend::Metal,
                    name: "Apple GPU".to_string(),
                    memory_total: None,
                    memory_free: None,
                    compute_version: None,
                },
                device,
            }),
            Err(e) => Err(format!("Metal init failed: {:?}", e)),
        }
    }
    
    /// Get device name
    pub fn name(&self) -> &str {
        &self.info.name
    }
    
    /// Check if GPU
    pub fn is_gpu(&self) -> bool {
        self.info.backend.is_gpu()
    }
    
    /// Get backend type
    pub fn backend(&self) -> GpuBackend {
        self.info.backend
    }
}

impl Default for GpuDevice {
    fn default() -> Self {
        Self::cpu()
    }
}

/// Get the best available device (prioritizes GPU)
#[cfg(feature = "candle-detection")]
pub fn get_best_device() -> GpuDevice {
    // Try CUDA first
    if let Ok(device) = GpuDevice::cuda(0) {
        log::info!("Using CUDA device: {}", device.name());
        return device;
    }
    
    // Try Metal (macOS/iOS)
    if let Ok(device) = GpuDevice::metal() {
        log::info!("Using Metal device: {}", device.name());
        return device;
    }
    
    // Fallback to CPU
    log::info!("Using CPU (no GPU available)");
    GpuDevice::cpu()
}

/// Get the best available device (CPU fallback when no candle)
#[cfg(not(feature = "candle-detection"))]
pub fn get_best_device() -> GpuDevice {
    GpuDevice::cpu()
}

/// List all available devices
#[cfg(feature = "candle-detection")]
pub fn list_devices() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    
    // Check CUDA devices
    for i in 0..8 {
        if let Ok(device) = GpuDevice::cuda(i) {
            devices.push(device);
        } else {
            break;
        }
    }
    
    // Check Metal
    if let Ok(device) = GpuDevice::metal() {
        devices.push(device);
    }
    
    // Always include CPU
    devices.push(GpuDevice::cpu());
    
    devices
}

#[cfg(not(feature = "candle-detection"))]
pub fn list_devices() -> Vec<GpuDevice> {
    vec![GpuDevice::cpu()]
}

// === Accelerated Operations ===

/// GPU-accelerated image preprocessing
#[cfg(feature = "candle-detection")]
pub struct GpuImageOps {
    device: Device,
}

#[cfg(feature = "candle-detection")]
impl GpuImageOps {
    /// Create with specific device
    pub fn new(device: &GpuDevice) -> Self {
        Self { device: device.device.clone() }
    }
    
    /// Preprocess RGB image to normalized tensor
    ///
    /// Converts [H, W, 3] u8 image to [1, 3, H, W] f32 tensor
    /// with values normalized to [-1, 1]
    pub fn preprocess_image(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
    ) -> CandleResult<Tensor> {
        let hw = (height * width) as usize;
        let mut chw_data = vec![0.0f32; 3 * hw];
        
        for i in 0..hw {
            let r = rgb_data[i * 3] as f32 / 127.5 - 1.0;
            let g = rgb_data[i * 3 + 1] as f32 / 127.5 - 1.0;
            let b = rgb_data[i * 3 + 2] as f32 / 127.5 - 1.0;
            
            chw_data[i] = r;
            chw_data[hw + i] = g;
            chw_data[2 * hw + i] = b;
        }
        
        let tensor = Tensor::from_vec(chw_data, (1, 3, height as usize, width as usize), &self.device)?;
        Ok(tensor)
    }
    
    /// Resize image tensor using bilinear interpolation
    pub fn resize(
        &self,
        tensor: &Tensor,
        new_height: usize,
        new_width: usize,
    ) -> CandleResult<Tensor> {
        // Use Candle's interpolate when available
        // For now, simple nearest-neighbor via gather
        let (_batch, channels, old_h, old_w) = tensor.dims4()?;
        
        let y_ratio = old_h as f32 / new_height as f32;
        let x_ratio = old_w as f32 / new_width as f32;
        
        let mut indices = Vec::with_capacity(new_height * new_width);
        for y in 0..new_height {
            for x in 0..new_width {
                let src_y = ((y as f32 * y_ratio) as usize).min(old_h - 1);
                let src_x = ((x as f32 * x_ratio) as usize).min(old_w - 1);
                indices.push((src_y * old_w + src_x) as u32);
            }
        }
        
        let idx_tensor = Tensor::from_vec(indices.clone(), new_height * new_width, &self.device)?;
        
        // Reshape and gather for each channel
        let mut channel_tensors = Vec::with_capacity(channels);
        for c in 0..channels {
            let channel = tensor.i((0, c, .., ..))?;
            let flat = channel.reshape((old_h * old_w,))?;
            let gathered = flat.index_select(&idx_tensor, 0)?;
            let reshaped = gathered.reshape((1, 1, new_height, new_width))?;
            channel_tensors.push(reshaped);
        }
        
        Tensor::cat(&channel_tensors, 1)
    }
    
    /// Apply mean and std normalization
    pub fn normalize(
        &self,
        tensor: &Tensor,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> CandleResult<Tensor> {
        let mean_t = Tensor::from_vec(mean.to_vec(), (1, 3, 1, 1), &self.device)?;
        let std_t = Tensor::from_vec(std.to_vec(), (1, 3, 1, 1), &self.device)?;
        
        let normalized = tensor.broadcast_sub(&mean_t)?.broadcast_div(&std_t)?;
        Ok(normalized)
    }
}

/// Batch processor for efficient GPU utilization
#[cfg(feature = "candle-detection")]
pub struct BatchProcessor {
    device: Device,
    batch_size: usize,
    buffer: Vec<Tensor>,
}

#[cfg(feature = "candle-detection")]
impl BatchProcessor {
    /// Create new batch processor
    pub fn new(device: &GpuDevice, batch_size: usize) -> Self {
        Self {
            device: device.device.clone(),
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        }
    }
    
    /// Add tensor to batch
    pub fn add(&mut self, tensor: Tensor) -> Option<Tensor> {
        self.buffer.push(tensor);
        
        if self.buffer.len() >= self.batch_size {
            self.flush()
        } else {
            None
        }
    }
    
    /// Flush batch (pad if needed)
    pub fn flush(&mut self) -> Option<Tensor> {
        if self.buffer.is_empty() {
            return None;
        }
        
        let batch = match Tensor::cat(&self.buffer, 0) {
            Ok(t) => t,
            Err(e) => {
                log::error!("Batch concat failed: {:?}", e);
                self.buffer.clear();
                return None;
            }
        };
        
        self.buffer.clear();
        Some(batch)
    }
    
    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// === Performance Metrics ===

/// GPU performance metrics
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    /// Total inference time (ms)
    pub total_inference_ms: f64,
    /// Number of inferences
    pub inference_count: u64,
    /// Memory peak (bytes)
    pub memory_peak_bytes: u64,
    /// Throughput (frames per second)
    pub throughput_fps: f64,
}

impl GpuMetrics {
    /// Update metrics after inference
    pub fn record_inference(&mut self, duration_ms: f64) {
        self.total_inference_ms += duration_ms;
        self.inference_count += 1;
        
        if self.inference_count > 0 {
            let avg_ms = self.total_inference_ms / self.inference_count as f64;
            self.throughput_fps = 1000.0 / avg_ms;
        }
    }
    
    /// Get average inference time
    pub fn avg_inference_ms(&self) -> f64 {
        if self.inference_count > 0 {
            self.total_inference_ms / self.inference_count as f64
        } else {
            0.0
        }
    }
    
    /// Reset metrics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_names() {
        assert_eq!(GpuBackend::Cuda.name(), "CUDA (NVIDIA GPU)");
        assert_eq!(GpuBackend::Metal.name(), "Metal (Apple GPU)");
        assert_eq!(GpuBackend::Cpu.name(), "CPU");
    }

    #[test]
    fn test_gpu_backend_is_gpu() {
        assert!(GpuBackend::Cuda.is_gpu());
        assert!(GpuBackend::Metal.is_gpu());
        assert!(!GpuBackend::Cpu.is_gpu());
    }

    #[test]
    fn test_gpu_device_cpu_default() {
        let device = GpuDevice::cpu();
        assert!(!device.is_gpu());
        assert_eq!(device.backend(), GpuBackend::Cpu);
    }

    #[test]
    fn test_get_best_device() {
        let device = get_best_device();
        // Should at least return CPU
        assert!(!device.name().is_empty());
    }

    #[test]
    fn test_list_devices() {
        let devices = list_devices();
        assert!(!devices.is_empty());
        // Always has CPU
        assert!(devices.iter().any(|d| d.backend() == GpuBackend::Cpu));
    }

    #[test]
    fn test_gpu_metrics() {
        let mut metrics = GpuMetrics::default();
        metrics.record_inference(10.0);
        metrics.record_inference(20.0);
        
        assert_eq!(metrics.inference_count, 2);
        assert!((metrics.avg_inference_ms() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_gpu_metrics_reset() {
        let mut metrics = GpuMetrics::default();
        metrics.record_inference(10.0);
        metrics.reset();
        
        assert_eq!(metrics.inference_count, 0);
        assert_eq!(metrics.total_inference_ms, 0.0);
    }
}
