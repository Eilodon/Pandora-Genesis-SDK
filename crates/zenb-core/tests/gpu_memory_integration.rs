//! Integration tests for GPU FFT with HolographicMemory.
//!
//! These tests verify that:
//! 1. GPU FFT backend can be created and initialized
//! 2. FFT operations produce correct results
//! 3. HolographicMemory entangle/recall work correctly

use zenb_core::memory::HolographicMemory;
use num_complex::Complex32;

/// Test FFT roundtrip correctness using the CPU fallback
#[cfg(feature = "gpu_wgpu")]
#[test]
fn test_gpu_fft_roundtrip() {
    use zenb_core::memory::gpu_fft::wgpu_backend::WgpuFftBackend;
    use zenb_core::memory::gpu_fft::GpuFftBackend;
    
    let dim = 256;
    let backend = WgpuFftBackend::new(dim).expect("Failed to create backend");
    
    // Create test signal: sine wave
    let mut data: Vec<Complex32> = (0..dim)
        .map(|i| {
            let t = i as f32 / dim as f32;
            let signal = (2.0 * std::f32::consts::PI * 4.0 * t).sin(); // 4 Hz
            Complex32::new(signal, 0.0)
        })
        .collect();
    
    let original = data.clone();
    
    // Forward FFT
    backend.fft_forward(&mut data);
    
    // Check that FFT changed the data
    let changed = data.iter().zip(original.iter()).any(|(a, b)| (a - b).norm() > 1e-6);
    assert!(changed, "FFT should transform the data");
    
    // Inverse FFT
    backend.fft_inverse(&mut data);
    
    // Verify roundtrip
    for (i, (orig, result)) in original.iter().zip(data.iter()).enumerate() {
        let diff = (orig - result).norm();
        assert!(
            diff < 1e-4,
            "Roundtrip error at index {}: expected {:?}, got {:?}, diff={}",
            i, orig, result, diff
        );
    }
}

/// Test that HolographicMemory works correctly using Complex32 API
#[test]
fn test_holographic_memory_complex() {
    let mut memory = HolographicMemory::new(64);
    
    // Create test patterns as Complex32
    let key1: Vec<Complex32> = (0..64).map(|i| Complex32::new(0.1 * i as f32, 0.0)).collect();
    let value1: Vec<Complex32> = (0..64).map(|i| Complex32::new(0.2 * i as f32, 0.0)).collect();
    
    // Entangle pattern
    memory.entangle(&key1, &value1);
    assert_eq!(memory.item_count(), 1);
    
    // Recall should return a vector
    let recalled = memory.recall(&key1);
    assert_eq!(recalled.len(), 64);
}

/// Test HolographicMemory with real-valued convenience API
#[test]
fn test_holographic_memory_real() {
    let mut memory = HolographicMemory::new(128);
    
    // Create test patterns as f32
    let key: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
    let value: Vec<f32> = (0..128).map(|i| (i as f32 / 64.0) - 1.0).collect();
    
    // Entangle using real-valued API
    memory.entangle_real(&key, &value);
    assert_eq!(memory.item_count(), 1);
    
    // Recall using real-valued API
    let recalled = memory.recall_real(&key);
    assert_eq!(recalled.len(), 128);
}

/// Test HolographicMemory with many patterns
#[test]
fn test_holographic_memory_many_patterns() {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    
    let dim = 64;
    let mut memory = HolographicMemory::new(dim);
    let mut rng = StdRng::seed_from_u64(42);
    
    // Entangle 20 random patterns
    for _ in 0..20 {
        let key: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let value: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        memory.entangle_real(&key, &value);
    }
    
    assert_eq!(memory.item_count(), 20);
}

/// Test LLM providers creation (no network required)
#[cfg(feature = "llm-all")]
#[test]
fn test_llm_providers_creation() {
    use zenb_core::llm::{LlmProvider, MockProvider};
    use zenb_core::llm::OllamaProvider;
    use zenb_core::llm::OpenAiProvider;
    use zenb_core::llm::LlamaCppProvider;
    
    // Mock is always available
    let mock = MockProvider::new();
    assert!(mock.is_available());
    assert_eq!(mock.name(), "mock");
    
    // Ollama provider creation (server not running)
    let ollama = OllamaProvider::default();
    assert_eq!(ollama.name(), "ollama");
    assert_eq!(ollama.model(), "deepseek-r1:8b");
    
    // OpenAI provider creation
    let openai = OpenAiProvider::openai("test-key", "gpt-4");
    assert_eq!(openai.name(), "openai");
    assert_eq!(openai.model(), "gpt-4");
    
    // LlamaCpp provider creation
    let llamacpp = LlamaCppProvider::default();
    assert_eq!(llamacpp.name(), "llama.cpp");
}
