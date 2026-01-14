//! Defines the trait for embedding models and provides a basic implementation for testing.

/// An abstract interface for any model that can convert raw byte data into a
/// semantic vector embedding.
///
/// This trait allows the Ālaya store to be agnostic to the specific embedding
/// technology used, enabling easy swapping of models (e.g., from a test model
/// to Sentence Transformers or OpenAI's API).
pub trait EmbeddingModel: Send + Sync + std::fmt::Debug {
    /// Transforms a slice of bytes into a vector of floating-point numbers.
    fn embed(&self, data: &[u8]) -> Vec<f32>;

    /// Returns the dimensionality of the vectors produced by this model.
    fn dimension(&self) -> usize;
}

/// A basic, deterministic embedding model based on hashing.
///
/// This is intended **for testing and demonstration purposes only**. It does not
/// produce semantically meaningful embeddings. Its value is in its speed and
/// predictability, allowing for fast, reproducible tests without network calls
/// or heavy model loading.
#[derive(Debug)]
pub struct HashEmbedding {
    dim: usize,
}

impl HashEmbedding {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl EmbeddingModel for HashEmbedding {
    fn embed(&self, data: &[u8]) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut embedding = vec![0.0; self.dim];
        
        if data.is_empty() {
            return embedding;
        }

        // A simple hash-based projection. This is NOT a good embedding algorithm.
        for chunk in data.chunks(8) {
            let mut hasher = DefaultHasher::new();
            chunk.hash(&mut hasher);
            let hash = hasher.finish();
            
            let idx = (hash as usize) % self.dim;
            // Use sine to create some variance in the values
            embedding[idx] += (hash as f32).sin();
        }
        
        // Normalize the vector (L2 normalization) to have a unit length.
        // This is a common practice in semantic search.
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 { // Avoid division by zero
            embedding.iter_mut().for_each(|x| *x /= norm);
        }
        
        embedding
    }
    
    fn dimension(&self) -> usize {
        self.dim
    }
}
