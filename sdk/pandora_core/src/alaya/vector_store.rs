//! The implementation of the Ālaya-vijñāna (store-house consciousness) using
//! an in-memory vector store for demonstration purposes.
//!
//! NOTE: This is a simplified in-memory implementation. For production use,
//! integrate with a real vector database like Qdrant or Milvus.

use crate::alaya::experience::Experience;
use parking_lot::Mutex;
use std::sync::Arc;

/// A client for the Ālaya store using in-memory storage.
///
/// This is a simplified implementation for demonstration and testing.
/// For production, this should be replaced with a proper vector database backend.
#[derive(Clone)]
pub struct AlayaStore {
    experiences: Arc<Mutex<Vec<Experience>>>,
    collection_name: String,
    embedding_dim: usize,
}

/// Statistics about the state of the Ālaya store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlayaStats {
    pub total_experiences: u64,
    pub collection_name: String,
}

impl AlayaStore {
    /// Initializes the Ālaya store with in-memory storage.
    ///
    /// # Parameters
    /// - `_url`: Unused in this implementation (for API compatibility)
    /// - `collection_name`: Name of the collection
    /// - `embedding_dim`: Dimensionality of embeddings
    pub async fn new(
        _url: &str,
        collection_name: String,
        embedding_dim: u64,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self {
            experiences: Arc::new(Mutex::new(Vec::new())),
            collection_name,
            embedding_dim: embedding_dim as usize,
        })
    }
    
    /// Stores a single `Experience` in the vector store.
    pub async fn store(&self, experience: Experience) -> Result<(), anyhow::Error> {
        let mut experiences = self.experiences.lock();
        experiences.push(experience);
        Ok(())
    }
    
    /// Retrieves experiences semantically similar to a query vector using cosine similarity.
    ///
    /// # Parameters
    /// - `query_embedding`: The query vector
    /// - `limit`: Maximum number of results to return
    /// - `min_karma_weight`: Optional filter for minimum karma weight
    ///
    /// # Returns
    /// Vector of (Experience, similarity_score) tuples, sorted by similarity (highest first)
    pub async fn retrieve_similar(
        &self,
        query_embedding: &[f32],
        limit: u64,
        min_karma_weight: Option<f32>,
    ) -> Result<Vec<(Experience, f32)>, anyhow::Error> {
        let experiences = self.experiences.lock();
        
        let mut scored: Vec<(Experience, f32)> = experiences
            .iter()
            .filter(|exp| {
                // Apply karma weight filter if specified
                if let Some(min_weight) = min_karma_weight {
                    exp.metadata.karma_weight >= min_weight
                } else {
                    true
                }
            })
            .map(|exp| {
                // Calculate cosine similarity
                let similarity = cosine_similarity(query_embedding, &exp.embedding);
                (exp.clone(), similarity)
            })
            .collect();
        
        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top `limit` results
        scored.truncate(limit as usize);
        
        Ok(scored)
    }

    /// Retrieves statistics about the collection.
    pub async fn stats(&self) -> Result<AlayaStats, anyhow::Error> {
        let experiences = self.experiences.lock();
        Ok(AlayaStats {
            total_experiences: experiences.len() as u64,
            collection_name: self.collection_name.clone(),
        })
    }
}

/// Calculates cosine similarity between two vectors.
///
/// Returns a value between -1.0 and 1.0, where:
/// - 1.0 means vectors point in the same direction
/// - 0.0 means vectors are orthogonal
/// - -1.0 means vectors point in opposite directions
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    
    dot_product / (norm_a * norm_b)
}

impl std::fmt::Debug for AlayaStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlayaStore")
            .field("collection_name", &self.collection_name)
            .field("embedding_dim", &self.embedding_dim)
            .field("num_experiences", &self.experiences.lock().len())
            .finish()
    }
}
