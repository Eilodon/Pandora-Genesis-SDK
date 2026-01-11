//! Ālaya-vijñāna (Store-house consciousness)
//!
//! This module provides the infrastructure for a long-term, semantic memory store,
//! allowing the system to learn from past experiences. It is the digital equivalent
//! of the "store-house consciousness" which holds all the seeds (bīja) of past karma.

pub mod vector_store;
pub mod experience;
pub mod embedding;

pub use vector_store::AlayaStore;
pub use experience::{Experience, ExperienceMetadata};
pub use embedding::{EmbeddingModel, HashEmbedding};
