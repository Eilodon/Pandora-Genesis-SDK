//! Defines the core `Experience` struct, the digital equivalent of a bīja (seed)
//! stored in the Ālaya-vijñāna.

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Represents a single, atomic experience or "seed" (bīja) to be stored in the Ālaya.
///
/// Each experience is a snapshot of a cognitive cycle, capturing not just the raw data,
/// but also the phenomenological qualities associated with it, such as its emotional
/// tone (karma_weight) and perceptual clarity (pattern_strength).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// A unique identifier for this experience.
    pub id: String,
    
    /// The semantic vector embedding of the experience, representing its "meaning".
    /// This is the primary key for similarity searches.
    pub embedding: Vec<f32>,
    
    /// The original, raw event data that triggered this experience.
    /// Kept for archival, debugging, and potential re-processing.
    pub event_data: Bytes,
    
    /// Rich metadata that provides context and allows for filtered retrieval.
    pub metadata: ExperienceMetadata,
    
    /// The timestamp when this experience was recorded in the Ālaya.
    pub timestamp: DateTime<Utc>,
}

/// Contextual metadata associated with an `Experience`.
///
/// This data captures the "karmic charge" and other qualitative aspects of the
/// experience, allowing for more nuanced retrieval queries (e.g., "find experiences
/// similar to this, but which felt negative").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceMetadata {
    /// The karmic weight (valence) from the Vedanā-skandha, ranging from -1.0 to 1.0.
    pub karma_weight: f32,
    
    /// The pattern strength or confidence from the Saññā-skandha, from 0.0 to 1.0.
    pub pattern_strength: f32,
    
    /// The name of the skandha stage that was the primary source of this experience.
    pub source_stage: String,
    
    /// A list of tags for easy categorical filtering.
    pub tags: Vec<String>,
    
    /// A flexible map for any other custom, string-based metadata.
    pub custom: std::collections::HashMap<String, String>,
}

impl Experience {
    /// Creates a new `Experience` with a generated UUID and current timestamp.
    pub fn new(
        embedding: Vec<f32>,
        event_data: Bytes,
        metadata: ExperienceMetadata,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            embedding,
            event_data,
            metadata,
            timestamp: Utc::now(),
        }
    }
}
