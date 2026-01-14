//! Stateful Vedanā implementation with mood tracking and long-term memory (Ālaya).

use crate::alaya::{AlayaStore, EmbeddingModel, Experience, ExperienceMetadata};
use crate::ontology::{EpistemologicalFlow, Vedana};
use crate::skandha_implementations::core::{
    state_management::MoodState,
    traits::{Skandha, StatefulSkandha, StatefulVedanaSkandha, VedanaSkandha},
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::debug;

/// Stateful Vedanā with both short-term mood and long-term experiential memory.
///
/// This advanced implementation integrates with the Ālaya vector store to allow
/// past experiences to influence present feelings.
pub struct StatefulVedana {
    name: String,
    inner_vedana: Arc<dyn VedanaSkandha>,
    state: Arc<Mutex<MoodState>>,
    alaya: Option<Arc<AlayaStore>>,
    embedding_model: Arc<dyn EmbeddingModel>,
}

impl StatefulVedana {
    /// Create a new `StatefulVedana` with an embedding model.
    pub fn new(
        name: &'static str,
        inner_vedana: Arc<dyn VedanaSkandha>,
        embedding_model: Arc<dyn EmbeddingModel>,
    ) -> Self {
        Self {
            name: name.to_string(),
            inner_vedana,
            state: Arc::new(Mutex::new(MoodState::default())),
            alaya: None,
            embedding_model,
        }
    }

    /// Builder method to attach an Ālaya store for long-term memory.
    pub fn with_alaya(mut self, alaya: Arc<AlayaStore>) -> Self {
        self.alaya = Some(alaya);
        self
    }

    /// Returns a clone of the current mood state for inspection.
    pub fn get_mood_state(&self) -> MoodState {
        self.state.lock().clone()
    }

    /// Adjusts the current feeling based on the average karma of similar past experiences.
    fn adjust_from_memory(&self, similar: &[(Experience, f32)], flow: &mut EpistemologicalFlow) {
        if similar.is_empty() {
            return;
        }

        // Calculate the average karma of past similar experiences.
        let avg_karma: f32 = similar
            .iter()
            .map(|(e, _score)| e.metadata.karma_weight)
            .sum::<f32>()
            / similar.len() as f32;

        debug!(
            "Adjusting feeling based on average past karma of {}",
            avg_karma
        );

        if let Some(vedana) = &mut flow.vedana {
            if let Vedana::Unpleasant { karma_weight } = vedana {
                // If past experiences were also significantly negative, amplify the current negative feeling.
                // This models trauma or conditioned negative responses.
                if avg_karma < -0.5 {
                    *karma_weight = (*karma_weight * 0.7 + avg_karma * 0.3).clamp(-1.0, 1.0);
                }
            }
            // Similar logic could be added for pleasant feelings.
        }
    }
}

impl std::fmt::Debug for StatefulVedana {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatefulVedana")
            .field("name", &self.name)
            .field("state", &self.state.lock())
            .field("has_alaya", &self.alaya.is_some())
            .finish()
    }
}

impl Skandha for StatefulVedana {
    fn name(&self) -> &'static str {
        "StatefulVedanaWithAlaya"
    }
}

// Note: The base `VedanaSkandha` trait is sync. For async operations, a new trait is needed.
// For now, we provide a blocking implementation for compatibility, though it's not ideal.
impl VedanaSkandha for StatefulVedana {
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        self.inner_vedana.feel(flow);
    }
}

impl StatefulSkandha for StatefulVedana {
    type State = MoodState;

    fn state(&self) -> &Self::State {
        // Note: Direct access to state under mutex requires unsafe or different API design.
        // This implementation maintains the mutex-based approach for thread safety.
        unimplemented!("Direct access to state under mutex is not provided for safety. Use state.lock() instead.");
    }

    fn state_mut(&mut self) -> &mut Self::State {
        unimplemented!("Direct access to state under mutex is not provided for safety. Use state.lock() instead.");
    }

    fn reset(&mut self) {
        let mut state = self.state.lock();
        state.valence = 0.0;
        state.arousal = 0.0;
    }
}

#[async_trait]
impl StatefulVedanaSkandha for StatefulVedana {
    /// The primary async method that incorporates both mood (short-term) and Ālaya (long-term) memory.
    async fn feel_with_state(&mut self, flow: &mut EpistemologicalFlow) {
        // 1. Get base feeling from the inner stateless skandha.
        self.inner_vedana.feel(flow);

        // 2. Query Ālaya for similar past experiences, if configured.
        if let (Some(alaya), Some(rupa)) = (&self.alaya, &flow.rupa) {
            let embedding = self.embedding_model.embed(rupa);
            if let Ok(similar) = alaya.retrieve_similar(&embedding, 5, None).await {
                // 3. Adjust current feeling based on past experiences.
                self.adjust_from_memory(&similar, flow);
            }
        }

        // 4. Update short-term mood state based on the (potentially adjusted) feeling.
        if let Some(vedana) = flow.vedana.clone() {
            let mut mood = self.state.lock();
            let (feeling_valence, feeling_intensity) = vedana.to_valence_intensity();
            mood.update(feeling_valence, feeling_intensity);

            // The final vedana in the flow reflects the updated mood.
            let final_vedana = Vedana::from_valence_intensity(mood.valence, mood.arousal);
            flow.vedana = Some(final_vedana);
        }

        // 5. Store this new experience back into Ālaya for the future.
        if let (Some(alaya), Some(rupa), Some(vedana)) = (&self.alaya, &flow.rupa, &flow.vedana) {
            let embedding = self.embedding_model.embed(rupa);
            let experience = Experience::new(
                embedding,
                rupa.clone(),
                ExperienceMetadata {
                    karma_weight: vedana.get_karma_weight(),
                    pattern_strength: 0.0, // Sanna would fill this
                    source_stage: "vedana".to_string(),
                    tags: vec![],
                    custom: Default::default(),
                },
            );
            if let Err(e) = alaya.store(experience).await {
                tracing::error!("Failed to store experience in Ālaya: {}", e);
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alaya::HashEmbedding;
    use crate::skandha_implementations::basic::BasicVedanaSkandha;
    use bytes::Bytes;
    use std::time::Duration;

    async fn create_test_alaya() -> Arc<AlayaStore> {
        let store = AlayaStore::new(
            "http://localhost:6333",
            "vedana-test-collection".to_string(),
            32,
        )
        .await
        .expect("Failed to connect to Qdrant");
        Arc::new(store)
    }

    fn create_flow(content: &'static str) -> EpistemologicalFlow {
        EpistemologicalFlow::from_bytes(Bytes::from(content))
    }

    #[tokio::test]
    async fn test_vedana_with_memory_amplifies_negative_feeling() {
        let alaya = create_test_alaya().await;
        let embedding_model = Arc::new(HashEmbedding::new(32));

        let mut vedana =
            StatefulVedana::new("TestVedana", Arc::new(BasicVedanaSkandha), embedding_model)
                .with_alaya(alaya);

        let error_event = "database connection error";

        // --- First pass ---
        let mut flow1 = create_flow(error_event);
        vedana.feel_with_state(&mut flow1).await;
        let karma1 = flow1.vedana.as_ref().unwrap().get_karma_weight();
        assert!(karma1 < 0.0, "Karma should be negative");

        // Allow time for the experience to be stored.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // --- Second pass ---
        // Now, when the same error occurs, Ālaya should find the previous negative experience.
        let mut flow2 = create_flow(error_event);
        vedana.feel_with_state(&mut flow2).await;
        let karma2 = flow2.vedana.as_ref().unwrap().get_karma_weight();

        // The feeling should be *more* negative due to memory of the last failure.
        assert!(
            karma2 < karma1,
            "Karma should become more negative after recalling a past failure"
        );
    }
}
