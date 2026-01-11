//! Factory for creating pre-configured skandha processors.
//! Hides complexity behind simple APIs for common use cases.

use crate::alaya::{AlayaStore, HashEmbedding};
use crate::skandha_implementations::{
    basic::*,
    stateful::{adapters::StatelessAdapter, StatefulSanna, StatefulVedana},
    processors::{LinearProcessor, RecurrentProcessor},
};
use std::sync::Arc;

/// Pre-configured processor presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorPreset {
    /// V1: Fast, stateless, deterministic.
    Linear,
    /// V2: Stateful mood and pattern memory, but no long-term Ālaya memory.
    StatefulOnly,
    /// V2 Full: Stateful with Ālaya long-term memory. (FOR TESTING)
    StatefulWithAlaya,
}

/// Main factory for creating processors.
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// Creates a LinearProcessor.
    pub fn create_linear() -> LinearProcessor {
        LinearProcessor::new(
            Box::new(BasicRupaSkandha::default()),
            Box::new(BasicVedanaSkandha::default()),
            Box::new(BasicSannaSkandha::default()),
            Box::new(BasicSankharaSkandha::default()),
            Box::new(BasicVinnanaSkandha::default()),
        )
    }

    /// Creates a RecurrentProcessor from a preset.
    /// Note: This is now an async function because creating the AlayaStore is async.
    pub async fn create_recurrent(preset: ProcessorPreset) -> RecurrentProcessor<StatefulVedana, StatefulSanna> {
        let embedding_model = Arc::new(HashEmbedding::new(64)); // Use a consistent test model

        match preset {
            ProcessorPreset::StatefulOnly => RecurrentProcessor::new(
                Box::new(BasicRupaSkandha::default()),
                StatefulVedana::new("StatefulVedana", Arc::new(BasicVedanaSkandha::default()), embedding_model.clone()),
                StatefulSanna::new("StatefulSanna", Arc::new(BasicSannaSkandha::default())),
                Box::new(StatelessAdapter::new(BasicSankharaSkandha::default())),
                Box::new(StatelessAdapter::new(BasicVinnanaSkandha::default())),
            ),
            ProcessorPreset::StatefulWithAlaya => {
                let collection_name = format!("pandora_test_{}", uuid::Uuid::new_v4());
                let alaya = Arc::new(
                    AlayaStore::new("http://localhost:6333", collection_name, 64)
                        .await
                        .expect("Factory failed to connect to Qdrant."),
                );

                let vedana = StatefulVedana::new(
                    "StatefulVedanaWithAlaya",
                    Arc::new(BasicVedanaSkandha::default()),
                    embedding_model.clone(),
                )
                .with_alaya(alaya);

                RecurrentProcessor::new(
                    Box::new(BasicRupaSkandha::default()),
                    vedana,
                    StatefulSanna::new("StatefulSanna", Arc::new(BasicSannaSkandha::default())),
                    Box::new(StatelessAdapter::new(BasicSankharaSkandha::default())),
                    Box::new(StatelessAdapter::new(BasicVinnanaSkandha::default())),
                )
            }
            _ => panic!("This preset is not for RecurrentProcessor"),
        }
    }
}
