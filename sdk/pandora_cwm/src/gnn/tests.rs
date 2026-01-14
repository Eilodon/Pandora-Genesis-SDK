//! Comprehensive tests for the GNN pipeline and message passing

use crate::gnn::types::{CausalEdge, CausalEdgeKind, GnnConfig};
use crate::model::{CausalEdgeType, CausalHypothesis, InterdependentCausalModel};
use fnv::FnvHashSet;
use pandora_core::ontology::{DataEidos, EpistemologicalFlow};
use std::sync::Arc;

#[cfg(test)]
mod gnn_pipeline_tests {
    use super::*;

    #[test]
    fn test_learn_relations_to_infer_context_flow() {
        let config = GnnConfig::new(64, 128, 3);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Create a flow with intent and perception
        let mut flow = EpistemologicalFlow::default();
        flow.sankhara = Some(Arc::from("move_forward"));

        // Add some perception data
        let mut sanna = pandora_core::ontology::DataEidos {
            active_indices: fnv::FnvHashSet::default(),
            dimensionality: 32,
        };
        sanna.active_indices.insert(1); // door_locked
        sanna.active_indices.insert(3); // has_key
        flow.sanna = Some(sanna);

        // Learn relations from the flow
        cwm.learn_relations(&flow).unwrap();

        // Verify that nodes and edges were added
        assert!(cwm.gnn().node_count() > 0);
        assert!(cwm.gnn().edge_count() > 0);

        // Now infer context to enrich the flow
        cwm.infer_context(&mut flow).unwrap();

        // Verify that the flow was enriched
        assert!(flow.sanna.is_some());
        assert!(flow.related_eidos.is_some());

        // Check that sanna was updated with new concepts
        let sanna = flow.sanna.as_ref().unwrap();
        assert!(!sanna.active_indices.is_empty());

        // Check that related_eidos was populated
        let related_eidos = flow.related_eidos.as_ref().unwrap();
        assert!(!related_eidos.is_empty());
    }

    #[test]
    fn test_multi_edge_type_message_passing() {
        let config = GnnConfig::new(32, 64, 2);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Create flows with different types of relationships
        let mut flow1 = EpistemologicalFlow::default();
        flow1.sankhara = Some(Arc::from("pick_up_key")); // This creates Cause edges

        let mut flow2 = EpistemologicalFlow::default();
        flow2.sanna = Some(DataEidos {
            active_indices: [1, 2, 3].iter().cloned().collect::<FnvHashSet<u32>>(),
            dimensionality: 3,
        }); // This creates Correlate edges

        // Learn relations from both flows
        cwm.learn_relations(&flow1).unwrap();
        cwm.learn_relations(&flow2).unwrap();

        // Verify different edge types were created
        let graph = cwm.gnn().graph();
        let mut edge_types = std::collections::HashSet::new();

        for edge_idx in graph.edge_indices() {
            if let Some(edge) = graph.edge_weight(edge_idx) {
                edge_types.insert(edge.kind.clone());
            }
        }

        // Should have at least Cause and Correlate edges
        assert!(edge_types.contains(&CausalEdgeKind::Cause));
        assert!(edge_types.contains(&CausalEdgeKind::Correlate));
    }

    #[test]
    fn test_message_passing_rounds() {
        let config = GnnConfig::new(16, 32, 2);
        let cwm = InterdependentCausalModel::new(config).unwrap();

        // Create a test embedding
        let input_embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        // Perform multiple rounds of message passing
        let mut current_embedding = input_embedding.clone();
        for _ in 0..3 {
            current_embedding = cwm
                .perform_message_passing_round(&current_embedding)
                .unwrap();
        }

        // Verify that the embedding changed (but not too drastically)
        assert_ne!(current_embedding, input_embedding);

        // Verify that values are in reasonable range (tanh activation)
        for &value in &current_embedding {
            assert!(value >= -1.0 && value <= 1.0);
        }
    }

    #[test]
    fn test_embedding_creation_consistency() {
        let config = GnnConfig::new(32, 64, 2);
        let cwm = InterdependentCausalModel::new(config.clone()).unwrap();

        // Test intent embedding consistency
        let intent1 = "move_forward";
        let intent2 = "move_forward";
        let intent3 = "move_backward";

        let embedding1 = cwm.create_intent_embedding(intent1);
        let embedding2 = cwm.create_intent_embedding(intent2);
        let embedding3 = cwm.create_intent_embedding(intent3);

        // Same intent should produce similar embeddings
        assert_eq!(embedding1[0], embedding2[0]); // First dimension should be identical

        // Different intents should produce different embeddings
        assert_ne!(embedding1[0], embedding3[0]);

        // All embeddings should have correct dimensions
        assert_eq!(embedding1.len(), config.hidden_dims);
        assert_eq!(embedding2.len(), config.hidden_dims);
        assert_eq!(embedding3.len(), config.hidden_dims);
    }

    #[test]
    fn test_concept_embedding_deterministic() {
        let config = GnnConfig::new(32, 64, 2);
        let cwm = InterdependentCausalModel::new(config.clone()).unwrap();

        // Test concept embedding consistency
        let concept1 = 42u32;
        let concept2 = 42u32;
        let concept3 = 43u32;

        let embedding1 = cwm.create_concept_embedding(concept1);
        let embedding2 = cwm.create_concept_embedding(concept2);
        let embedding3 = cwm.create_concept_embedding(concept3);

        // Same concept should produce identical embeddings
        assert_eq!(embedding1, embedding2);

        // Different concepts should produce different embeddings
        assert_ne!(embedding1, embedding3);

        // All embeddings should have correct dimensions
        assert_eq!(embedding1.len(), config.hidden_dims);
    }

    #[test]
    fn test_flow_enrichment_preserves_original_data() {
        let config = GnnConfig::new(32, 64, 2);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Create a flow with specific data
        let mut flow = EpistemologicalFlow::default();
        flow.sankhara = Some(Arc::from("test_intent"));

        let mut sanna = pandora_core::ontology::DataEidos {
            active_indices: fnv::FnvHashSet::default(),
            dimensionality: 32,
        };
        sanna.active_indices.insert(5);
        sanna.active_indices.insert(10);
        flow.sanna = Some(sanna);

        // Learn relations first
        cwm.learn_relations(&flow).unwrap();

        // Store original data
        let original_sankhara = flow.sankhara.clone();
        let original_sanna_indices = flow.sanna.as_ref().map(|s| s.active_indices.clone());

        // Infer context
        cwm.infer_context(&mut flow).unwrap();

        // Verify original data is preserved
        assert_eq!(flow.sankhara, original_sankhara);

        if let (Some(original), Some(current)) = (original_sanna_indices, &flow.sanna) {
            // Original indices should still be present
            for &index in &original {
                assert!(current.active_indices.contains(&index));
            }
        }
    }

    #[test]
    fn test_error_handling_invalid_flow() {
        let config = GnnConfig::new(32, 64, 2);
        let cwm = InterdependentCausalModel::new(config).unwrap();

        // Test with empty flow
        let mut empty_flow = EpistemologicalFlow::default();
        let result = cwm.infer_context(&mut empty_flow);
        assert!(result.is_ok());

        // Test with flow that has no sankhara or sanna
        let mut minimal_flow = EpistemologicalFlow::default();
        minimal_flow.sankhara = None;
        minimal_flow.sanna = None;
        minimal_flow.related_eidos = None;

        let result = cwm.infer_context(&mut minimal_flow);
        assert!(result.is_ok());
    }

    #[test]
    fn test_large_scale_learning() {
        let config = GnnConfig::new(64, 128, 3);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Create many flows to test scalability
        for i in 0..50 {
            let mut flow = EpistemologicalFlow::default();
            flow.sankhara = Some(Arc::from(format!("action_{}", i)));

            let mut sanna = pandora_core::ontology::DataEidos {
                active_indices: fnv::FnvHashSet::default(),
                dimensionality: 32,
            };
            sanna.active_indices.insert(i as u32 % 20); // Cycle through concepts
            flow.sanna = Some(sanna);

            cwm.learn_relations(&flow).unwrap();
        }

        // Verify that the graph grew appropriately
        assert!(cwm.gnn().node_count() > 50);
        assert!(cwm.gnn().edge_count() > 50);

        // Test that inference still works with large graph
        let mut test_flow = EpistemologicalFlow::default();
        test_flow.sankhara = Some(Arc::from("test_action"));

        let result = cwm.infer_context(&mut test_flow);
        assert!(result.is_ok());
    }

    #[test]
    fn test_causal_link_detection() {
        let config = GnnConfig::new(32, 64, 2);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Add some nodes
        let node1 = cwm.gnn_mut().add_node(vec![1.0, 2.0, 3.0]);
        let node2 = cwm.gnn_mut().add_node(vec![4.0, 5.0, 6.0]);

        // Add an edge between them
        let edge = CausalEdge::new(CausalEdgeKind::Cause, 0.8);
        cwm.gnn_mut().add_edge(node1, node2, edge).unwrap();

        // Test causal link detection
        assert!(cwm.has_causal_link(node1.index(), node2.index()));
        assert!(!cwm.has_causal_link(node2.index(), node1.index())); // Directed edge

        // Test with non-existent nodes
        assert!(!cwm.has_causal_link(999, 1000));
    }

    #[test]
    fn test_crystallize_causal_link() {
        let config = GnnConfig::new(32, 64, 2);
        let mut cwm = InterdependentCausalModel::new(config).unwrap();

        // Create a causal hypothesis
        let hypothesis = CausalHypothesis {
            from_node_index: 0,
            to_node_index: 1,
            strength: 0.9,
            confidence: 0.95,
            edge_type: CausalEdgeType::Direct,
        };

        // Crystallize the link
        cwm.crystallize_causal_link(&hypothesis).unwrap();

        // Verify the link was added
        assert!(cwm.has_causal_link(0, 1));
        assert_eq!(cwm.gnn().edge_count(), 1);
    }
}
