//! Integration tests for CWM v3.0 core functionality
//!
//! This test suite verifies that the InterdependentCausalModel can be instantiated,
//! that causal relationships can be added to the graph, and that the model can
//! perform basic inference operations.

use pandora_core::ontology::EpistemologicalFlow;
use pandora_cwm::gnn::types::{CausalEdge, CausalEdgeKind, GnnConfig};
use pandora_cwm::model::InterdependentCausalModel;

/// Test basic model instantiation and configuration
#[test]
fn test_cwm_v3_model_instantiation() {
    let config = GnnConfig::new(64, 128, 3);
    let model_result = InterdependentCausalModel::new(config);

    assert!(model_result.is_ok(), "Model should be created successfully");

    let model = model_result.unwrap();
    assert_eq!(model.gnn().node_count(), 0, "Initial graph should be empty");
    assert_eq!(
        model.gnn().edge_count(),
        0,
        "Initial graph should have no edges"
    );
}

/// Test adding causal relationships to the model
#[test]
fn test_causal_relationship_creation() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    // Add nodes representing concepts/events
    let node_a = model.gnn_mut().add_node(vec![1.0, 2.0, 3.0, 4.0]);
    let node_b = model.gnn_mut().add_node(vec![5.0, 6.0, 7.0, 8.0]);
    let node_c = model.gnn_mut().add_node(vec![9.0, 10.0, 11.0, 12.0]);

    assert_eq!(model.gnn().node_count(), 3, "Should have 3 nodes");

    // Add causal relationships: A -[Precondition]-> B -[Cause]-> C
    let precondition_edge = CausalEdge::new(CausalEdgeKind::Precondition, 0.8);
    let cause_edge = CausalEdge::new(CausalEdgeKind::Cause, 1.0);

    let edge1_result = model.gnn_mut().add_edge(node_a, node_b, precondition_edge);
    let edge2_result = model.gnn_mut().add_edge(node_b, node_c, cause_edge);

    assert!(
        edge1_result.is_ok(),
        "Precondition edge should be added successfully"
    );
    assert!(
        edge2_result.is_ok(),
        "Cause edge should be added successfully"
    );

    assert_eq!(model.gnn().edge_count(), 2, "Should have 2 edges");
}

/// Test learning relations from epistemological flow
#[test]
fn test_learn_relations() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    // Create a dummy epistemological flow
    let flow = EpistemologicalFlow::default();

    // Test that learn_relations doesn't panic and returns Ok
    let result = model.learn_relations(&flow);
    assert!(result.is_ok(), "Learning relations should succeed");
}

/// Test context inference
#[test]
fn test_infer_context() {
    let config = GnnConfig::new(64, 128, 3);
    let model = InterdependentCausalModel::new(config).unwrap();

    // Create a mutable epistemological flow
    let mut flow = EpistemologicalFlow::default();

    // Test that infer_context doesn't panic and returns Ok
    let result = model.infer_context(&mut flow);
    assert!(result.is_ok(), "Context inference should succeed");
}

/// Test complex causal graph construction
#[test]
fn test_complex_causal_graph() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    // Create a more complex causal graph:
    // A -[Cause]-> B -[Cause]-> C
    // A -[Enable]-> D -[Inhibit]-> C
    // B -[Precondition]-> E

    let node_a = model.gnn_mut().add_node(vec![1.0; 64]);
    let node_b = model.gnn_mut().add_node(vec![2.0; 64]);
    let node_c = model.gnn_mut().add_node(vec![3.0; 64]);
    let node_d = model.gnn_mut().add_node(vec![4.0; 64]);
    let node_e = model.gnn_mut().add_node(vec![5.0; 64]);

    // Add various causal relationships
    let edges = vec![
        (node_a, node_b, CausalEdgeKind::Cause, 1.0),
        (node_b, node_c, CausalEdgeKind::Cause, 1.0),
        (node_a, node_d, CausalEdgeKind::Enable, 0.7),
        (node_d, node_c, CausalEdgeKind::Inhibit, -0.6),
        (node_b, node_e, CausalEdgeKind::Precondition, 0.8),
    ];

    for (from, to, kind, weight) in edges {
        let edge = CausalEdge::new(kind, weight);
        let result = model.gnn_mut().add_edge(from, to, edge);
        assert!(result.is_ok(), "Edge should be added successfully");
    }

    assert_eq!(model.gnn().node_count(), 5, "Should have 5 nodes");
    assert_eq!(model.gnn().edge_count(), 5, "Should have 5 edges");
}

/// Test edge type weight defaults
#[test]
fn test_edge_type_default_weights() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    let node_a = model.gnn_mut().add_node(vec![1.0; 64]);
    let node_b = model.gnn_mut().add_node(vec![2.0; 64]);

    // Test different edge types with default weights
    let edge_types = vec![
        CausalEdgeKind::Cause,
        CausalEdgeKind::Precondition,
        CausalEdgeKind::Inhibit,
        CausalEdgeKind::Enable,
        CausalEdgeKind::Correlate,
    ];

    for edge_kind in edge_types {
        let edge = CausalEdge::with_default_weight(edge_kind);
        let result = model.gnn_mut().add_edge(node_a, node_b, edge);
        assert!(
            result.is_ok(),
            "Edge with default weight should be added successfully"
        );
    }

    assert_eq!(
        model.gnn().edge_count(),
        5,
        "Should have 5 edges with different types"
    );
}

/// Test error handling for invalid operations
#[test]
fn test_error_handling() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    let node_a = model.gnn_mut().add_node(vec![1.0; 64]);
    let invalid_node = petgraph::graph::NodeIndex::new(999); // Non-existent node

    let edge = CausalEdge::new(CausalEdgeKind::Cause, 1.0);
    let result = model.gnn_mut().add_edge(node_a, invalid_node, edge);

    assert!(
        result.is_err(),
        "Adding edge to non-existent node should fail"
    );
}

/// Test model configuration validation
#[test]
fn test_config_validation() {
    // Test valid configuration
    let valid_config = GnnConfig::new(64, 128, 3);
    assert!(
        valid_config.validate().is_ok(),
        "Valid config should pass validation"
    );

    // Test invalid configurations
    let invalid_configs = vec![
        GnnConfig {
            input_dims: 0,
            hidden_dims: 128,
            num_layers: 3,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        },
        GnnConfig {
            input_dims: 64,
            hidden_dims: 0,
            num_layers: 3,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        },
        GnnConfig {
            input_dims: 64,
            hidden_dims: 128,
            num_layers: 0,
            learning_rate: 0.001,
            dropout_rate: 0.1,
            use_attention: true,
        },
        GnnConfig {
            input_dims: 64,
            hidden_dims: 128,
            num_layers: 3,
            learning_rate: 0.0,
            dropout_rate: 0.1,
            use_attention: true,
        },
        GnnConfig {
            input_dims: 64,
            hidden_dims: 128,
            num_layers: 3,
            learning_rate: 0.001,
            dropout_rate: 1.0,
            use_attention: true,
        },
    ];

    for invalid_config in invalid_configs {
        assert!(
            invalid_config.validate().is_err(),
            "Invalid config should fail validation"
        );
    }
}

/// Test edge metadata functionality
#[test]
fn test_edge_metadata() {
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    let node_a = model.gnn_mut().add_node(vec![1.0; 64]);
    let node_b = model.gnn_mut().add_node(vec![2.0; 64]);

    let edge = CausalEdge::with_metadata(
        CausalEdgeKind::Cause,
        0.9,
        "Test causal relationship".to_string(),
    );

    let result = model.gnn_mut().add_edge(node_a, node_b, edge);
    assert!(
        result.is_ok(),
        "Edge with metadata should be added successfully"
    );

    // Verify the edge was added with metadata
    let graph = model.gnn().graph();
    let edge_indices: Vec<_> = graph.edge_indices().collect();
    assert_eq!(edge_indices.len(), 1, "Should have one edge");

    let edge_data = &graph[edge_indices[0]];
    assert_eq!(
        edge_data.metadata,
        Some("Test causal relationship".to_string())
    );
}

/// Test the complete workflow: create model, add causal relationships, learn, and infer
#[test]
fn test_complete_workflow() {
    // Step 1: Create model
    let config = GnnConfig::new(64, 128, 3);
    let mut model = InterdependentCausalModel::new(config).unwrap();

    // Step 2: Add causal relationships
    let node_a = model.gnn_mut().add_node(vec![1.0; 64]);
    let node_b = model.gnn_mut().add_node(vec![2.0; 64]);
    let node_c = model.gnn_mut().add_node(vec![3.0; 64]);

    let edge1 = CausalEdge::new(CausalEdgeKind::Cause, 1.0);
    let edge2 = CausalEdge::new(CausalEdgeKind::Precondition, 0.8);

    model.gnn_mut().add_edge(node_a, node_b, edge1).unwrap();
    model.gnn_mut().add_edge(node_b, node_c, edge2).unwrap();

    // Step 3: Learn from epistemological flow
    let flow = EpistemologicalFlow::default();
    let learn_result = model.learn_relations(&flow);
    assert!(learn_result.is_ok(), "Learning should succeed");

    // Step 4: Infer context
    let mut flow_for_inference = EpistemologicalFlow::default();
    let infer_result = model.infer_context(&mut flow_for_inference);
    assert!(infer_result.is_ok(), "Context inference should succeed");

    // Verify final state (learning may add a flow node)
    assert!(
        model.gnn().node_count() >= 3,
        "Should have at least 3 nodes"
    );
    assert_eq!(model.gnn().edge_count(), 2, "Should have 2 edges");
}
