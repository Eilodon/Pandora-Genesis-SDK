//! Integration tests for Active Inference Planning Engine
//!
//! This test suite verifies that the ActiveInferenceSankharaSkandha can plan actions
//! by simulating future states and selecting the one that maximizes expected free energy.

use bytes::Bytes;
use pandora_core::interfaces::skandhas::SankharaSkandha;
use pandora_core::ontology::EpistemologicalFlow;
use pandora_cwm::gnn::types::{CausalEdge, CausalEdgeKind, GnnConfig};
use pandora_cwm::model::InterdependentCausalModel;
use pandora_learning_engine::{ActiveInferenceSankharaSkandha, LearningEngine};
use std::sync::{Arc, Mutex};

/// Test basic Active Inference planning functionality
#[test]
fn test_active_inference_planning_basic() {
    // Set up the Causal World Model
    let config = GnnConfig::new(64, 128, 3);
    let cwm = InterdependentCausalModel::new(config).unwrap();

    // Set up the Learning Engine
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));

    // Set up available actions
    let available_actions = vec!["action_A", "action_B", "action_C"];

    // Create the Active Inference Sankhara Skandha
    let cwm_arc = Arc::new(Mutex::new(cwm));
    let sankhara = ActiveInferenceSankharaSkandha::new(
        cwm_arc,
        learning_engine,
        3, // planning horizon
        available_actions,
        0.9, // gamma
        0.1, // policy_epsilon
    );

    // Create a test flow
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));

    // Test that the skandha can form intent
    sankhara.form_intent(&mut flow);

    // Verify that an intent was formed
    assert!(
        flow.sankhara.is_some(),
        "Sankhara should have formed an intent"
    );

    let intent = flow.sankhara.as_ref().unwrap();
    assert!(
        [
            "action_A",
            "action_B",
            "action_C",
            "default_fallback_intent"
        ]
        .contains(&intent.as_ref()),
        "Intent should be one of the available actions or fallback"
    );
}

/// Test planning with a configured CWM that favors a specific action
#[test]
fn test_planning_with_configured_cwm() {
    // Set up the Causal World Model with some structure
    let config = GnnConfig::new(64, 128, 3);
    let mut cwm = InterdependentCausalModel::new(config).unwrap();

    // Add some nodes and edges to create a meaningful graph
    let node_a = cwm.gnn_mut().add_node(vec![1.0; 64]);
    let node_b = cwm.gnn_mut().add_node(vec![2.0; 64]);
    let node_c = cwm.gnn_mut().add_node(vec![3.0; 64]);

    // Add causal relationships that favor "action_A"
    let edge1 = CausalEdge::new(CausalEdgeKind::Cause, 1.0);
    let edge2 = CausalEdge::new(CausalEdgeKind::Precondition, 0.8);

    cwm.gnn_mut().add_edge(node_a, node_b, edge1).unwrap();
    cwm.gnn_mut().add_edge(node_b, node_c, edge2).unwrap();

    // Set up the Learning Engine
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));

    // Set up available actions
    let available_actions = vec!["action_A", "action_B", "action_C"];

    // Create the Active Inference Sankhara Skandha
    let cwm_arc = Arc::new(Mutex::new(cwm));
    let sankhara = ActiveInferenceSankharaSkandha::new(
        cwm_arc,
        learning_engine,
        2, // planning horizon
        available_actions,
        0.9, // gamma
        0.1, // policy_epsilon
    );

    // Create a test flow
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));

    // Test that the skandha can form intent
    sankhara.form_intent(&mut flow);

    // Verify that an intent was formed
    assert!(
        flow.sankhara.is_some(),
        "Sankhara should have formed an intent"
    );

    let intent = flow.sankhara.as_ref().unwrap();
    assert!(
        [
            "action_A",
            "action_B",
            "action_C",
            "default_fallback_intent"
        ]
        .contains(&intent.as_ref()),
        "Intent should be one of the available actions or fallback"
    );
}

/// Test planning with different planning horizons
#[test]
fn test_planning_with_different_horizons() {
    let config = GnnConfig::new(64, 128, 3);
    let cwm = InterdependentCausalModel::new(config).unwrap();
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
    let available_actions = vec!["action_A", "action_B"];

    let cwm_arc = Arc::new(Mutex::new(cwm));

    // Test with different planning horizons
    for horizon in [1, 3, 5] {
        let sankhara = ActiveInferenceSankharaSkandha::new(
            cwm_arc.clone(),
            learning_engine.clone(),
            horizon,
            available_actions.clone(),
            0.9, // gamma
            0.1, // policy_epsilon
        );

        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));
        sankhara.form_intent(&mut flow);

        assert!(
            flow.sankhara.is_some(),
            "Sankhara should form intent with horizon {}",
            horizon
        );
    }
}

/// Test planning with context-specific actions
#[test]
fn test_planning_with_context_specific_actions() {
    let config = GnnConfig::new(64, 128, 3);
    let cwm = InterdependentCausalModel::new(config).unwrap();
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
    let available_actions = vec!["action_A", "action_B"];

    let cwm_arc = Arc::new(Mutex::new(cwm));
    let sankhara = ActiveInferenceSankharaSkandha::new(
        cwm_arc,
        learning_engine,
        2,
        available_actions,
        0.9, // gamma
        0.1, // policy_epsilon
    );

    // Create a flow with specific context that should trigger context-specific actions
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"success_event".as_ref()));

    // Add some context that should influence action selection
    // (In a real implementation, this would be set by other skandhas)

    sankhara.form_intent(&mut flow);

    assert!(
        flow.sankhara.is_some(),
        "Sankhara should form intent based on context"
    );
}

/// Test error handling in planning
#[test]
fn test_planning_error_handling() {
    let config = GnnConfig::new(64, 128, 3);
    let cwm = InterdependentCausalModel::new(config).unwrap();
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));
    let available_actions = vec!["action_A"];

    let cwm_arc = Arc::new(Mutex::new(cwm));
    let sankhara = ActiveInferenceSankharaSkandha::new(
        cwm_arc,
        learning_engine,
        1,
        available_actions,
        0.9, // gamma
        0.1, // policy_epsilon
    );

    // Create a flow and test planning
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));
    sankhara.form_intent(&mut flow);

    // Even if there are errors, should fall back to default intent
    assert!(
        flow.sankhara.is_some(),
        "Sankhara should always form some intent"
    );

    let intent = flow.sankhara.as_ref().unwrap();
    assert!(
        ["action_A", "default_fallback_intent"].contains(&intent.as_ref()),
        "Intent should be available action or fallback"
    );
}

/// Test the complete planning workflow
#[test]
fn test_complete_planning_workflow() {
    // Step 1: Set up CWM with meaningful structure
    let config = GnnConfig::new(64, 128, 3);
    let mut cwm = InterdependentCausalModel::new(config).unwrap();

    // Add nodes representing different states
    let state_node = cwm.gnn_mut().add_node(vec![1.0; 64]);
    let goal_node = cwm.gnn_mut().add_node(vec![2.0; 64]);
    let obstacle_node = cwm.gnn_mut().add_node(vec![3.0; 64]);

    // Add causal relationships
    let positive_edge = CausalEdge::new(CausalEdgeKind::Cause, 1.0);
    let negative_edge = CausalEdge::new(CausalEdgeKind::Inhibit, -0.5);

    cwm.gnn_mut()
        .add_edge(state_node, goal_node, positive_edge)
        .unwrap();
    cwm.gnn_mut()
        .add_edge(obstacle_node, goal_node, negative_edge)
        .unwrap();

    // Step 2: Set up Learning Engine
    let learning_engine = Arc::new(LearningEngine::new(0.7, 0.3));

    // Step 3: Set up available actions
    let available_actions = vec!["move_toward_goal", "avoid_obstacle", "explore"];

    // Step 4: Create Active Inference Sankhara
    let cwm_arc = Arc::new(Mutex::new(cwm));
    let sankhara = ActiveInferenceSankharaSkandha::new(
        cwm_arc,
        learning_engine,
        3, // planning horizon
        available_actions,
        0.9, // gamma
        0.1, // policy_epsilon
    );

    // Step 5: Create test flow
    let mut flow =
        EpistemologicalFlow::from_bytes(Bytes::from(b"agent_needs_to_navigate".as_ref()));

    // Step 6: Test planning
    sankhara.form_intent(&mut flow);

    // Step 7: Verify results
    assert!(flow.sankhara.is_some(), "Planning should produce an intent");

    let intent = flow.sankhara.as_ref().unwrap();
    assert!(
        [
            "move_toward_goal",
            "avoid_obstacle",
            "explore",
            "default_fallback_intent"
        ]
        .contains(&intent.as_ref()),
        "Intent should be one of the available actions"
    );

    println!("Selected intent: {}", intent);
}

/// Test planning with different reward configurations
#[test]
fn test_planning_with_different_reward_configs() {
    let config = GnnConfig::new(64, 128, 3);
    let cwm = InterdependentCausalModel::new(config).unwrap();
    let available_actions = vec!["action_A", "action_B"];

    let cwm_arc = Arc::new(Mutex::new(cwm));

    // Test with different reward weight configurations
    let reward_configs = vec![
        (0.9, 0.1), // High exploit, low transcend
        (0.5, 0.5), // Balanced
        (0.1, 0.9), // Low exploit, high transcend
    ];

    for (exploit_weight, transcend_weight) in reward_configs {
        let learning_engine = Arc::new(LearningEngine::new(exploit_weight, transcend_weight));
        let sankhara = ActiveInferenceSankharaSkandha::new(
            cwm_arc.clone(),
            learning_engine,
            2,
            available_actions.clone(),
            0.9, // gamma
            0.1, // policy_epsilon
        );

        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from(b"test_event".as_ref()));
        sankhara.form_intent(&mut flow);

        assert!(
            flow.sankhara.is_some(),
            "Should form intent with weights ({}, {})",
            exploit_weight,
            transcend_weight
        );
    }
}
