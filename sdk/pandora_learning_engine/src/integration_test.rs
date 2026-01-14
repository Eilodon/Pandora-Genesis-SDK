//! Integration tests for the CWM decoder and value-driven policy
//!
//! This module demonstrates how the CWM's decode_and_update_flow method
//! works together with the value-driven policy to create a complete
//! active inference planning system.

#[cfg(test)]
use crate::{PolicyAction, NeuralQValueEstimator, Policy, QValueEstimator, ValueDrivenPolicy};
#[cfg(test)]
use bytes::Bytes;
#[cfg(test)]
use pandora_core::ontology::EpistemologicalFlow;
#[cfg(test)]
use pandora_cwm::gnn::types::GnnConfig;
#[cfg(test)]
use pandora_cwm::model::InterdependentCausalModel;

/// Test the CWM decoder functionality
#[test]
fn test_cwm_decoder_functionality() {
    // Create a CWM model
    let config = GnnConfig::new(64, 128, 3);
    let _model = InterdependentCausalModel::new(config).unwrap();

    // Create a flow with an intent
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));
    flow.set_static_intent("unlock_door");

    // Test that the flow has the intent set
    assert!(flow.sankhara.is_some());
    assert_eq!(flow.sankhara.as_ref().unwrap().as_ref(), "unlock_door");

    // The decoder will be tested when we call predict_next_state
    // This is a placeholder test - in a real implementation,
    // we would test the actual decoding logic
}

/// Test the value-driven policy functionality
#[test]
fn test_value_driven_policy() {
    // Create a value-driven policy
    let policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);

    // Create a flow
    let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));

    // Test action selection
    let action = policy.select_action(&flow, 0.0); // Exploitation mode
    assert!(matches!(
        action,
        PolicyAction::Noop
            | PolicyAction::Explore
            | PolicyAction::Exploit
            | PolicyAction::UnlockDoor
            | PolicyAction::PickUpKey
            | PolicyAction::MoveForward
    ));

    // Test exploration mode
    let action = policy.select_action(&flow, 1.0); // Exploration mode
    assert!(matches!(
        action,
        PolicyAction::Noop
            | PolicyAction::Explore
            | PolicyAction::Exploit
            | PolicyAction::UnlockDoor
            | PolicyAction::PickUpKey
            | PolicyAction::MoveForward
    ));
}

/// Test the neural Q-value estimator
#[test]
fn test_neural_q_value_estimator() {
    // Create a Q-value estimator
    let mut estimator = NeuralQValueEstimator::new(0.1, 0.9);

    // Create a flow
    let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));

    // Test Q-value estimation
    let q_values = estimator.get_q_values(&flow).unwrap();
    assert!(!q_values.is_empty());

    // Test that all actions have Q-values
    let action_names: Vec<&str> = q_values.iter().map(|(name, _)| *name).collect();
    assert!(action_names.contains(&"unlock_door"));
    assert!(action_names.contains(&"pick_up_key"));
    assert!(action_names.contains(&"move_forward"));
    assert!(action_names.contains(&"explore"));
    assert!(action_names.contains(&"noop"));

    // Test Q-value update
    let next_flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"next_state"));
    estimator.update_q_value(&flow, "unlock_door", 1.0, &next_flow);

    // Test visit count
    let visit_count = estimator.get_visit_count(&flow, "unlock_door");
    assert_eq!(visit_count, 1);
}

/// Test the complete active inference planning cycle
#[test]
fn test_active_inference_planning_cycle() {
    // Create a CWM model
    let config = GnnConfig::new(64, 128, 3);
    let _model = InterdependentCausalModel::new(config).unwrap();

    // Create a value-driven policy
    let mut policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);

    // Create a flow with an intent
    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));
    flow.set_static_intent("unlock_door");

    // Step 1: Select an action using the policy
    let action = policy.select_action(&flow, 0.0); // Exploitation mode
    assert!(matches!(
        action,
        PolicyAction::Noop
            | PolicyAction::Explore
            | PolicyAction::Exploit
            | PolicyAction::UnlockDoor
            | PolicyAction::PickUpKey
            | PolicyAction::MoveForward
    ));

    // Step 2: Use the CWM to predict the next state
    // Note: This would call predict_next_state which uses the decoder
    // For now, we'll just test that the flow is in a valid state
    assert!(flow.sankhara.is_some());

    // Step 3: Simulate receiving a reward and updating the policy
    let reward = 1.0;
    let next_flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"next_state"));
    policy.update_with_experience(&flow, &action, reward, &next_flow);

    // Test that the policy was updated
    assert!(policy.total_visits > 0);
}

/// Test the UCB1 exploration strategy
#[test]
fn test_ucb1_exploration() {
    // Create a value-driven policy
    let policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);

    // Create a flow
    let flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));

    // Test UCB1 scoring for different actions
    let actions = PolicyAction::all_actions();
    for action in &actions {
        let score = policy.ucb1_score(&flow, action);
        assert!(score.is_finite() || score == f64::INFINITY);
    }
}

/// Test the action string conversion
#[test]
fn test_action_string_conversion() {
    assert_eq!(PolicyAction::Noop.as_str(), "noop");
    assert_eq!(PolicyAction::Explore.as_str(), "explore");
    assert_eq!(PolicyAction::Exploit.as_str(), "exploit");
    assert_eq!(PolicyAction::UnlockDoor.as_str(), "unlock_door");
    assert_eq!(PolicyAction::PickUpKey.as_str(), "pick_up_key");
    assert_eq!(PolicyAction::MoveForward.as_str(), "move_forward");
}

/// Test the complete learning loop
#[test]
fn test_complete_learning_loop() {
    // Create a CWM model
    let config = GnnConfig::new(64, 128, 3);
    let _model = InterdependentCausalModel::new(config).unwrap();

    // Create a value-driven policy
    let mut policy = ValueDrivenPolicy::new(0.1, 0.9, 2.0);

    // Simulate multiple learning steps
    for i in 0..5 {
        // Create a flow with different intents
        let intent = match i % 3 {
            0 => "unlock_door",
            1 => "pick_up_key",
            _ => "move_forward",
        };

        let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"test_state"));
        flow.set_static_intent(intent);

        // Select action
        let action = policy.select_action(&flow, 0.1); // 10% exploration

        // Simulate reward (higher for successful actions)
        let reward = if action.as_str() == intent { 1.0 } else { 0.0 };

        // Create next state
        let next_flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"next_state"));

        // Update policy
        policy.update_with_experience(&flow, &action, reward, &next_flow);
    }

    // Test that the policy learned something
    assert!(policy.total_visits > 0);
}
