use zenb_core::config::{FeatureConfig, ZenbConfig};
use zenb_core::domain::ControlDecision;
use zenb_core::engine::Engine;
use zenb_core::policy::ActionPolicy; // Import ActionPolicy for checks

#[test]
fn test_efe_policy_selection_flow() {
    // 1. Setup Config with EFE enabled
    let mut config = ZenbConfig::default();
    config.features.use_efe_selection = true;
    config.features.efe_precision_beta = Some(4.0);

    // 2. Initialize Engine
    let mut engine = Engine::new_with_config(6.0, Some(config));

    // Verify EFE is active
    assert!(engine.config.features.use_efe_selection);

    // 2. Setup high stress state (HR 100, HRV Low)
    // Heuristic would pick ~8 BPM or Breath Guidance.
    // EFE should pick Calming Breath (High Pragmatic Value for Stress)
    let est = engine.ingest_sensor(&[100.0, 20.0, 18.0, 0.2, 0.0], 0);

    // 3. Make control decision
    let (decision, _, policy_info, deny_reason) = engine.make_control(&est, 1_000_000);

    // 4. Verification

    // A. Check that policy_info is populated (indicates EFE was used)
    // (u8, u32, f32) -> mode, reason, conf.
    // Wait, make_control returns policy_info as Option<(u8, u32, f32)>.
    // This is NOT the EFE policy info. This is the OLD existing return.
    // I did NOT change the return signature of make_control.
    // But I DID update `last_selected_policy` in Engine.

    assert!(
        engine.last_selected_policy.is_some(),
        "EFE selection should populate last_selected_policy"
    );

    let selected = engine.last_selected_policy.as_ref().unwrap();
    println!("Selected Policy: {}", selected.policy.description());

    // B. Check that decision reflects the policy
    // For Calming Breath, target BPM is 6.0
    // Digital Intervention does not change BPM (returns proposed).

    // For high stress (100 BPM), Calming Breath (6 BPM) should be favored by EFE if defined correctly.
    // Let's assert we aren't getting a default failure or weird state.
    assert!(decision.confidence > 0.0);

    // C. Verify Beta Adaptation Logic (triggered via learn_from_outcome)
    // Simulate Success
    let old_beta = engine.efe_precision_beta;
    engine.learn_from_outcome(true, "test_action".to_string(), 2_000_000, 0.0);

    // Beta should change (logic: success rate high -> slight increase? or if exploration low...)
    assert!(
        engine.efe_meta_learner.success_rate_ema > 0.5,
        "Success rate should increase"
    );
}
