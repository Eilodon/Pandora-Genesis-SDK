//! Debug Test: Why Does Vajra Always Converge to Energize?
//!
//! This test traces the FULL data flow from sensors to belief to identify
//! where the bias toward Energize mode is introduced.

use zenb_core::domain::{BioMetrics, Observation};
use zenb_core::engine::Engine;

#[test]
fn debug_vajra_energize_bias() {
    println!("\n=== DEBUG: Vajra Energize Bias Investigation ===\n");

    // Test with CALM data (should converge to Calm, not Energize)
    let calm_obs = Observation {
        timestamp_us: 0,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(62.0),           // Low HR (calm)
            hrv_rmssd: Some(55.0),        // High HRV (calm)
            respiratory_rate: Some(10.0), // Slow breathing (calm)
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    };

    println!("Input Data (CALM scenario):");
    println!("  HR: 62 BPM (calm)");
    println!("  HRV: 55 ms (high = calm)");
    println!("  RR: 10 BPM (slow = calm)");
    println!();

    // Baseline (no Vajra)
    let mut baseline = Engine::new_for_test(6.0);
    baseline.config.features.vajra_enabled = false;

    let bio = calm_obs.bio_metrics.as_ref().unwrap();
    let raw_features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
    ];

    println!("--- BASELINE PATH ---");
    println!("1. Raw Features: {:?}", raw_features);

    let est_baseline = baseline.ingest_sensor(&raw_features, 0);
    println!(
        "2. Estimate: hr={:?}, hrv={:?}, rr={:?}, conf={:.3}",
        est_baseline.hr_bpm, est_baseline.rmssd, est_baseline.rr_bpm, est_baseline.confidence
    );

    let _ = baseline.make_control(&est_baseline, 0);
    println!("3. Belief State: {:?}", baseline.belief_state.p);
    println!(
        "   Mode: {:?} (idx={})",
        baseline.belief_state.mode,
        baseline
            .belief_state
            .p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    );
    println!();

    // Vajra (with Sheaf)
    let mut vajra = Engine::new_for_test(6.0);
    vajra.config.features.vajra_enabled = true;

    let vajra_features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
        0.9, // quality
        0.1, // motion
    ];

    println!("--- VAJRA PATH ---");
    println!(
        "1. Raw Features (with quality/motion): {:?}",
        vajra_features
    );

    let est_vajra = vajra.ingest_sensor(&vajra_features, 0);
    println!("2. After Sheaf:");
    println!("   Sheaf Energy: {:.3}", vajra.last_sheaf_energy);
    println!(
        "   Estimate: hr={:?}, hrv={:?}, rr={:?}, conf={:.3}",
        est_vajra.hr_bpm, est_vajra.rmssd, est_vajra.rr_bpm, est_vajra.confidence
    );

    // Check if PhysioState is different
    if let Some(ref phys) = vajra.last_phys {
        println!(
            "   PhysioState: hr={:?}, hrv={:?}, rr={:?}, conf={:.3}",
            phys.hr_bpm, phys.rmssd, phys.rr_bpm, phys.confidence
        );
    }

    let _ = vajra.make_control(&est_vajra, 0);
    println!("3. Belief State: {:?}", vajra.belief_state.p);
    println!(
        "   Mode: {:?} (idx={})",
        vajra.belief_state.mode,
        vajra
            .belief_state
            .p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    );
    println!(
        "   FEP State: mu={:?}, sigma={:?}, FE_ema={:.3}",
        vajra.fep_state.mu, vajra.fep_state.sigma, vajra.fep_state.free_energy_ema
    );
    println!();

    // Run 10 iterations to see convergence
    println!("--- CONVERGENCE TRACKING (10 iterations) ---");
    for i in 1..=10 {
        let ts = i * 1_000_000;

        // Baseline
        let est_b = baseline.ingest_sensor(&raw_features, ts);
        let _ = baseline.make_control(&est_b, ts);

        // Vajra
        let est_v = vajra.ingest_sensor(&vajra_features, ts);
        let _ = vajra.make_control(&est_v, ts);

        let baseline_mode_idx = baseline
            .belief_state
            .p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let vajra_mode_idx = vajra
            .belief_state
            .p
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        println!(
            "Iter {}: Baseline mode={} (p={:.3}), Vajra mode={} (p={:.3})",
            i,
            baseline_mode_idx,
            baseline.belief_state.p[baseline_mode_idx],
            vajra_mode_idx,
            vajra.belief_state.p[vajra_mode_idx]
        );
    }

    println!("\n--- FINAL ANALYSIS ---");
    let baseline_mode_idx = baseline
        .belief_state
        .p
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let vajra_mode_idx = vajra
        .belief_state
        .p
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!(
        "Baseline final: mode={}, belief={:?}",
        baseline_mode_idx, baseline.belief_state.p
    );
    println!(
        "Vajra final: mode={}, belief={:?}",
        vajra_mode_idx, vajra.belief_state.p
    );

    if vajra_mode_idx == 4 {
        println!("\n❌ CONFIRMED: Vajra converges to Energize (mode=4) even with CALM data!");
        println!("Expected: Calm (mode=0) or Sleepy (mode=3)");

        // Hypothesis tests
        println!("\n🔬 HYPOTHESIS TESTS:");

        // H1: Sheaf filter is drastically changing sensor values
        let hr_diff = (est_baseline.hr_bpm.unwrap_or(0.0) - est_vajra.hr_bpm.unwrap_or(0.0)).abs();
        let hrv_diff = (est_baseline.rmssd.unwrap_or(0.0) - est_vajra.rmssd.unwrap_or(0.0)).abs();
        println!("H1 (Sheaf alters sensors significantly):");
        println!("    HR diff: {:.2} BPM", hr_diff);
        println!("    HRV diff: {:.2} ms", hrv_diff);
        if hr_diff > 5.0 || hrv_diff > 10.0 {
            println!("    → Likely cause! Sheaf is drastically changing sensor readings.");
        } else {
            println!("    → Not significant. Sheaf changes are small.");
        }

        // H2: FEP state has Energize bias in initialization
        println!("\nH2 (FEP initialization bias toward Energize):");
        println!("    Vajra FEP mu: {:?}", vajra.fep_state.mu);
        if vajra.fep_state.mu[4] > 0.5 {
            println!("    → Likely cause! mu[4] (Energize) is already high at start.");
        } else {
            println!("    → FEP initialization looks balanced.");
        }

        // H3: BeliefEngine has different behavior with Vajra-processed data
        println!("\nH3 (BeliefEngine reacts differently to Vajra data):");
        println!("    Baseline belief: {:?}", baseline.belief_state.p);
        println!("    Vajra belief: {:?}", vajra.belief_state.p);
        println!("    → Need to inspect BeliefEngine.update_fep_with_config() internals.");
    } else {
        println!(
            "\n✅ Vajra converged to mode={} (not Energize)",
            vajra_mode_idx
        );
    }
}

#[test]
fn debug_sheaf_output_vs_raw() {
    println!("\n=== DEBUG: Sheaf Filter Output Analysis ===\n");

    let test_cases = vec![
        ("Calm", vec![62.0, 55.0, 10.0]),
        ("Stress", vec![95.0, 22.0, 18.0]),
        ("Contradictory", vec![55.0, 15.0, 10.0]), // HR calm but HRV stressed
    ];

    for (name, raw) in test_cases {
        println!("--- {} Scenario ---", name);
        println!("Raw: HR={:.1}, HRV={:.1}, RR={:.1}", raw[0], raw[1], raw[2]);

        let mut engine = Engine::new_for_test(6.0);
        engine.config.features.vajra_enabled = true;

        let features = vec![raw[0], raw[1], raw[2], 0.9, 0.1];
        let est = engine.ingest_sensor(&features, 0);

        println!("After Sheaf:");
        println!(
            "  HR={:?}, HRV={:?}, RR={:?}",
            est.hr_bpm, est.rmssd, est.rr_bpm
        );
        println!("  Sheaf Energy: {:.3}", engine.last_sheaf_energy);
        println!(
            "  Changes: HR {:+.1}, HRV {:+.1}, RR {:+.1}",
            est.hr_bpm.unwrap_or(0.0) - raw[0],
            est.rmssd.unwrap_or(0.0) - raw[1],
            est.rr_bpm.unwrap_or(0.0) - raw[2]
        );
        println!();
    }
}
