//! FAIR Benchmark: Skandha Pipeline vs Baseline
//!
//! This test corrects the unfair comparison in skandha_benchmark.rs.
//!
//! Previous test compared:
//! - Baseline: Adaptive FEP (Bayesian inference, multi-agent consensus)
//! - Skandha: Rule-based if-else (DefaultVinnana with hard thresholds)
//!
//! This is like comparing a PhD AI system with freshman code!
//!
//! This test ensures FAIR comparison by:
//! 1. Using SAME belief update mechanism (FEP) in both
//! 2. Only testing architectural difference (piecemeal vs unified pipeline)
//! 3. Measuring pipeline-specific benefits (Sheaf consensus, Holographic memory, Dharma ethics)

use zenb_core::engine::Engine;
use zenb_core::domain::{Observation, BioMetrics};
// BeliefState not needed directly
use std::time::Instant;

/// Generate observation with EXTREME contradiction (tests Sheaf)
fn contradictory_observation(timestamp_us: i64) -> Observation {
    Observation {
        timestamp_us,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(55.0),      // Says CALM (low HR)
            hrv_rmssd: Some(15.0),   // Says STRESSED (very low HRV!)
            respiratory_rate: Some(10.0),  // Says calm
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    }
}

/// Generate clean calm observation
fn calm_observation(timestamp_us: i64) -> Observation {
    Observation {
        timestamp_us,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(62.0),
            hrv_rmssd: Some(55.0),  // High HRV (calm)
            respiratory_rate: Some(10.0),
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    }
}

/// Generate stress observation
fn stress_observation(timestamp_us: i64) -> Observation {
    Observation {
        timestamp_us,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(95.0),     // High HR
            hrv_rmssd: Some(22.0),  // Low HRV
            respiratory_rate: Some(18.0),  // Fast breathing
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    }
}

/// Compute entropy of belief distribution (lower = more confident)
fn belief_entropy(belief: &[f32; 5]) -> f32 {
    let mut entropy = 0.0f32;
    for &p in belief.iter() {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Find dominant mode index
fn dominant_mode(belief: &[f32; 5]) -> usize {
    belief.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[test]
fn test_sheaf_consensus_benefit() {
    println!("\n=== SHEAF CONSENSUS TEST (Contradictory Sensors) ===\n");
    println!("Scenario: HR=55 (calm) but HRV=15 (stressed) - CONTRADICTORY!");
    println!("Expected: Sheaf should reduce ambiguity via Laplacian consensus\n");

    let observations: Vec<Observation> = (0..50)
        .map(|i| contradictory_observation(i * 1_000_000))
        .collect();

    // Baseline (no Sheaf)
    let mut baseline_engine = Engine::new_for_test(6.0);
    baseline_engine.use_vajra_architecture = false;

    for obs in &observations {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
        ];
        let est = baseline_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = baseline_engine.make_control(&est, obs.timestamp_us);
    }

    let baseline_belief = baseline_engine.belief_state.clone();
    let baseline_entropy = belief_entropy(&baseline_belief.p);
    let baseline_mode = dominant_mode(&baseline_belief.p);

    println!("Baseline (no Sheaf):");
    println!("  Belief: {:?}", baseline_belief.p);
    println!("  Mode: {} (idx={})", format!("{:?}", baseline_belief.mode), baseline_mode);
    println!("  Confidence: {:.3}", baseline_belief.conf);
    println!("  Entropy: {:.3} (lower = more decisive)", baseline_entropy);

    // Vajra with Sheaf
    let mut vajra_engine = Engine::new_for_test(6.0);
    vajra_engine.use_vajra_architecture = true;  // Enable Sheaf

    for obs in &observations {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
            0.9,  // quality
            0.1,  // motion
        ];
        let est = vajra_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = vajra_engine.make_control(&est, obs.timestamp_us);
    }

    let vajra_belief = vajra_engine.belief_state.clone();
    let vajra_entropy = belief_entropy(&vajra_belief.p);
    let vajra_mode = dominant_mode(&vajra_belief.p);

    println!("\nVajra (with Sheaf):");
    println!("  Belief: {:?}", vajra_belief.p);
    println!("  Mode: {} (idx={})", format!("{:?}", vajra_belief.mode), vajra_mode);
    println!("  Confidence: {:.3}", vajra_belief.conf);
    println!("  Entropy: {:.3} (lower = more decisive)", vajra_entropy);

    println!("\n📊 RESULT:");
    let entropy_reduction = (baseline_entropy - vajra_entropy) / baseline_entropy * 100.0;
    if vajra_entropy < baseline_entropy {
        println!("  ✅ Sheaf REDUCES ambiguity by {:.1}%", entropy_reduction);
    } else {
        println!("  ❌ Sheaf INCREASES ambiguity by {:.1}%", -entropy_reduction);
    }

    // Sheaf should reduce entropy (make more decisive decision)
    // NOTE: This might fail if Sheaf integration is broken!
}

#[test]
fn test_convergence_speed() {
    println!("\n=== CONVERGENCE SPEED TEST ===\n");
    println!("Scenario: Clean calm data (HR=62, HRV=55)");
    println!("Expected: Both converge to Calm mode, measure speed\n");

    let observations: Vec<Observation> = (0..100)
        .map(|i| calm_observation(i * 1_000_000))
        .collect();

    // Baseline
    let mut baseline_engine = Engine::new_for_test(6.0);
    baseline_engine.use_vajra_architecture = false;

    let mut baseline_converged_at = None;
    for (i, obs) in observations.iter().enumerate() {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
        ];
        let est = baseline_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = baseline_engine.make_control(&est, obs.timestamp_us);

        // Check convergence (Calm mode, confidence > 0.5)
        if baseline_converged_at.is_none()
            && dominant_mode(&baseline_engine.belief_state.p) == 0  // Calm
            && baseline_engine.belief_state.p[0] > 0.5 {
            baseline_converged_at = Some(i);
        }
    }

    // Vajra
    let mut vajra_engine = Engine::new_for_test(6.0);
    vajra_engine.use_vajra_architecture = true;

    let mut vajra_converged_at = None;
    for (i, obs) in observations.iter().enumerate() {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
            0.9,
            0.1,
        ];
        let est = vajra_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = vajra_engine.make_control(&est, obs.timestamp_us);

        if vajra_converged_at.is_none()
            && dominant_mode(&vajra_engine.belief_state.p) == 0
            && vajra_engine.belief_state.p[0] > 0.5 {
            vajra_converged_at = Some(i);
        }
    }

    println!("Baseline:");
    println!("  Converged at: {} observations", baseline_converged_at.unwrap_or(999));
    println!("  Final belief: {:?}", baseline_engine.belief_state.p);

    println!("\nVajra:");
    println!("  Converged at: {} observations", vajra_converged_at.unwrap_or(999));
    println!("  Final belief: {:?}", vajra_engine.belief_state.p);

    if let (Some(b), Some(v)) = (baseline_converged_at, vajra_converged_at) {
        let speedup = (b as f32 - v as f32) / b as f32 * 100.0;
        if v < b {
            println!("\n📊 RESULT: ✅ Vajra converges {:.1}% FASTER", speedup);
        } else {
            println!("\n📊 RESULT: ❌ Vajra converges {:.1}% SLOWER", -speedup);
        }
    }
}

#[test]
fn test_pattern_memory_benefit() {
    println!("\n=== HOLOGRAPHIC MEMORY TEST ===\n");
    println!("Scenario: Train on stress pattern (20x), then test recall");
    println!("Expected: Holographic memory should recognize pattern faster\n");

    let stress_pattern: Vec<Observation> = (0..20)
        .map(|i| stress_observation(i * 1_000_000))
        .collect();

    // Baseline (no Holographic memory)
    let mut baseline_engine = Engine::new_for_test(6.0);
    baseline_engine.use_vajra_architecture = false;

    // Train
    for obs in &stress_pattern {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
        ];
        let est = baseline_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = baseline_engine.make_control(&est, obs.timestamp_us);
    }

    // Test: Present pattern again after delay
    let test_obs = stress_observation(100_000_000);
    let bio = test_obs.bio_metrics.as_ref().unwrap();
    let features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
    ];
    let est = baseline_engine.ingest_sensor(&features, test_obs.timestamp_us);
    let _ = baseline_engine.make_control(&est, test_obs.timestamp_us);

    let baseline_stress_conf = baseline_engine.belief_state.p[1];  // Stress mode

    println!("Baseline (no memory):");
    println!("  Stress confidence after 20 training: {:.3}", baseline_stress_conf);

    // Vajra (with Holographic memory)
    let mut vajra_engine = Engine::new_for_test(6.0);
    vajra_engine.use_vajra_architecture = true;

    // Train
    for obs in &stress_pattern {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
            0.9,
            0.1,
        ];
        let est = vajra_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = vajra_engine.make_control(&est, obs.timestamp_us);
    }

    // Test
    let features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
        0.9,
        0.1,
    ];
    let est = vajra_engine.ingest_sensor(&features, test_obs.timestamp_us);
    let _ = vajra_engine.make_control(&est, test_obs.timestamp_us);

    let vajra_stress_conf = vajra_engine.belief_state.p[1];

    println!("\nVajra (with Holographic memory):");
    println!("  Stress confidence after 20 training: {:.3}", vajra_stress_conf);

    let improvement = (vajra_stress_conf - baseline_stress_conf) / baseline_stress_conf * 100.0;
    if vajra_stress_conf > baseline_stress_conf {
        println!("\n📊 RESULT: ✅ Memory improves recognition by {:.1}%", improvement);
    } else {
        println!("\n📊 RESULT: ❌ Memory WORSENS recognition by {:.1}%", -improvement);
    }

    // NOTE: This test might show no benefit if Holographic memory is not properly integrated!
}

#[test]
fn test_dharma_ethical_veto() {
    println!("\n=== DHARMA FILTER TEST ===\n");
    println!("Scenario: Propose extreme actions (very high/low BPM)");
    println!("Expected: Dharma should veto harmful actions\n");

    // Create scenario that would trigger extreme action
    let extreme_obs = Observation {
        timestamp_us: 0,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(120.0),  // Very high (panic?)
            hrv_rmssd: Some(10.0),  // Very low
            respiratory_rate: Some(25.0),  // Very fast
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    };

    // Baseline (no Dharma filter)
    let mut baseline_engine = Engine::new_for_test(6.0);
    baseline_engine.use_vajra_architecture = false;

    let bio = extreme_obs.bio_metrics.as_ref().unwrap();
    let features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
    ];
    let est = baseline_engine.ingest_sensor(&features, extreme_obs.timestamp_us);
    let (baseline_decision, _, _, baseline_deny) = baseline_engine.make_control(&est, extreme_obs.timestamp_us);

    println!("Baseline (no Dharma):");
    println!("  Proposed BPM: {:.2}", baseline_decision.target_rate_bpm);
    println!("  Denied: {:?}", baseline_deny.is_some());

    // Vajra (with Dharma filter)
    let mut vajra_engine = Engine::new_for_test(6.0);
    vajra_engine.use_vajra_architecture = true;

    let features = vec![
        bio.hr_bpm.unwrap(),
        bio.hrv_rmssd.unwrap(),
        bio.respiratory_rate.unwrap(),
        0.9,
        0.1,
    ];
    let est = vajra_engine.ingest_sensor(&features, extreme_obs.timestamp_us);
    let (vajra_decision, _, _, vajra_deny) = vajra_engine.make_control(&est, extreme_obs.timestamp_us);

    println!("\nVajra (with Dharma):");
    println!("  Proposed BPM: {:.2}", vajra_decision.target_rate_bpm);
    println!("  Denied: {:?}", vajra_deny.is_some());

    let bpm_diff = (baseline_decision.target_rate_bpm - vajra_decision.target_rate_bpm).abs();
    if bpm_diff > 0.5 {
        println!("\n📊 RESULT: ✅ Dharma modified action (diff={:.2} BPM)", bpm_diff);
    } else {
        println!("\n📊 RESULT: ⚠️ Dharma had no effect (diff={:.2} BPM)", bpm_diff);
    }
}

#[test]
fn test_latency_comparison() {
    println!("\n=== LATENCY BENCHMARK ===\n");

    let observations: Vec<Observation> = (0..1000)
        .map(|i| calm_observation(i * 1_000_000))
        .collect();

    // Baseline
    let start = Instant::now();
    let mut baseline_engine = Engine::new_for_test(6.0);
    baseline_engine.use_vajra_architecture = false;

    for obs in &observations {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
        ];
        let est = baseline_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = baseline_engine.make_control(&est, obs.timestamp_us);
    }
    let baseline_time = start.elapsed();

    // Vajra
    let start = Instant::now();
    let mut vajra_engine = Engine::new_for_test(6.0);
    vajra_engine.use_vajra_architecture = true;

    for obs in &observations {
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap(),
            bio.hrv_rmssd.unwrap(),
            bio.respiratory_rate.unwrap(),
            0.9,
            0.1,
        ];
        let est = vajra_engine.ingest_sensor(&features, obs.timestamp_us);
        let _ = vajra_engine.make_control(&est, obs.timestamp_us);
    }
    let vajra_time = start.elapsed();

    println!("Baseline: {:.3}s ({:.1} µs/observation)",
             baseline_time.as_secs_f64(),
             baseline_time.as_micros() as f64 / observations.len() as f64);
    println!("Vajra:    {:.3}s ({:.1} µs/observation)",
             vajra_time.as_secs_f64(),
             vajra_time.as_micros() as f64 / observations.len() as f64);

    let speedup = (baseline_time.as_micros() as f64 / vajra_time.as_micros() as f64 - 1.0) * 100.0;
    if vajra_time < baseline_time {
        println!("\n📊 RESULT: ✅ Vajra is {:.1}% faster", speedup);
    } else {
        println!("\n📊 RESULT: ❌ Vajra is {:.1}% slower", -speedup);
    }
}
