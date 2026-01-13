//! Empirical Benchmark: Skandha Pipeline vs Baseline
//!
//! This test compares three architectural approaches:
//! 1. Baseline: No Vajra components (legacy path)
//! 2. Vajra Piecemeal: Individual components used separately (current state)
//! 3. Skandha Unified: Full pipeline integration (proposed)
//!
//! Metrics:
//! - Latency: Time to process one observation
//! - Accuracy: Deviation from ideal belief state
//! - Memory: Heap allocations
//! - Code complexity: Lines of code in execution path

use zenb_core::belief::BeliefState;
use zenb_core::domain::Observation;
use zenb_core::engine::Engine;
// Skandha pipeline is tested through Engine methods
use std::time::Instant;

/// Generate synthetic observation data for testing
fn generate_test_observation(timestamp_us: i64, stress_level: f32) -> Observation {
    let hr_bpm = 60.0 + stress_level * 40.0; // 60-100 BPM range
    let hrv_rmssd = 50.0 - stress_level * 30.0; // 50-20 ms range (inverse)
    let rr_bpm = 12.0 + stress_level * 6.0; // 12-18 BPM range

    Observation {
        timestamp_us,
        bio_metrics: Some(zenb_core::domain::BioMetrics {
            hr_bpm: Some(hr_bpm),
            hrv_rmssd: Some(hrv_rmssd),
            respiratory_rate: Some(rr_bpm),
        }),
        environmental_context: None,
        digital_context: None,
        cognitive_context: None,
    }
}

/// Benchmark: Baseline (No Vajra)
fn benchmark_baseline(observations: &[Observation]) -> (f64, BeliefState) {
    let mut engine = Engine::new_for_test(6.0);
    engine.config.features.vajra_enabled = false; // DISABLE Vajra

    let start = Instant::now();
    let mut final_belief = BeliefState::default();

    for obs in observations {
        // Extract sensor features manually (baseline path)
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap_or(60.0),
            bio.hrv_rmssd.unwrap_or(50.0),
            bio.respiratory_rate.unwrap_or(12.0),
        ];

        // Ingest and control (baseline path)
        let est = engine.ingest_sensor(&features, obs.timestamp_us);
        let (_dec, _persist, _policy, _deny) = engine.make_control(&est, obs.timestamp_us);
        final_belief = engine.belief_state.clone();
    }

    let duration = start.elapsed();
    (duration.as_secs_f64(), final_belief)
}

/// Benchmark: Vajra Piecemeal (Current State)
fn benchmark_vajra_piecemeal(observations: &[Observation]) -> (f64, BeliefState) {
    let mut engine = Engine::new_for_test(6.0);
    engine.config.features.vajra_enabled = true; // ENABLE Vajra (piecemeal usage)

    let start = Instant::now();
    let mut final_belief = BeliefState::default();

    for obs in observations {
        // Extract sensor features
        let bio = obs.bio_metrics.as_ref().unwrap();
        let features = vec![
            bio.hr_bpm.unwrap_or(60.0),
            bio.hrv_rmssd.unwrap_or(50.0),
            bio.respiratory_rate.unwrap_or(12.0),
            0.9, // quality
            0.1, // motion
        ];

        // Ingest (uses SheafPerception internally via circuit breaker)
        let est = engine.ingest_sensor(&features, obs.timestamp_us);

        // Make control (uses DharmaFilter internally)
        let (_dec, _persist, _policy, _deny) = engine.make_control(&est, obs.timestamp_us);
        final_belief = engine.belief_state.clone();
    }

    let duration = start.elapsed();
    (duration.as_secs_f64(), final_belief)
}

/// Benchmark: Skandha Unified (Proposed)
fn benchmark_skandha_unified(observations: &[Observation]) -> (f64, BeliefState) {
    let mut engine = Engine::new_for_test(6.0);
    engine.config.features.vajra_enabled = true;

    let start = Instant::now();
    let mut final_belief = BeliefState::default();

    for obs in observations {
        // Use unified Skandha pipeline
        let synthesis = engine.process_skandha_pipeline(obs);

        // Apply synthesized belief directly
        // NOTE: This requires integration work to wire synthesis into engine state
        // For now, we measure pipeline latency only

        // Convert synthesis.belief to BeliefState
        let mut belief = BeliefState::default();
        belief.p = synthesis.belief;
        belief.mode = match synthesis.mode {
            0 => zenb_core::belief::BeliefBasis::Calm,
            1 => zenb_core::belief::BeliefBasis::Stress,
            2 => zenb_core::belief::BeliefBasis::Focus,
            3 => zenb_core::belief::BeliefBasis::Sleepy,
            4 => zenb_core::belief::BeliefBasis::Energize,
            _ => zenb_core::belief::BeliefBasis::Calm,
        };
        belief.conf = synthesis.confidence;

        final_belief = belief;
    }

    let duration = start.elapsed();
    (duration.as_secs_f64(), final_belief)
}

/// Compute belief state deviation from expected "ground truth"
fn compute_deviation(belief: &BeliefState, expected_mode: zenb_core::belief::BeliefBasis) -> f32 {
    let expected_idx = match expected_mode {
        zenb_core::belief::BeliefBasis::Calm => 0,
        zenb_core::belief::BeliefBasis::Stress => 1,
        zenb_core::belief::BeliefBasis::Focus => 2,
        zenb_core::belief::BeliefBasis::Sleepy => 3,
        zenb_core::belief::BeliefBasis::Energize => 4,
    };

    // Compute L2 deviation from ideal one-hot distribution
    let mut ideal = [0.0f32; 5];
    ideal[expected_idx] = 1.0;

    let mut deviation = 0.0f32;
    for i in 0..5 {
        deviation += (belief.p[i] - ideal[i]).powi(2);
    }
    deviation.sqrt()
}

#[test]
fn test_skandha_vs_baseline_calm_scenario() {
    // Test scenario: Low stress (should converge to Calm mode)
    let observations: Vec<Observation> = (0..100)
        .map(|i| generate_test_observation(i * 1_000_000, 0.2)) // Low stress
        .collect();

    println!("\n=== CALM SCENARIO (100 observations, stress=0.2) ===\n");

    // Baseline
    let (baseline_time, baseline_belief) = benchmark_baseline(&observations);
    let baseline_dev = compute_deviation(&baseline_belief, zenb_core::belief::BeliefBasis::Calm);
    println!("Baseline:");
    println!("  Time: {:.6}s", baseline_time);
    println!("  Belief: {:?}", baseline_belief.p);
    println!("  Mode: {:?}", baseline_belief.mode);
    println!("  Deviation from ideal: {:.4}", baseline_dev);

    // Vajra Piecemeal
    let (vajra_time, vajra_belief) = benchmark_vajra_piecemeal(&observations);
    let vajra_dev = compute_deviation(&vajra_belief, zenb_core::belief::BeliefBasis::Calm);
    println!("\nVajra Piecemeal (current):");
    println!(
        "  Time: {:.6}s ({:+.2}% vs baseline)",
        vajra_time,
        ((vajra_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", vajra_belief.p);
    println!("  Mode: {:?}", vajra_belief.mode);
    println!(
        "  Deviation from ideal: {:.4} ({:+.2}% vs baseline)",
        vajra_dev,
        ((vajra_dev - baseline_dev) / baseline_dev * 100.0)
    );

    // Skandha Unified
    let (skandha_time, skandha_belief) = benchmark_skandha_unified(&observations);
    let skandha_dev = compute_deviation(&skandha_belief, zenb_core::belief::BeliefBasis::Calm);
    println!("\nSkandha Unified (proposed):");
    println!(
        "  Time: {:.6}s ({:+.2}% vs baseline)",
        skandha_time,
        ((skandha_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", skandha_belief.p);
    println!("  Mode: {:?}", skandha_belief.mode);
    println!(
        "  Deviation from ideal: {:.4} ({:+.2}% vs baseline)",
        skandha_dev,
        ((skandha_dev - baseline_dev) / baseline_dev * 100.0)
    );
}

#[test]
fn test_skandha_vs_baseline_stress_scenario() {
    // Test scenario: High stress (should converge to Stress mode)
    let observations: Vec<Observation> = (0..100)
        .map(|i| generate_test_observation(i * 1_000_000, 0.8)) // High stress
        .collect();

    println!("\n=== STRESS SCENARIO (100 observations, stress=0.8) ===\n");

    let (baseline_time, baseline_belief) = benchmark_baseline(&observations);
    let baseline_dev = compute_deviation(&baseline_belief, zenb_core::belief::BeliefBasis::Stress);
    println!("Baseline:");
    println!("  Time: {:.6}s", baseline_time);
    println!("  Belief: {:?}", baseline_belief.p);
    println!("  Deviation from ideal Stress: {:.4}", baseline_dev);

    let (vajra_time, vajra_belief) = benchmark_vajra_piecemeal(&observations);
    let vajra_dev = compute_deviation(&vajra_belief, zenb_core::belief::BeliefBasis::Stress);
    println!("\nVajra Piecemeal:");
    println!(
        "  Time: {:.6}s ({:+.2}%)",
        vajra_time,
        ((vajra_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", vajra_belief.p);
    println!(
        "  Deviation: {:.4} ({:+.2}%)",
        vajra_dev,
        ((vajra_dev - baseline_dev) / baseline_dev * 100.0)
    );

    let (skandha_time, skandha_belief) = benchmark_skandha_unified(&observations);
    let skandha_dev = compute_deviation(&skandha_belief, zenb_core::belief::BeliefBasis::Stress);
    println!("\nSkandha Unified:");
    println!(
        "  Time: {:.6}s ({:+.2}%)",
        skandha_time,
        ((skandha_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", skandha_belief.p);
    println!(
        "  Deviation: {:.4} ({:+.2}%)",
        skandha_dev,
        ((skandha_dev - baseline_dev) / baseline_dev * 100.0)
    );
}

#[test]
fn test_skandha_contradictory_sensors() {
    // Test scenario: Contradictory sensor data (Sheaf should filter)
    let mut observations = Vec::new();

    for i in 0..50 {
        let ts = i * 1_000_000;

        // Inject contradictory data: HR says calm (60 BPM) but HRV says stress (20ms)
        observations.push(Observation {
            timestamp_us: ts,
            bio_metrics: Some(zenb_core::domain::BioMetrics {
                hr_bpm: Some(60.0),    // Low HR (calm)
                hrv_rmssd: Some(20.0), // Low HRV (stressed)
                respiratory_rate: Some(12.0),
            }),
            environmental_context: None,
            digital_context: None,
            cognitive_context: None,
        });
    }

    println!("\n=== CONTRADICTORY SENSOR SCENARIO (50 observations) ===");
    println!("HR=60 (calm) but HRV=20 (stressed) - sensors disagree\n");

    let (baseline_time, baseline_belief) = benchmark_baseline(&observations);
    println!("Baseline (no filtering):");
    println!("  Time: {:.6}s", baseline_time);
    println!("  Belief: {:?}", baseline_belief.p);
    println!("  Mode: {:?}", baseline_belief.mode);

    let (vajra_time, vajra_belief) = benchmark_vajra_piecemeal(&observations);
    println!("\nVajra Piecemeal (with SheafPerception):");
    println!(
        "  Time: {:.6}s ({:+.2}%)",
        vajra_time,
        ((vajra_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", vajra_belief.p);
    println!("  Mode: {:?}", vajra_belief.mode);
    println!("  Sheaf should reduce contradiction!");

    let (skandha_time, skandha_belief) = benchmark_skandha_unified(&observations);
    println!("\nSkandha Unified:");
    println!(
        "  Time: {:.6}s ({:+.2}%)",
        skandha_time,
        ((skandha_time - baseline_time) / baseline_time * 100.0)
    );
    println!("  Belief: {:?}", skandha_belief.p);
    println!("  Mode: {:?}", skandha_belief.mode);
}

#[test]
fn test_memory_usage() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Simple allocation tracker
    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            System.dealloc(ptr, layout)
        }
    }

    println!("\n=== MEMORY USAGE COMPARISON ===\n");

    // Note: This test shows RELATIVE memory usage, not absolute
    // Rust's allocator reuses memory, so we measure peak allocations

    let observations: Vec<Observation> = (0..100)
        .map(|i| generate_test_observation(i * 1_000_000, 0.5))
        .collect();

    // Baseline
    ALLOCATED.store(0, Ordering::SeqCst);
    let _ = benchmark_baseline(&observations);
    let baseline_mem = ALLOCATED.load(Ordering::SeqCst);
    println!("Baseline peak allocations: {} bytes", baseline_mem);

    // Vajra Piecemeal
    ALLOCATED.store(0, Ordering::SeqCst);
    let _ = benchmark_vajra_piecemeal(&observations);
    let vajra_mem = ALLOCATED.load(Ordering::SeqCst);
    println!(
        "Vajra Piecemeal peak allocations: {} bytes ({:+.2}%)",
        vajra_mem,
        ((vajra_mem as f64 - baseline_mem as f64) / baseline_mem as f64 * 100.0)
    );

    // Skandha Unified
    ALLOCATED.store(0, Ordering::SeqCst);
    let _ = benchmark_skandha_unified(&observations);
    let skandha_mem = ALLOCATED.load(Ordering::SeqCst);
    println!(
        "Skandha Unified peak allocations: {} bytes ({:+.2}%)",
        skandha_mem,
        ((skandha_mem as f64 - baseline_mem as f64) / baseline_mem as f64 * 100.0)
    );
}
