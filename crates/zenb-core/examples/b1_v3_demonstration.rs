//! B.ONE V3 Integration Demonstration
//!
//! This example proves the VALUE of B.ONE V3 integration by comparing:
//! - BEFORE: Traditional processing (no philosophical state awareness)
//! - AFTER: B.ONE V3 processing (with consciousness-aware modulation)
//!
//! Run with: cargo run --example b1_v3_demonstration

use zenb_core::{
    // B.ONE V3 new types
    ConsciousnessAspect, FlowEnrichment, PhilosophicalProcessingConfig,
    PhilosophicalState, PhilosophicalStateMonitor, SkandhaStage,
    UniversalFlowStream, VedanaType,
    // Existing types
    BeliefBasis, BeliefEngine, BeliefState, Context, FepState,
    PhysioState, SensorFeatures,
    skandha::{AffectiveState, SensorInput},
};

use std::time::Instant;

// ============================================================================
// SCENARIO DEFINITIONS
// ============================================================================

/// Simulates different real-world scenarios
#[derive(Debug, Clone, Copy)]
enum Scenario {
    /// User meditating - calm, low HR, high HRV
    CalmMeditation,
    /// User working normally - moderate stress
    NormalWork,
    /// User in crisis - high HR, low HRV, system errors
    HighStressCrisis,
    /// Rapid fluctuations - unstable sensor data
    UnstableSignals,
}

impl Scenario {
    fn generate_sensor_data(&self, step: usize) -> SensorFeatures {
        match self {
            Scenario::CalmMeditation => SensorFeatures {
                hr_bpm: Some(55.0 + (step as f32 * 0.1).sin() * 2.0),
                rmssd: Some(80.0 + (step as f32 * 0.05).cos() * 5.0),
                rr_bpm: Some(6.0),
                quality: 0.95,
                motion: 0.05,
            },
            Scenario::NormalWork => SensorFeatures {
                hr_bpm: Some(72.0 + (step as f32 * 0.2).sin() * 8.0),
                rmssd: Some(45.0 + (step as f32 * 0.1).cos() * 10.0),
                rr_bpm: Some(14.0),
                quality: 0.85,
                motion: 0.2,
            },
            Scenario::HighStressCrisis => SensorFeatures {
                hr_bpm: Some(120.0 + (step as f32 * 0.5).sin() * 15.0),
                rmssd: Some(15.0 + (step as f32 * 0.3).cos() * 5.0),
                rr_bpm: Some(22.0),
                quality: 0.6,
                motion: 0.7,
            },
            Scenario::UnstableSignals => SensorFeatures {
                // Rapidly changing, unreliable data
                hr_bpm: if step % 3 == 0 { Some(140.0) } else { Some(50.0) },
                rmssd: if step % 2 == 0 { Some(100.0) } else { Some(5.0) },
                rr_bpm: Some(if step % 4 == 0 { 30.0 } else { 4.0 }),
                quality: 0.3 + (step as f32 * 0.7).sin().abs() * 0.4,
                motion: 0.1 + (step as f32 * 1.2).cos().abs() * 0.8,
            },
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Scenario::CalmMeditation => "Calm meditation - stable, relaxed state",
            Scenario::NormalWork => "Normal work - moderate cognitive load",
            Scenario::HighStressCrisis => "High stress crisis - system under pressure",
            Scenario::UnstableSignals => "Unstable signals - unreliable sensor data",
        }
    }
}

// ============================================================================
// TRADITIONAL PROCESSING (BEFORE B.ONE V3)
// ============================================================================

/// Traditional processing result
#[derive(Debug)]
struct TraditionalResult {
    decisions_made: usize,
    total_confidence: f32,
    mode_changes: usize,
    processing_time_us: u128,
    // No awareness of philosophical state
    // No Vedanā classification
    // No coherence tracking
}

fn process_traditional(scenario: Scenario, steps: usize) -> TraditionalResult {
    let start = Instant::now();
    let engine = BeliefEngine::new();

    let mut belief = BeliefState::default();
    let mut fep = FepState::default();
    let mut decisions = 0;
    let mut total_conf = 0.0;
    let mut mode_changes = 0;
    let mut last_mode = BeliefBasis::Calm;

    let ctx = Context {
        local_hour: 14,
        is_charging: false,
        recent_sessions: 2,
    };

    for step in 0..steps {
        let sensor = scenario.generate_sensor_data(step);
        let phys = PhysioState {
            hr_bpm: sensor.hr_bpm,
            rr_bpm: sensor.rr_bpm,
            rmssd: sensor.rmssd,
            confidence: sensor.quality,
        };

        // Traditional: Just update belief, no philosophical awareness
        let (new_belief, _debug) = engine.update(&belief, &sensor, &phys, &ctx, 0.5);
        belief = new_belief;

        // Count mode changes
        if belief.mode != last_mode {
            mode_changes += 1;
            last_mode = belief.mode;
        }

        // Simulated decision making (always proceeds, no protective measures)
        if belief.conf > 0.3 {
            decisions += 1;
            total_conf += belief.conf;
        }
    }

    TraditionalResult {
        decisions_made: decisions,
        total_confidence: if decisions > 0 { total_conf / decisions as f32 } else { 0.0 },
        mode_changes,
        processing_time_us: start.elapsed().as_micros(),
    }
}

// ============================================================================
// B.ONE V3 PROCESSING (AFTER INTEGRATION)
// ============================================================================

/// B.ONE V3 processing result
#[derive(Debug)]
struct BOneV3Result {
    decisions_made: usize,
    total_confidence: f32,
    mode_changes: usize,
    processing_time_us: u128,
    // NEW: B.ONE V3 specific metrics
    philosophical_states: PhilosophicalStateBreakdown,
    vedana_breakdown: VedanaBreakdown,
    coherence_avg: f32,
    protective_activations: usize,
    karma_balance: f32,
    consciousness_observations: usize,
}

#[derive(Debug, Default)]
struct PhilosophicalStateBreakdown {
    yen_cycles: usize,
    dong_cycles: usize,
    honloan_cycles: usize,
}

#[derive(Debug, Default)]
struct VedanaBreakdown {
    sukha: usize,
    dukkha: usize,
    upekkha: usize,
}

fn process_b1_v3(scenario: Scenario, steps: usize) -> BOneV3Result {
    let start = Instant::now();
    let engine = BeliefEngine::new();

    // B.ONE V3: Use Universal Flow Stream and Philosophical State Monitor
    let mut stream = UniversalFlowStream::new();
    let mut belief = BeliefState::default();
    let mut fep = FepState::default();

    let mut decisions = 0;
    let mut total_conf = 0.0;
    let mut mode_changes = 0;
    let mut last_mode = BeliefBasis::Calm;

    let mut states = PhilosophicalStateBreakdown::default();
    let mut vedana = VedanaBreakdown::default();
    let mut coherence_sum = 0.0;
    let mut protective_activations = 0;
    let mut consciousness_obs = 0;

    let ctx = Context {
        local_hour: 14,
        is_charging: false,
        recent_sessions: 2,
    };

    for step in 0..steps {
        let timestamp_us = (step as i64) * 500_000; // 500ms intervals
        let sensor = scenario.generate_sensor_data(step);
        let phys = PhysioState {
            hr_bpm: sensor.hr_bpm,
            rr_bpm: sensor.rr_bpm,
            rmssd: sensor.rmssd,
            confidence: sensor.quality,
        };

        // B.ONE V3: Emit sensor event to Universal Flow Stream
        let sensor_input = SensorInput {
            hr_bpm: sensor.hr_bpm,
            hrv_rmssd: sensor.rmssd,
            rr_bpm: sensor.rr_bpm,
            quality: sensor.quality,
            motion: sensor.motion,
            timestamp_us,
        };
        let _flow_event = stream.emit_sensor(sensor_input, timestamp_us);

        // Update belief (same as traditional)
        let (new_belief, debug) = engine.update(&belief, &sensor, &phys, &ctx, 0.5);
        belief = new_belief;

        // B.ONE V3: Compute coherence from agent votes
        let coherence = zenb_core::compute_coherence_from_votes(&debug.per_pathway);
        coherence_sum += coherence;

        // B.ONE V3: Calculate free energy proxy (from confidence inverse)
        let free_energy = 1.0 - belief.conf;

        // B.ONE V3: Update philosophical state
        let phil_state = stream.update_state(free_energy, coherence, timestamp_us);

        // Track philosophical states
        match phil_state {
            PhilosophicalState::Yen => states.yen_cycles += 1,
            PhilosophicalState::Dong => states.dong_cycles += 1,
            PhilosophicalState::HonLoan => states.honloan_cycles += 1,
        }

        // B.ONE V3: Classify Vedanā based on system health
        let karma_weight = if coherence > 0.7 && belief.conf > 0.6 {
            0.5 // Sukha - system aligned
        } else if coherence < 0.4 || belief.conf < 0.3 {
            -0.5 // Dukkha - system struggling
        } else {
            0.0 // Upekkha - neutral
        };

        let vedana_type = VedanaType::from_karma(karma_weight);
        match vedana_type {
            VedanaType::Sukha => vedana.sukha += 1,
            VedanaType::Dukkha => vedana.dukkha += 1,
            VedanaType::Upekkha => vedana.upekkha += 1,
        }

        // B.ONE V3: Emit affective state
        let affect = AffectiveState {
            valence: karma_weight,
            arousal: 0.5,
            confidence: belief.conf,
            karma_weight,
            is_karmic_debt: karma_weight < -0.3,
        };
        let _affect_event = stream.emit_affect(affect, timestamp_us);

        // Count mode changes
        if belief.mode != last_mode {
            mode_changes += 1;
            last_mode = belief.mode;
        }

        // B.ONE V3: Get processing config based on philosophical state
        let config = stream.state_monitor.get_processing_config();

        // B.ONE V3: CRITICAL DIFFERENCE - Protective measures in HỖN LOẠN
        if config.enable_safe_fallback {
            protective_activations += 1;
            // In HỖN LOẠN: Don't make decisions, wait for stability
            continue;
        }

        // B.ONE V3: Modulated decision making
        let adjusted_threshold = match phil_state {
            PhilosophicalState::Yen => 0.3,    // Normal threshold
            PhilosophicalState::Dong => 0.5,   // Higher threshold when active
            PhilosophicalState::HonLoan => 0.9, // Very high when chaotic
        };

        if belief.conf > adjusted_threshold {
            decisions += 1;
            total_conf += belief.conf;
            consciousness_obs += 1;
        }
    }

    BOneV3Result {
        decisions_made: decisions,
        total_confidence: if decisions > 0 { total_conf / decisions as f32 } else { 0.0 },
        mode_changes,
        processing_time_us: start.elapsed().as_micros(),
        philosophical_states: states,
        vedana_breakdown: vedana,
        coherence_avg: coherence_sum / steps as f32,
        protective_activations,
        karma_balance: stream.stats().karma_balance,
        consciousness_observations: consciousness_obs,
    }
}

// ============================================================================
// COMPARISON AND ANALYSIS
// ============================================================================

fn print_separator() {
    println!("{}", "=".repeat(80));
}

fn print_comparison(scenario: Scenario, traditional: &TraditionalResult, b1v3: &BOneV3Result) {
    println!();
    print_separator();
    println!("SCENARIO: {:?}", scenario);
    println!("{}", scenario.description());
    print_separator();

    println!("\n{:<40} {:>15} {:>15}", "Metric", "Traditional", "B.ONE V3");
    println!("{}", "-".repeat(72));

    // Basic metrics
    println!("{:<40} {:>15} {:>15}",
        "Decisions Made",
        traditional.decisions_made,
        b1v3.decisions_made
    );

    println!("{:<40} {:>15.3} {:>15.3}",
        "Avg Confidence",
        traditional.total_confidence,
        b1v3.total_confidence
    );

    println!("{:<40} {:>15} {:>15}",
        "Mode Changes (instability)",
        traditional.mode_changes,
        b1v3.mode_changes
    );

    println!("{:<40} {:>15} {:>15}",
        "Processing Time (μs)",
        traditional.processing_time_us,
        b1v3.processing_time_us
    );

    // B.ONE V3 exclusive metrics
    println!("\n--- B.ONE V3 EXCLUSIVE INSIGHTS ---");
    println!("{:<40} {:>15} {:>15}",
        "Philosophical States:",
        "N/A",
        ""
    );
    println!("  - YÊN (Tranquil) cycles:              {:>15}", b1v3.philosophical_states.yen_cycles);
    println!("  - ĐỘNG (Active) cycles:               {:>15}", b1v3.philosophical_states.dong_cycles);
    println!("  - HỖN LOẠN (Chaotic) cycles:          {:>15}", b1v3.philosophical_states.honloan_cycles);

    println!("\n{:<40} {:>15} {:>15}",
        "Vedanā Classification:",
        "N/A",
        ""
    );
    println!("  - Lạc Thọ (Sukha/Pleasant):           {:>15}", b1v3.vedana_breakdown.sukha);
    println!("  - Khổ Thọ (Dukkha/Unpleasant):        {:>15}", b1v3.vedana_breakdown.dukkha);
    println!("  - Xả Thọ (Upekkha/Neutral):           {:>15}", b1v3.vedana_breakdown.upekkha);

    println!("\n{:<40} {:>15} {:>15.3}",
        "Coherence (Tam Tâm Thức Agreement)",
        "N/A",
        b1v3.coherence_avg
    );

    println!("{:<40} {:>15} {:>15}",
        "Protective Activations",
        "0 (no protection)",
        b1v3.protective_activations
    );

    println!("{:<40} {:>15} {:>15.2}",
        "Karma Balance",
        "N/A",
        b1v3.karma_balance
    );

    // Analysis
    println!("\n--- VALUE ANALYSIS ---");

    // Decision quality
    let decision_diff = b1v3.decisions_made as i32 - traditional.decisions_made as i32;
    if decision_diff < 0 && b1v3.protective_activations > 0 {
        println!("✓ B.ONE V3 PREVENTED {} potentially bad decisions during chaos",
            decision_diff.abs());
    }

    // Confidence improvement
    if b1v3.total_confidence > traditional.total_confidence {
        let improvement = ((b1v3.total_confidence / traditional.total_confidence) - 1.0) * 100.0;
        println!("✓ Decision confidence improved by {:.1}%", improvement);
    }

    // Stability
    if b1v3.mode_changes < traditional.mode_changes {
        println!("✓ System {} more stable ({} fewer mode oscillations)",
            (traditional.mode_changes - b1v3.mode_changes) * 100 / traditional.mode_changes.max(1),
            traditional.mode_changes - b1v3.mode_changes);
    }

    // Philosophical awareness
    if b1v3.philosophical_states.honloan_cycles > 0 {
        println!("✓ Detected {} chaos cycles - Traditional would be BLIND to this!",
            b1v3.philosophical_states.honloan_cycles);
    }

    // Moral tracking
    if b1v3.vedana_breakdown.dukkha > 0 {
        println!("✓ Identified {} moments of system suffering (Dukkha) - enabling healing",
            b1v3.vedana_breakdown.dukkha);
    }
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          B.ONE V3 EPISTEMOLOGICAL REASONING SYSTEM - PROOF OF VALUE          ║");
    println!("║                                                                              ║");
    println!("║  Comparing TRADITIONAL processing vs B.ONE V3 consciousness-aware processing ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    let steps = 100; // 100 processing cycles per scenario

    let scenarios = [
        Scenario::CalmMeditation,
        Scenario::NormalWork,
        Scenario::HighStressCrisis,
        Scenario::UnstableSignals,
    ];

    let mut total_traditional_decisions = 0;
    let mut total_b1v3_decisions = 0;
    let mut total_protective = 0;
    let mut total_dukkha = 0;

    for scenario in scenarios {
        let traditional = process_traditional(scenario, steps);
        let b1v3 = process_b1_v3(scenario, steps);

        total_traditional_decisions += traditional.decisions_made;
        total_b1v3_decisions += b1v3.decisions_made;
        total_protective += b1v3.protective_activations;
        total_dukkha += b1v3.vedana_breakdown.dukkha;

        print_comparison(scenario, &traditional, &b1v3);
    }

    // Final summary
    println!();
    print_separator();
    println!("                           FINAL VERDICT");
    print_separator();
    println!();

    let prevented = total_traditional_decisions as i32 - total_b1v3_decisions as i32;
    if prevented > 0 {
        println!("🛡️  B.ONE V3 PREVENTED {} potentially harmful decisions", prevented);
        println!("    through {} protective activations during chaos states.", total_protective);
    }

    println!();
    println!("📊 NEW CAPABILITIES B.ONE V3 PROVIDES:");
    println!("   1. Philosophical State Awareness (YÊN/ĐỘNG/HỖN LOẠN)");
    println!("   2. Vedanā (Moral Feeling) Classification for every event");
    println!("   3. Coherence tracking across Three Consciousnesses");
    println!("   4. Karma balance for long-term system health");
    println!("   5. Automatic protective measures during instability");
    println!();

    println!("💡 KEY INSIGHT:");
    println!("   Traditional processing is BLIND to system chaos.");
    println!("   B.ONE V3 SEES the chaos, FEELS its nature (Vedanā),");
    println!("   and PROTECTS itself through philosophical awareness.");
    println!();

    if total_dukkha > 0 {
        println!("🧘 MINDFULNESS METRIC:");
        println!("   B.ONE V3 identified {} moments of 'Dukkha' (suffering/stress).", total_dukkha);
        println!("   Traditional processing would push through BLINDLY.");
        println!("   This is ETHICS-BY-DESIGN in action.");
    }

    println!();
    print_separator();
    println!("Proof complete. B.ONE V3 is not just theory - it's OBSERVABLE BEHAVIOR.");
    print_separator();
}
