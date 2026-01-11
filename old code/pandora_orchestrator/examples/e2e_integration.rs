//! 🚀 Pandora Genesis SDK - End-to-End Integration Example
//!
//! This example demonstrates the complete cognitive system in action.

use bytes::Bytes;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "ml")]
use pandora_core::interfaces::skandhas::SankharaSkandha;
#[cfg(feature = "ml")]
use pandora_core::ontology::EpistemologicalFlow;
#[cfg(feature = "ml")]
use pandora_cwm::gnn::types::GnnConfig;
#[cfg(feature = "ml")]
use pandora_cwm::model::InterdependentCausalModel;
#[cfg(feature = "ml")]
use pandora_learning_engine::{ActiveInferenceSankharaSkandha, LearningEngine};
#[cfg(feature = "ml")]
use pandora_mcg::enhanced_mcg::EnhancedMetaCognitiveGovernor;
#[cfg(feature = "ml")]
use pandora_orchestrator::AutomaticScientistOrchestrator;

/// Main integration example
#[cfg(feature = "ml")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Pandora Genesis SDK - End-to-End Integration Example");
    println!("========================================================");

    // Initialize system
    let system = tokio::runtime::Runtime::new()?.block_on(initialize_system())?;

    // Run demonstrations
    tokio::runtime::Runtime::new()?.block_on(run_automatic_scientist_demo(&system))?;
    tokio::runtime::Runtime::new()?.block_on(run_active_inference_demo(&system))?;
    tokio::runtime::Runtime::new()?.block_on(run_performance_analysis(&system))?;

    println!("\n🎉 Integration example completed successfully!");
    Ok(())
}

/// Initialize the cognitive system
#[cfg(feature = "ml")]
async fn initialize_system() -> Result<CognitiveSystem, Box<dyn std::error::Error>> {
    println!("\n🔧 Initializing cognitive system...");

    let config = GnnConfig::new(128, 256, 5);
    let cwm = Arc::new(Mutex::new(InterdependentCausalModel::new(config)?));

    let learning_engine = Arc::new(LearningEngine::new(0.8, 0.2));

    let available_actions = vec![
        "explore",
        "exploit",
        "unlock_door",
        "pick_up_key",
        "move_forward",
        "turn_on_switch",
        "turn_off_switch",
        "observe",
        "plan",
        "wait",
    ];

    let sankhara = Arc::new(Mutex::new(ActiveInferenceSankharaSkandha::new(
        cwm.clone(),
        learning_engine.clone(),
        5,
        available_actions,
        0.9,
        0.15,
    )));

    let mcg = Arc::new(Mutex::new(EnhancedMetaCognitiveGovernor::new()));

    let orchestrator = AutomaticScientistOrchestrator::new(
        cwm.clone(),
        learning_engine.clone(),
        sankhara.clone(),
        mcg.clone(),
    );

    println!("   ✅ All components initialized");

    Ok(CognitiveSystem {
        cwm,
        learning_engine,
        sankhara,
        mcg,
        orchestrator,
    })
}

/// Run Automatic Scientist demonstration
#[cfg(feature = "ml")]
async fn run_automatic_scientist_demo(
    system: &CognitiveSystem,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Running Automatic Scientist Demo");
    println!("===================================");

    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"demo"));
    flow.set_static_intent("discover_causal_relationships");

    println!("🎯 Scenario: Discover hidden causal relationships");

    for cycle in 1..=10 {
        println!("🔄 Discovery Cycle {}: Testing relationships", cycle);

        let experimental_condition = cycle % 4;
        match experimental_condition {
            0 => flow.set_static_intent("activate_system_a"),
            1 => flow.set_static_intent("deactivate_system_a"),
            2 => flow.set_static_intent("enable_system_b"),
            _ => flow.set_static_intent("disable_system_b"),
        }

        system.orchestrator.run_cycle(&mut flow).await?;

        if cycle % 5 == 0 {
            println!("   📊 Progress check: Analyzing patterns...");
        }
    }

    println!("✅ Automatic Scientist demo completed");
    Ok(())
}

/// Run Active Inference demonstration
#[cfg(feature = "ml")]
async fn run_active_inference_demo(
    system: &CognitiveSystem,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧠 Running Active Inference Planning Demo");
    println!("=========================================");

    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"planning_demo"));

    println!("🎯 Scenario: Multi-step planning");
    println!("   - Get key from room A to open box in room C");
    println!("   - Ignore distraction in room B");

    let mut current_room = "start";
    let mut has_key = false;
    let mut box_opened = false;

    for step in 1..=10 {
        println!(
            "🔄 Step {}: Room = {}, Has key = {}, Box opened = {}",
            step, current_room, has_key, box_opened
        );

        system.sankhara.lock().unwrap().form_intent(&mut flow);

        let planned_action = flow.sankhara.as_ref().map(|s| s.as_ref()).unwrap_or("wait");

        println!("   Planned action: {}", planned_action);

        // Simulate action execution
        match planned_action {
            "move_to_room_a" => {
                current_room = "room_a";
                println!("   ✅ Moved to room A");
            }
            "pick_up_key" => {
                if current_room == "room_a" {
                    has_key = true;
                    println!("   ✅ Picked up key");
                }
            }
            "move_to_room_c" => {
                current_room = "room_c";
                println!("   ✅ Moved to room C");
            }
            "open_box" => {
                if current_room == "room_c" && has_key {
                    box_opened = true;
                    println!("   🎉 Opened box!");
                }
            }
            _ => println!("   ⏸️  Action: {}", planned_action),
        }

        if box_opened {
            println!("   🎯 GOAL ACHIEVED!");
            break;
        }
    }

    println!("✅ Active Inference demo completed");
    Ok(())
}

/// Run Performance Analysis
#[cfg(feature = "ml")]
async fn run_performance_analysis(
    system: &CognitiveSystem,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Running Performance Analysis");
    println!("===============================");

    let mut flow = EpistemologicalFlow::from_bytes(Bytes::from_static(b"perf_test"));

    let start_time = Instant::now();

    for i in 1..=1000 {
        system.sankhara.lock().unwrap().form_intent(&mut flow);

        if i % 200 == 0 {
            let elapsed = start_time.elapsed();
            let avg_time = elapsed / i;
            println!("   Processed {} cycles, average: {:?}", i, avg_time);
        }
    }

    let total_time = start_time.elapsed();
    let avg_time_per_cycle = total_time / 1000;

    println!("✅ Performance results:");
    println!("   Total time: {:?}", total_time);
    println!("   Average per cycle: {:?}", avg_time_per_cycle);
    println!(
        "   Cycles per second: {:.2}",
        1000.0 / total_time.as_secs_f64()
    );

    Ok(())
}

/// Cognitive system structure
#[cfg(feature = "ml")]
#[allow(dead_code)]
struct CognitiveSystem {
    cwm: Arc<Mutex<InterdependentCausalModel>>,
    learning_engine: Arc<LearningEngine>,
    sankhara: Arc<Mutex<ActiveInferenceSankharaSkandha>>,
    mcg: Arc<Mutex<EnhancedMetaCognitiveGovernor>>,
    orchestrator: AutomaticScientistOrchestrator,
}

/// Fallback for when ML features are not enabled
#[cfg(not(feature = "ml"))]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("❌ ML features not enabled. Please run with --features ml");
    println!("Example: cargo run --example e2e_integration --features ml");
    Ok(())
}
