use crate::engine::Engine;
use crate::causal::Variable;
use crate::config::ZenbConfig;

#[test]
fn test_scientist_wires_discovery_to_causal_graph() {
    // 1. Setup Engine with Scientist enabled
    let mut config = ZenbConfig::default();
    config.features.scientist_enabled = Some(true);
    // Lower thresholds for faster testing
    config.features.thermo_enabled = Some(false); // Disable thermo to reduce noise
    
    let mut eng = Engine::new(6.0);
    eng.config = config;
    
    // Configure scientist for rapid discovery
    // We can't easily configure the internal scientist config without exposing it
    // But defaults are min_observations=30, correlation_threshold=0.3
    
    // 2. Feed correlated data: HR causes HRV (negative correlation for stress)
    // HR high -> HRV low
    // HR low -> HRV high
    
    // We need at least 30 observations to start proposing
    // Then 5 experiment steps
    // Then 1 tick to verify and wire
    
    let iterations = 60;
    
    for i in 0..iterations {
        let hr_norm = (i % 10) as f32 / 10.0;
        let hrv_norm = 1.0 - hr_norm; // Perfect negative correlation
        
        let obs = crate::domain::Observation {
            timestamp_us: i as i64 * 1_000_000,
            bio_metrics: Some(crate::domain::BioMetrics {
                hr_bpm: Some(hr_norm * 200.0),
                hrv_rmssd: Some(hrv_norm * 100.0),
                respiratory_rate: Some(15.0),
            }),
            ..Default::default()
        };
        
        eng.ingest_observation(obs);
        eng.tick(1_000_000);
        
        // Debug output to see state transitions
        if i % 10 == 0 {
            // println!("Tick {}: Scientist State = {:?}", i, eng.scientist.state_name());
        }
    }
    
    // 3. Verify Wiring
    // Scientist internal variables: 0=HR, 1=HRV
    // Hypothesis: 0 -> 1 (strength ~1.0)
    // CausalGraph: HeartRate -> HeartRateVariability
    
    let effect = eng.causal_graph.get_effect(Variable::HeartRate, Variable::HeartRateVariability);
    let edges = eng.causal_graph.get_effects(Variable::HeartRate);
    
    // Just checking we have SOME learned edges from HR
    let has_learned_edges = !edges.is_empty();
    
    // Print for debugging if failed
    if !has_learned_edges {
        println!("Scientist State: {:?}", eng.scientist.state_name());
        println!("Crystallized: {:?}", eng.scientist.crystallized);
        println!("Rejected: {:?}", eng.scientist.rejected);
    }
    
    assert!(has_learned_edges, "Scientist should have wired a causal link from HeartRate");
    
    // Check specific link if possible
    // Note: AutomaticScientist might find 0->1 or 1->0 depending on which it checks first
    let hr_to_hrv = eng.causal_graph.get_effect(Variable::HeartRate, Variable::HeartRateVariability);
    let hrv_to_hr = eng.causal_graph.get_effect(Variable::HeartRateVariability, Variable::HeartRate);
    
    assert!(hr_to_hrv > 0.0 || hrv_to_hr > 0.0, "Should detect relationship between HR and HRV");
}
