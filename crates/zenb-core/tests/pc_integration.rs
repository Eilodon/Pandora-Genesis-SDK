use zenb_core::engine::Engine;
use zenb_core::config::ZenbConfig;

#[test]
fn test_pc_learning_pipeline_configured() {
    // Configure Engine with PC learning enabled
    let mut config = ZenbConfig::default();
    config.features.pc_learning_enabled = true;
    
    let engine = Engine::new_with_config(6.0, Some(config));
    
    // Verify PC components are initialized
    assert_eq!(engine.config.features.pc_learning_enabled, true, "PC learning should be enabled");
    assert_eq!(engine.pc_change_detector.min_samples, 50, "PC detector should be configured with min_samples");
    assert_eq!(engine.observation_buffer.len(), 0, "Buffer should start empty");
    assert_eq!(engine.last_pc_run_ts, 0, "PC should not have run yet");
    
    println!("✅ PC Integration Test PASSED");
    println!("PC detector min_samples: {}", engine.pc_change_detector.min_samples);
    println!("PC detector trigger_interval: {}", engine.pc_change_detector.trigger_interval);
}

#[test]
fn test_pc_buffer_accumulation() {
    let mut config = ZenbConfig::default();
    config.features.pc_learning_enabled = true;
    
    let mut engine = Engine::new_with_config(6.0, Some(config));
    
    // Simulate observation ingestion
    use zenb_core::domain::{Observation, BioMetrics};
    
    for i in 0..10 {
        let obs = Observation {
            timestamp_us: i * 1_000_000,
            bio_metrics: Some(BioMetrics {
                hr_bpm: Some(70.0 + (i as f32)),
                hrv_rmssd: Some(50.0),
                respiratory_rate: Some(6.0),
            }),
            digital_context: None,
            environmental_context: None,
            cognitive_context: None,
        };
        
        engine.ingest_observation(obs);
        engine.tick(1_000_000);
    }
    
    // Verify observations were buffered
    assert_eq!(engine.observation_buffer.len(), 10, "Should have buffered 10 observations");
    
    println!("✅ PC Buffer Accumulation Test PASSED");
}
