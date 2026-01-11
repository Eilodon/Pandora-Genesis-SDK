use zenb_core::config::{ZenbConfig, SotaConfig};
// use zenb_core::estimators::{UkfConfig, ukf::AukfConfig}; // imports removed as they were unused in the test body?
// Wait, I assign config.sota.ukf_config... which uses AukfConfig implicitly but struct name might not be needed?
// Actually compiler warned `unused imports: UkfConfig and ukf::AukfConfig`.
// Let's remove them if they are truly unused.
use zenb_core::engine::Engine;

#[test]
fn test_aukf_q_adaptation_in_engine() {
    // 1. Setup Config with UKF enabled and Q adaptation active
    let mut config = ZenbConfig::default();
    config.sota.use_ukf = true;
    config.sota.ukf_config.adapt_q = true;
    config.sota.ukf_config.min_samples = 10;
    config.sota.ukf_config.ukf.q_scale = 0.01;

    // 2. Initialize Engine
    let mut engine = Engine::new_with_config(6.0, Some(config));
    
    // Verify UKF is initialized
    assert!(engine.ukf_estimator.is_some(), "UKF should be initialized when use_ukf is true");

    // 3. Simulate noisy sine wave input
    // Q adaptation should react to the noise
    let mut q_initial = [0.0f32; 5];
    let mut q_final = [0.0f32; 5];
    
    println!("Simulating 50 steps of data...");
    
    for i in 0..50 {
        let t = i as f32 * 0.1;
        let signal = (t).sin() * 20.0 + 70.0; // HR base
        let noise = (i % 3) as f32 - 1.0; // Simple deterministic noise -1, 0, 1
        let hr = signal + noise;
        
        let features = vec![
            hr,             // HR
            50.0 + noise,   // RMSSD
            15.0,           // RR
            1.0,            // Quality
            0.0             // Motion
        ];
        
        // Ingest
        let ts_us = i * 100_000; // 0.1s steps
        let _est = engine.ingest_sensor(&features, ts_us as i64);
        
        // Capture initial Q after a few samples
        if i == 11 {
            q_initial = engine.ukf_estimator.as_ref().unwrap().get_q_diagonal();
        }
        
        // Check telemetry
        if i > 0 && i % 10 == 0 {
            assert!(engine.last_aukf_telemetry.is_some(), "Telemetry should be generated every 10 samples");
            let telemetry = engine.last_aukf_telemetry.as_ref().unwrap();
            let count = telemetry["aukf_sample_count"].as_u64().unwrap();
            // Count lags by 1 because first sample (i=0) has dt=0 and is skipped by engine
            assert_eq!(count, i as u64);
        }
    }
    
    // 4. Check Q adaptation results
    q_final = engine.ukf_estimator.as_ref().unwrap().get_q_diagonal();
    
    println!("Initial Q (step 11): {:?}", q_initial);
    println!("Final Q (step 50):   {:?}", q_final);
    
    // Verify Q has changed (adaptation occurred)
    let has_changed = q_initial.iter().zip(q_final.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(has_changed, "Q matrix should adapt over time with noisy input");
    
    // Verify values are within reasonable bounds (not NaN, not exploding)
    for q in q_final.iter() {
        assert!(!q.is_nan(), "Q should not be NaN");
        assert!(*q >= 0.0, "Q variance should be non-negative");
        assert!(*q < 1.0, "Q should remain small for this scale");
    }
}
