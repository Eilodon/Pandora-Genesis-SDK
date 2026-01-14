//! Realistic load test scenarios simulating production usage

use pandora_orchestrator::{CircuitBreakerConfig, Orchestrator, OrchestratorTrait, SkillRegistry};
use pandora_tools::skills::arithmetic_skill::AdaptiveArithmeticEngine;
use pandora_tools::skills::logical_reasoning_skill::LogicalReasoningSkill;
use pandora_tools::PatternMatchingSkill;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration, Instant};

/// Simulate a realistic web service workload
/// - 70% arithmetic requests
/// - 20% pattern matching
/// - 10% complex reasoning
#[tokio::test]
#[ignore] // Run with: cargo test --test load_scenarios -- --ignored
async fn test_realistic_web_service_load() {
    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));
    registry.register_arc(Arc::new(PatternMatchingSkill));
    registry.register_arc(Arc::new(LogicalReasoningSkill));

    let orchestrator = Arc::new(Orchestrator::new(Arc::new(registry)));

    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    let duration = Duration::from_secs(30);
    let start = Instant::now();

    println!(
        "🚀 Starting realistic web service load test for {:?}",
        duration
    );
    println!("📊 Request distribution: 70% arithmetic, 20% pattern, 10% logic");

    let mut handles = vec![];
    let mut request_id = 0;

    while start.elapsed() < duration {
        let orch = orchestrator.clone();
        let success = success_count.clone();
        let errors = error_count.clone();

        // Determine request type based on distribution
        let rand = fastrand::u32(0..100);
        let (skill, input) = if rand < 70 {
            // 70% arithmetic
            let a = fastrand::i32(0..1000);
            let b = fastrand::i32(0..1000);
            (
                "arithmetic",
                json!({"expression": format!("{} + {}", a, b)}),
            )
        } else if rand < 90 {
            // 20% pattern matching
            let patterns = ["a*b", "x*y*z", "test*"];
            let pattern = patterns[fastrand::usize(0..patterns.len())];
            (
                "pattern_matching",
                json!({
                    "pattern": pattern,
                    "candidates": ["ab", "xyz", "test123", "other"]
                }),
            )
        } else {
            // 10% logical reasoning
            (
                "logical_reasoning",
                json!({
                    "ast": {
                        "type": "AND",
                        "children": [
                            {"type": "CONST", "value": true},
                            {"type": "CONST", "value": true}
                        ]
                    },
                    "context": {}
                }),
            )
        };

        let handle = tokio::spawn(async move {
            match orch.process_request(skill, input) {
                Ok(_) => success.fetch_add(1, Ordering::Relaxed),
                Err(_) => errors.fetch_add(1, Ordering::Relaxed),
            };
        });

        handles.push(handle);
        request_id += 1;

        // Control request rate: ~100 req/sec
        if request_id % 10 == 0 {
            sleep(Duration::from_millis(100)).await;
        }
    }

    // Wait for all requests to complete
    for handle in handles {
        let _ = handle.await;
    }

    let total_requests =
        success_count.load(Ordering::Relaxed) + error_count.load(Ordering::Relaxed);
    let success_rate =
        (success_count.load(Ordering::Relaxed) as f64 / total_requests as f64) * 100.0;

    println!("\n📈 Load Test Results:");
    println!("  Total requests: {}", total_requests);
    println!("  Successful: {}", success_count.load(Ordering::Relaxed));
    println!("  Failed: {}", error_count.load(Ordering::Relaxed));
    println!("  Success rate: {:.2}%", success_rate);
    println!(
        "  Requests/sec: {:.2}",
        total_requests as f64 / duration.as_secs_f64()
    );

    // Assert success rate > 95%
    assert!(
        success_rate > 95.0,
        "Success rate too low: {:.2}%",
        success_rate
    );
}

/// Simulate burst traffic scenario
#[tokio::test]
#[ignore]
async fn test_burst_traffic_scenario() {
    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));

    let config = CircuitBreakerConfig {
        failure_threshold: 10,
        ..Default::default()
    };

    let orchestrator = Arc::new(Orchestrator::with_config(Arc::new(registry), config));

    println!("🚀 Starting burst traffic test");

    // Normal load: 10 req/sec
    println!("📊 Phase 1: Normal load (10 req/sec) for 5 seconds");
    let mut handles = vec![];
    for i in 0..50 {
        let orch = orchestrator.clone();
        handles.push(tokio::spawn(async move {
            let input = json!({"expression": format!("{} + 1", i)});
            orch.process_request("arithmetic", input)
        }));
        if i % 10 == 0 {
            sleep(Duration::from_secs(1)).await;
        }
    }

    for handle in handles {
        let _ = handle.await;
    }

    // Burst: 1000 req in 1 second
    println!("📊 Phase 2: Burst (1000 req/sec) for 1 second");
    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..1000 {
        let orch = orchestrator.clone();
        handles.push(tokio::spawn(async move {
            let input = json!({"expression": format!("{} * 2", i)});
            orch.process_request("arithmetic", input)
        }));
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successes += 1;
        }
    }

    let elapsed = start.elapsed();

    println!("📈 Burst Results:");
    println!("  Total: 1000");
    println!("  Successes: {}", successes);
    println!("  Failures: {}", 1000 - successes);
    println!("  Duration: {:?}", elapsed);
    println!(
        "  Throughput: {:.2} req/sec",
        1000.0 / elapsed.as_secs_f64()
    );

    // We expect most requests to succeed even during burst
    assert!(
        successes > 900,
        "Too many failures during burst: {}",
        1000 - successes
    );
}
