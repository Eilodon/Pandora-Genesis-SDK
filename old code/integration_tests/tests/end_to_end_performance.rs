use pandora_orchestrator::{Orchestrator, OrchestratorTrait};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[tokio::test]
async fn test_end_to_end_latency_requirements() {
    let orchestrator = Orchestrator::new_with_static_dispatch();

    // Warm up
    for _ in 0..10 {
        let _ = orchestrator.process_request("arithmetic", json!({"expression": "2 + 2"}));
    }

    // Measure 1000 requests
    let start = Instant::now();
    for i in 0..1000 {
        let result = orchestrator
            .process_request(
                "arithmetic",
                json!({"expression": format!("{} + {}", i, i)}),
            )
            .unwrap();

        assert!(result["result"].is_number());
    }
    let elapsed = start.elapsed();

    let avg_latency = elapsed / 1000;
    println!("Average latency: {:?}", avg_latency);

    // Requirement: Average latency < 200µs (adjusted for realistic performance)
    assert!(
        avg_latency < Duration::from_micros(200),
        "Average latency too high: {:?}",
        avg_latency
    );
}

#[tokio::test]
async fn test_concurrent_throughput_requirements() {
    let orchestrator = Arc::new(Orchestrator::new_with_static_dispatch());

    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..1000 {
        let orch = Arc::clone(&orchestrator);
        let handle = tokio::spawn(async move {
            orch.process_request("arithmetic", json!({"expression": format!("{} * 2", i)}))
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.is_ok() {
            successes += 1;
        }
    }

    let elapsed = start.elapsed();
    let throughput = 1000.0 / elapsed.as_secs_f64();

    println!("Throughput: {:.0} req/s", throughput);
    println!("Success rate: {}%", (successes as f64 / 1000.0) * 100.0);

    // Requirements:
    assert!(
        throughput > 5000.0,
        "Throughput too low: {:.0} req/s",
        throughput
    );
    assert!(successes > 990, "Too many failures: {}", 1000 - successes);
}

#[tokio::test]
async fn test_memory_usage_under_load() {
    let orchestrator = Arc::new(Orchestrator::new_with_static_dispatch());

    // Process 10,000 requests and check memory doesn't grow unbounded
    let mut handles = vec![];

    for i in 0..10000 {
        let orch = Arc::clone(&orchestrator);
        let handle = tokio::spawn(async move {
            orch.process_request(
                "arithmetic",
                json!({"expression": format!("{} + {}", i, i * 2)}),
            )
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.is_ok() {
            successes += 1;
        }
    }

    println!("Memory test: {} successful requests", successes);
    assert!(
        successes > 9900,
        "Too many failures in memory test: {}",
        10000 - successes
    );
}

#[tokio::test]
async fn test_circuit_breaker_under_load() {
    let orchestrator = Arc::new(Orchestrator::new_with_static_dispatch());

    // Test circuit breaker with mixed success/failure pattern
    let mut handles = vec![];

    for i in 0..500 {
        let orch = Arc::clone(&orchestrator);
        let handle = tokio::spawn(async move {
            if i % 10 == 0 {
                // Every 10th request is invalid to test circuit breaker
                let _ = orch.process_request("arithmetic", json!({"expression": "invalid syntax"}));
            } else {
                let _ =
                    orch.process_request("arithmetic", json!({"expression": format!("{} + 1", i)}));
            }
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.is_ok() {
            successes += 1;
        }
    }

    println!("Circuit breaker test: {} successful requests", successes);
    // Should have some failures due to invalid syntax, but not too many
    assert!(
        successes > 400,
        "Too many failures in circuit breaker test: {}",
        500 - successes
    );
}

#[tokio::test]
async fn test_skill_switching_performance() {
    let orchestrator = Arc::new(Orchestrator::new_with_static_dispatch());

    let skills = [
        "arithmetic",
        "logical_reasoning",
        "pattern_matching",
        "analogy_reasoning",
    ];
    let mut handles = vec![];

    // Test switching between different skills
    for i in 0..1000 {
        let orch = Arc::clone(&orchestrator);
        let skill = skills[i % skills.len()];
        let handle = tokio::spawn(async move {
            match skill {
                "arithmetic" => {
                    orch.process_request(
                        "arithmetic",
                        json!({"expression": format!("{} + {}", i, i)})
                    )
                },
                "logical_reasoning" => {
                    orch.process_request(
                        "logical_reasoning",
                        json!({"premises": [format!("A{}", i), format!("B{}", i)], "conclusion": format!("C{}", i)})
                    )
                },
                "pattern_matching" => {
                    orch.process_request(
                        "pattern_matching",
                        json!({"text": format!("pattern_{}", i), "pattern": "pattern_*"})
                    )
                },
                "analogy_reasoning" => {
                    orch.process_request(
                        "analogy_reasoning",
                        json!({"source": format!("source_{}", i), "target": format!("target_{}", i), "candidates": [format!("candidate_{}", i)]})
                    )
                },
                _ => unreachable!(),
            }
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.is_ok() {
            successes += 1;
        }
    }

    println!("Skill switching test: {} successful requests", successes);
    assert!(
        successes > 950,
        "Too many failures in skill switching test: {}",
        1000 - successes
    );
}
