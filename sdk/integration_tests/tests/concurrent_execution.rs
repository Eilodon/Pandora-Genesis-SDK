use pandora_orchestrator::{Orchestrator, OrchestratorTrait, SkillRegistry};
use pandora_tools::skills::arithmetic_skill::AdaptiveArithmeticEngine;
use serde_json::json;
use std::sync::Arc;
use tokio::time::Duration;

#[tokio::test]
async fn test_concurrent_skill_execution() {
    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));
    let orchestrator = Arc::new(Orchestrator::new(Arc::new(registry)));

    // 100 request song song
    let mut handles = vec![];

    for i in 0..100 {
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move {
            let input = json!({"expression": format!("{} + {}", i, i)});
            orch.process_request("arithmetic", input)
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successes += 1;
        }
    }

    assert_eq!(successes, 100);
}

#[tokio::test]
async fn test_concurrent_circuit_breaker_isolation() {
    use pandora_orchestrator::CircuitBreakerConfig;

    struct WorkingSkill;
    struct FailingSkill;

    #[async_trait::async_trait]
    impl pandora_core::interfaces::skills::SkillModule for WorkingSkill {
        fn descriptor(&self) -> pandora_core::interfaces::skills::SkillDescriptor {
            pandora_core::interfaces::skills::SkillDescriptor {
                name: "working".to_string(),
                description: "Works".to_string(),
                input_schema: "{}".to_string(),
                output_schema: "{}".to_string(),
            }
        }

        async fn execute(
            &self,
            _: serde_json::Value,
        ) -> pandora_core::interfaces::skills::SkillOutput {
            Ok(json!({"status": "ok"}))
        }
    }

    #[async_trait::async_trait]
    impl pandora_core::interfaces::skills::SkillModule for FailingSkill {
        fn descriptor(&self) -> pandora_core::interfaces::skills::SkillDescriptor {
            pandora_core::interfaces::skills::SkillDescriptor {
                name: "failing".to_string(),
                description: "Fails".to_string(),
                input_schema: "{}".to_string(),
                output_schema: "{}".to_string(),
            }
        }

        async fn execute(
            &self,
            _: serde_json::Value,
        ) -> pandora_core::interfaces::skills::SkillOutput {
            Err(pandora_error::PandoraError::skill_exec("failing", "Error"))
        }
    }

    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(WorkingSkill));
    registry.register_arc(Arc::new(FailingSkill));

    let config = CircuitBreakerConfig {
        failure_threshold: 3,
        ..Default::default()
    };

    let orchestrator = Arc::new(Orchestrator::with_config(Arc::new(registry), config));

    let mut handles = vec![];

    for _ in 0..3 {
        let orch = orchestrator.clone();
        handles.push(tokio::spawn(async move {
            orch.process_request("failing", json!({}))
        }));
    }

    for _ in 0..10 {
        let orch = orchestrator.clone();
        handles.push(tokio::spawn(async move {
            orch.process_request("working", json!({}))
        }));
    }

    for handle in handles {
        let _ = handle.await;
    }

    let result = orchestrator.process_request("working", json!({}));
    assert!(result.is_ok());

    let result = orchestrator.process_request("failing", json!({}));
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_cleanup_task() {
    use pandora_orchestrator::CircuitBreakerConfig;

    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));

    let config = CircuitBreakerConfig {
        state_ttl_secs: 1,
        ..Default::default()
    };

    let orchestrator = Arc::new(Orchestrator::with_config(Arc::new(registry), config));

    let _cleanup_handle = orchestrator.clone().start_cleanup_task();

    for i in 0..10 {
        let skill_name = format!("skill_{}", i);
        let _ = orchestrator.process_request(&skill_name, json!({}));
    }

    let stats_before = orchestrator.circuit_stats();
    assert!(stats_before.total_circuits >= 1);

    tokio::time::sleep(Duration::from_secs(2)).await;

    let stats_after = orchestrator.circuit_stats();
    assert!(stats_after.total_circuits <= stats_after.capacity);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_high_concurrency_stress() {
    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(AdaptiveArithmeticEngine::new()));
    let orchestrator = Arc::new(Orchestrator::new(Arc::new(registry)));

    let mut handles = vec![];

    for i in 0..1000 {
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move {
            let input = json!({"expression": format!("{} * 2", i)});
            orch.process_request("arithmetic", input)
        });
        handles.push(handle);
    }

    let mut successes = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => successes += 1,
            _ => {}
        }
    }

    assert!(successes > 950, "Only {} successes out of 1000", successes);
}
