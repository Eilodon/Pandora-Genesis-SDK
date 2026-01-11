use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pandora_orchestrator::{CircuitBreakerConfig, Orchestrator, OrchestratorTrait, SkillRegistry};
use pandora_tools::ArithmeticSkill;
use serde_json::json;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn benchmark_single_skill_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_skill");

    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(ArithmeticSkill::new()));
    let orchestrator = Orchestrator::new(Arc::new(registry));

    let rt = Runtime::new().unwrap();

    group.bench_function("arithmetic_simple", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator
                .process_request("arithmetic", json!({"expression": "2 + 2"}))
                .unwrap()
        });
    });

    group.bench_function("arithmetic_complex", |b| {
        b.to_async(&rt).iter(|| async {
            orchestrator
                .process_request(
                    "arithmetic",
                    json!({"expression": "(123 + 456) * (789 - 321) / 2"}),
                )
                .unwrap()
        });
    });

    group.finish();
}

fn benchmark_concurrent_requests(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_requests");

    let mut registry = SkillRegistry::new();
    registry.register_arc(Arc::new(ArithmeticSkill::new()));
    let orchestrator = Arc::new(Orchestrator::new(Arc::new(registry)));

    let rt = Runtime::new().unwrap();

    // Benchmark different concurrency levels
    for concurrency in [1u64, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| {
                    let orch = orchestrator.clone();
                    async move {
                        let mut handles = vec![];

                        for i in 0..concurrency {
                            let orch = orch.clone();
                            let handle = tokio::spawn(async move {
                                let input = json!({"expression": format!("{} + {}", i, i)});
                                orch.process_request("arithmetic", input)
                            });
                            handles.push(handle);
                        }

                        for handle in handles {
                            let _ = handle.await;
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_circuit_breaker_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker");

    let rt = Runtime::new().unwrap();

    // Without circuit breaker (baseline)
    let mut registry_no_cb = SkillRegistry::new();
    registry_no_cb.register_arc(Arc::new(ArithmeticSkill::new()));
    let orch_no_cb = Orchestrator::new(Arc::new(registry_no_cb));

    group.bench_function("without_circuit_breaker", |b| {
        b.to_async(&rt).iter(|| async {
            orch_no_cb
                .process_request("arithmetic", json!({"expression": "2 + 2"}))
                .unwrap()
        });
    });

    // With circuit breaker
    let mut registry_with_cb = SkillRegistry::new();
    registry_with_cb.register_arc(Arc::new(ArithmeticSkill::new()));
    let orch_with_cb =
        Orchestrator::with_config(Arc::new(registry_with_cb), CircuitBreakerConfig::default());

    group.bench_function("with_circuit_breaker", |b| {
        b.to_async(&rt).iter(|| async {
            orch_with_cb
                .process_request("arithmetic", json!({"expression": "2 + 2"}))
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_single_skill_execution,
    benchmark_concurrent_requests,
    benchmark_circuit_breaker_overhead,
);
criterion_main!(benches);
