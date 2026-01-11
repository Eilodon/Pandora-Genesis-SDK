use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pandora_orchestrator::circuit_breaker::{
    CircuitBreakerConfig, LegacyCircuitBreakerManager, ShardedCircuitBreakerManager,
};
use std::sync::Arc;
use std::thread;

fn benchmark_circuit_breaker_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker_contention");

    for thread_count in [1, 2, 4, 8, 16, 32].iter() {
        // Benchmark sharded version
        group.bench_with_input(
            BenchmarkId::new("sharded", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let config = CircuitBreakerConfig::default();
                    let manager = Arc::new(ShardedCircuitBreakerManager::new(config));

                    let mut handles = vec![];
                    for i in 0..thread_count {
                        let mgr = Arc::clone(&manager);
                        let handle = thread::spawn(move || {
                            let skill_name = format!("skill_{}", i % 10);
                            for _ in 0..100 {
                                black_box(mgr.is_open(&skill_name));
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );

        // Benchmark legacy version for comparison
        group.bench_with_input(
            BenchmarkId::new("legacy", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let config = CircuitBreakerConfig::default();
                    let manager = Arc::new(LegacyCircuitBreakerManager::new(config));

                    let mut handles = vec![];
                    for i in 0..thread_count {
                        let mgr = Arc::clone(&manager);
                        let handle = thread::spawn(move || {
                            let skill_name = format!("skill_{}", i % 10);
                            for _ in 0..100 {
                                black_box(mgr.is_open(&skill_name));
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_circuit_breaker_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("circuit_breaker_operations");

    // Benchmark individual operations
    group.bench_function("sharded_is_open", |b| {
        let config = CircuitBreakerConfig::default();
        let manager = ShardedCircuitBreakerManager::new(config);
        let skill_name = "test_skill";

        b.iter(|| {
            black_box(manager.is_open(skill_name));
        });
    });

    group.bench_function("sharded_record_failure", |b| {
        let config = CircuitBreakerConfig::default();
        let manager = ShardedCircuitBreakerManager::new(config);
        let skill_name = "test_skill";

        b.iter(|| {
            manager.record_failure(skill_name);
        });
    });

    group.bench_function("sharded_record_success", |b| {
        let config = CircuitBreakerConfig::default();
        let manager = ShardedCircuitBreakerManager::new(config);
        let skill_name = "test_skill";

        b.iter(|| {
            manager.record_success(skill_name);
        });
    });

    group.bench_function("sharded_stats", |b| {
        let config = CircuitBreakerConfig::default();
        let manager = ShardedCircuitBreakerManager::new(config);

        // Pre-populate with some circuits
        for i in 0..50 {
            manager.record_failure(&format!("skill_{}", i));
        }

        b.iter(|| {
            black_box(manager.stats());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_circuit_breaker_contention,
    benchmark_circuit_breaker_operations
);
criterion_main!(benches);
