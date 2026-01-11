use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pandora_mcg::enhanced_mcg::{EnhancedMetaCognitiveGovernor, ResourceMetrics, SystemMetrics};

fn bench_monitor(c: &mut Criterion) {
    let mut mcg = EnhancedMetaCognitiveGovernor::new();
    let metrics = SystemMetrics {
        uncertainty: 0.55,
        compression_reward: 0.75,
        novelty_score: 0.62,
        performance: 0.5,
        resource_usage: ResourceMetrics {
            cpu_usage: 0.4,
            memory_usage: 0.5,
            latency_ms: 12.0,
        },
    };
    c.bench_function("enhanced_mcg_monitor", |b| {
        b.iter(|| {
            let _ = mcg.monitor_comprehensive(black_box(&metrics));
        })
    });
}

criterion_group!(benches, bench_monitor);
criterion_main!(benches);
