//! Phase 2: Stateful vs. Stateless Benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pandora_core::skandha_implementations::{
    basic::*,
    core::*,
    stateful::*,
    processors::*,
};
use pandora_core::alaya::{HashEmbedding, EmbeddingModel};  // For StatefulVedana embedding
use std::sync::Arc;

fn create_linear_processor() -> LinearProcessor {
    LinearProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        Box::new(BasicVedanaSkandha::default()),
        Box::new(BasicSannaSkandha::default()),
        Box::new(BasicSankharaSkandha::default()),
        Box::new(BasicVinnanaSkandha::default()),
    )
}

fn create_recurrent_processor() -> RecurrentProcessor<StatefulVedana, StatefulSanna> {
    let embedding = Arc::new(HashEmbedding::new(128)) as Arc<dyn EmbeddingModel>;  // 128-dim embeddings for benchmarking
    RecurrentProcessor::new(
        Box::new(BasicRupaSkandha::default()),
        StatefulVedana::new("StatefulVedana", Arc::new(BasicVedanaSkandha::default()), embedding),
        StatefulSanna::new("StatefulSanna", Arc::new(BasicSannaSkandha::default())),
        Box::new(adapters::StatelessAdapter::new(BasicSankharaSkandha::default())),
        Box::new(adapters::StatelessAdapter::new(BasicVinnanaSkandha::default())),
    )
}

fn bench_processors(c: &mut Criterion) {
    let mut group = c.benchmark_group("Processor Comparison");
    let event = b"A typical log event with some information to process.".to_vec();

    let linear = create_linear_processor();
    group.bench_function("LinearProcessor", |b| {
        b.iter(|| linear.run_cycle(black_box(event.clone())))
    });

    let mut recurrent = create_recurrent_processor();
    group.bench_function("RecurrentProcessor (No Loops)", |b| {
        b.iter(|| recurrent.run_cycle(black_box(event.clone()), EnergyBudget::default_budget()))
    });

    group.finish();
}

criterion_group!(benches, bench_processors);
criterion_main!(benches);
