use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic::*;  // Updated from deprecated basic_skandhas
use tokio::runtime::Runtime;

fn make_processor() -> SkandhaProcessor {
    SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    )
}

fn benchmark_sync_cycle(c: &mut Criterion) {
    let processor = make_processor();
    let event = b"test event with error".to_vec();

    c.bench_function("sync_cycle", |b| {
        b.iter(|| {
            let _ = processor.run_epistemological_cycle(black_box(event.clone()));
        });
    });
}

fn benchmark_async_cycle(c: &mut Criterion) {
    let processor = make_processor();
    let event = b"test event with error".to_vec();
    let rt = Runtime::new().unwrap();

    c.bench_function("async_cycle", |b| {
        b.to_async(&rt).iter(|| async {
            let _ = processor
                .run_epistemological_cycle_async(black_box(event.clone()))
                .await;
        });
    });
}

criterion_group!(benches, benchmark_sync_cycle, benchmark_async_cycle);
criterion_main!(benches);
