use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pandora_core::intents;
use pandora_core::ontology::EpistemologicalFlow;
use std::sync::Arc;

fn benchmark_flow_creation(c: &mut Criterion) {
    c.bench_function("flow_from_bytes", |b| {
        let data = vec![0u8; 1024];
        b.iter(|| {
            let flow = EpistemologicalFlow::from_bytes(black_box(bytes::Bytes::from(data.clone())));
            black_box(flow);
        });
    });

    // Baseline-like variants to compare against
    c.bench_function("flow_from_vec_clone", |b| {
        let data = vec![0u8; 1024];
        b.iter(|| {
            // Simulate old path: allocate a Vec and clone into a new Vec (heap)
            let owned: Vec<u8> = black_box(data.clone());
            let flow = EpistemologicalFlow::from_bytes(bytes::Bytes::from(owned));
            black_box(flow);
        });
    });

    c.bench_function("flow_set_static_intent", |b| {
        let mut flow = EpistemologicalFlow::default();
        b.iter(|| {
            flow.set_static_intent(black_box(intents::constants::REPORT_ERROR));
        });
    });

    c.bench_function("flow_set_owned_string_intent", |b| {
        let mut flow = EpistemologicalFlow::default();
        b.iter(|| {
            // Simulate old path: allocate new String each iteration
            let s = String::from("REPORT_ERROR");
            let arc: Arc<str> = Arc::from(s);
            flow.set_interned_intent(black_box(arc));
        });
    });

    c.bench_function("flow_set_interned_intent", |b| {
        let mut flow = EpistemologicalFlow::default();
        let intent: Arc<str> = Arc::from("CUSTOM_INTENT");
        b.iter(|| {
            flow.set_interned_intent(black_box(intent.clone()));
        });
    });
}

criterion_group!(benches, benchmark_flow_creation);
criterion_main!(benches);
