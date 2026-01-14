#![allow(clippy::field_reassign_with_default)]
use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};
use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::core::{  // Updated to use new core traits
    RupaSkandha, SankharaSkandha, SannaSkandha, VedanaSkandha, VinnanaSkandha,
};
use pandora_core::ontology::EpistemologicalFlow;
use pandora_core::skandha_implementations::basic::*;  // Updated from deprecated basic_skandhas
use std::time::Duration;

fn benchmark_skandha_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("skandha_pipeline");
    group.measurement_time(Duration::from_secs(10));

    let processor = SkandhaProcessor::new(
        Box::new(BasicRupaSkandha),
        Box::new(BasicVedanaSkandha),
        Box::new(BasicSannaSkandha),
        Box::new(BasicSankharaSkandha),
        Box::new(BasicVinnanaSkandha),
    );

    for size in [10, 100, 1000, 10_000].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::new("full_cycle", size), size, |b, &size| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter_batched(
                    || vec![0u8; size],
                    |event| async { processor.run_epistemological_cycle(black_box(event)) },
                    BatchSize::SmallInput,
                );
        });
    }

    group.finish();
}

fn benchmark_individual_skandhas(c: &mut Criterion) {
    let mut group = c.benchmark_group("individual_skandhas");

    let rupa = BasicRupaSkandha;
    let vedana = BasicVedanaSkandha;
    let sanna = BasicSannaSkandha;
    let sankhara = BasicSankharaSkandha;
    let vinnana = BasicVinnanaSkandha;

    group.bench_function("rupa_process_event", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter_batched(
                || vec![0u8; 100],
                |event| async { rupa.process_event(black_box(event)) },
                BatchSize::SmallInput,
            );
    });

    group.bench_function("vedana_feel", |b| {
        let mut flow = EpistemologicalFlow::default();
        flow.rupa = Some(b"test error event".to_vec().into());

        b.iter(|| {
            let mut test_flow = flow.clone();
            vedana.feel(black_box(&mut test_flow));
        });
    });

    group.bench_function("sanna_perceive", |b| {
        let mut flow = EpistemologicalFlow::default();
        flow.rupa = Some(b"test event with patterns".to_vec().into());

        b.iter(|| {
            let mut test_flow = flow.clone();
            sanna.perceive(black_box(&mut test_flow));
        });
    });

    group.bench_function("sankhara_form_intent", |b| {
        let mut flow = EpistemologicalFlow::default();
        flow.vedana = Some(pandora_core::ontology::Vedana::Unpleasant { karma_weight: -1.0 });

        b.iter(|| {
            let mut test_flow = flow.clone();
            sankhara.form_intent(black_box(&mut test_flow));
        });
    });

    group.bench_function("vinnana_synthesize", |b| {
        let mut flow = EpistemologicalFlow::default();
        flow.sankhara = Some(std::sync::Arc::from("TEST_INTENT"));

        b.iter(|| vinnana.synthesize(black_box(&flow)));
    });

    group.finish();
}

fn benchmark_memory_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocations");

    use pandora_core::string_interner::StringInterner;

    group.bench_function("string_intern_new", |b| {
        let interner = StringInterner::new();
        let strings: Vec<String> = (0..100).map(|i| format!("string_{}", i)).collect();

        b.iter(|| {
            for s in &strings {
                interner.intern(black_box(s));
            }
        });
    });

    group.bench_function("string_intern_existing", |b| {
        let interner = StringInterner::new();
        let s = "common_string";
        interner.intern(s);

        b.iter(|| interner.intern(black_box(s)));
    });

    group.bench_function("flow_from_bytes", |b| {
        let data = vec![0u8; 1024];

        b.iter(|| EpistemologicalFlow::from_bytes(black_box(data.clone().into())));
    });

    group.finish();
}

fn benchmark_hashmaps(c: &mut Criterion) {
    use fnv::FnvHashMap;
    use std::collections::HashMap;

    let mut group = c.benchmark_group("hashmap_comparison");

    let keys: Vec<String> = (0..1000).map(|i| format!("key_{}", i)).collect();

    group.bench_function("std_hashmap_insert_1000", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut map = HashMap::new();
                for key in keys {
                    map.insert(black_box(key), black_box(42));
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("fnv_hashmap_insert_1000", |b| {
        b.iter_batched(
            || keys.clone(),
            |keys| {
                let mut map = FnvHashMap::default();
                for key in keys {
                    map.insert(black_box(key), black_box(42));
                }
                map
            },
            BatchSize::SmallInput,
        );
    });

    let mut std_map = HashMap::new();
    let mut fnv_map = FnvHashMap::default();
    for (i, key) in keys.iter().enumerate() {
        std_map.insert(key.clone(), i);
        fnv_map.insert(key.clone(), i);
    }

    group.bench_function("std_hashmap_lookup_1000", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(std_map.get(black_box(key)));
            }
        });
    });

    group.bench_function("fnv_hashmap_lookup_1000", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(fnv_map.get(black_box(key)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_skandha_pipeline,
    benchmark_individual_skandhas,
    benchmark_memory_allocations,
    benchmark_hashmaps,
);
criterion_main!(benches);
