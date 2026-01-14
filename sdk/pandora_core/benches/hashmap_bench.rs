use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fnv::FnvHashMap;
use std::collections::HashMap;

fn bench_string_keys(c: &mut Criterion) {
    let keys: Vec<String> = (0..1000).map(|i| format!("skill_{}", i)).collect();

    c.bench_function("std_hashmap_insert_string", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for key in &keys {
                map.insert(key.clone(), black_box(42));
            }
        });
    });

    c.bench_function("fnv_hashmap_insert_string", |b| {
        b.iter(|| {
            let mut map = FnvHashMap::default();
            for key in &keys {
                map.insert(key.clone(), black_box(42));
            }
        });
    });

    let mut std_map = HashMap::new();
    let mut fnv_map = FnvHashMap::default();
    for key in &keys {
        std_map.insert(key.clone(), 42);
        fnv_map.insert(key.clone(), 42);
    }

    c.bench_function("std_hashmap_lookup_string", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(std_map.get(key));
            }
        });
    });

    c.bench_function("fnv_hashmap_lookup_string", |b| {
        b.iter(|| {
            for key in &keys {
                black_box(fnv_map.get(key));
            }
        });
    });
}

fn bench_u64_keys(c: &mut Criterion) {
    let keys: Vec<u64> = (0..1000).collect();

    c.bench_function("std_hashmap_insert_u64", |b| {
        b.iter(|| {
            let mut map = HashMap::new();
            for &key in &keys {
                map.insert(key, black_box(42));
            }
        });
    });

    c.bench_function("fnv_hashmap_insert_u64", |b| {
        b.iter(|| {
            let mut map = FnvHashMap::default();
            for &key in &keys {
                map.insert(key, black_box(42));
            }
        });
    });
}

criterion_group!(benches, bench_string_keys, bench_u64_keys);
criterion_main!(benches);
