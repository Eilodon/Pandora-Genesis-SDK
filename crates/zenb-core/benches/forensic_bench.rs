use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use zenb_core::config::ZenbConfig;
use zenb_core::engine::Engine;
use zenb_core::skandha::{zenb, SensorInput};
use zenb_core::memory::HolographicMemory;
use num_complex::Complex32;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

fn benchmark_engine_ingest(c: &mut Criterion) {
    let mut engine = Engine::new(60.0);
    // Warmup
    let features = vec![70.0, 50.0, 15.0, 1.0, 0.0];
    engine.ingest_sensor(&features, 0);

    c.bench_function("engine_ingest_sensor", |b| {
        b.iter(|| {
            engine.ingest_sensor(black_box(&features), black_box(100));
        })
    });
}

fn benchmark_skandha_pipeline_process(c: &mut Criterion) {
    let config = ZenbConfig::default();
    let mut pipeline = zenb::zenb_pipeline(&config);
    let input = SensorInput {
        hr_bpm: Some(70.0),
        hrv_rmssd: Some(50.0),
        rr_bpm: Some(15.0),
        quality: 1.0,
        motion: 0.0,
        timestamp_us: 100,
    };

    c.bench_function("skandha_process", |b| {
        b.iter(|| {
            pipeline.process(black_box(&input));
        })
    });
}

/// Benchmark: HolographicMemory 10K-item entangle + recall
/// Target: <10ms for recall after 10K patterns stored
fn benchmark_holographic_memory_10k(c: &mut Criterion) {
    let dim = 256; // Typical dimension for production
    let mut memory = HolographicMemory::new(dim);
    let mut rng = StdRng::seed_from_u64(42);

    // Pre-generate 10K patterns for entanglement
    let patterns: Vec<(Vec<Complex32>, Vec<Complex32>)> = (0..10_000)
        .map(|_| {
            let key: Vec<Complex32> = (0..dim)
                .map(|_| Complex32::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5))
                .collect();
            let val = key.clone();
            (key, val)
        })
        .collect();

    // Entangle all patterns (setup)
    for (key, val) in &patterns {
        memory.entangle(key, val);
    }

    // Benchmark recall after 10K entanglements
    let query = &patterns[5000].0; // Query a middle pattern
    
    c.bench_function("holographic_10k_recall", |b| {
        b.iter(|| {
            memory.recall(black_box(query))
        })
    });
}

/// Benchmark: HolographicMemory entangle operation
fn benchmark_holographic_entangle(c: &mut Criterion) {
    let dim = 256;
    let mut memory = HolographicMemory::new(dim);
    let mut rng = StdRng::seed_from_u64(123);

    let key: Vec<Complex32> = (0..dim)
        .map(|_| Complex32::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5))
        .collect();
    let val = key.clone();

    c.bench_function("holographic_entangle", |b| {
        b.iter(|| {
            memory.entangle(black_box(&key), black_box(&val))
        })
    });
}

/// Benchmark: HolographicMemory decay operation (forgetting)
fn benchmark_holographic_decay(c: &mut Criterion) {
    let dim = 256;
    let mut memory = HolographicMemory::new(dim);
    let mut rng = StdRng::seed_from_u64(987);

    // Entangle some patterns first
    for _ in 0..100 {
        let key: Vec<Complex32> = (0..dim)
            .map(|_| Complex32::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5))
            .collect();
        memory.entangle(&key, &key);
    }

    c.bench_function("holographic_decay", |b| {
        b.iter(|| {
            memory.decay(black_box(0.99))
        })
    });
}

/// Benchmark: Memory scaling with pattern count
fn benchmark_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");
    let dim = 128;
    let mut rng = StdRng::seed_from_u64(456);

    for pattern_count in [100, 1000, 5000, 10000].iter() {
        let mut memory = HolographicMemory::new(dim);
        
        // Pre-populate
        let patterns: Vec<Vec<Complex32>> = (0..*pattern_count)
            .map(|_| {
                (0..dim)
                    .map(|_| Complex32::new(rng.gen::<f32>() - 0.5, rng.gen::<f32>() - 0.5))
                    .collect()
            })
            .collect();

        for p in &patterns {
            memory.entangle(p, p);
        }

        let query = &patterns[pattern_count / 2];
        
        group.bench_with_input(
            BenchmarkId::new("recall", pattern_count),
            pattern_count,
            |b, _| {
                b.iter(|| memory.recall(black_box(query)))
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_engine_ingest,
    benchmark_skandha_pipeline_process,
    benchmark_holographic_memory_10k,
    benchmark_holographic_entangle,
    benchmark_holographic_decay,
    benchmark_memory_scaling,
);
criterion_main!(benches);

