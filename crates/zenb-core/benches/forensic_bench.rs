use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zenb_core::config::ZenbConfig;
use zenb_core::engine::Engine;
use zenb_core::skandha::{zenb, SensorInput, SkandhaPipeline};

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
    let pipeline = zenb::zenb_pipeline(&config);
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

criterion_group!(
    benches,
    benchmark_engine_ingest,
    benchmark_skandha_pipeline_process
);
criterion_main!(benches);
