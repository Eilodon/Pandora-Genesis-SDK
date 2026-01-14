use metrics_exporter_prometheus::PrometheusBuilder;
use pandora_orchestrator::OrchestratorTrait;
use pandora_orchestrator::{Orchestrator, SkillRegistry};
use pandora_tools::skills::arithmetic_skill::ArithmeticSkill;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};

fn init_logging() {
    let mut filter = EnvFilter::from_default_env();
    if let Ok(d) = "pandora_core=info".parse() {
        filter = filter.add_directive(d);
    }
    if let Ok(d) = "pandora_simulation=info".parse() {
        filter = filter.add_directive(d);
    }
    if let Ok(d) = "pandora_orchestrator=info".parse() {
        filter = filter.add_directive(d);
    }

    fmt().with_env_filter(filter).init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logging();

    // Init Prometheus exporter at 0.0.0.0:9000 -> /metrics
    let builder = PrometheusBuilder::new();
    let _handle = builder
        .with_http_listener(([0, 0, 0, 0], 9000))
        .install_recorder()?;
    info!("Prometheus metrics exporter listening on :9000/metrics");

    // Setup orchestrator
    let mut registry = SkillRegistry::new();
    registry.register(Arc::new(ArithmeticSkill));

    let orchestrator = Arc::new(Orchestrator::new(Arc::new(registry)));

    // Start cleanup task
    let _cleanup_handle = orchestrator.clone().start_cleanup_task();

    // Simulate load
    info!("🚀 Starting load simulation...");

    for i in 0..100 {
        let orch = orchestrator.clone();
        tokio::spawn(async move {
            let input = serde_json::json!({"expression": format!("{} + {}", i, i)});
            let _ = orch.process_request("arithmetic", input).await;
        });
    }

    // Monitor stats periodically
    for _ in 0..10 {
        sleep(Duration::from_secs(2)).await;
        let stats = orchestrator.circuit_stats();
        info!(
            total = stats.total_circuits,
            closed = stats.closed,
            open = stats.open,
            half_open = stats.half_open,
            capacity = stats.capacity,
            "📊 Circuits"
        );
    }

    Ok(())
}
