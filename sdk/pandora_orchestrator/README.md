# Pandora Orchestrator

High-performance cognitive skill orchestrator with static dispatch and circuit breaker patterns.

## Features

- **Static Dispatch**: Zero-cost abstraction for built-in skills
- **Circuit Breaker**: Automatic failure detection and recovery
- **Retry Policies**: Configurable exponential backoff
- **Timeout Management**: Per-request timeout controls
- **Metrics**: Prometheus-compatible metrics
- **Schema Validation**: Optional JSON schema validation

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Static Dispatch | ~100µs | 10k/sec |
| Dynamic Dispatch | ~200µs | 5k/sec |
| Circuit Breaker Check | ~90ns | - |
| Skill Registration | ~1µs | - |

## Usage

### Basic Usage

```rust
use pandora_orchestrator::{Orchestrator, OrchestratorTrait};

// Create orchestrator with static dispatch (recommended)
let orchestrator = Orchestrator::new_with_static_dispatch();

// Process a request
let result = orchestrator.process_request(
    "arithmetic",
    serde_json::json!({"expression": "2 + 2"})
).await?;

println!("Result: {}", result["result"]);
```

### Advanced: Custom Configuration

```rust
use pandora_orchestrator::{
    Orchestrator, OrchestratorConfig, RetryPolicy, TimeoutPolicy
};

let config = OrchestratorConfig {
    retry: RetryPolicy {
        max_retries: 3,
        initial_backoff_ms: 100,
        backoff_multiplier: 2.0,
        max_backoff_ms: 5000,
        jitter_ms: 50,
    },
    timeout: TimeoutPolicy {
        timeout_ms: 5000,
    },
    circuit: CircuitConfig {
        failure_threshold: 5,
        open_cooldown_ms: 10000,
        half_open_trial: 2,
    },
};

let orchestrator = Orchestrator::new_with_static_dispatch()
    .with_retry_policy(config.retry)
    .with_timeout_policy(config.timeout);
```

### Routing with Fallbacks

```rust
use pandora_orchestrator::RoutingPolicy;

let routing = RoutingPolicy {
    primary: "arithmetic".to_string(),
    fallbacks: vec!["backup_calculator".to_string()],
};

let result = orchestrator.process_with_policy(routing, input).await?;
```

## Built-in Skills

The orchestrator comes with these pre-registered skills:

- **arithmetic**: Mathematical expression evaluation
- **logical_reasoning**: Boolean logic and inference
- **pattern_matching**: Text pattern recognition
- **analogy_reasoning**: Analogical reasoning

## Circuit Breaker

The circuit breaker automatically opens when failure rate exceeds threshold:

```rust
// Check circuit breaker status
let stats = orchestrator.circuit_stats();
println!("Open circuits: {}", stats.open);
println!("Closed circuits: {}", stats.closed);
```

## Metrics

Prometheus metrics are automatically collected:

- `cognitive_requests_total`: Total requests by skill
- `cognitive_requests_failed_total`: Failed requests by skill  
- `cognitive_request_duration_seconds`: Request duration histogram

## Testing

```bash
# Unit tests
cargo test -p pandora_orchestrator

# Integration tests
cargo test -p integration_tests

# Benchmarks
cargo bench -p pandora_orchestrator
```

## Configuration

Configuration can be loaded from:

1. `orchestrator.toml` file
2. Environment variables with `ORCH_` prefix
3. Programmatic configuration

Example `orchestrator.toml`:

```toml
[retry]
max_retries = 3
initial_backoff_ms = 100
backoff_multiplier = 2.0
max_backoff_ms = 5000
jitter_ms = 50

[timeout]
timeout_ms = 5000

[circuit]
failure_threshold = 5
open_cooldown_ms = 10000
half_open_trial = 2
```

## See Also

- [Pandora Core](../pandora_core/README.md) - Core interfaces
- [Pandora Tools](../pandora_tools/README.md) - Skill implementations
