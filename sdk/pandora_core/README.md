# Pandora Core

The philosophical "soul" of the SDK, defining core interfaces and data structures.

## Architecture

```
pandora_core/
├── interfaces/        # Core traits (FepCell, Skandhas, Skills)
├── ontology/         # Data structures (EpistemologicalFlow, DataEidos)
├── fep_cell/         # Skandha processor implementation
├── skandha_implementations/  # Basic and advanced skandha variants
└── world_model/      # World model traits and reward structures
```

## Key Concepts

### Free Energy Principle (FEP)

Every cognitive entity implements the `FepCell` trait:

```rust
pub trait FepCell {
    type Belief;
    type Observation;
    type Action;
    
    fn get_internal_model(&self) -> &Self::Belief;
    async fn perceive(&mut self, observation: Self::Observation) -> f64;
    async fn act(&mut self) -> Option<Self::Action>;
}
```

### Five Skandhas Pipeline

1. **Rupa (Form)**: Raw input → `EpistemologicalFlow`
2. **Vedana (Feeling)**: Assign moral valence (Pleasant/Unpleasant/Neutral)
3. **Sanna (Perception)**: Pattern matching against knowledge base
4. **Sankhara (Formations)**: Intent formation based on patterns
5. **Vinnana (Consciousness)**: Synthesis and potential rebirth

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Skandha Cycle (sync) | ~30-40µs | 30k/sec |
| Skandha Cycle (async) | ~45-55µs | 20k/sec |
| String Interning (new) | ~50ns | - |
| String Interning (cached) | ~10ns | - |

## Usage

### Basic Example

```rust
use pandora_core::fep_cell::SkandhaProcessor;
use pandora_core::skandha_implementations::basic_skandhas::*;

let processor = SkandhaProcessor::new(
    Box::new(BasicRupaSkandha),
    Box::new(BasicVedanaSkandha),
    Box::new(BasicSannaSkandha),
    Box::new(BasicSankharaSkandha),
    Box::new(BasicVinnanaSkandha),
);

let event = b"error detected".to_vec();
let result = processor.run_epistemological_cycle(event);
// Error events produce reborn events with corrective intent
assert!(result.is_some());
```

### Async Processing

```rust
# async fn example() {
let event = b"normal operation".to_vec();
let result = processor.run_epistemological_cycle_async(event).await;
// Normal events typically don't produce reborn events
assert!(result.is_none());
# }
```

### Advanced: Custom Skandha

```rust
use pandora_core::interfaces::skandhas::VedanaSkandha;

struct CustomVedanaSkandha {
    sensitivity: f32,
}

impl VedanaSkandha for CustomVedanaSkandha {
    fn feel(&self, flow: &mut EpistemologicalFlow) {
        // Custom logic...
    }
}
```

## Testing

```bash
# Unit tests
cargo test -p pandora_core

# Property tests
cargo test -p pandora_core --test skandha_properties

# Benchmarks
cargo bench -p pandora_core
```

## See Also

- [Architecture Guide](../../docs/architecture.md)
- [API Documentation](https://docs.rs/pandora-core)
