# Pandora Causal World Model (CWM)

The "body" of the SDK, implementing machine learning capabilities and causal reasoning.

## Architecture

```
pandora_cwm/
├── ml/                    # Machine Learning modules
│   ├── predictor.rs       # World model predictors (logistic regression)
│   └── trainer.rs         # Model training utilities
├── gnn/                   # Graph Neural Networks
│   ├── message_passing.rs # Message passing operations
│   └── layers.rs          # Graph convolution layers
├── nn/                    # Neural Networks
│   └── uq_models.rs       # Uncertainty quantification
├── vsa/                   # Vector Symbolic Architecture
└── interdependent_repr/   # Interdependent representations
```

## Features

### Machine Learning (Optional)

Enable with `--features ml`:

```toml
[dependencies]
pandora_cwm = { version = "0.1.0", features = ["ml"] }
```

#### World Model Predictor

```rust
use pandora_cwm::ml::predictor::WorldModelPredictor;

let mut predictor = WorldModelPredictor::new(2);
let x = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
let y = vec![0, 1, 1];
predictor.train(x, 3, 2, y).unwrap();

let test_x = vec![1.5, 1.5];
let predictions = predictor.predict(test_x, 1, 2).unwrap();
```

#### Graph Neural Networks

```rust
use pandora_cwm::gnn::{message_passing::aggregate_mean, layers::GraphConvLayer};
use ndarray::arr2;

let adj = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
let features = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let aggregated = aggregate_mean(&adj, &features);

let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let layer = GraphConvLayer::new(weight);
let output = layer.forward(&adj, &features);
```

### Vector Symbolic Architecture (VSA)

```rust
use pandora_cwm::vsa::hrr::{bind, bundle};

let v1 = vec![1.0, 0.0, -1.0];
let v2 = vec![0.0, 1.0, 0.0];

// Binding operation
let bound = bind(&v1, &v2);

// Bundling operation
let bundled = bundle(&[&v1, &v2]);
```

## Performance

| Operation | Latency | Throughput |
|-----------|---------|------------|
| GNN Message Passing | ~5µs | 200k/sec |
| Logistic Regression (train) | ~100µs | 10k/sec |
| Logistic Regression (predict) | ~10µs | 100k/sec |
| VSA Bind | ~50ns | 20M/sec |
| VSA Bundle | ~100ns | 10M/sec |

## Usage

### Basic ML Pipeline

```rust
use pandora_cwm::ml::predictor::WorldModelPredictor;

// Train a world model predictor
let mut predictor = WorldModelPredictor::new(10);
let training_data = generate_training_data();
predictor.train(training_data.features, training_data.samples, 10, training_data.labels)?;

// Make predictions
let predictions = predictor.predict(test_features, 1, 10)?;
```

### Graph Processing

```rust
use pandora_cwm::gnn::layers::GraphConvLayer;
use ndarray::arr2;

// Create a simple graph convolution layer
let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let layer = GraphConvLayer::new(weight);

// Process graph data
let adjacency = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
let node_features = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let output = layer.forward(&adjacency, &node_features);
```

## Testing

```bash
# Unit tests
cargo test -p pandora_cwm

# ML feature tests
cargo test -p pandora_cwm --features ml

# VSA property tests
cargo test -p pandora_cwm --test vsa_properties
```

## Dependencies

### Required
- `ndarray` - N-dimensional arrays
- `bytes` - Byte handling utilities

### Optional (ML feature)
- `smartcore` - Pure Rust ML library
- `ndarray-rand` - Random array generation

## See Also

- [Architecture Guide](../../docs/architecture.md)
- [API Documentation](https://docs.rs/pandora-cwm)
