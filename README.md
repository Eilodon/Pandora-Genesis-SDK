# AGOLOS 🌌

[![CI](https://github.com/Eilodon/ZenB-Rust/workflows/CI/badge.svg)](https://github.com/Eilodon/ZenB-Rust/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

**Cognitive AI SDK for Autonomous Systems** — bridging Active Inference, Causal Reasoning, and Deterministic Control.

AGOLOS is not just a state engine; it is a complete toolkit for building **intelligent agents** that learn from experience, predict user behavior, and operate safely in uncertain environments.

---

## 🌟 Key Capabilities

### 🧠 Cognitive Engine (`zenb-core`)
- **Active Inference**: Prioritized Experience Replay for efficient learning from prediction errors.
- **Behavior Prediction**: PrefixSpan algorithm for mining sequential user patterns.
- **Belief Propagation**: Free Energy Principle (FEP) fusion of sensors and internal models.
- **Causal Reasoning**: DAG-based causal graph validation.

### 🛡️ Operational Safety
- **Deterministic State**: BLAKE3 hashing ensures full reproducibility of every decision.
- **Safety Swarm**: Consensus-based guard rails with trauma memory.
- **Crypto-Shredding**: Per-event XChaCha20-Poly1305 encryption with secure key deletion.
- **Circuit Breakers**: Automatic fault isolation for resilient operation.

### 🔬 Simulation & Verification
- **GridWorld**: Built-in 2D environment for training and verifying agent logic.
- **Partial Observability**: Native support for sensor occlusion and uncertainty.
- **Deterministic Replay**: Re-run any incident log with 100% fidelity.

### 📊 Production Observability
- **Prometheus Metrics**: Native exporters for Latency, Throughput, Confidence, and Error Rate.
- **Structured Logging**: Context-aware audit trails.

---

## 📦 Workspace Structure

```
AGOLOS/
├── crates/
│   ├── zenb-core/          # THE BRAIN: Cognitive SDK (Learning, Sim, Belief)
│   │   ├── learning/       # Experience Buffer & Pattern Mining (NEW)
│   │   ├── simulation/     # GridWorld & Environments (NEW)
│   │   ├── monitoring/     # Prometheus Metrics (NEW)
│   │   ├── belief/         # FEP State Machine
│   │   └── causal/         # Causal Graphs
│   ├── zenb-store/         # THE MEMORY: Encrypted Event Store (SQLite)
│   ├── zenb-signals/       # THE SENSES: DSP & Biosignal Processing
│   ├── zenb-p2p/           # THE VOICE: Peer-to-Peer Networking
│   └── zenb-uniffi/        # THE BODY: Cross-platform Bindings
└── sdk/                    # Pandora reference implementation
```

---

## 🚀 Usage Examples

### 1. Active Inference (Learning)
Train an agent to optimize decisions by replaying high-error experiences.

```rust
use zenb_core::learning::{PriorityExperienceBuffer, ExperienceSample};

// Create prioritized buffer
let mut buffer = PriorityExperienceBuffer::with_capacity(1000);

// Push experience with priority = prediction error
buffer.push(
    ExperienceSample { state, reward, ts_us },
    prediction_error // Higher error = higher replay probability
);

// Sample for training
let idx = buffer.sample_index(seed);
```

### 2. User Behavior Prediction
Predict what the user will do next based on their history.

```rust
use zenb_core::learning::{TemporalPrefixSpanEngine, Sequence};

let mut engine = TemporalPrefixSpanEngine::new(2, 5)?; // min_support=2, max_len=5
engine.mine_patterns(&user_history)?;

let output = engine.predict_next_action(&current_session)?;
println!("Next likely action: {} ({:.1}%)", 
    output[0].predicted_action, 
    output[0].confidence * 100.0
);
```

### 3. Simulation (Unit Testing)
Verify agent logic in a controlled environment.

```rust
use zenb_core::simulation::{GridWorld, Action, ObservabilityMode};

// Create a maze with partial visibility
let mut world = GridWorld::simple_maze(10, 10, ObservabilityMode::Partial { range: 3 });

// Agent acts
let result = world.submit_action(Action::Move(Direction::East));
let observation = world.get_world_state(); // Agent only sees 3 tiles away
```

### 4. Production Metrics
Enable observability in `Cargo.toml`:
```toml
zenb-core = { version = "*", features = ["prometheus"] }
```

```rust
zenb_core::monitoring::register_metrics();
// ... system runs ...
let metrics_text = zenb_core::monitoring::gather_metrics();
```

---

## 🔌 Domain Modules

AGOLOS supports pluggable architecture where specific business logic is isolated from the cognitive core:

| Domain | Status | Use Case |
|--------|--------|----------|
| **Biofeedback** | ✅ Native | Breath guidance, Heart Rate Variability (HRV) loops |
| **Trading** | 📘 Example | Market algorithmic trading with causal constraints |
| **Robotics** | 🚧 Planned | Sensor fusion navigation |

---

## 🛠️ Development

### Prerequisites
- Rust 1.70+
- SQLite 3.x

### Build & Test
```bash
# Build workspace
cargo build --release

# Run all tests (Core + Learning + Sim)
cargo test --all

# Run benchmarks
cargo bench
```

## 🔒 Security & Privacy
- **AES-256 equivalent** encryption for all data at rest.
- **Zero-knowledge** architecture capability (keys managed by client).
- **Audit-ready** cryptographic logs.

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.

*Built with ❤️ by the Eilodon Team.*
