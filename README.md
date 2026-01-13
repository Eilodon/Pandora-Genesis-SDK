# AGOLOS 🌌

[![CI](https://github.com/Eilodon/ZenB-Rust/workflows/CI/badge.svg)](https://github.com/Eilodon/ZenB-Rust/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

**Deeptech SDK for building autonomous adaptive systems** - featuring causal inference, belief propagation (Free Energy Principle), and deterministic state management for mission-critical control loops.

---

## 🌟 Features

### Core Capabilities
- ✅ **Deterministic Engine** - Cross-platform reproducible state with BLAKE3 hashing
- ✅ **Event Sourcing** - Encrypted SQLite store with crypto-shredding
- ✅ **Belief Engine** - Free Energy Principle with multi-pathway fusion
- ✅ **Safety Swarm** - Multi-guard consensus with trauma tracking
- ✅ **Resonance Tracking** - Goertzel-based phase detection
- ✅ **Async Worker** - Retry queue with emergency dump mechanism
- ✅ **Database Migration** - Safe schema evolution (v0→v2)

### Security & Safety
- 🔒 **XChaCha20-Poly1305** encryption per event
- 🔒 **TOCTOU-safe** transactions with IMMEDIATE lock
- 🔒 **Guard conflict** validation
- 🔒 **Trauma registry** with exponential decay
- 🔒 **Zero unsafe** code blocks

### Performance
- ⚡ **2x faster** hashing with fixed-point arithmetic
- ⚡ **Burst filtering** for sensor noise (<10ms)
- ⚡ **Bounded backpressure** (50-capacity channel)
- ⚡ **Mobile-optimized** (~70KB memory footprint)
- ⚡ **WAL mode** SQLite with batch append

---

## 📦 Architecture

```
zenb-rust/
├── crates/
│   ├── zenb-core/          # Deterministic domain logic
│   │   ├── belief/         # FEP belief engine
│   │   ├── safety_swarm/   # Multi-guard consensus
│   │   ├── resonance/      # Phase detection
│   │   └── breath_engine/  # Oscillator/Rhythm core (Reference Implementation)
│   ├── zenb-store/         # Encrypted event store
│   │   └── migration/      # Schema evolution
│   ├── zenb-projectors/    # Read models (Dashboard, Stats)
│   ├── zenb-uniffi/        # FFI runtime + async worker
│   ├── zenb-cli/           # CLI tools
│   └── zenb-wasm-demo/     # Web demo
├── docs/                   # Documentation
└── scripts/                # Build scripts
```

---

## 🔌 Domain Modules (v2.1)

AGOLOS supports **pluggable application domains** through a trait-based abstraction:

| Domain | Status | Variables | Modes | Use Case |
|--------|--------|-----------|-------|----------|
| `biofeedback` | ✅ Reference | 12 | 5 | Breath guidance, HRV, physiological signals |
| `trading` | ✅ Example | 7 | 5 | Market analysis, algorithmic trading |
| `robotics` | 📋 Planned | - | - | Autonomous systems, sensor fusion |
| `industrial` | 📋 Planned | - | - | IoT/sensor control, process automation |

### Create Your Own Domain

Implement **four traits** to create a custom domain:

```rust
use zenb_core::core::{Domain, OscillatorConfig, SignalVariable, ActionKind, BeliefMode};

// Define signal variables for causal modeling
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MyVariable { Sensor1, Sensor2, Output }
impl SignalVariable for MyVariable { /* ... */ }

// Define belief modes
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MyMode { Normal, Alert, Critical }
impl BeliefMode for MyMode { /* ... */ }

// Define actions
enum MyAction { Activate, Deactivate, Hold }
impl ActionKind for MyAction { /* ... */ }

// Tie everything together
struct MyDomain;
impl Domain for MyDomain {
    type Config = MyConfig;      // OscillatorConfig
    type Variable = MyVariable;  // SignalVariable
    type Action = MyAction;      // ActionKind
    type Mode = MyMode;          // BeliefMode (NEW!)
    
    fn name() -> &'static str { "my_domain" }
    fn default_priors() -> fn(usize, usize) -> f32 { |_, _| 0.0 }
}

// Use GenericCausalGraph with your variable type
type MyCausalGraph = zenb_core::core::GenericCausalGraph<MyVariable>;
```

See the [**trading domain**](crates/zenb-core/src/domains/trading/) as a complete example.

---

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+ (2021 edition)
- SQLite 3.x
- (Optional) `uniffi-bindgen` for FFI bindings

### Installation

```bash
# Clone repository
git clone https://github.com/Eilodon/ZenB-Rust.git
cd ZenB-Rust

# Build all crates
cargo build --all --release

# Run tests
cargo test --all --tests

# Run clippy
cargo clippy --all -- -D warnings

# Format code
cargo fmt -- --check
```

### Basic Usage

```rust
use zenb_core::{Engine, SessionId};
use zenb_store::EventStore;

// Initialize store
let master_key = [0u8; 32]; // Use secure key in production
let store = EventStore::open("zenb.db", master_key)?;

// Create session
let session_id = SessionId::new();
store.create_session_key(&session_id)?;

// Initialize engine (using oscillator reference implementation)
let mut engine = Engine::new(6.0); // 6.0 Hz/BPM primary frequency
engine.update_context(zenb_core::belief::Context {
    local_hour: 12,
    is_charging: true,
    recent_sessions: 0,
});

// Ingest sensor data
let features = vec![60.0, 40.0, 6.0, 0.9, 0.1]; // HR, RMSSD, RR, quality, motion
let estimate = engine.ingest_sensor(&features, timestamp_us);

// Make control decision
let (decision, changed, policy, deny_reason) = 
    engine.make_control(&estimate, timestamp_us, Some(&store));

if let Some(reason) = deny_reason {
    println!("Decision denied: {}", reason);
} else {
    println!("Target Rate: {:.2}", decision.target_rate_bpm);
}
```

### Using Async Worker

```rust
use zenb_uniffi::async_worker::AsyncWorker;

// Start async worker
let worker = AsyncWorker::start(store);

// Submit appends (non-blocking)
worker.submit_append(session_id, envelopes)?;

// Flush and wait
worker.flush_sync()?;

// Get metrics
let metrics = worker.metrics();
println!("Success: {}, Retries: {}, Dumps: {}",
    metrics.appends_success,
    metrics.retries,
    metrics.emergency_dumps
);

// Graceful shutdown
worker.shutdown();
```

---

## 📚 Documentation

### Core Documentation
- **[AUDIT_REPORT.md](AUDIT_REPORT.md)** - Comprehensive project audit (400+ lines)
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - P0.1-P0.7 implementation details
- **[IMPLEMENTATION_PART2_SUMMARY.md](IMPLEMENTATION_PART2_SUMMARY.md)** - P0.2, P0.6, P0.8, P0.9 details
- **[BLUEPRINT.md](docs/BLUEPRINT.md)** - High-level architecture
- **[TECH_SPEC.md](docs/TECH_SPEC.md)** - Technical specifications
- **[BELIEF_ENGINE.md](docs/BELIEF_ENGINE.md)** - Belief engine details
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history

### API Documentation

```bash
# Generate and open API docs
cargo doc --all --no-deps --open
```

---

## 🧪 Testing

### Run All Tests

```bash
# Full test suite
cargo test --all --tests

# With output
cargo test --all --tests -- --nocapture

# Specific crate
cargo test --package zenb-core
cargo test --package zenb-store

# Specific test
cargo test --package zenb-core tests_determinism -- --nocapture
```

### Test Coverage

| Crate | Tests | Coverage |
|-------|-------|----------|
| zenb-core | 20+ | ~70% |
| zenb-store | 13+ | ~75% |
| zenb-uniffi | 12+ | ~65% |
| **Total** | **45+** | **~70%** |

---

## 🔧 Development

### Developer Commands

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Run clippy
cargo clippy --all -- -D warnings

# Build release
cargo build --all --release

# Generate uniffi bindings
bash scripts/uniffi_gen.sh

# Watch mode (requires cargo-watch)
cargo watch -x "test --all"
```

### CI/CD

GitHub Actions workflow runs on every push:
1. ✅ Format check (`cargo fmt`)
2. ✅ Linting (`cargo clippy -D warnings`)
3. ✅ Test suite (`cargo test --all`)
4. ✅ UniFFI binding generation

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details.

---

## 🎯 P0 Improvements (v2.0 Gold + P0)

### Recently Completed (Jan 2026)

#### P0.1: Floating Point Determinism ✅
- Fixed-point arithmetic (1M scale, 6 decimals)
- Manual BLAKE3 hashing (2x faster)
- Cross-platform consistency
- **Tests:** 7 comprehensive tests

#### P0.2 & P0.8: Async Worker ✅
- Bounded channel (50 capacity)
- Retry queue with exponential backoff
- Emergency dump to JSON
- Atomic metrics tracking
- **Tests:** 3 integration tests

#### P0.3: Ironclad Transaction ✅
- IMMEDIATE lock for TOCTOU prevention
- INSERT OR IGNORE for idempotency
- Sequence validation in transaction
- Complete audit trail (append_log)
- **Tests:** 9 storage tests

#### P0.4: Enhanced Error Types ✅
- Structured error variants
- Contextual error messages
- Session ID in errors
- Observability improvements

#### P0.6: Guard Conflict Validation ✅
- Unsatisfiable range detection
- Invalid constraint validation
- Detailed error logging
- Production safety

#### P0.7: Estimator dt=0 Fix ✅
- Burst filtering (<10ms threshold)
- First sample initialization (alpha=1.0)
- Cached estimates
- **Tests:** 8 estimator tests

#### P0.9: Database Migration ✅
- Version tracking (v0→v1→v2)
- Idempotent migrations
- Atomic IMMEDIATE transactions
- Zero data loss
- **Tests:** 4 migration tests

**Total:** 8/8 P0 improvements, 31 new tests, 3,608 lines of code

---

## 📊 Performance Benchmarks

| Operation | Time | Frequency | Impact |
|-----------|------|-----------|--------|
| Hash (P0.1) | ~70µs | Per state change | Low |
| Belief Update | ~50µs | 1-2 Hz | Low |
| Resonance Calc | ~150µs | 1-2 Hz | Low |
| Guard Consensus | ~10µs | 1-2 Hz | Negligible |
| Event Encrypt | ~80µs | 2-4 Hz | Low |
| Batch Append | ~2ms | 0.1-1 Hz | Low |
| **Total CPU** | **<1%** | - | **Excellent** |

---

## 🔐 Security

### Encryption
- **Algorithm:** XChaCha20-Poly1305 (AEAD)
- **Key Size:** 256-bit
- **Nonce:** 192-bit random per event
- **AAD:** session_id + seq + event_type + ts_us + BLAKE3(meta)

### Key Management
- Per-session keys wrapped with master key
- Crypto-shredding via `delete_session_keys()`
- Zeroize for sensitive data cleanup

### Safety Guarantees
- No unsafe code blocks
- TOCTOU-safe transactions
- Guard conflict validation
- Trauma tracking with decay

---

## 🤝 Contributing

### Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- Follow Rust style guide
- Run `cargo fmt` before commit
- Ensure `cargo clippy` passes
- Add tests for new features
- Update documentation

### Commit Convention
```
feat: Add new feature
fix: Fix bug
docs: Update documentation
test: Add tests
refactor: Refactor code
perf: Performance improvement
chore: Maintenance tasks
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Free Energy Principle** - Karl Friston
- **Goertzel Algorithm** - Gerald Goertzel
- **Rust Community** - For excellent tooling and libraries
- **UniFFI** - Mozilla for FFI bindings

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Eilodon/ZenB-Rust/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Eilodon/ZenB-Rust/discussions)
- **Documentation:** [Project Wiki](https://github.com/Eilodon/ZenB-Rust/wiki)

---

## 🗺️ Roadmap

### Completed ✅
- [x] Core deterministic engine
- [x] Belief engine with FEP
- [x] Safety swarm with guards
- [x] Encrypted event store
- [x] Resonance tracking
- [x] All P0 improvements (P0.1-P0.9)
- [x] Async worker with retry queue
- [x] Database migration system
- [x] Domain-agnostic architecture (v2.1)
- [x] Generic CausalGraph with SignalVariable
- [x] BeliefMode trait with GenericBeliefState
- [x] Trading domain example

### In Progress 🚧
- [ ] Performance benchmarking suite
- [ ] Mobile platform integration
- [ ] Real-time dashboard

### Planned 📋
- [ ] Machine learning integration
- [ ] Multi-user support
- [ ] Cloud sync capabilities
- [ ] Advanced analytics

---

**Built with ❤️ using Rust** 🦀

*Last Updated: January 13, 2026*
