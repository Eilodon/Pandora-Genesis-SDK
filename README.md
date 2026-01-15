# AGOLOS 🌌

[![CI](https://github.com/Eilodon/ZenB-Rust/workflows/CI/badge.svg)](https://github.com/Eilodon/ZenB-Rust/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

**Cognitive Biometric AI Platform** — A comprehensive SDK combining Active Inference, rPPG Signal Processing, and Buddhist-inspired Cognitive Architecture for building intelligent, embodied AI systems.

AGOLOS (Autonomous Goal-Oriented Learning Operating System) is a full-stack platform for:
- **Biometric Signal Processing**: Heart rate, HRV, respiration from camera (rPPG)
- **Cognitive AI Engine**: Active Inference with Free Energy Principle
- **Cross-Platform SDK**: Native bindings for iOS, Android, WASM

---

## 🌟 Core Capabilities

### 🧠 Cognitive Engine (`zenb-core`)

**Five Skandhas Architecture** — Buddhist-inspired cognitive pipeline:
- **Rupa (Form)**: `SheafPerception` - Geometric sensor fusion
- **Vedana (Feeling)**: `BeliefEngine` - Emotional valence from FEP
- **Sanna (Perception)**: `HolographicMemory` - Fourier-domain associative memory
- **Sankhara (Formation)**: `DharmaFilter` - Ethical action filtering
- **Vinnana (Consciousness)**: `Engine` - Supreme orchestrator

**Philosophical State Machine**:
- **YÊN (Tranquil)**: Low free energy, high coherence
- **ĐỘNG (Active)**: Moderate engagement, enhanced attention
- **HỖN LOẠN (Chaotic)**: High entropy, protective fallbacks

**Advanced Estimators**:
- **Unscented Kalman Filter (UKF)**: 5D state estimation (Arousal, Valence, Attention, Rhythm)
- **Liquid Time-Constant Networks (LTC)**: Adaptive temporal prediction
- **Hyperdimensional Computing (HDC)**: Binary vector memory for NPU acceleration

### 📡 Signal Processing (`zenb-signals`)

**rPPG Algorithms** (State-of-the-Art 2025):
- **PRISM**: Adaptive plane-orthogonal pulse extraction
- **APON**: Noise direction estimation via PCA
- **EnsembleProcessor**: SNR-weighted fusion of CHROM, POS, PRISM
- **Multi-ROI**: Forehead + cheeks landmark-based extraction

**Physiological Estimators**:
- **HRV Analysis**: RMSSD, SDNN, pNN50 from pulse waveform
- **Respiration Rate**: AM/FM/BW/CWT modulation fusion
- **Fatigue/Stress Fusion**: Multi-signal wellness scoring

**Beauty & Attention Module**:
- **468 MediaPipe Landmarks**: Canonical face normalization
- **Face Measurements**: 22 geometric ratios for shape classification
- **Attention Tracking**: EAR, PERCLOS, gaze direction
- **Quality Gating**: Pose, lighting, occlusion detection

### 🛡️ Safety & Security

- **DharmaFilter**: LTL-based ethical action constraints
- **TraumaGuard**: Pattern-based harm prevention with memory
- **Safety Swarm**: Consensus voting from multiple guard agents
- **Circuit Breakers**: Automatic fault isolation
- **Crypto-Shredding**: XChaCha20-Poly1305 per-event encryption

### 🔬 Learning & Simulation

- **Priority Experience Buffer**: Active Inference replay
- **PrefixSpan**: Sequential pattern mining for behavior prediction
- **GridWorld**: 2D environment with partial observability
- **Causal Discovery**: DAG-based intervention reasoning

---

## 📦 Workspace Structure

```
AGOLOS/
├── crates/
│   ├── zenb-core/           # 🧠 Cognitive Engine
│   │   ├── skandha/         # Five Skandhas pipeline
│   │   ├── memory/          # Holographic + HDC memory
│   │   ├── estimators/      # UKF, LTC predictors
│   │   ├── safety/          # DharmaFilter, LTL monitor
│   │   ├── causal/          # Causal graphs & intervention
│   │   ├── learning/        # Experience buffer, PrefixSpan
│   │   ├── simulation/      # GridWorld environment
│   │   └── domains/         # Pluggable domain modules
│   │
│   ├── zenb-signals/        # 📡 Biometric Signal Processing
│   │   ├── rppg/            # PRISM, APON, Ensemble, Multi-ROI
│   │   ├── physio/          # HRV, Respiration estimators
│   │   ├── dsp/             # Filtering, quality, motion
│   │   ├── wavelet/         # Morlet CWT, ALDTF denoising
│   │   ├── vision/          # Face detection, ROI extraction
│   │   └── beauty/          # Landmarks, measurements, attention
│   │
│   ├── zenb-store/          # 💾 Encrypted Event Store (SQLite)
│   ├── zenb-uniffi/         # 📱 Cross-Platform FFI (iOS/Android)
│   ├── zenb-verticals/      # 🏢 Vertical Market Modules (NEW)
│   ├── zenb-p2p/            # 🌐 Peer-to-Peer Networking
│   ├── zenb-cli/            # ⌨️ Command Line Interface
│   └── zenb-wasm-demo/      # 🌍 WebAssembly Demo
│
└── docs/                    # 📚 Documentation
    └── VERTICAL_MARKET_PLAN_*.md  # Expansion roadmap
```

---

## 🚀 Quick Start

### 1. rPPG Heart Rate Extraction

```rust
use zenb_signals::{EnsembleProcessor, EnsembleResult};

let mut processor = EnsembleProcessor::new();

// Feed RGB samples from face ROI
for frame in video_frames {
    processor.add_sample(frame.r, frame.g, frame.b);
}

// Get heart rate with confidence
if let Some(result) = processor.process() {
    println!("Heart Rate: {:.1} BPM (confidence: {:.0}%)", 
        result.heart_rate_bpm, 
        result.confidence * 100.0
    );
}
```

### 2. Cognitive Engine with Skandha Pipeline

```rust
use zenb_core::{Engine, SensorInput, PhilosophicalState};

let mut engine = Engine::new(config);

// Ingest biometric data
let input = SensorInput {
    hr_bpm: Some(72.0),
    hrv_rmssd: Some(45.0),
    quality: 0.9,
    motion: 0.1,
    timestamp_us: now_us,
    ..Default::default()
};

let (decision, state) = engine.ingest_sensor_with_context(input, context);

// Check philosophical state
match engine.philosophical_state() {
    PhilosophicalState::Yen => println!("System tranquil"),
    PhilosophicalState::Dong => println!("System active"),
    PhilosophicalState::HonLoan => println!("System chaotic - fallback active"),
}
```

### 3. HRV & Stress Analysis

```rust
use zenb_signals::{HrvEstimator, HrvConfig};
use ndarray::Array1;

let estimator = HrvEstimator::with_config(HrvConfig {
    sample_rate: 30.0,
    min_hr: 40.0,
    max_hr: 180.0,
    ..Default::default()
});

let pulse_signal = Array1::from_vec(pulse_data);
if let Some(hrv) = estimator.estimate(&pulse_signal) {
    println!("RMSSD: {:.1} ms", hrv.rmssd_ms);
    println!("Mean HR: {:.1} BPM", hrv.mean_hr_bpm);
}
```

### 4. Face Landmark Analysis

```rust
use zenb_signals::{BeautyAnalyzer, BeautyInput, normalize_to_canonical};

let mut analyzer = BeautyAnalyzer::new();

// From 468 MediaPipe landmarks
let canonical = normalize_to_canonical(&raw_landmarks, inter_ocular_px);
let input = BeautyInput {
    landmarks: canonical,
    timestamp_us: now_us,
    ..Default::default()
};

let result = analyzer.process_frame(&input);
println!("Face shape: {:?}", result.face_shape);
println!("Attention score: {:.2}", result.attention.score);
```

---

## 🏢 Vertical Markets

AGOLOS supports expansion into specialized verticals via `zenb-verticals`:

| Vertical | Status | Key Features |
|----------|--------|--------------|
| **Liveness Detection** | 🚧 Planned | rPPG pulse verification, texture analysis, challenge-response |
| **Driver Monitoring** | 🚧 Planned | PERCLOS drowsiness, gaze distraction, cardiac emergency |
| **Retail Analytics** | 🚧 Planned | Emotion tracking, engagement scoring, purchase intent |
| **Fintech Fraud** | 🚧 Planned | Cardiac fingerprinting, stress anomaly, coercion detection |
| **Exam Proctoring** | 🚧 Planned | Identity verification, gaze tracking, behavior scoring |

See `docs/VERTICAL_MARKET_PLAN_*.md` for detailed implementation roadmap.

---

## ⚙️ Feature Flags

```toml
[dependencies]
zenb-core = { version = "0.1", features = ["vajra_void", "prometheus"] }
zenb-signals = { version = "0.1", features = ["parallel", "image-processing"] }
```

| Feature | Crate | Description |
|---------|-------|-------------|
| `vajra_void` | zenb-core | Enable signal processing integration |
| `prometheus` | zenb-core | Production metrics export |
| `skandha_pipeline` | zenb-core | Debug visualization for cognitive pipeline |
| `parallel` | zenb-signals | Rayon parallelization for rPPG |
| `image-processing` | zenb-signals | Image loading for ROI extraction |

---

## 🛠️ Development

### Prerequisites
- Rust 1.70+
- SQLite 3.x (bundled)

### Build & Test
```bash
# Build entire workspace
cargo build --release

# Run all tests
cargo test --workspace

# Run with all features
cargo test --workspace --all-features

# Benchmarks
cargo bench -p zenb-core

# Generate documentation
cargo doc --workspace --open
```

### Cross-Platform Bindings
```bash
# Generate UniFFI bindings
cd crates/zenb-uniffi
cargo build --release

# Swift (iOS)
uniffi-bindgen generate src/zenb.udl --language swift

# Kotlin (Android)
uniffi-bindgen generate src/zenb.udl --language kotlin
```

---

## 🔒 Security & Privacy

- **XChaCha20-Poly1305**: Per-event encryption with secure key derivation
- **BLAKE3**: Deterministic state hashing for audit trails
- **Crypto-Shredding**: Secure deletion via key destruction
- **Zero-Knowledge Ready**: Client-managed keys architecture
- **GDPR/CCPA Compliant**: Data retention and consent utilities

---

## 📊 Architecture Highlights

```
┌─────────────────────────────────────────────────────────────┐
│                    AGOLOS Platform                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ zenb-uniffi │  │  zenb-cli   │  │   zenb-wasm-demo    │  │
│  │  (iOS/And)  │  │   (CLI)     │  │      (Web)          │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────┴────────────────┴─────────────────────┴──────────┐  │
│  │                     zenb-core                          │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐   │  │
│  │  │ Engine  │ │ Skandha │ │ Memory  │ │ DharmaFilter│   │  │
│  │  │ (Vinna) │ │ Pipeline│ │ (Holo)  │ │  (Safety)   │   │  │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘   │  │
│  │       └───────────┴───────────┴─────────────┘          │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │                    zenb-signals                        │  │
│  │  ┌──────┐ ┌───────┐ ┌───────┐ ┌────────┐ ┌──────────┐  │  │
│  │  │ rPPG │ │ Physio│ │  DSP  │ │ Vision │ │  Beauty  │  │  │
│  │  │PRISM │ │  HRV  │ │ FFT   │ │  ROI   │ │ Landmark │  │  │
│  │  └──────┘ └───────┘ └───────┘ └────────┘ └──────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │                    zenb-store                          │  │
│  │            SQLite + XChaCha20-Poly1305                 │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ by the Eilodon Team*

**AGOLOS** — *Where Cognitive AI Meets Biometric Intelligence*
