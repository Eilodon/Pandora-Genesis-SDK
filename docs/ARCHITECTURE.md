# AGOLOS Architecture Guide

> Comprehensive technical architecture for the Autonomous Goal-Oriented Learning Operating System

## System Overview

```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        UNIFFI[zenb-uniffi<br/>iOS/Android FFI]
        CLI[zenb-cli<br/>Terminal UI]
        WASM[zenb-wasm-demo<br/>Browser]
    end
    
    subgraph Core["Cognitive Core"]
        ENGINE[Engine<br/>Vinnana/Orchestrator]
        SKANDHA[Skandha Pipeline<br/>5-Stage Cognition]
        MEMORY[HolographicMemory<br/>Fourier Associative]
        SAFETY[DharmaFilter<br/>Ethical Constraints]
        CAUSAL[CausalHypergraph<br/>Intervention Reasoning]
    end
    
    subgraph Signals["Signal Processing"]
        RPPG[rPPG Ensemble<br/>PRISM/APON/CHROM]
        PHYSIO[HRV/Respiration<br/>Wellness Fusion]
        BEAUTY[Landmark Analysis<br/>468 MediaPipe Points]
    end
    
    subgraph Storage["Persistence"]
        STORE[zenb-store<br/>SQLite + XChaCha20]
        P2P[zenb-p2p<br/>GossipSub Network]
    end
    
    UNIFFI --> ENGINE
    CLI --> ENGINE
    WASM --> ENGINE
    
    ENGINE --> SKANDHA
    SKANDHA --> MEMORY
    SKANDHA --> SAFETY
    ENGINE --> CAUSAL
    
    RPPG --> ENGINE
    PHYSIO --> ENGINE
    BEAUTY --> ENGINE
    
    ENGINE --> STORE
    ENGINE --> P2P
```

---

## Five Skandhas Pipeline

Buddhist-inspired cognitive processing stages:

| Skandha | Sanskrit | Function | Implementation |
|---------|----------|----------|----------------|
| **Rupa** | Form | Sensory input | `SheafPerception` - Geometric fusion |
| **Vedana** | Feeling | Affective valence | `BeliefEngine` - FEP emotional state |
| **Sanna** | Perception | Pattern recognition | `HolographicMemory` - Fourier recall |
| **Sankhara** | Formation | Action preparation | `IntentTracker` - Goal formation |
| **Vinnana** | Consciousness | Executive control | `Engine` - Decision orchestration |

### Data Flow

```mermaid
sequenceDiagram
    participant Sensor
    participant Rupa
    participant Vedana
    participant Sanna
    participant Sankhara
    participant Vinnana
    
    Sensor->>Rupa: Raw RGB/physiological
    Rupa->>Vedana: Fused percept
    Vedana->>Sanna: + Valence annotation
    Sanna->>Sankhara: + Memory context
    Sankhara->>Vinnana: Action proposal
    Vinnana->>Vinnana: DharmaFilter check
    Vinnana-->>Sensor: Execute/Inhibit
```

---

## Philosophical State Machine

Three-state cognitive regulation based on Free Energy Principle:

```mermaid
stateDiagram-v2
    [*] --> YEN
    YEN --> DONG: free_energy > threshold
    DONG --> YEN: coherence > 0.8
    DONG --> HONLOAN: entropy spike
    HONLOAN --> DONG: stabilization
    HONLOAN --> YEN: circuit_breaker
    
    YEN: 🧘 Tranquil<br/>Low FE, high coherence
    DONG: ⚡ Active<br/>Moderate engagement
    HONLOAN: 🌀 Chaotic<br/>High entropy, fallback mode
```

| State | Free Energy | Coherence | Behavior |
|-------|-------------|-----------|----------|
| **YÊN** | Low | High | Minimal intervention, energy conservation |
| **ĐỘNG** | Moderate | Moderate | Active learning, exploration |
| **HỖN LOẠN** | High | Low | Circuit breaker, safe defaults |

---

## Memory Architecture

### HolographicMemory

Fourier-domain associative memory with GPU acceleration:

```
┌─────────────────────────────────────────────────┐
│             HolographicMemory                   │
├─────────────────────────────────────────────────┤
│  entangle(pattern)                              │
│    1. FFT(pattern) → frequency domain           │
│    2. Superimpose onto hologram                 │
│    3. Normalize energy                          │
├─────────────────────────────────────────────────┤
│  recall(cue)                                    │
│    1. FFT(cue) → frequency domain               │
│    2. Multiply with hologram conjugate          │
│    3. IFFT → reconstructed pattern              │
│    4. Similarity scoring                        │
└─────────────────────────────────────────────────┘
```

### HDC (Hyperdimensional Computing)

Binary vector memory for NPU acceleration:

- 10,000-dim binary vectors
- XOR binding, majority bundling
- Hamming distance similarity

---

## Safety Architecture

### DharmaFilter

LTL (Linear Temporal Logic) constraint monitoring:

```rust
// Example ethical constraint
□(harm_detected → ¬action_executed)  // Always: if harm detected, don't execute

// Implementation as TraumaPattern
TraumaPattern {
    signature: [...],
    severity: HarmLevel::Critical,
    response: Response::Inhibit,
}
```

### Safety Swarm

Multi-agent consensus voting:

```
┌─────────────────────────────────────────┐
│           Safety Swarm (3-of-5)         │
├─────────────────────────────────────────┤
│  [Guard 1] ─┐                           │
│  [Guard 2] ─┼─→ Majority Vote → Decision│
│  [Guard 3] ─┤                           │
│  [Guard 4] ─┤                           │
│  [Guard 5] ─┘                           │
└─────────────────────────────────────────┘
```

---

## Crate Dependencies

```mermaid
graph LR
    UNIFFI[zenb-uniffi] --> CORE[zenb-core]
    CLI[zenb-cli] --> CORE
    WASM[zenb-wasm] --> CORE
    CORE --> SIGNALS[zenb-signals]
    CORE --> STORE[zenb-store]
    CORE --> P2P[zenb-p2p]
    VERTICALS[zenb-verticals] --> CORE
    VERTICALS --> SIGNALS
```

---

## Key Invariants

1. **Free Energy Never Negative**: `assert!(free_energy >= 0.0)`
2. **Causality Preserved**: No cycles in CausalHypergraph
3. **Safety Always Checked**: Every action passes DharmaFilter
4. **Memory Bounded**: Hologram energy normalized after each entangle
5. **Encryption At Rest**: All persisted data uses XChaCha20-Poly1305

---

## Performance Characteristics

| Operation | Latency | Throughput |
|-----------|---------|------------|
| rPPG frame process | ~2ms | 500 fps |
| Skandha pipeline | ~5ms | 200 Hz |
| Holographic recall | ~1ms (CPU), ~0.1ms (GPU) | - |
| Safety filter check | <100μs | - |

---

*For API details, see [rustdoc](cargo doc --workspace --open)*
