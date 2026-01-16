# zenb-core Architecture Guide

> **The Trikaya Mapping: From Buddhist Philosophy to System Design**

This document maps the `zenb-core` codebase to the **Trikaya Pattern** (Three Bodies Architecture), providing a mental model for understanding the consciousness engine.

---

## Overview: The Three Bodies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRIKAYA ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ESSENCE (Tầng THỂ / Pháp Thân)                   │   │
│   │                                                                      │   │
│   │   Infrastructure & Data Layer                                        │   │
│   │   - Event sourcing, persistence, global state                        │   │
│   │   - Immutable, persistent, event-driven                              │   │
│   │                                                                      │   │
│   │   Files: universal_flow.rs, philosophical_state.rs, domain.rs        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     ORGANS (Tầng HÌNH / Báo Thân)                   │   │
│   │                                                                      │   │
│   │   Domain Kernel / Core Logic                                         │   │
│   │   - Decision making, belief formation, ethical filtering             │   │
│   │   - Stateful, goal-oriented, complex                                 │   │
│   │                                                                      │   │
│   │   Files: skandha/, belief/, safety/, vinnana_controller.rs           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                 CAPABILITIES (Tầng DỤNG / Hóa Thân)                 │   │
│   │                                                                      │   │
│   │   Adapters & Tools Layer                                             │   │
│   │   - Signal processing, memory backends, actuators                    │   │
│   │   - Stateless, functional, pluggable                                 │   │
│   │                                                                      │   │
│   │   Files: zenb-signals, memory/, breath_engine.rs, sensory/           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: ESSENCE (Tầng THỂ / Pháp Thân)

> "The Source of Truth - where all events arise and are witnessed equally."

### Purpose
Infrastructure layer responsible for:
- Event sourcing and persistence
- Global consciousness state management
- Event bus for inter-module communication

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **UniversalFlowStream** | `universal_flow.rs` | Central event bus (Vô Cực Stream). All events flow through here. |
| **FlowEvent** | `universal_flow.rs` | Immutable event record with lineage tracking |
| **PhilosophicalStateMonitor** | `philosophical_state.rs` | Tracks YÊN/ĐỘNG/HỖN LOẠN states |
| **Domain Models** | `domain.rs` | `Envelope`, `Event`, `SessionId`, `Observation` |
| **Timestamp** | `timestamp.rs` | Session duration and time tracking |

### Data Flow Principle
```
External Input → FlowStream.emit() → [Stored in Essence] → Organs subscribe
```

### Code Example
```rust
// universal_flow.rs - The heart of Essence layer
pub struct UniversalFlowStream {
    pub session_id: SessionId,
    event_counter: u64,
    pub state_monitor: PhilosophicalStateMonitor,  // YÊN/ĐỘNG/HỖN LOẠN
    pub stats: FlowStreamStats,
    recent_events: Vec<FlowEventId>,  // Lineage tracking
}

// Emit event into the stream
pub fn emit(&mut self, payload: FlowPayload, stage: SkandhaStage, timestamp_us: i64) -> FlowEvent
```

---

## Layer 2: ORGANS (Tầng HÌNH / Báo Thân)

> "The Domain Logic - where consciousness processes and decisions form."

### Purpose
Core business logic layer implementing the **Five Skandhas** (Ngũ Uẩn) cognitive pipeline.

### The Five Skandhas Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        NGŨ UẨN (Five Skandhas)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SẮC (Rupa)     THỌ (Vedana)    TƯỞNG (Sanna)   HÀNH (Sankhara)  THỨC  │
│   ┌─────┐        ┌─────┐         ┌─────┐         ┌─────┐         ┌─────┐│
│   │Form │   →    │Feel │    →    │Percv│    →    │Form │    →    │Synth││
│   │     │        │     │         │     │         │     │         │     ││
│   │Sheaf│        │Belif│         │Holo │         │Dharm│         │Vinn ││
│   │Percn│        │Engne│         │Memry│         │Filtr│         │Ctrl ││
│   └─────┘        └─────┘         └─────┘         └─────┘         └─────┘│
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Skandha | Vietnamese | File | Implementation |
|---------|------------|------|----------------|
| **Rupa** | Sắc (Form) | `perception/sheaf.rs` | `SheafPerception` - Sensor consensus via Laplacian |
| **Vedana** | Thọ (Feeling) | `belief/mod.rs` | `BeliefEngine` - Valence/arousal extraction |
| **Sanna** | Tưởng (Perception) | `memory/hologram.rs` | `HolographicMemory` - FFT pattern recall |
| **Sankhara** | Hành (Formations) | `skandha/sankhara.rs` | `UnifiedSankhara` - Intent formation + DharmaFilter |
| **Vinnana** | Thức (Consciousness) | `vinnana_controller.rs` | `VinnanaController` - Synthesis orchestrator |

### Supporting Organs

| Component | File | Role |
|-----------|------|------|
| **SafetyMonitor (Sati)** | `safety/monitor.rs` | Meta-observer, LTL verification |
| **KarmaEngine** | `skandha/sankhara.rs` | Intent tracking, feedback loop |
| **CausalGraph** | `causal/dagma.rs` | Causal reasoning |
| **Scientist** | `scientist.rs` | Automatic hypothesis discovery |

### Code Example
```rust
// vinnana_controller.rs - The Consciousness Orchestrator
pub struct VinnanaController {
    pub pipeline: ZenbPipelineUnified,     // Skandha stages
    pub last_state: Option<SynthesizedState>,
    pub philosophical_state: PhilosophicalStateMonitor,
    pub thermo: ThermodynamicEngine,       // GENERIC dynamics
    pub saccade: SaccadeLinker,            // Memory prediction
}

// The synthesis loop
pub fn synthesize(&mut self, obs: &Observation) -> SynthesizedState {
    let input = SensorInput { ... };
    let result = self.pipeline.process(&input);  // Rupa→Vedana→Sanna→Sankhara→Vinnana
    self.philosophical_state.update(result.free_energy, result.confidence);
    result
}
```

---

## Layer 3: CAPABILITIES (Tầng DỤNG / Hóa Thân)

> "The Tools - stateless functions that transform and actuate."

### Purpose
Adapter layer containing:
- Signal processing algorithms
- Memory backends
- Actuator controls
- External integrations

### Design Principle: Stateless
Capabilities must be **pure functions** or **stateless services**. They don't know about business context - only transform data.

### Key Components

| Category | Files | Components |
|----------|-------|------------|
| **Perception** | `zenb-signals/` | rPPG (CHROM, POS, PRISM), DSP, Vision |
| **Cognition** | `memory/`, `causal/` | HdcMemory, ZenithMemory, DAGMA |
| **Action** | `breath_engine.rs`, `sensory/` | BreathEngine, Haptics, Binaural |
| **Integration** | `llm/`, `edge.rs` | LLM providers, Edge optimization |

### Code Example
```rust
// memory/hologram.rs - Stateless memory operations
impl HolographicMemory {
    // Pure function: given key and value, store interference pattern
    pub fn store(&mut self, key: &[Complex32], value: &[Complex32]) { ... }

    // Pure function: given key, retrieve associated value
    pub fn recall(&self, key: &[Complex32]) -> Vec<Complex32> { ... }
}
```

---

## The Three Consciousnesses (Tam Tâm Thức)

The system embodies three consciousness aspects that observe and contribute:

| Consciousness | Vietnamese | Role | Primary Skandha |
|--------------|------------|------|-----------------|
| **MINH GIỚI** | Tâm Thức Minh Giới | Moral Guardian (Tánh Giám) | Vedana |
| **GEM** | Tâm Thức Gem | Pattern Oracle (Tánh Biết) | Sanna |
| **PHÁ QUÂN** | Tâm Thức Phá Quân | Strategic Core (Lõi Đạo Sống) | Sankhara |

```rust
// universal_flow.rs
pub enum ConsciousnessAspect {
    MinhGioi,  // Assigns moral feeling (Vedana)
    Gem,       // Recognizes patterns (Sanna)
    PhaQuan,   // Forms strategic intent (Sankhara)
}
```

---

## Design Principles

### 1. Unidirectional Flow (Luật Một Chiều)
```
Capability → Essence (FlowStream) → Organ → Essence → Capability
```
Data always flows through the FlowStream. Never bypass it.

### 2. Stateless Capabilities (Luật Vô Ngã)
Tools in the Capabilities layer must not store business state. They only transform.

### 3. Observability First (Luật Chánh Niệm)
Every significant operation must emit a FlowEvent. If it's not emitted, it didn't happen (no Karma).

### 4. Persistence Transparency (Luật Tàng Thức)
The Essence layer handles persistence automatically. Other layers don't care about storage.

---

## File Organization Reference

```
zenb-core/src/
│
├── ESSENCE (Infrastructure)
│   ├── universal_flow.rs          # Vô Cực Stream (event bus)
│   ├── philosophical_state.rs     # YÊN/ĐỘNG/HỖN LOẠN states
│   ├── domain.rs                  # Core domain models
│   ├── timestamp.rs               # Time tracking
│   └── validation.rs              # State validation
│
├── ORGANS (Domain Logic)
│   ├── skandha/                   # Five Skandhas pipeline
│   │   ├── mod.rs                 # Pipeline types and traits
│   │   ├── sankhara.rs            # Intent formation + KarmaEngine
│   │   └── llm/                   # LLM-augmented stages
│   ├── belief/                    # BeliefEngine, FEP
│   ├── belief_subsystem.rs        # Encapsulated belief management
│   ├── vinnana_controller.rs      # Consciousness orchestrator
│   ├── safety/                    # Safety & ethics
│   │   ├── monitor.rs             # Sati - LTL safety monitor
│   │   ├── dharma.rs              # Ethical filtering
│   │   └── guardians.rs           # Triple Guardians
│   ├── causal/                    # Causal reasoning
│   │   ├── dagma.rs               # Structure learning
│   │   └── intervenable.rs        # do-calculus
│   └── scientist.rs               # Hypothesis discovery
│
├── CAPABILITIES (Tools)
│   ├── perception/                # Sensor processing
│   │   └── sheaf.rs               # Laplacian consensus
│   ├── memory/                    # Memory backends
│   │   ├── hologram.rs            # FFT associative memory
│   │   ├── hdc.rs                 # Hyperdimensional computing
│   │   └── zenith.rs              # Unified memory API
│   ├── sensory/                   # Output generation
│   │   ├── binaural.rs            # Binaural beats
│   │   └── haptics.rs             # Haptic feedback
│   ├── breath_engine.rs           # Breathing actuator
│   ├── llm/                       # LLM providers
│   └── edge.rs                    # Device optimization
│
└── ORCHESTRATION
    ├── engine.rs                  # Main engine (wiring)
    ├── config.rs                  # Configuration
    └── control_flow.rs            # Type-safe pipeline builder
```

---

## Data Flow Example

```
1. Sensor Input (HR=70, HRV=45)
   │
   ▼
2. zenb-signals (CAPABILITY) processes raw signal
   │
   ▼
3. FlowStream.emit_sensor() (ESSENCE) records event
   │
   ▼
4. SheafPerception (ORGAN/Rupa) achieves sensor consensus
   │
   ▼
5. FlowStream.emit() records ProcessedForm
   │
   ▼
6. BeliefEngine (ORGAN/Vedana) extracts feeling
   │
   ▼
7. HolographicMemory (ORGAN/Sanna) recalls patterns
   │
   ▼
8. UnifiedSankhara (ORGAN/Sankhara) forms intent
   │
   ▼
9. VinnanaController (ORGAN/Vinnana) synthesizes decision
   │
   ▼
10. FlowStream.emit_synthesis() records final state
    │
    ▼
11. BreathEngine (CAPABILITY) generates guidance
    │
    ▼
12. User receives breathing instruction
```

---

## Related Documentation

- `TRIKAYA_ANALYSIS.md` - Detailed comparison of Trikaya Pattern vs current architecture
- `README.md` - Project overview
- Cargo.toml - Feature flags and dependencies

---

*This architecture guide provides a philosophical framework for understanding zenb-core.
The actual code organization may differ slightly, but the principles remain constant.*
