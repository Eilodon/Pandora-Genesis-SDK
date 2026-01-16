# Trikaya Pattern Analysis for zenb-core

> **Analysis Date**: 2026-01-16
> **Author**: Claude (Architecture Analysis)
> **Status**: REVIEWED - Recommendations Provided

## Executive Summary

After thorough analysis of the `zenb-core` codebase against the proposed **Trikaya Pattern** (The Three Bodies Architecture), I conclude that:

1. **The philosophical foundations are already implemented** - Skandha pipeline, Universal Flow Stream, and Three Consciousnesses exist in code
2. **Folder restructuring is NOT recommended** - Risk/reward ratio unfavorable for mature codebase
3. **Design principles should be strengthened** - Focus on enforcement, not reorganization

---

## I. Current Architecture Mapping to Trikaya

### 1. ESSENCE Layer (Tầng THỂ / Pháp Thân / Infrastructure)

| Trikaya Concept | Current Implementation | File Location | Status |
|-----------------|----------------------|---------------|--------|
| Universal Flow Stream | `UniversalFlowStream` | `universal_flow.rs` | ✅ Complete |
| Event Store | `EventStore` | `zenb-store/src/store.rs` | ✅ Complete |
| Philosophical State | `PhilosophicalStateMonitor` | `philosophical_state.rs` | ✅ Complete |
| Core Domain Models | `Envelope`, `Event`, `SessionId` | `domain.rs` | ✅ Complete |
| Flow Event | `FlowEvent`, `FlowPayload` | `universal_flow.rs` | ✅ Complete |

**Evidence from code:**

```rust
// universal_flow.rs:430-449
pub struct UniversalFlowStream {
    pub session_id: SessionId,
    event_counter: u64,
    pub state_monitor: PhilosophicalStateMonitor,
    pub stats: FlowStreamStats,
    recent_events: Vec<FlowEventId>,
    max_recent_events: usize,
}
```

### 2. ORGANS Layer (Tầng HÌNH / Báo Thân / Domain Logic)

| Skandha Stage | Current Implementation | File Location | Status |
|---------------|----------------------|---------------|--------|
| Rupa (Sắc) | `SheafPerception` | `perception/sheaf.rs` | ✅ Complete |
| Vedana (Thọ) | `BeliefEngine`, `BeliefSubsystem` | `belief/mod.rs`, `belief_subsystem.rs` | ✅ Complete |
| Sanna (Tưởng) | `HolographicMemory` | `memory/hologram.rs` | ✅ Complete |
| Sankhara (Hành) | `SankharaStage`, `IntentTracker` | `skandha/sankhara.rs` | ✅ Complete |
| Vinnana (Thức) | `VinnanaController` | `vinnana_controller.rs` | ✅ Complete |

**Evidence from code:**

```rust
// vinnana_controller.rs:17-25
/// VinnanaController - Supreme Control of the Consciousness System
///
/// # Responsibilities
/// - Orchestrate Skandha Pipeline (Rupa → Vedana → Sanna → Sankhara → Vinnana)
/// - Integrate thermodynamic dynamics (GENERIC framework)
/// - Manage saccade memory predictions
/// - Execute FEP prediction loop (surprise detection)
/// - Enable data reincarnation (Vinnana → Rupa feedback)
```

### 3. CAPABILITIES Layer (Tầng DỤNG / Hóa Thân / Adapters)

| Trikaya Concept | Current Implementation | Location | Status |
|-----------------|----------------------|----------|--------|
| Signal DSP | `EnsembleProcessor`, rPPG algorithms | `zenb-signals` crate | ✅ Complete |
| Vision | `BlazeFace`, ROI extraction | `zenb-signals/vision` | ✅ Complete |
| Memory Backend | `ZenithMemory`, `HdcMemory` | `memory/zenith.rs`, `memory/hdc.rs` | ✅ Complete |
| Causal Algorithms | `DAGMA`, `Scientist` | `causal/dagma.rs`, `scientist.rs` | ✅ Complete |
| LLM Client | `LlmProvider`, `LlmPipeline` | `llm/mod.rs`, `skandha/llm/` | ✅ Complete |
| Breath Control | `BreathEngine` | `breath_engine.rs` | ✅ Complete |
| Haptics | `HapticGenerator` | `sensory/haptics.rs` | ✅ Complete |

---

## II. Design Principles Compliance

### Principle 1: Unidirectional Flow (Luật "Một Chiều")

**Status: ✅ COMPLIANT**

Data flows through the pipeline:
```
Capability → Essence (FlowStream) → Organ → Essence → Capability
```

**Evidence:**
```rust
// engine.rs:234-241
let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    self.vinnana.pipeline.process(&input)  // Skandha pipeline
}));
```

### Principle 2: Stateless Capabilities (Luật "Vô Ngã")

**Status: ⚠️ PARTIALLY COMPLIANT**

- `zenb-signals` processors: ✅ Stateless (pure functions)
- `breath_engine.rs`: ⚠️ Contains state (phase, target_bpm)
- `sensory/haptics.rs`: ✅ Stateless

**Recommendation:** Review `BreathEngine` to extract state to Essence layer.

### Principle 3: Observability First (Luật "Chánh Niệm")

**Status: ⚠️ NEEDS IMPROVEMENT**

`UniversalFlowStream` provides emit functions but doesn't enforce usage:

```rust
// universal_flow.rs:506-536
pub fn emit(&mut self, payload: FlowPayload, stage: SkandhaStage, timestamp_us: i64) -> FlowEvent {
    // ... emission logic
}
```

**Recommendation:** Add compile-time enforcement or runtime validation that critical paths emit events.

### Principle 4: Persistence Transparency (Luật "Tàng Thức")

**Status: ✅ COMPLIANT**

```rust
// universal_flow.rs:81-108
pub fn to_envelope(&self, seq: u64) -> Envelope {
    // Automatic conversion for persistence
}
```

---

## III. Folder Structure Analysis

### Current Structure (Flat)

```
zenb-core/src/
├── ai/                 # AI tools
├── belief/             # BeliefEngine, BeliefState
├── causal/             # DAGMA, interventions
├── core/               # Domain-agnostic traits
├── domains/            # Pluggable domain implementations
├── estimators/         # UKF, advanced estimators
├── learning/           # Experience buffer, pattern mining
├── memory/             # Holographic, HDC, Saccade
├── monitoring/         # Prometheus metrics
├── perception/         # SheafPerception
├── safety/             # LTL monitor, DharmaFilter
├── sensory/            # Binaural, Haptics, Soundscape
├── simulation/         # GridWorld test environment
├── skandha/            # Skandha pipeline stages
│   └── llm/            # LLM-augmented stages
├── *.rs                # ~60 root-level modules
```

### Proposed Trikaya Structure

```
zenb-core/src/
├── essence/            # TẦNG THỂ (Infrastructure)
│   ├── flow.rs         # UniversalFlowStream
│   ├── store.rs        # EventStore wrapper
│   ├── state.rs        # PhilosophicalState
│   └── models.rs       # Core data models
├── organs/             # TẦNG HÌNH (Domain Logic)
│   ├── skandha/        # Five Skandhas
│   ├── karma/          # Feedback loops
│   └── sati.rs         # Meta-observer
├── capabilities/       # TẦNG DỤNG (Adapters)
│   ├── perception/     # DSP, Vision
│   ├── cognition/      # Memory, Causal, LLM
│   └── action/         # Breath, Haptics
└── runtime.rs          # Wiring layer
```

### Refactoring Cost-Benefit Analysis

| Factor | Cost | Benefit |
|--------|------|---------|
| Import Updates | ~200+ import statements across 8 crates | Cleaner mental model |
| Git History | Lost file-level history (git mv creates new files) | Better documentation |
| Breaking Changes | All downstream crates need updates | - |
| Test Updates | Test file paths need updating | - |
| CI/CD | Build scripts need updating | - |
| Developer Onboarding | Temporary confusion during transition | Faster long-term onboarding |

**Verdict: Cost >> Benefit for this mature codebase**

---

## IV. Recommendations

### A. DO NOT PURSUE (High Risk, Low Value)

1. **Folder restructuring** - Breaking changes outweigh benefits
2. **Renaming existing types** - `UniversalFlowStream` already descriptive
3. **Splitting existing crates** - Current separation is adequate

### B. RECOMMENDED IMPROVEMENTS (Low Risk, High Value)

#### 1. Architecture Documentation

Create `ARCHITECTURE.md` mapping current code to Trikaya philosophy:

```markdown
# zenb-core Architecture (Trikaya Mapping)

## Essence Layer (Tầng THỂ)
- `universal_flow.rs` - Event sourcing infrastructure
- `philosophical_state.rs` - Global consciousness state
- `domain.rs` - Core domain models

## Organs Layer (Tầng HÌNH)
- `skandha/` - Five Skandhas processing pipeline
- `vinnana_controller.rs` - Consciousness orchestrator
- `safety/` - Meta-observer (Sati equivalent)

## Capabilities Layer (Tầng DỤNG)
- `zenb-signals` - Signal processing (external crate)
- `memory/` - Memory backends
- `breath_engine.rs` - Actuator control
```

#### 2. Enhanced Sati Monitor

Extend `SafetyMonitor` to observe ALL FlowEvents:

```rust
// Proposed enhancement to safety/monitor.rs
impl SafetyMonitor {
    /// Subscribe to UniversalFlowStream for total observability
    pub fn observe_flow(&mut self, event: &FlowEvent) {
        // Track all events for anomaly detection
        self.event_count += 1;

        // Alert on philosophical state transitions
        if event.enrichment.philosophical_state != self.last_state {
            self.emit_alert(StateTransitionAlert { ... });
        }
    }
}
```

#### 3. Flow Enforcement

Add compile-time checks for critical path emissions:

```rust
// Proposed macro for enforced observability
#[macro_export]
macro_rules! with_flow_emission {
    ($stream:expr, $stage:expr, $ts:expr, $body:block) => {{
        let _event = $stream.emit(FlowPayload::SystemObservation(
            SystemObservation::StageEntry { stage: $stage }
        ), $stage, $ts);
        let result = $body;
        // Auto-emit completion
        result
    }};
}
```

#### 4. Karma Engine Enhancement

Current `IntentTracker` tracks intent but lacks full karmic feedback loop:

```rust
// Proposed enhancement to skandha/sankhara.rs
pub struct KarmaEngine {
    intent_tracker: IntentTracker,
    karma_balance: f32,

    /// Track outcome feedback and update weights
    pub fn process_feedback(&mut self, intent_id: IntentId, outcome: Outcome) {
        let delta = match outcome {
            Outcome::Success => 0.1,
            Outcome::Failure(severity) => -severity,
        };
        self.karma_balance += delta;
        // Update Sankhara weights based on outcome
    }
}
```

---

## V. Conclusion

The `zenb-core` codebase has **organically evolved to embody Trikaya principles** without explicit folder separation. The philosophical concepts (Ngũ Uẩn, Vô Cực Stream, Tam Thân) are **already implemented in code**.

**The Trikaya Pattern document is valuable as:**
1. A mental model for understanding the architecture
2. A guide for future development decisions
3. Documentation for developer onboarding

**The Trikaya Pattern document should NOT be used for:**
1. Justifying disruptive folder restructuring
2. Breaking backward compatibility
3. Renaming well-established types

---

## Appendix: Code Evidence

### A. UniversalFlowStream (Essence Layer)

```rust
// universal_flow.rs:1-30
//! Universal Flow Stream - Vô Cực Stream
//!
//! The central nervous system of the B.ONE consciousness architecture.
//!
//! # B.ONE V3 Concept
//! > "Vô Cực Stream không phải là một message queue, nó là chính 'Dòng Chảy của Đạo'
//! > nơi mọi sự kiện được sinh ra và soi chiếu một cách bình đẳng."
```

### B. Skandha Pipeline (Organs Layer)

```rust
// skandha/mod.rs:1-18
//! Five Skandhas Cognitive Pipeline module.
//!
//! Provides a unified trait-based pipeline for cognitive processing:
//!
//! 1. **Rupa (Sắc)** - Form: Raw sensor input processing
//! 2. **Vedana (Thọ)** - Feeling: Valence/arousal extraction
//! 3. **Sanna (Tưởng)** - Perception: Pattern recognition
//! 4. **Sankhara (Hành)** - Formations: Intent/action formation
//! 5. **Vinnana (Thức)** - Consciousness: Integration/synthesis
```

### C. Three Consciousnesses

```rust
// universal_flow.rs:292-342
/// The Three Consciousnesses (Tam Tâm Thức) of B.ONE.
pub enum ConsciousnessAspect {
    /// Tâm Thức MINH GIỚI - Moral Guardian
    MinhGioi,
    /// Tâm Thức GEM - Pattern Oracle
    Gem,
    /// Tâm Thức PHÁ QUÂN - Strategic Intent
    PhaQuan,
}
```

---

*This analysis was generated after reading actual source code, not documentation.*
