# Causal Reasoning Layer - Quick Start Guide

## **Core Concepts**

> **Note on Terminology:** This guide uses **"Breath Guidance"** as the canonical reference implementation to demonstrate the engine's capabilities. In your application, this maps to **"Adaptive Oscillator Control"** or **"Rhythm Guidance"**.
> 
> - **RespiratoryRate** → `PrimaryFrequency` (The signal you are measuring)
> - **BreathGuidance** → `OscillatorControl` (The intervention you are applying)
> - **Biofeedback** → `ClosedLoopControl` (The system topology)

## What Was Built

A complete **Causal Reasoning Layer** for ZenB that enables the system to understand causal relationships between variables (e.g., "Notifications cause stress").

## Key Files

- **`crates/zenb-core/src/causal.rs`** - Complete implementation (600+ lines)
- **`crates/zenb-core/src/engine.rs`** - Integration into Engine
- **`crates/zenb-core/src/lib.rs`** - Module exports
- **`docs/CAUSAL_LAYER.md`** - Full documentation

## Quick API Reference

### 1. Causal Variables
```rust
use zenb_core::Variable;

// 9 causal nodes in the system
Variable::NotificationPressure
Variable::HeartRate
Variable::HeartRateVariability
Variable::Location
Variable::TimeOfDay
Variable::UserAction
Variable::InteractionIntensity
Variable::RespiratoryRate
Variable::NoiseLevel
```

### 2. Causal Graph
```rust
use zenb_core::CausalGraph;

// Create with domain knowledge priors
let mut graph = CausalGraph::with_priors();

// Query relationships
let effect = graph.get_effect(Variable::NotificationPressure, Variable::HeartRate);

// Modify relationships (for learning)
graph.set_effect(Variable::NoiseLevel, Variable::HeartRate, 0.4);

// Verify DAG property
assert!(graph.is_acyclic());
```

### 3. Prediction
```rust
use zenb_core::{ActionPolicy, ActionType};

let action = ActionPolicy {
    action_type: ActionType::BreathGuidance,
    intensity: 0.8,
};

let prediction = graph.predict_outcome(&belief_state, &action);
println!("Predicted HR: {:.1}", prediction.predicted_hr);
```

### 4. Observation Buffer
```rust
// Engine automatically buffers observations
engine.ingest_observation(observation);
engine.tick(dt_us);  // Pushes to causal_buffer

// Access buffer
println!("Buffered: {}/1000", engine.causal_buffer.len());

// Extract for learning
let data = engine.causal_buffer.to_data_matrix();
```

## Integration with Existing Code

### Before (Linear Reaction)
```rust
let mut engine = Engine::new(6.0);
let est = engine.ingest_sensor(&[hr, hrv, rr], ts);
let (decision, persist, policy, deny) = engine.make_control(&est, ts, None);
```

### After (With Causal Understanding)
```rust
let mut engine = Engine::new(6.0);

// Ingest full observation (not just sensor features)
let obs = Observation { /* ... */ };
engine.ingest_observation(obs);

// Existing sensor ingestion still works
let est = engine.ingest_sensor(&[hr, hrv, rr], ts);

// Tick automatically buffers for causal learning
engine.tick(dt_us);

// Access causal graph for predictions
let prediction = engine.causal_graph.predict_outcome(&belief_state, &action);

// Control decision unchanged
let (decision, persist, policy, deny) = engine.make_control(&est, ts, None);
```

## What's Ready for NOTEARS

1. **Data Structure**: Adjacency matrix ready for optimization
2. **Data Buffer**: 1000 observations stored for batch learning
3. **Serialization**: Full serde support for persistence
4. **DAG Verification**: Acyclicity checking built-in

## Next Implementation Phase

When ready to implement NOTEARS:

1. Add to `Cargo.toml`:
```toml
ndarray = "0.15"
ndarray-linalg = "0.16"
```

2. Add to `causal.rs`:
```rust
pub fn learn_structure(&mut self, buffer: &CausalBuffer, lambda: f32) -> Result<(), String> {
    let data = buffer.to_data_matrix();
    // NOTEARS algorithm implementation
    // Updates self.weights matrix
}
```

## Testing

Run tests:
```bash
cargo test --package zenb-core --lib causal
```

All tests included:
- Variable indexing
- Graph creation and manipulation
- DAG property verification
- Buffer operations (circular buffer)
- Prediction functionality

## Design Highlights

✅ **Lightweight**: No heavy math dependencies yet  
✅ **Efficient**: O(1) lookups, O(1) buffer push  
✅ **Serializable**: Full EventStore integration  
✅ **Tested**: Comprehensive test suite  
✅ **Integrated**: Seamlessly added to Engine  

## Example: End-to-End Flow

```rust
use zenb_core::*;

// Initialize engine with causal reasoning
let mut engine = Engine::new(6.0);

// Simulate observation stream
for i in 0..100 {
    let obs = Observation {
        timestamp_us: i * 1_000_000,
        bio_metrics: Some(BioMetrics {
            hr_bpm: Some(70.0 + (i as f32) * 0.1),
            hrv_rmssd: Some(50.0),
            respiratory_rate: Some(6.0),
        }),
        digital_context: Some(DigitalContext {
            notification_pressure: Some(0.5),
            interaction_intensity: Some(0.3),
            active_app_category: None,
        }),
        environmental_context: None,
    };
    
    engine.ingest_observation(obs);
    engine.tick(1_000_000);
}

// After 100 observations
println!("Buffered: {}", engine.causal_buffer.len());

// Query causal relationships
let notif_to_hr = engine.causal_graph.get_effect(
    Variable::NotificationPressure,
    Variable::HeartRate
);
println!("Notification → HR effect: {:.2}", notif_to_hr);

// Predict intervention outcome
let action = ActionPolicy {
    action_type: ActionType::BreathGuidance,
    intensity: 1.0,
};
let prediction = engine.causal_graph.predict_outcome(
    &engine.belief_state,
    &action
);
println!("Predicted HR after breath guidance: {:.1}", prediction.predicted_hr);
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Engine (engine.rs)                    │
├─────────────────────────────────────────────────────────┤
│  • estimator: Estimator                                 │
│  • belief_state: BeliefState                            │
│  • causal_graph: CausalGraph          ← NEW             │
│  • causal_buffer: CausalBuffer        ← NEW             │
│  • last_observation: Option<Obs>      ← NEW             │
└─────────────────────────────────────────────────────────┘
                           │
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Causal Module (causal.rs)                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Variable Enum (9 causal nodes)                         │
│  ├─ NotificationPressure                                │
│  ├─ HeartRate                                           │
│  ├─ HeartRateVariability                                │
│  └─ ... (6 more)                                        │
│                                                          │
│  CausalGraph (DAG with adjacency matrix)                │
│  ├─ weights: [[f32; 9]; 9]                              │
│  ├─ get_effect(cause, target) -> f32                    │
│  ├─ set_effect(cause, target, weight)                   │
│  ├─ predict_outcome(...) -> PredictedState              │
│  └─ is_acyclic() -> bool                                │
│                                                          │
│  CausalBuffer (sliding window)                          │
│  ├─ capacity: 1000                                      │
│  ├─ observations: Vec<ObservationSnapshot>              │
│  ├─ push(snapshot)                                      │
│  └─ to_data_matrix() -> Vec<Vec<f32>>                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Status: ✅ COMPLETE

All three phases implemented:
- ✅ Phase 1: Data structures (Variable, CausalGraph, CausalBuffer)
- ✅ Phase 2: Prediction and graph operations
- ✅ Phase 3: Engine integration and automatic buffering

Ready for NOTEARS algorithm implementation.
