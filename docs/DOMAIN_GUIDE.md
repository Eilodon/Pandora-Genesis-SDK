# Domain Implementation Guide

This guide explains how to implement custom domains for the AGOLOS adaptive control engine.

## Overview

AGOLOS uses a **trait-based abstraction** to support multiple application domains:

| Domain | Status | Description |
|--------|--------|-------------|
| `biofeedback` | ✅ Reference | Breath guidance, HRV, physiological signals |
| `industrial` | 🚧 Example | Temperature/pressure control |
| `trading` | 🚧 Example | Financial signal processing |

## Core Traits

### 1. `OscillatorConfig`

Defines the primary frequency control for your domain:

```rust
use zenb_core::core::OscillatorConfig;

#[derive(Clone, Debug, Default)]
pub struct MyConfig {
    pub target_hz: f32,
}

impl OscillatorConfig for MyConfig {
    fn target_frequency(&self) -> f32 {
        self.target_hz * 60.0  // Convert Hz to CPM
    }
    
    fn set_target_frequency(&mut self, freq: f32) {
        self.target_hz = freq / 60.0;
    }
    
    fn min_frequency(&self) -> f32 { 0.1 }
    fn max_frequency(&self) -> f32 { 100.0 }
}
```

### 2. `SignalVariable`

Defines the causal variables in your domain:

```rust
use zenb_core::core::SignalVariable;
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum MyVariable {
    Temperature,
    Pressure,
    FlowRate,
}

impl SignalVariable for MyVariable {
    fn index(&self) -> usize {
        match self {
            Self::Temperature => 0,
            Self::Pressure => 1,
            Self::FlowRate => 2,
        }
    }
    
    fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Temperature),
            1 => Some(Self::Pressure),
            2 => Some(Self::FlowRate),
            _ => None,
        }
    }
    
    fn count() -> usize { 3 }
    
    fn all() -> &'static [Self] {
        &[Self::Temperature, Self::Pressure, Self::FlowRate]
    }
}
```

### 3. `ActionKind`

Defines the interventions your domain can perform:

```rust
use zenb_core::core::ActionKind;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MyAction {
    AdjustHeater { delta: f32 },
    OpenValve { percent: f32 },
    Alert { message: String },
    DoNothing,
}

impl ActionKind for MyAction {
    fn description(&self) -> String {
        match self {
            Self::AdjustHeater { delta } => format!("Adjust heater by {:.1}°", delta),
            Self::OpenValve { percent } => format!("Open valve to {}%", percent),
            Self::Alert { message } => format!("Alert: {}", message),
            Self::DoNothing => "No action".to_string(),
        }
    }
    
    fn intrusiveness(&self) -> f32 {
        match self {
            Self::DoNothing => 0.0,
            Self::Alert { .. } => 0.2,
            Self::AdjustHeater { .. } => 0.5,
            Self::OpenValve { .. } => 0.7,
        }
    }
}
```

### 4. `Domain`

Ties everything together:

```rust
use zenb_core::core::Domain;

pub struct IndustrialDomain;

impl Domain for IndustrialDomain {
    type Config = MyConfig;
    type Variable = MyVariable;
    type Action = MyAction;
    
    fn name() -> &'static str { "industrial" }
    
    fn default_priors() -> fn(cause: usize, effect: usize) -> f32 {
        |cause, effect| {
            // Encode domain knowledge
            match (cause, effect) {
                (0, 1) => 0.8,  // Temperature → Pressure
                (2, 0) => 0.3,  // FlowRate → Temperature
                _ => 0.0,
            }
        }
    }
}
```

## Using Your Domain

```rust
use zenb_core::Engine;

// The engine currently uses BiofeedbackDomain by default
let engine = Engine::new(6.0);

// Future: Engine<IndustrialDomain> support
```

## Reference: Biofeedback Domain

See `src/domains/biofeedback/` for a complete working example:

- [`config.rs`](file:///home/ybao/B.1/AGOLOS/crates/zenb-core/src/domains/biofeedback/config.rs) - OscillatorConfig implementation
- [`variables.rs`](file:///home/ybao/B.1/AGOLOS/crates/zenb-core/src/domains/biofeedback/variables.rs) - SignalVariable implementation  
- [`actions.rs`](file:///home/ybao/B.1/AGOLOS/crates/zenb-core/src/domains/biofeedback/actions.rs) - ActionKind implementation
- [`mod.rs`](file:///home/ybao/B.1/AGOLOS/crates/zenb-core/src/domains/biofeedback/mod.rs) - Domain implementation with causal priors
