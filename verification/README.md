# AGOLOS Formal Verification

This directory contains scaffolding for formal verification using [Aeneas](https://github.com/AeneasVerif/aeneas) and Lean 4.

## Overview

Aeneas translates Rust code to Lean 4 for formal verification. The goal is to prove critical safety properties about AGOLOS components.

## Integration Path

1. **Annotate**: Add contract annotations to Rust code (`requires!`, `ensures!`, `invariant!`)
2. **Translate**: Run Aeneas to generate Lean 4 code from annotated Rust
3. **Prove**: Write Lean 4 tactics to prove the contracts
4. **CI**: Integrate proof checking into CI pipeline

## Target Components

### High Priority
- `DharmaFilter::sanction` - Prove ethical veto correctness
- `HolographicMemory::entangle` - Prove energy bounds
- `HamiltonianGuard::check_metar` - Prove safety veto correctness

### Medium Priority
- `ThermodynamicEngine::step` - Prove degeneracy conditions
- `SheafPerception::diffuse` - Prove convergence

## Setup

```bash
# Install Aeneas
opam install aeneas

# Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Generate Lean from Rust (future)
aeneas -opaque-as-axiom crates/zenb-core/src/safety/dharma.rs
```

## Stub Proofs

See:
- `dharma_safety.lean` - Proof stubs for DharmaFilter
- `memory_bounds.lean` - Proof stubs for HolographicMemory energy bounds
