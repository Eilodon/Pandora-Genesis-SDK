# PR1 Changelog: Unify Truth

## Summary
Eliminated split-brain BeliefState definitions and established canonical type hierarchy.

## Changes

### Type Refactoring
- **RENAMED:** `domain::BeliefState` → `domain::CausalBeliefState`
  - 3-factor representation (bio/cognitive/social)
  - Used ONLY by causal reasoning layer
  - Clear documentation added explaining non-canonical status

- **CANONICAL:** `belief::BeliefState` (unchanged)
  - 5-mode collapsed representation (Calm, Stress, Focus, Sleepy, Energize)
  - Used by: Engine, BeliefEngine, safety guards, policy logic
  - This is the ONE TRUE runtime belief state

### Module Updates

#### `crates/zenb-core/src/domain.rs`
- Renamed `BeliefState` → `CausalBeliefState`
- Added PR1 documentation comment explaining canonical decision
- All methods updated (most_likely_bio_state, etc.)

#### `crates/zenb-core/src/causal.rs`
- Import: `use crate::domain::CausalBeliefState`
- `predict_outcome()`: now accepts `&crate::belief::BeliefState`
- `extract_state_values()`: now accepts `&crate::belief::BeliefState`
- Updated mapping: 5-mode → causal variable space
  - Stress mode → HR/HRV
  - Focus mode → notification pressure (inverse)
- `ObservationSnapshot.belief_state`: now `Option<CausalBeliefState>`
- Test updated to use canonical type

#### `crates/zenb-core/src/engine.rs`
- `tick()`: Added explicit conversion from canonical to CausalBeliefState
- Mapping logic documented:
  ```
  bio_state: [Calm, Stress, Sleepy]
  cognitive_state: [Focus, 1-Focus, 0]
  social_state: [0.33, 0.33, 0.33] (uniform prior)
  ```

#### `crates/zenb-core/src/lib.rs`
- **REMOVED:** All 15 wildcard re-exports (`pub use module::*`)
- **ADDED:** Curated explicit exports organized by subsystem
- **DOCUMENTED:** Canonical BeliefState in module-level comment
- Export groups:
  - Domain types (event sourcing)
  - Belief engine (CANONICAL BeliefState here)
  - Safety swarm
  - Causal reasoning
  - Controller, estimator, resonance, etc.

## Impact

### Breaking Changes
- `domain::BeliefState` renamed to `domain::CausalBeliefState`
  - External code using this type must update imports
  - Internal code already updated

### Non-Breaking Changes
- `belief::BeliefState` unchanged (canonical type)
- All public API exports now explicit (no functional change)
- Improved type safety and clarity

## Verification Checklist

- [x] No duplicate type names in exports
- [x] One canonical BeliefState (belief::BeliefState)
- [x] Intentional export surface (no wildcards)
- [x] All call sites updated
- [x] Documentation added
- [ ] `cargo fmt` (pending user environment)
- [ ] `cargo clippy -D warnings` (pending user environment)
- [ ] `cargo test --workspace` (pending user environment)

## Files Modified
1. `crates/zenb-core/src/domain.rs`
2. `crates/zenb-core/src/causal.rs`
3. `crates/zenb-core/src/engine.rs`
4. `crates/zenb-core/src/lib.rs`

## Files Created
1. `docs/PR1-unify-truth.md`
2. `CHANGELOG_PR1.md`
