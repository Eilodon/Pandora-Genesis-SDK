# Panic Safety Guarantees

## Policy

Library code in `pandora_*` crates MUST NOT panic in normal operation.
Panics are only acceptable for:

1. Unrecoverable bugs – use `unreachable!()` with clear explanation
2. Test code – any panics allowed in `#[cfg(test)]`
3. Example code – `.unwrap()` acceptable for clarity

## Public API Guarantees

### Guaranteed Panic-Free

These functions will never panic with any input:

```rust
pub fn bind(x: &[f64], y: &[f64]) -> Result<Vec<f64>, pandora_error::PandoraError>
pub fn bundle(vectors: &[Vec<f64>]) -> Result<Vec<f64>, pandora_error::PandoraError>
```

### May Panic (documented)

```rust
// Panics if buffer size < required; caller must ensure invariant
pub unsafe fn write_unchecked(buf: &mut [u8], offset: usize, val: u8)
```

## Audit Status (2025-10-01)

| Crate | Status | Panics | Notes |
|-------|--------|--------|-------|
| pandora_core | Clean | 0 | Assertions removed |
| pandora_cwm | Clean | 0 | VSA functions return Result |
| pandora_tools | Clean | 0 | Skills handle errors |
| pandora_orchestrator | Clean | 0 | Retry/circuit breaker safe |
| pandora_error | Clean | 0 | Error construction safe |
| pandora_mcg | Clean | 0 | Rule evaluation safe |
| pandora_sie | Clean | 0 | Strategy execution safe |
| pandora_learning_engine | Clean | 0 | WorldModel trait safe |

## Testing

```
./scripts/audit_panics.sh
cargo test --workspace
```


