# PR4: Strict Time — Eliminate Timestamp Wraparound

**Status:** ✅ COMPLETE  
**Date:** 2026-01-03

---

## Executive Summary

This PR eliminates timestamp wraparound vulnerabilities by introducing safe time delta helpers (`dt_us`, `dt_sec`) that use saturating subtraction. All unsafe `(a - b) as u64` casts have been replaced with these helpers, ensuring the system remains stable even if clocks go backwards.

**Key Achievement:** Out-of-order timestamps cannot destabilize the control loop. Clock skew returns 0 delta instead of wrapping to huge values.

---

## Changes Made

### 1. Time Delta Helper Functions

**File:** `crates/zenb-core/src/domain.rs`

**Added:**
```rust
/// PR4: Compute time delta with saturating subtraction to prevent wraparound.
/// If clocks go backwards (now < last), returns 0 instead of wrapping to huge value.
#[inline]
pub fn dt_us(now_us: i64, last_us: i64) -> u64 {
    if now_us >= last_us {
        (now_us - last_us) as u64
    } else {
        // Clock went backwards - return 0 instead of wrapping
        0
    }
}

/// PR4: Compute time delta in seconds with saturating subtraction.
/// Convenience wrapper around dt_us for floating-point calculations.
#[inline]
pub fn dt_sec(now_us: i64, last_us: i64) -> f32 {
    (dt_us(now_us, last_us) as f32) / 1_000_000.0
}
```

**Rationale:**
- i64 timestamps can go backwards (NTP sync, device sleep, manual clock change)
- Naive `(now - last) as u64` wraps to huge positive value when `now < last`
- Huge delta causes control loop instability (massive dt_sec values)
- Saturating to 0 is safe: "no time passed" is correct interpretation

### 2. Replace Unsafe Casts in Runtime

**File:** `crates/zenb-uniffi/src/lib.rs`

**Before:**
```rust
pub fn tick(&mut self, ts_us: i64) {
    let dt_us = match self.last_ts_us {
        Some(last) => (ts_us - last) as u64,  // ❌ Can wrap!
        None => 0u64,
    };
```

**After:**
```rust
pub fn tick(&mut self, ts_us: i64) {
    // PR4: Use dt_us helper to prevent wraparound if clocks go backwards
    let dt_us = match self.last_ts_us {
        Some(last) => zenb_core::domain::dt_us(ts_us, last),  // ✅ Safe
        None => 0u64,
    };
```

### 3. Replace Unsafe Casts in Engine

**File:** `crates/zenb-core/src/engine.rs`

**Before:**
```rust
let dt_sec = match self.last_control_ts_us {
    Some(last) => (((ts_us - last).max(0)) as f32) / 1_000_000f32,  // ❌ max(0) not enough
    None => 0.0,
};
```

**After:**
```rust
// PR4: Use dt_sec helper to prevent wraparound if clocks go backwards
let dt_sec = match self.last_control_ts_us {
    Some(last) => crate::domain::dt_sec(ts_us, last),  // ✅ Safe
    None => 0.0,
};
```

**Note:** The old code used `.max(0)` which prevents negative i64 but doesn't prevent wraparound when casting to u64.

### 4. Export Helpers for Public Use

**File:** `crates/zenb-core/src/lib.rs`

**Added to exports:**
```rust
pub use domain::{
    // ... other exports ...
    dt_us, dt_sec, // PR4: Time delta helpers (prevent wraparound)
};
```

**Benefit:** External code can use these helpers for consistent time handling.

### 5. Comprehensive Tests

**File:** `crates/zenb-core/src/domain.rs`

**Added Tests:**
- `test_dt_us_normal_forward` - Normal case: time moves forward
- `test_dt_us_backwards_clock` - Clock went backwards → returns 0
- `test_dt_us_same_timestamp` - Edge case: same timestamp → returns 0
- `test_dt_sec_conversion` - Verify microseconds to seconds conversion
- `test_dt_sec_backwards_clock` - Backwards clock in dt_sec → returns 0.0

**Coverage:**
- Normal operation (forward time)
- Clock skew (backwards time)
- Edge cases (same timestamp)
- Unit conversion (microseconds → seconds)

---

## Files Touched

1. `crates/zenb-core/src/domain.rs` — dt_us/dt_sec helpers + tests
2. `crates/zenb-uniffi/src/lib.rs` — Use dt_us in tick()
3. `crates/zenb-core/src/engine.rs` — Use dt_sec in make_control()
4. `crates/zenb-core/src/lib.rs` — Export dt_us and dt_sec
5. `docs/PR4-strict-time.md` — This document

---

## Why This Fixes Wraparound

### Before (VULNERABLE):

```rust
// Runtime.tick()
let dt_us = match self.last_ts_us {
    Some(last) => (ts_us - last) as u64,  // ❌ DANGER
    None => 0u64,
};

// Example: Clock goes backwards
// ts_us = 1000000 (1 second)
// last = 2000000 (2 seconds)
// (1000000 - 2000000) = -1000000 (i64)
// -1000000 as u64 = 18446744073708551616 (wraparound!)
// dt_us = 18446744073708551616 microseconds = 584 million years!
```

**Impact:**
- Belief engine receives dt_sec = 584 million years
- EMA calculations explode (alpha ≈ 1.0)
- Control decisions become unstable
- System crashes or produces garbage output

### After (SAFE):

```rust
// Runtime.tick()
let dt_us = zenb_core::domain::dt_us(ts_us, last);

// Example: Clock goes backwards
// ts_us = 1000000 (1 second)
// last = 2000000 (2 seconds)
// dt_us(1000000, 2000000) = 0 (saturated)
// dt_us = 0 microseconds (safe!)
```

**Impact:**
- Belief engine receives dt_sec = 0.0
- EMA calculations use alpha = 0.0 (no update)
- Control decisions remain stable (use previous state)
- System continues operating normally

---

## Timestamp Type Consistency

### Current State:

**Timestamps are i64 throughout:**
- `Envelope.ts_us: i64`
- `Observation.timestamp_us: i64`
- `Runtime.tick(ts_us: i64)`
- `Engine.make_control(ts_us: i64)`

**Deltas are u64:**
- `Engine.tick(dt_us: u64)`
- `BreathEngine.tick(dt_us: u64)`

**This is correct:**
- Timestamps can be negative (relative to epoch) → i64
- Deltas are always non-negative → u64
- Conversion happens via `dt_us()` helper (safe)

### No Changes Needed:

The existing type system is sound. We only needed to fix the **conversion** from i64 timestamps to u64 deltas.

---

## Clock Skew Scenarios

### Scenario 1: NTP Sync (Common)

**What happens:**
- Device syncs with NTP server
- Clock jumps backwards by a few seconds
- Next tick: `ts_us < last_ts_us`

**Old behavior:** Wraparound → system crash  
**New behavior:** dt = 0 → system pauses briefly, then resumes

### Scenario 2: Device Sleep/Wake (Mobile)

**What happens:**
- Device goes to sleep
- Wakes up with slightly earlier timestamp (clock drift)
- Next tick: `ts_us < last_ts_us`

**Old behavior:** Wraparound → garbage data  
**New behavior:** dt = 0 → no update, wait for next tick

### Scenario 3: Manual Clock Change (Rare)

**What happens:**
- User manually sets clock backwards
- Next tick: `ts_us << last_ts_us` (large backwards jump)

**Old behavior:** Massive wraparound → system unusable  
**New behavior:** dt = 0 → system freezes updates until clock catches up

### Scenario 4: Timezone Change (Edge Case)

**What happens:**
- Device crosses timezone boundary
- If using local time (we use UTC) → clock jumps

**Old behavior:** Wraparound if backwards  
**New behavior:** dt = 0 if backwards, normal if forwards

**Note:** We use UTC timestamps everywhere, so timezone changes don't affect us.

---

## Performance Impact

**Negligible:**
- `dt_us()` is marked `#[inline]` → zero overhead in release builds
- Single branch prediction (forward time is common case)
- No allocations, no syscalls

**Benchmark (estimated):**
- Old: `(a - b) as u64` → 1 instruction
- New: `if a >= b { (a - b) as u64 } else { 0 }` → 2-3 instructions
- Difference: < 1 nanosecond per call

**Call frequency:**
- `Runtime.tick()`: ~60 Hz (every frame)
- `Engine.make_control()`: ~2 Hz (control loop)
- Total: ~62 calls/second → ~62 nanoseconds/second overhead

**Conclusion:** Performance impact is unmeasurable.

---

## Verification Commands

```bash
# Format code
cargo fmt

# Check warnings
cargo clippy -D warnings

# Run tests (including new dt_us tests)
cargo test --workspace

# Verify no unsafe casts remain
rg "\(.*-.*\) as u64" crates/ --type rust

# Verify dt_us is used
rg "dt_us\(" crates/ --type rust

# Verify dt_sec is used
rg "dt_sec\(" crates/ --type rust

# Run domain tests specifically
cargo test -p zenb-core test_dt_us
```

**Expected Results:**
- All tests pass (including 5 new dt_us tests)
- No `(a - b) as u64` patterns for time deltas
- dt_us/dt_sec used in tick() and make_control()

---

## Acceptance Criteria

- [x] dt_us and dt_sec helpers implemented
- [x] All time delta calculations use helpers
- [x] No `(a - b) as u64` remains for time deltas
- [x] Tests cover backwards clock scenarios
- [x] Helpers exported for public use
- [x] Out-of-order clocks cannot destabilize control loop
- [x] Performance impact negligible

---

## Remaining Time-Related Code

### Safe Patterns (No Changes Needed):

**breath_engine.rs:**
```rust
let cycle_us = (60_000_000f32 / bpm).round() as u64;
```
- This is BPM → microseconds conversion, not a time delta
- Safe: always positive, no wraparound risk

**async_worker.rs:**
```rust
let timestamp = chrono::Utc::now().timestamp_micros();
```
- This is timestamp generation, not delta calculation
- Safe: monotonic within session

**Safety.rs, Projectors, Store:**
- No time delta calculations found
- Only timestamp comparisons (safe)

---

## Future Enhancements

1. **Monotonic Clock Source:**
   - Use `std::time::Instant` for deltas (guaranteed monotonic)
   - Keep i64 timestamps for persistence
   - Convert at boundaries

2. **Clock Skew Detection:**
   - Log warning when `dt_us()` returns 0 due to backwards clock
   - Emit telemetry event for monitoring
   - Track frequency of clock skew events

3. **Maximum Delta Sanity Check:**
   - Add `dt_us_clamped(now, last, max_us)` variant
   - Clamp to reasonable maximum (e.g., 10 seconds)
   - Prevents issues if device sleeps for hours

4. **Timestamp Validation:**
   - Add `validate_timestamp(ts_us)` helper
   - Check timestamp is within reasonable range
   - Reject timestamps from year 1970 or 2100+

---

**End of PR4 Documentation**
