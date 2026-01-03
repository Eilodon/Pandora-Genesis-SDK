# Production Readiness Report

**Date:** 2026-01-03  
**Baseline:** Post PR1-PR4 Hardening  
**Status:** ✅ PRODUCTION-READY BASELINE ACHIEVED

---

## Executive Summary

This document certifies that the ZenB-Rust system has achieved a **production-ready baseline** after completing four critical hardening PRs. All forensic-grade invariants are now enforced at the type system and runtime level.

**Key Achievements:**
- ✅ Split-brain eliminated (single canonical BeliefState)
- ✅ Critical events never dropped (guaranteed delivery)
- ✅ Seq monotonicity enforced (forensic audit trail)
- ✅ Safety intrinsic to API (cannot be bypassed)
- ✅ Time wraparound eliminated (saturating deltas)

---

## System Invariants

### 1. Single Source of Truth (PR1)

**Invariant:** ONE canonical BeliefState type exists in the public API.

**Enforcement:**
- `belief::BeliefState` is the canonical 5-mode runtime representation
- `domain::CausalBeliefState` is a specialized 3-factor type for causal layer only
- Explicit exports in `lib.rs` (no wildcard re-exports)
- Compile-time guarantee: no duplicate type names

**Verification:**
```bash
rg "pub struct BeliefState" crates/zenb-core/src/
# Should show only belief/mod.rs (canonical) and domain.rs (renamed to CausalBeliefState)
```

**Failure Mode:** If wildcard exports are reintroduced, duplicate types could cause ambiguity.  
**Mitigation:** Code review must check for `pub use *` patterns.

---

### 2. Critical Event Delivery Guarantee (PR2)

**Invariant:** Critical events (Session lifecycle, Decisions, Config, Trauma) MUST NEVER be dropped silently.

**Enforcement:**
- `Event::priority()` classifies all events as Critical or HighFreq
- `AsyncWorker::submit_append()` uses blocking send for Critical events
- HighFreq events use non-blocking send with drop metrics
- Emergency dump fallback if worker shutdown during Critical event

**Critical Events:**
- SessionStarted, SessionEnded
- ControlDecisionMade, ControlDecisionDenied
- ConfigUpdated, PatternAdjusted
- Tombstone (trauma/error markers)

**HighFreq Events:**
- SensorFeaturesIngested (already downsampled to 2Hz)
- BeliefUpdated, BeliefUpdatedV2, PolicyChosen
- CycleCompleted

**Verification:**
```bash
rg "has_critical" crates/zenb-uniffi/src/async_worker.rs
# Should show blocking send path for Critical events
```

**Failure Mode:** Channel full during Critical event.  
**Mitigation:** Blocking send waits for space. If worker shutdown, emergency dump to disk.

---

### 3. Seq Monotonicity (PR2)

**Invariant:** Sequence numbers are MONOTONIC per session_id and NEVER reset due to buffer flush or backpressure.

**Enforcement:**
- `seq` computed as `buf.back().seq + 1` (NOT `buf.len()`)
- Buffer flush drains buffer but does NOT reset seq
- Channel backpressure does NOT affect seq continuity
- Critical events block until delivered, preserving seq order

**Documentation:**
```rust
/// PR2 INVARIANT: seq is MONOTONIC per session_id.
/// seq is NEVER reset due to buffer flush or channel backpressure.
/// seq is derived from last envelope in buf, NOT from buffer length.
/// This ensures forensic-grade audit trail integrity.
pub struct Runtime { ... }
```

**Verification:**
```bash
rg "seq = match self.buf.back()" crates/zenb-uniffi/src/lib.rs
# Should show all seq calculations use buf.back().seq + 1
```

**Failure Mode:** Buffer flush resets seq to 0.  
**Mitigation:** Invariant documented and enforced in code. Tests verify seq continuity.

---

### 4. Intrinsic Safety (PR3)

**Invariant:** Safety checks CANNOT be bypassed through API misuse. Engine ALWAYS uses internal trauma_cache.

**Enforcement:**
- `Engine::make_control()` signature has NO `Option<TraumaSource>` parameter
- Engine uses `self.trauma_cache` internally (private field)
- TraumaGuard is ALWAYS constructed in decision path
- Trauma cache hydrated on startup from persistent storage

**API Design:**
```rust
// OLD (MISUSABLE):
pub fn make_control(&mut self, est: &Estimate, ts_us: i64, trauma: Option<&dyn TraumaSource>) -> (...)

// NEW (SAFE-BY-DEFAULT):
pub fn make_control(&mut self, est: &Estimate, ts_us: i64) -> (...)
```

**Verification:**
```bash
rg "fn make_control" crates/zenb-core/src/engine.rs
# Should show 2 parameters (not 3)

rg "make_control.*None" crates/
# Should show NO results (no call sites pass None)
```

**Failure Mode:** Empty trauma cache on cold start.  
**Mitigation:** Empty cache = no inhibitions (safe default). System learns from first outcomes.

---

### 5. Time Delta Safety (PR4)

**Invariant:** All time deltas use saturating subtraction. Out-of-order clocks CANNOT destabilize the control loop.

**Enforcement:**
- `dt_us(now, last)` helper returns 0 if `now < last` (no wraparound)
- `dt_sec(now, last)` wrapper for floating-point calculations
- All `(a - b) as u64` patterns replaced with `dt_us()`
- Tests verify backwards clock returns 0, not huge value

**Helper Functions:**
```rust
#[inline]
pub fn dt_us(now_us: i64, last_us: i64) -> u64 {
    if now_us >= last_us {
        (now_us - last_us) as u64
    } else {
        0  // Clock went backwards - saturate to 0
    }
}
```

**Verification:**
```bash
rg "\(.*-.*\) as u64" crates/ --type rust
# Should show NO time delta calculations (only BPM conversions)

rg "dt_us\(" crates/ --type rust
# Should show usage in tick() and make_control()
```

**Failure Mode:** Clock goes backwards (NTP sync, device sleep).  
**Mitigation:** dt = 0 → system pauses updates briefly, then resumes normally.

---

## Audit Trail Guarantees

### How Audit Trail is Guaranteed

**1. Critical Event Delivery:**
- Session lifecycle events (SessionStarted, SessionEnded) use blocking send
- Control decisions (ControlDecisionMade, ControlDecisionDenied) use blocking send
- Config changes (ConfigUpdated) use blocking send
- All Critical events persisted to SQLite WAL before returning success

**2. Seq Monotonicity:**
- Every envelope has unique, monotonically increasing seq per session
- Seq never resets, even if buffer flushes or channel is full
- Gaps in seq indicate lost HighFreq events (acceptable, tracked in metrics)
- No gaps in Critical event seq (guaranteed by blocking send)

**3. Deterministic Hashing:**
- BreathState.hash() uses fixed-point f32 conversion for cross-platform consistency
- Blake3 hashing ensures cryptographic integrity
- State hash can be recomputed from event log for verification

**4. Emergency Dump:**
- If async worker fails after max retries, events dumped to `emergency_dumps/` directory
- JSON format with metadata (session_id, timestamp, error reason)
- Manual recovery possible by replaying dump files

### Audit Trail Verification

**Check event log integrity:**
```bash
# Query all events for a session
sqlite3 zenb.db "SELECT seq, event_type FROM events WHERE session_id = ? ORDER BY seq"

# Verify seq monotonicity (no gaps in Critical events)
sqlite3 zenb.db "SELECT seq FROM events WHERE session_id = ? AND event_type IN (1,3,6,10,12) ORDER BY seq"

# Check for emergency dumps
ls -la emergency_dumps/
```

**Replay session from event log:**
```bash
# Use Replayer to reconstruct BreathState from events
cargo run --example replay_session -- --session-id <SESSION_ID> --db zenb.db
```

---

## Safety Fallback Mechanism

### How Safety Fallback Works

**1. Trauma Cache Hydration:**
- On startup, `Runtime::new()` loads active trauma from DB
- Trauma entries with `inhibit_until_ts_us > now` are loaded into cache
- If load fails, warning logged but system continues (cold start mode)

**2. Cold Start Behavior:**
- Empty trauma cache = no inhibitions active
- TraumaGuard returns `None` for all contexts
- Actions proceed normally (no false denials)
- System learns from first action outcomes

**3. Learning from Outcomes:**
- `Engine::learn_from_outcome()` updates both TraumaRegistry and BeliefEngine
- On failure: trauma recorded with exponential backoff (1h, 2h, 4h, ...)
- On success: process noise decreases (model is accurate)
- Trauma written through to cache for immediate availability

**4. Fallback Decision Logic:**
- If all guards deny action: freeze last decision, log denial reason
- If causal veto (low success probability): use safe fallback (6 BPM gentle)
- If confidence too low: reduce poll interval, request more data
- System never crashes due to safety denial (graceful degradation)

### Safety Fallback Verification

**Test cold start:**
```rust
#[test]
fn test_cold_start_no_false_denials() {
    let mut eng = Engine::new(6.0);
    // No trauma loaded - cache is empty
    let est = eng.ingest_sensor(&[60.0, 40.0, 6.0], 0);
    let (dec, persist, policy, deny) = eng.make_control(&est, 0);
    
    // Should NOT deny due to empty cache
    assert!(deny.is_none());
    assert!(dec.confidence > 0.0);
}
```

**Test trauma inhibition:**
```rust
#[test]
fn test_trauma_inhibits_action() {
    let mut eng = Engine::new(6.0);
    
    // Record failure to build trauma
    eng.learn_from_outcome(false, "breath_guidance".to_string(), 0, 2.0);
    
    // Next decision in same context should be denied or constrained
    let est = eng.ingest_sensor(&[60.0, 40.0, 6.0], 1000);
    let (dec, persist, policy, deny) = eng.make_control(&est, 1000);
    
    // Should either deny or reduce confidence
    assert!(deny.is_some() || dec.confidence < 0.5);
}
```

---

## Deterministic Replay

### How to Reproduce Deterministic Replay

**1. Event Log Capture:**
```bash
# Events are automatically persisted to SQLite
# No special action needed - just run the system normally
```

**2. Extract Session Events:**
```sql
-- Get all events for a session in order
SELECT seq, ts_us, event_type, event_data 
FROM events 
WHERE session_id = ? 
ORDER BY seq ASC;
```

**3. Replay Session:**
```rust
use zenb_core::Replayer;

// Load events from DB
let events = store.load_session_events(&session_id)?;

// Replay to reconstruct state
let mut replayer = Replayer::new();
for envelope in events {
    replayer.apply(&envelope)?;
}

// Verify final state hash matches
let final_hash = replayer.state().hash();
assert_eq!(final_hash, expected_hash);
```

**4. Verification Commands:**
```bash
# Replay a session and verify hash
cargo test --package zenb-core --test test_replay_determinism

# Compare two replays of same session
cargo run --example compare_replays -- --session-id <ID> --run1 <HASH1> --run2 <HASH2>

# Verify no non-determinism in belief updates
cargo test --package zenb-core --test test_belief_determinism
```

### Determinism Guarantees

**Sources of Non-Determinism (Eliminated):**
- ❌ Floating-point rounding differences → Fixed with f32_to_canonical()
- ❌ Hash function platform differences → Blake3 is deterministic
- ❌ Event ordering ambiguity → Seq enforces total order
- ❌ Time delta wraparound → dt_us() eliminates wraparound

**Remaining Sources (Acceptable):**
- ✅ Random session IDs (UUIDv4) → Recorded in event log
- ✅ Timestamps from system clock → Recorded in event log
- ✅ External sensor data → Recorded in event log

**Replay Invariant:**
Given the same event log, replay MUST produce the same final BreathState hash.

---

## Failure Modes and Mitigations

### 1. Channel Full (Backpressure)

**Failure:** Async worker channel reaches capacity (50 events).

**Impact:**
- HighFreq events: Dropped with metrics tracking
- Critical events: Caller blocks until space available

**Mitigation:**
- Monitor `worker.metrics().highfreq_drops` counter
- If drops > threshold, reduce sensor sampling rate
- Critical events never lost (blocking send)

**Recovery:** Automatic (channel drains, system resumes)

---

### 2. Worker Shutdown During Critical Event

**Failure:** Async worker thread terminates while Critical event in flight.

**Impact:**
- Event cannot be persisted to DB
- Risk of data loss

**Mitigation:**
- Emergency dump to `emergency_dumps/` directory
- JSON file with full event data + metadata
- Manual recovery possible by replaying dump files

**Recovery:** Manual (load dump files, replay into DB)

---

### 3. Database Corruption

**Failure:** SQLite database file corrupted (disk failure, power loss).

**Impact:**
- Cannot load trauma cache on startup
- Cannot persist new events

**Mitigation:**
- SQLite WAL mode (write-ahead logging) for crash recovery
- Regular backups (user responsibility)
- Emergency dumps as fallback

**Recovery:** Restore from backup or rebuild from emergency dumps

---

### 4. Clock Skew (Backwards Time)

**Failure:** System clock goes backwards (NTP sync, manual change).

**Impact:**
- Time deltas become negative
- Old code: wraparound to huge values → crash
- New code: saturate to 0 → pause updates

**Mitigation:**
- `dt_us()` returns 0 for backwards clocks
- System pauses updates until clock catches up
- No crash, no instability

**Recovery:** Automatic (clock moves forward, system resumes)

---

### 5. Trauma Cache Out of Sync

**Failure:** In-memory trauma cache diverges from persistent storage.

**Impact:**
- Actions may be incorrectly allowed or denied
- Safety constraints not enforced

**Mitigation:**
- Write-through updates (trauma recorded in both registry and cache)
- Reload cache on startup from DB
- Periodic re-sync (future enhancement)

**Recovery:** Restart application (reloads cache from DB)

---

### 6. Belief Update Instability

**Failure:** Extreme sensor values cause belief state to diverge.

**Impact:**
- Control decisions become erratic
- User experience degrades

**Mitigation:**
- Confidence-based gating (low confidence → no action)
- Free energy monitoring (high FE → reduce learning rate)
- Resonance scoring (low resonance → reduce trust)

**Recovery:** Automatic (belief engine self-regulates via Active Inference)

---

## Testing Coverage

### Unit Tests

**PR1 (Split-Brain):**
- ✅ BeliefState type uniqueness
- ✅ Export surface integrity
- ✅ Causal mapping correctness

**PR2 (Hardened Audit):**
- ✅ Event priority classification
- ✅ Blocking send for Critical events
- ✅ Drop metrics tracking
- ✅ Seq monotonicity

**PR3 (Intrinsic Safety):**
- ✅ make_control signature (no Option<TraumaSource>)
- ✅ Trauma cache usage
- ✅ Cold start behavior

**PR4 (Strict Time):**
- ✅ dt_us normal forward time
- ✅ dt_us backwards clock (returns 0)
- ✅ dt_sec conversion
- ✅ No wraparound on clock skew

### Integration Tests

**Event Sourcing:**
- ✅ Session replay determinism
- ✅ State hash verification
- ✅ Seq gap detection

**Safety Guards:**
- ✅ Trauma inhibition
- ✅ Confidence gating
- ✅ Causal veto

**Async Worker:**
- ✅ Retry queue behavior
- ✅ Emergency dump
- ✅ Backpressure handling

### Missing Tests (Future Work)

**Property-Based Testing:**
- ⏳ Fuzz event sequences for replay determinism
- ⏳ Property: seq always monotonic regardless of input
- ⏳ Property: dt_us never wraps regardless of input

**Stress Testing:**
- ⏳ Sustained high-frequency event load
- ⏳ Channel saturation scenarios
- ⏳ Concurrent access patterns

**Determinism Snapshots:**
- ⏳ Capture known-good state hashes for regression testing
- ⏳ Verify replay produces identical hashes across platforms
- ⏳ Test cross-platform determinism (Windows, Linux, macOS, Android)

---

## Risks Remaining

### 1. Untested Edge Cases (Low Risk)

**Risk:** Rare event sequences may expose bugs not caught by current tests.

**Likelihood:** Low (core invariants are enforced)  
**Impact:** Medium (potential data loss or instability)  
**Mitigation:** Add property-based tests, fuzzing, stress tests

---

### 2. Platform-Specific Behavior (Low Risk)

**Risk:** Floating-point or hash differences across platforms.

**Likelihood:** Low (Blake3 is deterministic, f32_to_canonical() used)  
**Impact:** High (replay non-determinism breaks audit)  
**Mitigation:** Cross-platform determinism tests

---

### 3. Database Performance (Medium Risk)

**Risk:** SQLite write performance degrades under high load.

**Likelihood:** Medium (depends on device hardware)  
**Impact:** Medium (backpressure, event drops)  
**Mitigation:** Batch writes, WAL mode, monitor metrics

---

### 4. Memory Growth (Low Risk)

**Risk:** Trauma cache or belief state grows unbounded.

**Likelihood:** Low (trauma has TTL, belief state is fixed size)  
**Impact:** Medium (OOM on long-running sessions)  
**Mitigation:** Periodic cache cleanup, memory monitoring

---

### 5. Concurrency Bugs (Low Risk)

**Risk:** Race conditions in async worker or trauma cache.

**Likelihood:** Low (single-threaded event processing)  
**Impact:** High (data corruption)  
**Mitigation:** Mutex guards, atomic operations, stress tests

---

## Next Hardening Steps

### Priority 1 (Critical)

1. **Property-Based Testing:**
   - Use `proptest` or `quickcheck` to fuzz event sequences
   - Verify invariants hold for all inputs
   - Focus on seq monotonicity and replay determinism

2. **Cross-Platform Determinism Tests:**
   - Run same event log on Windows, Linux, macOS, Android
   - Verify identical state hashes
   - Document any platform-specific quirks

3. **Stress Testing:**
   - Sustained 100Hz sensor input for 1 hour
   - Channel saturation scenarios
   - Concurrent Runtime access patterns

### Priority 2 (Important)

4. **Determinism Snapshots:**
   - Capture known-good state hashes for regression testing
   - Store in `tests/snapshots/` directory
   - CI fails if replay produces different hash

5. **Fuzzing:**
   - Use `cargo-fuzz` to fuzz event deserialization
   - Fuzz belief update with extreme sensor values
   - Fuzz trauma cache with malicious inputs

6. **Memory Profiling:**
   - Profile long-running sessions (24+ hours)
   - Verify no memory leaks
   - Document peak memory usage

### Priority 3 (Nice-to-Have)

7. **Formal Verification:**
   - Model key invariants in TLA+ or Alloy
   - Verify seq monotonicity formally
   - Verify safety guard logic

8. **Performance Benchmarking:**
   - Benchmark event throughput (events/sec)
   - Benchmark replay speed (events/sec)
   - Benchmark belief update latency (ms)

9. **Chaos Engineering:**
   - Inject random failures (DB errors, clock skew, OOM)
   - Verify graceful degradation
   - Verify recovery mechanisms

---

## Conclusion

The ZenB-Rust system has achieved a **production-ready baseline** with the following guarantees:

✅ **Single Source of Truth:** No split-brain, canonical types enforced  
✅ **Audit Integrity:** Critical events never lost, seq monotonic  
✅ **Intrinsic Safety:** Cannot be bypassed, trauma-aware by default  
✅ **Time Safety:** No wraparound, stable under clock skew  

**Remaining work** focuses on **testing depth** (property tests, fuzzing, stress tests) rather than **core correctness** (invariants are enforced).

**Recommendation:** System is ready for **controlled production deployment** with monitoring and gradual rollout.

---

**End of Production Readiness Report**
