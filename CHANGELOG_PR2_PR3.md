# PR2 + PR3 Changelog: Hardened Audit + Intrinsic Safety

## Summary
Implemented forensic-grade audit guarantees (PR2) and intrinsic safety enforcement (PR3) to eliminate silent data loss and API misuse.

---

## PR2: Hardened Audit — No Loss of Critical Events

### Event Priority Classification
- **Added:** `EventPriority` enum (Critical, HighFreq)
- **Added:** `Event::priority()` method for classification
- **Critical events:** SessionStarted, SessionEnded, ControlDecisionMade, ControlDecisionDenied, ConfigUpdated, PatternAdjusted, Tombstone
- **HighFreq events:** SensorFeaturesIngested, BeliefUpdated, BeliefUpdatedV2, PolicyChosen, CycleCompleted

### Guaranteed Delivery Implementation
- **Modified:** `AsyncWorker::submit_append()` to use priority-aware delivery
- **Critical path:** Uses `tx.send()` (blocking) - GUARANTEED delivery
- **HighFreq path:** Uses `tx.try_send()` (non-blocking) - can drop with visibility
- **Fallback:** Emergency dump to disk if worker shutdown during Critical event

### Drop Visibility Metrics
- **Added:** `WorkerMetrics::highfreq_drops` counter
- **Added:** `WorkerMetrics::highfreq_coalesced` counter (future use)
- **Added:** Warning logs for every HighFreq drop
- **Updated:** `MetricsSnapshot` to include new counters

### Seq Monotonicity Hardening
- **Added:** Invariant comments documenting seq monotonicity guarantee
- **Documented:** seq is NEVER reset due to buffer flush or backpressure
- **Documented:** seq is derived from `buf.back().seq + 1`, NOT buffer length

### Files Modified (PR2)
1. `crates/zenb-core/src/domain.rs` — EventPriority enum + Event::priority()
2. `crates/zenb-uniffi/src/async_worker.rs` — Priority-aware delivery + metrics
3. `crates/zenb-uniffi/src/lib.rs` — Seq invariant comments + updated flush
4. `crates/zenb-core/src/lib.rs` — Export EventPriority

---

## PR3: Intrinsic Safety — API Cannot Be Misused

### API Simplification
- **Removed:** `Option<&dyn TraumaSource>` parameter from `Engine::make_control()`
- **Before:** `make_control(&mut self, est: &Estimate, ts_us: i64, trauma: Option<&dyn TraumaSource>)`
- **After:** `make_control(&mut self, est: &Estimate, ts_us: i64)`
- **Rationale:** Engine always uses `self.trauma_cache` - no need for external source

### Intrinsic Safety Enforcement
- **Guarantee:** Safety checks cannot be bypassed through API misuse
- **Guarantee:** Engine ALWAYS uses internal `trauma_cache`
- **Guarantee:** TraumaGuard is ALWAYS active in decision path
- **Cold start:** Empty cache = no inhibitions (safe default), not disabled safety

### Call Site Updates
- **Updated:** `crates/zenb-uniffi/src/lib.rs` to remove `None` parameter
- **Removed:** Misleading comment about "trauma checks disabled"
- **Verified:** Trauma cache hydration on startup (already implemented)

### Test Updates
- **Updated:** `engine_decision_flow()` test to match new signature
- **Added:** PR3 comment explaining intrinsic safety

### Files Modified (PR3)
1. `crates/zenb-core/src/engine.rs` — Removed Option<TraumaSource> parameter
2. `crates/zenb-uniffi/src/lib.rs` — Updated call site
3. `crates/zenb-core/src/engine.rs` — Updated test

---

## Impact Analysis

### Breaking Changes
- **PR2:** None (additive changes only)
- **PR3:** `Engine::make_control()` signature changed (internal API only)

### Performance Impact
- **PR2 Critical events:** May block briefly under extreme backpressure (acceptable - rare events)
- **PR2 HighFreq events:** No change (still non-blocking)
- **PR3:** No performance impact (removed unused parameter)

### Safety Improvements
- **PR2:** Critical events NEVER lost silently (forensic-grade audit)
- **PR2:** HighFreq drops are visible via metrics and logs
- **PR3:** Safety cannot be bypassed through API misuse
- **PR3:** Simpler API reduces cognitive load and mistakes

---

## Verification Checklist

### PR2
- [x] EventPriority enum defined
- [x] Event::priority() method implemented
- [x] Critical events use blocking send
- [x] HighFreq events track drops
- [x] Seq monotonicity documented
- [x] Metrics exposed
- [ ] Tests pass (pending user environment)

### PR3
- [x] Option<TraumaSource> removed from make_control
- [x] Engine uses self.trauma_cache
- [x] Call sites updated
- [x] Tests updated
- [x] No way to bypass safety
- [ ] Tests pass (pending user environment)

---

## Files Created
1. `docs/PR2-hardened-audit.md` — Full PR2 documentation
2. `docs/PR3-intrinsic-safety.md` — Full PR3 documentation
3. `CHANGELOG_PR2_PR3.md` — This file

---

## Next Steps

### Immediate (User Action Required)
1. Run `cargo fmt`
2. Run `cargo clippy -D warnings`
3. Run `cargo test --workspace`
4. Verify all tests pass
5. Commit changes if verification succeeds

### Future PRs
1. **PR4:** Fix time delta wrapping (use `saturating_sub` everywhere)
2. **PR5:** Periodic health event summarizing drop metrics
3. **PR6:** Smart coalescing for HighFreq events
4. **PR7:** Adaptive backpressure (reduce sampling under load)

---

**End of Changelog**
