# Learning Mechanism - Quick Start Guide

## What Was Built

A complete **Learning Mechanism** that closes the feedback loop from Android action outcomes to AI model updates. The system learns from success/failure and becomes **more conservative on failure**.

### **Terminology Note**
> This system uses `breath_guidance` as the reference implementation for the **Oscillator Control** signal.
> - **ActionType::BreathGuidance** → `ActionType::OscillatorControl`
> - **Breath Pattern** → `Oscillation Pattern`

## Key Files Modified/Created

### Created
- `docs/LEARNING_MECHANISM.md` - Full technical documentation

### Modified
1. **`crates/zenb-core/src/safety_swarm.rs`** - Added `TraumaRegistry` with exponential backoff
2. **`crates/zenb-core/src/belief/mod.rs`** - Added `BeliefEngine::process_feedback()`
3. **`crates/zenb-core/src/engine.rs`** - Added `Engine::learn_from_outcome()`
4. **`crates/zenb-uniffi/src/lib.rs`** - Wired up `report_action_outcome()`
5. **`crates/zenb-store/src/lib.rs`** - Added `record_trauma_with_inhibit()`

## Quick API Reference

### Android Side

```kotlin
// Report action outcome
val outcome = JSONObject().apply {
    put("action_id", "action_123")
    put("success", true)  // or false
    put("result_type", "Success")  // or "Error", "UserCancelled", etc.
    put("action_type", "BreathGuidance")
    put("timestamp_us", System.currentTimeMillis() * 1000)
}

zenbCore.reportActionOutcome(outcome.toString())
```

### Rust Core

```rust
// Engine automatically learns from outcomes
engine.learn_from_outcome(
    false,                      // success
    "BreathGuidance".into(),    // action_type
    timestamp_us,               // timestamp
    2.0,                        // severity (0.0-5.0)
);

// Check trauma registry
println!("Blocked contexts: {}", engine.trauma_registry.len());

// Check belief state
println!("Process noise: {:.4}", engine.config.fep.process_noise);
println!("Learning rate: {:.3}", engine.fep_state.lr);
```

## How It Works

### On Success ✅

1. **Process Noise**: Decreases (model is accurate)
2. **Learning Rate**: Slightly increases (model is on track)
3. **Uncertainty**: Reduces (more confident predictions)

### On Failure ❌

1. **Trauma Registry**: Records failure with exponential backoff
   - 1st failure: 1 hour block
   - 2nd failure: 2 hours block
   - 3rd failure: 4 hours block
   - Capped at: 24 hours

2. **Process Noise**: Increases (acknowledge uncertainty)
3. **Learning Rate**: Decreases (be more cautious)
4. **Free Energy**: Increases (reflect surprise)

## Exponential Backoff Example

```
Context: "Calm mode, 10pm, at home, breath pattern #5"

Failure 1 → Blocked for 1 hour  (until 11pm)
Failure 2 → Blocked for 2 hours (until 1am)
Failure 3 → Blocked for 4 hours (until 5am)
Failure 4 → Blocked for 8 hours (until 1pm)
...
Failure N → Blocked for 24 hours (max)
```

## Safety Guarantees

✅ **No Immediate Retry**: System doesn't just try the same action again  
✅ **Context-Specific**: Failure in one context doesn't affect others  
✅ **Conservative**: Multiple layers of caution on failure  
✅ **Bounded**: Inhibit duration capped at 24 hours  

## Integration Flow

```
Android ActionDispatcher
        ↓
    report_action_outcome(JSON)
        ↓
    ZenbCoreApi (uniffi)
        ↓
    Engine::learn_from_outcome()
        ↓
    ┌───────────────┬──────────────┐
    ↓               ↓              ↓
TraumaRegistry  BeliefEngine  EventStore
(exponential)   (Active Inf)  (persist)
```

## Result Types & Severity

| Result Type | Severity | Description |
|-------------|----------|-------------|
| Success | 0.0 | Action succeeded |
| Timeout | 1.0 | Action timed out |
| Error | 1.5 | Technical error |
| UserCancelled | 2.0 | User rejected action |
| Rejected | 2.5 | System rejected (most severe) |

## Testing

```rust
#[test]
fn test_learning_from_failure() {
    let mut engine = Engine::new(6.0);
    let initial_noise = engine.config.fep.process_noise;
    
    // Report failure
    engine.learn_from_outcome(false, "test".into(), 0, 2.0);
    
    // Verify trauma recorded
    assert_eq!(engine.trauma_registry.len(), 1);
    
    // Verify noise increased (more conservative)
    assert!(engine.config.fep.process_noise > initial_noise);
}
```

## Monitoring

The system logs all learning events:

```
TRAUMA RECORDED: action=BreathGuidance, count=1, severity=2.00, inhibit_until=+1h
FEEDBACK: FAILURE → process_noise=0.0240 (increased), lr=0.425 (reduced), FE=0.150
```

## Key Metrics

1. **Trauma Registry Size** - Number of blocked contexts
2. **Process Noise** - Model uncertainty (higher = less confident)
3. **Learning Rate** - Adaptation speed (lower = more cautious)
4. **Success Rate** - Overall action effectiveness

## Example: Complete Flow

```kotlin
// Android: Execute breath guidance
fun startBreathGuidance() {
    val actionId = "action_${System.currentTimeMillis()}"
    
    try {
        breathController.start()
        
        // Report success
        val outcome = JSONObject().apply {
            put("action_id", actionId)
            put("success", true)
            put("result_type", "Success")
            put("action_type", "BreathGuidance")
            put("timestamp_us", System.currentTimeMillis() * 1000)
        }
        zenbCore.reportActionOutcome(outcome.toString())
        
    } catch (e: Exception) {
        // Report failure
        val outcome = JSONObject().apply {
            put("action_id", actionId)
            put("success", false)
            put("result_type", "Error")
            put("action_type", "BreathGuidance")
            put("message", e.message)
            put("timestamp_us", System.currentTimeMillis() * 1000)
        }
        zenbCore.reportActionOutcome(outcome.toString())
    }
}
```

## Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│                    Engine                              │
├────────────────────────────────────────────────────────┤
│  • belief_state: BeliefState                           │
│  • fep_state: FepState                                 │
│  • config: ZenbConfig                                  │
│  • trauma_registry: TraumaRegistry        ← NEW        │
│                                                         │
│  learn_from_outcome(success, type, ts, severity)       │
│    ├─→ BeliefEngine::process_feedback()                │
│    │     ├─→ Adjust process_noise                      │
│    │     ├─→ Update learning_rate                      │
│    │     └─→ Modify posterior uncertainty              │
│    │                                                    │
│    └─→ TraumaRegistry::record_negative_feedback()      │
│          ├─→ Exponential backoff                       │
│          ├─→ Context hashing                           │
│          └─→ Severity EMA                              │
└────────────────────────────────────────────────────────┘
```

## Status: ✅ COMPLETE

All three phases implemented:
- ✅ Phase 1: Trauma Update Logic (exponential backoff)
- ✅ Phase 2: Belief Update Logic (Active Inference)
- ✅ Phase 3: Engine Integration & Android wiring

The system now learns from action outcomes and becomes more conservative on failure.

## Next Steps

### Immediate Use
1. Update Android `ActionDispatcher` to call `reportActionOutcome()`
2. Monitor trauma registry size and process noise
3. Verify exponential backoff behavior in production

### Future Enhancements
1. **Automatic Persistence**: Periodic trauma sync to EventStore
2. **Decay & Forgetting**: Reduce trauma severity over time
3. **Multi-Armed Bandit**: Thompson Sampling for action selection
4. **Causal Integration**: Connect to causal graph for counterfactual reasoning

## Documentation

- **Full Docs**: `docs/LEARNING_MECHANISM.md`
- **This Guide**: `LEARNING_QUICKSTART.md`
- **Causal Layer**: `docs/CAUSAL_LAYER.md`
