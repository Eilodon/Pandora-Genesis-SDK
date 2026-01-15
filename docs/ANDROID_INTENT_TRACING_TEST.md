# Android End-to-End Testing Guide: Intent ID Tracing

**B.ONE V3: Di Hồn Đại Pháp - Karmic Feedback Loop**

This guide provides step-by-step instructions for testing the complete intent tracing flow on Android to verify the karmic feedback loop works in production.

---

## 🎯 Test Objective

Verify that:
1. Rust core generates unique `IntentId` for each decision
2. Android receives `intent_id` in `ControlDecision`
3. Android can report outcomes with `intent_id`
4. Rust core learns from outcomes with correct context
5. Trauma registry and policy adaptation work correctly

---

## 📋 Prerequisites

### Rust Side (Already Complete)
- ✅ `ControlDecision` has `intent_id: Option<u64>` field
- ✅ `Engine::make_control()` populates intent_id
- ✅ `UnifiedSankhara::deliberate()` tracks all intents (including denied)
- ✅ `report_action_outcome()` accepts `intent_id` in JSON

### Android Side (Needs Update)
- ⚠️ Kotlin data classes need `intentId` field
- ⚠️ Action execution needs to capture `intentId`
- ⚠️ Outcome reporting needs to include `intentId`

---

## 🔧 Android Code Changes Required

### 1. Update ControlDecision Data Class

**Location:** `android/app/src/main/java/com/agolos/zenb/core/domain/ControlDecision.kt`

```kotlin
data class ControlDecision(
    val targetRateBpm: Float,
    val confidence: Float,
    val recommendedPollIntervalMs: Long,
    
    // B.ONE V3: Intent ID for karmic feedback
    val intentId: Long? = null  // NEW FIELD
)
```

### 2. Update ActionExecution to Track IntentId

**Location:** `android/app/src/main/java/com/agolos/zenb/breath/BreathGuidanceService.kt`

```kotlin
class BreathGuidanceService {
    // Store current intent_id when starting guidance
    private var currentIntentId: Long? = null
    
    fun startBreathGuidance(decision: ControlDecision) {
        // Capture intent_id at the start
        currentIntentId = decision.intentId
        
        Log.d("BreathGuidance", "Starting guidance with IntentId: $currentIntentId")
        
        // ... existing guidance logic ...
    }
    
    fun stopBreathGuidance(reason: StopReason) {
        // Use captured intent_id when reporting
        val intentId = currentIntentId
        currentIntentId = null // Clear after use
        
        reportOutcome(
            intentId = intentId,  // Pass to outcome reporting
            reason = reason
        )
    }
}
```

### 3. Update Outcome Reporting

**Location:** `android/app/src/main/java/com/agolos/zenb/core/ActionOutcomeReporter.kt`

```kotlin
data class ActionOutcome(
    val actionId: String,
    val intentId: Long?,  // NEW: Include intent_id
    val success: Boolean,
    val resultType: String,
    val actionType: String,
    val timestampUs: Long
)

fun reportActionOutcome(outcome: ActionOutcome) {
    val json = JSONObject().apply {
        put("action_id", outcome.actionId)
        
        // B.ONE V3: Include intent_id if available
        outcome.intentId?.let { 
            put("intent_id", it) 
            Log.d("KarmicFeedback", "Reporting outcome with IntentId: $it")
        }
        
        put("success", outcome.success)
        put("result_type", outcome.resultType)
        put("action_type", outcome.actionType)
        put("timestamp_us", outcome.timestampUs)
    }
    
    zenbCore.reportActionOutcome(json.toString())
}
```

---

## 🧪 Test Scenarios

### Scenario 1: Successful Breath Guidance

**Setup:**
1. User has elevated heart rate (75+ BPM)
2. System is in good state (no trauma, high confidence)

**Test Steps:**

```kotlin
@Test
fun testSuccessfulBreathGuidance_withIntentTracking() = runBlocking {
    // 1. Ingest sensor data
    val features = listOf(75f, 40f, 14f, 0.9f, 0.2f)
    zenbCore.ingestSensorWithContext(
        tsUs = System.currentTimeMillis() * 1000,
        features = features,
        localHour = 14,
        isCharging = false,
        recentSessions = 5
    )
    
    // 2. Tick engine
    zenbCore.tick(System.currentTimeMillis() * 1000)
    
    // 3. Get control decision
    val decision = zenbCore.getLastDecision()
    
    // VERIFY: Decision has intent_id
    assertNotNull(decision.intentId, "Decision must have intent_id")
    val intentId = decision.intentId!!
    
    Log.d("TEST", "Received decision with IntentId: $intentId")
    
    // 4. Execute breath guidance
    breathService.startBreathGuidance(decision)
    delay(5000) // User follows guidance for 5 seconds
    breathService.stopBreathGuidance(StopReason.COMPLETED)
    
    // 5. Report success outcome
    reportActionOutcome(ActionOutcome(
        actionId = "test_${System.currentTimeMillis()}",
        intentId = intentId,  // Include the intent_id
        success = true,
        resultType = "Success",
        actionType = "BreathGuidance",
        timestampUs = System.currentTimeMillis() * 1000
    ))
    
    // VERIFY: Check logs for IntentTracker confirmation
    // Expected log: "Outcome recorded: intent_id=XXX, success=true"
}
```

**Expected Results:**
- ✅ Decision contains valid `intent_id` (non-null, > 0)
- ✅ Outcome report includes same `intent_id`
- ✅ Rust logs show: `"Outcome recorded: intent_id=XXX, success=true"`
- ✅ No crash or errors

---

### Scenario 2: User Cancelled Action

**Test Steps:**

```kotlin
@Test
fun testUserCancellation_withIntentTracking() = runBlocking {
    val features = listOf(80f, 35f, 16f, 0.85f, 0.3f)
    zenbCore.ingestSensorWithContext(...)
    zenbCore.tick(...)
    
    val decision = zenbCore.getLastDecision()
    val intentId = decision.intentId!!
    
    // User starts but cancels immediately
    breathService.startBreathGuidance(decision)
    delay(500)
    breathService.stopBreathGuidance(StopReason.USER_CANCELLED)
    
    // Report failure
    reportActionOutcome(ActionOutcome(
        actionId = "test_cancel",
        intentId = intentId,
        success = false,
        resultType = "UserCancelled",
        actionType = "BreathGuidance",
        timestampUs = System.currentTimeMillis() * 1000
    ))
    
    // VERIFY: This should trigger trauma registry update
    // Next decision should be affected by this failure
    
    // Make another decision immediately
    zenbCore.ingestSensorWithContext(...)
    val nextDecision = zenbCore.getLastDecision()
    
    // Expected: System becomes more conservative
    assertTrue(
        nextDecision.targetRateBpm >= decision.targetRateBpm,
        "After cancellation, system should be more conservative"
    )
}
```

**Expected Results:**
- ✅ Failure recorded with correct `intent_id`
- ✅ Trauma registry updated
- ✅ Subsequent decisions show learned caution
- ✅ Policy adapter may mask the failed action temporarily

---

### Scenario 3: Denied Action (Safety Guard)

**Setup:**
- Force a condition that triggers safety guards (e.g., extreme breath rate)

**Test Steps:**

```kotlin
@Test
fun testDeniedAction_stillTracked() = runBlocking {
    // Create extreme scenario that will be denied
    val features = listOf(120f, 20f, 25f, 0.5f, 0.8f) // Very high HR, low HRV
    zenbCore.ingestSensorWithContext(...)
    zenbCore.tick(...)
    
    val decision = zenbCore.getLastDecision()
    
    // VERIFY: Even if action was denied, intent_id exists
    assertNotNull(decision.intentId, "Denied decisions should still have intent_id")
    
    Log.d("TEST", "Denied decision with IntentId: ${decision.intentId}")
    
    // Decision might be fallback (6 BPM baseline)
    assertEquals(6f, decision.targetRateBpm, delta = 0.1f)
    
    // Even denied intents are tracked for learning
}
```

**Expected Results:**
- ✅ Denied decisions have `intent_id`
- ✅ Intent is tracked in IntentTracker
- ✅ System can learn from denial patterns

---

## 📊 Verification Checklist

After running tests, verify in Rust logs:

### ✅ Intent Generation
```
[DEBUG] IntentId 42 tracked successfully
```

### ✅ Outcome Recording
```
[INFO] Outcome recorded: intent_id=42, success=true
```

### ✅ Context Preserved
```
[DEBUG] Intent 42 - Original context: mode=0, confidence=0.85
[DEBUG] Outcome received after 5000ms
```

### ✅ Learning Applied
```
[INFO] PolicyAdapter: Updated success rate for BreathGuidance
[INFO] TraumaRegistry: Updated entry for pattern_id=1, mode=2
```

---

## 🔍 Debugging Tips

### Issue: `intent_id` is always `null`

**Possible causes:**
1. Android FFI binding not updated - regenerate with `cargo uniffi-bindgen`
2. Kotlin data class missing field - check generated code
3. Serialization issue - check JSON encoding

**Fix:**
```bash
# Regenerate Kotlin bindings
cd crates/zenb-uniffi
cargo run --features=uniffi/cli --bin uniffi-bindgen generate src/zenb.udl --language kotlin
```

### Issue: IntentTracker shows 0 items

**Possible causes:**
1. `deliberate()` returning early before tracking
2. Safety guards denying all actions (now fixed in Option B)
3. Engine not calling `deliberate()` properly

**Check:**
```kotlin
// Add debug logging in Rust (if accessible)
println!("IntentTracker size: {}", tracker.len());
```

### Issue: Outcomes not recorded

**Possible causes:**
1. `intent_id` mismatch between decision and outcome
2. Intent expired (> 5 min old)
3. JSON parsing error in `report_action_outcome`

**Verify JSON format:**
```json
{
  "action_id": "action_1234567890_5678",
  "intent_id": 42,
  "success": true,
  "result_type": "Success",
  "action_type": "BreathGuidance",
  "timestamp_us": 1704268800000000
}
```

---

## 📈 Success Metrics

After 1 week of production testing:

- ✅ **Coverage:** > 90% of decisions have `intent_id`
- ✅ **Traceability:** > 80% of outcomes mapped to original intent
- ✅ **Learning:** Measurable improvement in success rate for repeated actions
- ✅ **Trauma Registry:** Accurate failure tracking with context
- ✅ **No Crashes:** Zero crashes related to intent tracking

---

## 🎉 Expected Benefits

Once fully deployed:

1. **Precise Learning:** System learns from exact decision context, not aggregate statistics
2. **Better Trauma Management:** Know exactly which decisions led to bad outcomes
3. **Policy Refinement:** Adapt action selection based on historical success/failure per context
4. **Debugging:** Full audit trail from sensor → decision → action → outcome
5. **Meta-Learning:** Understand when the AI makes good vs. bad decisions

---

## 📝 Next Steps After Android Testing

1. **Monitor Production:** Track `intent_id` coverage and outcome mapping rate
2. **Analyze Patterns:** Look for intents that frequently fail
3. **Tune Parameters:** Adjust trauma decay, policy masking thresholds
4. **A/B Test:** Compare learning rate with vs. without intent tracing
5. **User Study:** Verify improved breath guidance effectiveness

---

## 🔗 Related Documentation

- Integration Test: `crates/zenb-uniffi/tests/intent_tracing_test.rs`
- Sankhara Implementation: `crates/zenb-core/src/skandha/sankhara.rs`
- UniFFI API: `crates/zenb-uniffi/src/lib.rs`
- FlowStream Events: `crates/zenb-core/src/universal_flow.rs`

---

**Questions or issues?** Check test results in `intent_tracing_test.rs` first - they demonstrate the expected behavior.

**Ready for production?** All Rust-side changes are complete and tested. Only Android-side integration remains.
