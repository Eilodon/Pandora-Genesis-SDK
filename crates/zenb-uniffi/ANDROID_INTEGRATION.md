# Android Integration Guide: ZenB Core FFI

This guide explains how to integrate the Rust `zenb-core` with Android using UniFFI and the `SensorFusionService`.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Android Application                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           SensorFusionService (Kotlin)                │  │
│  │  - Aggregates multi-dimensional context               │  │
│  │  - Normalizes Android sensor data                     │  │
│  │  - Serializes to JSON                                 │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │ JSON Observation                        │
│                   ▼                                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ZenbCoreApi (UniFFI Wrapper)                  │  │
│  │  - Thread-safe FFI boundary                           │  │
│  │  - Error handling & type conversion                   │  │
│  └────────────────┬─────────────────────────────────────┘  │
└───────────────────┼──────────────────────────────────────────┘
                    │ FFI Call
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Rust Core (zenb-core)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Runtime (Event Sourcing)                 │  │
│  │  - Parses Observation struct                          │  │
│  │  - Extracts features for Active Inference             │  │
│  │  - Updates belief state                               │  │
│  │  - Selects action policies                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Setup Instructions

### 1. Build Rust Library with UniFFI

```bash
cd crates/zenb-uniffi

# Generate UniFFI bindings
cargo build --release

# Generate Kotlin bindings (requires uniffi-bindgen)
cargo run --bin uniffi-bindgen generate src/zenb.udl --language kotlin --out-dir ../android/app/src/main/java/uniffi/zenb/
```

This generates:
- `libzenb_uniffi.so` (native library for Android)
- Kotlin bindings in the specified output directory

### 2. Add Native Library to Android Project

```kotlin
// In your app's build.gradle.kts
android {
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

dependencies {
    implementation("net.java.dev.jna:jna:5.13.0@aar")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
}
```

Copy the compiled `.so` files to:
```
app/src/main/jniLibs/
├── arm64-v8a/
│   └── libzenb_uniffi.so
├── armeabi-v7a/
│   └── libzenb_uniffi.so
└── x86_64/
    └── libzenb_uniffi.so
```

### 3. Initialize ZenbCoreApi

```kotlin
// In your Application class or DI module
class PandoraApplication : Application() {
    private lateinit var zenbCoreApi: ZenbCoreApi
    
    override fun onCreate() {
        super.onCreate()
        
        // Initialize Rust core
        val dbPath = getDatabasePath("zenb.db").absolutePath
        val masterKey = generateOrLoadMasterKey() // 32-byte key
        
        zenbCoreApi = ZenbCoreApi(dbPath, masterKey.toList())
        
        // Provide via DI (Hilt/Dagger)
        // ...
    }
    
    private fun generateOrLoadMasterKey(): ByteArray {
        // SECURITY: Never hardcode master keys.
        // Store a per-install random key using Android Keystore (via Jetpack Security).
        //
        // Imports:
        // import android.util.Base64
        // import androidx.security.crypto.EncryptedSharedPreferences
        // import androidx.security.crypto.MasterKey
        // import java.security.SecureRandom
        val masterKey = MasterKey.Builder(this)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        val prefs = EncryptedSharedPreferences.create(
            this,
            "zenb_keys",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
        val existingB64 = prefs.getString("zenb_master_key_b64", null)
        if (existingB64 != null) {
            return Base64.decode(existingB64, Base64.NO_WRAP)
        }
        val fresh = ByteArray(32).also { SecureRandom().nextBytes(it) }
        prefs.edit()
            .putString("zenb_master_key_b64", Base64.encodeToString(fresh, Base64.NO_WRAP))
            .apply()
        // NOTE: The master key will exist in plaintext in process memory while passed to Rust.
        // Treat rooted/hooked devices as untrusted; consider best-effort root/hook detection if needed.
        return fresh
    }
}
```

### 4. Integrate SensorFusionService

```kotlin
// Hilt/Dagger module
@Module
@InstallIn(SingletonComponent::class)
object SensorModule {
    
    @Provides
    @Singleton
    fun provideSensorFusionService(
        @ApplicationContext context: Context,
        locationAwareness: LocationAwareness,
        userActivityAnalyzer: UserActivityAnalyzer,
        appUsageIntelligence: AppUsageIntelligence,
        zenbCoreApi: ZenbCoreApi
    ): SensorFusionService {
        val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
        return SensorFusionService(
            context,
            locationAwareness,
            userActivityAnalyzer,
            appUsageIntelligence,
            zenbCoreApi,
            serviceScope
        )
    }
}
```

### 5. Start Sensor Fusion

```kotlin
// In your MainActivity or foreground service
class MainActivity : ComponentActivity() {
    @Inject lateinit var sensorFusionService: SensorFusionService
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Start continuous sensor fusion
        lifecycleScope.launch {
            lifecycle.repeatOnLifecycle(Lifecycle.State.STARTED) {
                sensorFusionService.startCollection()
            }
        }
        
        // Observe observations (optional, for debugging)
        lifecycleScope.launch {
            sensorFusionService.observationFlow.collect { observation ->
                Log.d("ZenB", "Observation: $observation")
            }
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        sensorFusionService.stopCollection()
    }
}
```

## Data Flow Example

### Input: Android Sensor Data

```kotlin
// From LocationAwareness
LocationData(
    type = "home",
    latitude = 37.7749,
    longitude = -122.4194,
    ambientNoiseDb = 45.0f
)

// From UserActivityAnalyzer (wearable)
ActivityData(
    heartRate = 72,
    hrv = 45,
    respiratoryRate = 14,
    stepCount = 1234,
    activityType = "walking"
)

// From AppUsageIntelligence
AppUsageData(
    currentAppCategory = "social",
    currentAppPackage = "com.instagram.android",
    touchEventsPerMinute = 30,
    notificationsPerHour = 12.0f,
    screenTimeMinutes = 45
)
```

### Output: JSON Observation (sent to Rust)

```json
{
  "timestamp_us": 1704268800000000,
  "bio_metrics": {
    "hr_bpm": 72.0,
    "hrv_rmssd": 45.0,
    "respiratory_rate": 14.0
  },
  "environmental_context": {
    "location_type": "Home",
    "noise_level": 0.45,
    "is_charging": true
  },
  "digital_context": {
    "active_app_category": "Social",
    "interaction_intensity": 0.5,
    "notification_pressure": 0.2
  }
}
```

### Rust Processing

The Rust core:
1. Parses JSON into `Observation` struct
2. Extracts feature vector: `[72.0, 45.0, 14.0, 0.9, 0.8]`
3. Updates `BeliefState` distributions using Active Inference
4. Selects `ActionPolicy` (e.g., `GuidanceBreath` or `DigitalIntervention`)
5. Returns policy to Android for execution

## Data Normalization

The `SensorFusionService` normalizes all Android values to `[0.0, 1.0]`:

| Android Value | Normalization | Range |
|---------------|---------------|-------|
| Screen brightness | `value / 255` | 0-255 → 0.0-1.0 |
| Noise level (dB) | `value / 100` | 0-100 dB → 0.0-1.0 |
| Touch events/min | `value / 60` | 0-60+ → 0.0-1.0 |
| Notifications/hour | `value / 60` | 0-60+ → 0.0-1.0 |

## Error Handling

### Kotlin Side

```kotlin
try {
    zenbCoreApi.ingestObservation(jsonPayload)
} catch (e: ZenbError.JsonParseError) {
    // Malformed JSON - log and skip
    Log.e(TAG, "Invalid JSON", e)
} catch (e: ZenbError.RuntimeError) {
    // Rust runtime error - may need restart
    Log.e(TAG, "Rust runtime error", e)
} catch (e: Exception) {
    // FFI panic or unexpected error
    Log.e(TAG, "FFI call failed", e)
}
```

### Rust Side

```rust
pub fn ingest_observation(&mut self, json_payload: &str) -> Result<(), RuntimeError> {
    let obs: Observation = serde_json::from_str(json_payload)?;
    // ... processing
    Ok(())
}
```

Errors are converted to `ZenbError` enum and propagated to Kotlin.

## Thread Safety

- **Kotlin**: All FFI calls run on `Dispatchers.IO` (background thread pool)
- **Rust**: `ZenbCoreApi` uses `Arc<Mutex<Runtime>>` for thread-safe access
- **Guarantee**: No main thread blocking, safe concurrent access

## Performance Considerations

1. **Sampling Rate**: 2 Hz (500ms interval) balances responsiveness and battery
2. **Batching**: Rust buffers events and flushes at 80ms or 20 events
3. **Downsampling**: High-frequency data is downsampled before persistence
4. **Memory**: Bounded buffers prevent memory leaks

## Testing

### Unit Test (Kotlin)

```kotlin
@Test
fun `test observation serialization`() = runTest {
    val observation = ObservationData(
        timestampUs = 1234567890000L,
        bioMetrics = BioMetricsData(hrBpm = 72.0f),
        environmentalContext = EnvironmentalContextData(isCharging = true),
        digitalContext = null
    )
    
    val json = Json.encodeToString(observation)
    assertTrue(json.contains("timestamp_us"))
    assertTrue(json.contains("bio_metrics"))
}
```

### Integration Test (Rust)

```rust
#[test]
fn test_observation_ingestion() {
    let json = r#"{
        "timestamp_us": 1234567890000,
        "bio_metrics": {"hr_bpm": 72.0, "hrv_rmssd": 45.0},
        "environmental_context": {"is_charging": true},
        "digital_context": null
    }"#;
    
    let mut rt = Runtime::new("test.db", [0u8; 32], SessionId::new()).unwrap();
    assert!(rt.ingest_observation(json).is_ok());
}
```

## Troubleshooting

### Issue: FFI call crashes app

**Solution**: Ensure native library is loaded before calling FFI:

```kotlin
companion object {
    init {
        System.loadLibrary("zenb_uniffi")
    }
}
```

### Issue: JSON parsing fails

**Solution**: Verify field names match exactly (snake_case in JSON, camelCase in Kotlin):

```kotlin
@Serializable
@SerialName("timestamp_us")
val timestampUs: Long
```

### Issue: High battery drain

**Solution**: Increase `COLLECTION_INTERVAL_MS` or use adaptive sampling:

```kotlin
private val adaptiveInterval: Long
    get() = if (batteryManager.isCharging) 500L else 2000L
```

## Next Steps

1. Implement `LocationAwareness`, `UserActivityAnalyzer`, `AppUsageIntelligence`
2. Add policy execution layer (receive `ActionPolicy` from Rust, execute on Android)
3. Integrate with Android Health Connect for biometric data
4. Add privacy controls and data encryption
5. Implement offline-first architecture with sync

## References

- [UniFFI Documentation](https://mozilla.github.io/uniffi-rs/)
- [Kotlin Coroutines](https://kotlinlang.org/docs/coroutines-guide.html)
- [Android NDK Guide](https://developer.android.com/ndk/guides)
- [Active Inference Framework](../docs/BELIEF_ENGINE.md)
