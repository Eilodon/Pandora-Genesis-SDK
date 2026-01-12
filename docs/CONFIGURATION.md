# ZenB Configuration Guide

Complete guide for configuring the ZenB system using external configuration files and environment variables.

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration Files](#configuration-files)
3. [Configuration Structure](#configuration-structure)
4. [Loading Configuration](#loading-configuration)
5. [Environment Variables](#environment-variables)
6. [Configuration Validation](#configuration-validation)
7. [Best Practices](#best-practices)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

ZenB uses a **layered configuration system** with the following priority (highest to lowest):

1. **Environment Variables** (highest priority)
2. **User Config File** (e.g., `~/.zenb/config.toml`)
3. **Default Config File** (e.g., `config/default.toml`)
4. **Built-in Defaults** (lowest priority)

This allows flexible configuration for different environments (development, testing, production) while maintaining sensible defaults.

---

## Configuration Files

### Available Configurations

ZenB provides pre-configured files for different use cases:

| File | Purpose | Use Case |
|------|---------|----------|
| `config/default.toml` | Default settings | General use, starting point |
| `config/production.toml` | Production-optimized | Stable, conservative settings |
| `config/development.toml` | Development-friendly | Fast feedback, permissive |
| `config/testing.toml` | Test-optimized | Deterministic, minimal delays |

### File Locations

**System-wide config:**
```
/etc/zenb/config.toml          # Linux/macOS
C:\ProgramData\ZenB\config.toml # Windows
```

**User-specific config:**
```
~/.zenb/config.toml            # Linux/macOS
%APPDATA%\ZenB\config.toml     # Windows
```

**Project-specific config:**
```
./config/default.toml
./config/production.toml
```

---

## Configuration Structure

### Complete Configuration Schema

```toml
[fep]
# Free Energy Principle parameters
process_noise = 0.02        # Process noise for belief update (0-1)
base_obs_var = 0.25         # Base observation variance (> 0)
lr_base = 0.6               # Base learning rate (0-1)
lr_min = 0.05               # Minimum learning rate (0-lr_max)
lr_max = 0.8                # Maximum learning rate (0-1)

[resonance]
# Resonance tracking parameters
window_size_sec = 12.0      # Window size for resonance calculation (> 0)
coherence_threshold = 0.35  # Coherence threshold (0-1)

[safety]
# Safety and trauma tracking parameters
trauma_hard_th = 1.5        # Hard trauma threshold (> trauma_soft_th)
trauma_soft_th = 0.7        # Soft trauma threshold (> 0)
trauma_decay_default = 0.7  # Default trauma decay rate (0-1)

### Oscillator Control (Reference: Breath)

> **Note:** The "Breath" configuration section controls the core oscillator engine. While currently named `[breath]` for historical reasons (referencing the reference implementation), these parameters govern the primary frequency generation for any adaptive loop.

[breath]
# Breath guidance parameters
default_target_bpm = 6.0    # Default target breathing rate (1-30)

[belief]
# Belief engine parameters
pathway_weights = [1.0, 0.8, 1.2]  # Weights for [Logical, Contextual, Biometric]
prior_logits = [0.0, 0.0, 0.0, 0.0, 0.0]  # Prior logits for belief states
smooth_tau_sec = 4.0        # Smoothing time constant (> 0)
enter_threshold = 0.6       # Hysteresis enter threshold (> exit_threshold)
exit_threshold = 0.4        # Hysteresis exit threshold (0-1)

[performance]
# Performance tuning parameters
batch_size = 20             # Number of events to batch (> 0)
flush_interval_ms = 80      # Flush interval in milliseconds
max_buffer_size = 100       # Maximum buffer size (>= batch_size)
enable_burst_filter = true  # Enable burst filtering
burst_filter_threshold_us = 10000  # Burst filter threshold in microseconds
```

### Parameter Descriptions

#### FEP (Free Energy Principle)

- **process_noise**: Controls how much the belief state can change between updates. Higher values allow faster adaptation but less stability.
- **base_obs_var**: Base observation variance. Lower values trust observations more.
- **lr_base**: Base learning rate for belief updates. Higher values = faster learning.
- **lr_min/lr_max**: Bounds for adaptive learning rate.

#### Resonance

- **window_size_sec**: Time window for resonance calculation. Longer windows = more stable but slower response.
- **coherence_threshold**: Minimum coherence score to consider resonance detected.

#### Safety

- **trauma_hard_th**: Severity threshold that completely blocks actions.
- **trauma_soft_th**: Severity threshold that triggers warnings.
- **trauma_decay_default**: How quickly trauma memories fade (0 = never, 1 = immediate).

#### Breath

- **default_target_bpm**: Default breathing rate in breaths per minute.

#### Belief

- **pathway_weights**: Relative importance of each reasoning pathway:
  - `[0]`: Logical pathway (rule-based heuristics)
  - `[1]`: Contextual pathway (time-of-day, charging, history)
  - `[2]`: Biometric pathway (sensor quality, motion)
- **prior_logits**: Initial belief distribution over 5 states (Calm, Stress, Focus, Sleepy, Energize)
- **smooth_tau_sec**: Time constant for exponential smoothing
- **enter_threshold**: Probability threshold to enter a new belief state
- **exit_threshold**: Probability threshold to exit current state (hysteresis)

#### Performance

- **batch_size**: Number of events to accumulate before database write
- **flush_interval_ms**: Maximum time between flushes
- **max_buffer_size**: Maximum events in buffer before forced flush
- **enable_burst_filter**: Filter out sensor samples < threshold apart
- **burst_filter_threshold_us**: Minimum time between samples (microseconds)

---

## Loading Configuration

### Method 1: Load from Single File

```rust
use zenb_core::config::ZenbConfig;

// Load from file
let config = ZenbConfig::from_file("config/production.toml")?;

// Use in engine
let engine = Engine::new_with_config(6.0, Some(config));
```

### Method 2: Load with Environment Overrides

```rust
// Load file + apply environment variables
let config = ZenbConfig::from_file_with_env("config/default.toml")?;
```

### Method 3: Layered Loading (Recommended)

```rust
use std::path::Path;

// Load with priority: env vars > user config > default config > built-in
let config = ZenbConfig::load_layered(
    Some(Path::new("config/default.toml")),
    Some(Path::new("~/.zenb/config.toml")),
)?;
```

### Method 4: Use Built-in Defaults

```rust
// Use default configuration
let config = ZenbConfig::default();
```

---

## Environment Variables

All configuration values can be overridden via environment variables prefixed with `ZENB_`.

### Naming Convention

```
ZENB_<SECTION>_<PARAMETER>
```

Examples:
```bash
ZENB_FEP_PROCESS_NOISE=0.03
ZENB_BELIEF_SMOOTH_TAU_SEC=5.0
ZENB_PERFORMANCE_BATCH_SIZE=30
```

### Complete Environment Variable List

#### FEP
```bash
ZENB_FEP_PROCESS_NOISE=0.02
ZENB_FEP_BASE_OBS_VAR=0.25
ZENB_FEP_LR_BASE=0.6
ZENB_FEP_LR_MIN=0.05
ZENB_FEP_LR_MAX=0.8
```

#### Resonance
```bash
ZENB_RESONANCE_WINDOW_SIZE_SEC=12.0
ZENB_RESONANCE_COHERENCE_THRESHOLD=0.35
```

#### Safety
```bash
ZENB_SAFETY_TRAUMA_HARD_TH=1.5
ZENB_SAFETY_TRAUMA_SOFT_TH=0.7
ZENB_SAFETY_TRAUMA_DECAY_DEFAULT=0.7
```

#### Breath
```bash
ZENB_BREATH_DEFAULT_TARGET_BPM=6.0
```

#### Belief
```bash
ZENB_BELIEF_SMOOTH_TAU_SEC=4.0
ZENB_BELIEF_ENTER_THRESHOLD=0.6
ZENB_BELIEF_EXIT_THRESHOLD=0.4
```

#### Performance
```bash
ZENB_PERFORMANCE_BATCH_SIZE=20
ZENB_PERFORMANCE_FLUSH_INTERVAL_MS=80
ZENB_PERFORMANCE_MAX_BUFFER_SIZE=100
```

### Using Environment Variables

**Linux/macOS:**
```bash
export ZENB_FEP_PROCESS_NOISE=0.03
export ZENB_BELIEF_SMOOTH_TAU_SEC=5.0
./zenb-app
```

**Windows (PowerShell):**
```powershell
$env:ZENB_FEP_PROCESS_NOISE="0.03"
$env:ZENB_BELIEF_SMOOTH_TAU_SEC="5.0"
.\zenb-app.exe
```

**Docker:**
```dockerfile
ENV ZENB_FEP_PROCESS_NOISE=0.03
ENV ZENB_BELIEF_SMOOTH_TAU_SEC=5.0
```

---

## Configuration Validation

All configurations are automatically validated when loaded. Validation checks:

### FEP Validation
- `process_noise` must be in (0, 1]
- `base_obs_var` must be positive
- `lr_min` must be in [0, lr_max]
- `lr_max` must be <= 1.0

### Resonance Validation
- `window_size_sec` must be positive
- `coherence_threshold` must be in [0, 1]

### Safety Validation
- `trauma_hard_th` must be > `trauma_soft_th`
- `trauma_decay_default` must be in [0, 1]

### Breath Validation
- `default_target_bpm` must be in [1, 30]

### Belief Validation
- `pathway_weights` must have exactly 3 elements
- All weights must be non-negative
- `smooth_tau_sec` must be positive
- `enter_threshold` must be > `exit_threshold`

### Performance Validation
- `batch_size` must be > 0
- `max_buffer_size` must be >= `batch_size`

### Error Handling

```rust
match ZenbConfig::from_file("config.toml") {
    Ok(config) => {
        println!("Config loaded successfully");
    }
    Err(e) => {
        eprintln!("Config error: {}", e);
        // Use defaults or exit
    }
}
```

---

## Best Practices

### 1. Use Layered Configuration

```rust
// Recommended approach
let config = ZenbConfig::load_layered(
    Some(Path::new("config/default.toml")),
    Some(Path::new(&user_config_path)),
)?;
```

**Benefits:**
- Defaults always available
- User can override specific values
- Environment variables for deployment-specific tweaks

### 2. Version Control

**Do commit:**
- `config/default.toml`
- `config/production.toml`
- `config/development.toml`
- `config/testing.toml`

**Don't commit:**
- User-specific configs (`~/.zenb/config.toml`)
- Secrets or API keys

### 3. Environment-Specific Configs

```bash
# Development
./zenb-app --config config/development.toml

# Production
./zenb-app --config config/production.toml

# Testing
ZENB_CONFIG=config/testing.toml cargo test
```

### 4. Document Custom Values

```toml
# config/custom.toml

[fep]
# Increased for high-stress scenarios
process_noise = 0.04  # Default: 0.02

[belief]
# Tuned for elderly users (slower adaptation)
smooth_tau_sec = 6.0  # Default: 4.0
```

### 5. Validate After Changes

```rust
// Load and validate
let config = ZenbConfig::from_file("config/custom.toml")?;
config.validate()?;

// Export to verify
println!("{}", config.to_toml_string()?);
```

---

## Examples

### Example 1: High-Performance Configuration

```toml
# config/high-performance.toml

[fep]
process_noise = 0.015  # Lower noise for stability
lr_base = 0.5          # Conservative learning

[performance]
batch_size = 50        # Large batches
flush_interval_ms = 200  # Infrequent flushes
max_buffer_size = 200
```

### Example 2: Sensitive User Configuration

```toml
# config/sensitive.toml

[safety]
trauma_hard_th = 1.0   # More conservative
trauma_soft_th = 0.5
trauma_decay_default = 0.9  # Slow decay

[belief]
smooth_tau_sec = 8.0   # Heavy smoothing
enter_threshold = 0.7  # Harder to change states
exit_threshold = 0.3
```

### Example 3: Testing Configuration

```toml
# config/test.toml

[belief]
pathway_weights = [1.0, 1.0, 1.0]  # Equal weights
smooth_tau_sec = 0.1   # Minimal smoothing

[performance]
batch_size = 1         # Immediate writes
flush_interval_ms = 1
enable_burst_filter = false  # Deterministic
```

### Example 4: Runtime Configuration Update

```rust
use zenb_core::config::ZenbConfig;

// Load initial config
let mut config = ZenbConfig::from_file("config/default.toml")?;

// Update specific values
config.fep.lr_base = 0.7;
config.belief.smooth_tau_sec = 5.0;

// Validate changes
config.validate()?;

// Save to new file
config.save_to_file("config/custom.toml")?;

// Use in engine
let engine = Engine::new_with_config(6.0, Some(config));
```

---

## Troubleshooting

### Issue: Config file not found

**Error:**
```
Config error: IO error: No such file or directory
```

**Solution:**
```rust
// Check if file exists before loading
use std::path::Path;

let config_path = Path::new("config/default.toml");
if !config_path.exists() {
    eprintln!("Config file not found, using defaults");
    config = ZenbConfig::default();
} else {
    config = ZenbConfig::from_file(config_path)?;
}
```

### Issue: Invalid TOML syntax

**Error:**
```
Config error: TOML parse error: expected `=`, found `:`
```

**Solution:**
- TOML uses `=` not `:`
- Check array syntax: `[1.0, 2.0, 3.0]`
- Check string quotes: `"value"`

### Issue: Validation failed

**Error:**
```
Config error: Config validation error: fep.lr_min must be in [0, lr_max]
```

**Solution:**
```toml
# Wrong
[fep]
lr_min = 0.9
lr_max = 0.8  # lr_min > lr_max!

# Correct
[fep]
lr_min = 0.05
lr_max = 0.8
```

### Issue: Environment variable not applied

**Problem:** Set `ZENB_FEP_PROCESS_NOISE=0.03` but still using 0.02

**Solution:**
```rust
// Use from_file_with_env() not from_file()
let config = ZenbConfig::from_file_with_env("config/default.toml")?;

// Or use load_layered()
let config = ZenbConfig::load_layered(
    Some(Path::new("config/default.toml")),
    None,
)?;
```

### Issue: Config changes not taking effect

**Problem:** Modified config file but behavior unchanged

**Checklist:**
1. ✅ Saved the file?
2. ✅ Restarted the application?
3. ✅ Loading the correct file?
4. ✅ Environment variables overriding?
5. ✅ Validation passing?

**Debug:**
```rust
let config = ZenbConfig::from_file("config/custom.toml")?;
println!("Loaded config:\n{}", config.to_toml_string()?);
```

---

## Migration from Hardcoded Values

### Before (Hardcoded)

```rust
// Old code with hardcoded values
let belief_engine = BeliefEngine::new();  // Uses hardcoded weights
```

### After (External Config)

```rust
// New code with external config
let config = ZenbConfig::from_file("config/production.toml")?;
let belief_engine = BeliefEngine::from_config(&config.belief);
```

### Gradual Migration

```rust
// Support both old and new
let config = match ZenbConfig::from_file("config.toml") {
    Ok(cfg) => cfg,
    Err(_) => {
        eprintln!("Using default configuration");
        ZenbConfig::default()  // Falls back to same values as before
    }
};
```

---

## API Reference

### ZenbConfig

```rust
impl ZenbConfig {
    // Load from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError>;
    
    // Load from file with env overrides
    pub fn from_file_with_env<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError>;
    
    // Load with layering
    pub fn load_layered(
        default_path: Option<&Path>,
        user_path: Option<&Path>,
    ) -> Result<Self, ConfigError>;
    
    // Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError>;
    
    // Export to TOML string
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error>;
    
    // Save to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError>;
}
```

### ConfigError

```rust
pub enum ConfigError {
    Io(std::io::Error),
    TomlParse(toml::de::Error),
    Validation(String),
    EnvVar(std::env::VarError),
}
```

---

## Summary

ZenB's configuration system provides:

✅ **Flexible**: TOML files + environment variables  
✅ **Layered**: Multiple config sources with clear priority  
✅ **Validated**: Automatic validation on load  
✅ **Type-safe**: Rust structs with serde  
✅ **Documented**: Comprehensive inline comments  
✅ **Tested**: Pre-configured files for all environments  

**Quick Start:**
```rust
let config = ZenbConfig::load_layered(
    Some(Path::new("config/default.toml")),
    Some(Path::new("~/.zenb/config.toml")),
)?;
let engine = Engine::new_with_config(6.0, Some(config));
```

---

**Last Updated:** January 3, 2026  
**Version:** 2.0 (External Configuration System)
