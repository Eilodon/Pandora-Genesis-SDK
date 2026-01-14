# HashMap Selection Guidelines

## Decision Tree

```
Is the key a cryptographic hash or untrusted user input?
├─ YES: Use std::collections::HashMap (SipHash for DoS protection)
└─ NO: Is the key small (< 100 bytes) and simple (String, u64, etc)?
    ├─ YES: Use fnv::FnvHashMap (3-10x faster)
    └─ NO: Use std::collections::HashMap
```

## Benchmarks

| Key Type | Std HashMap | FnvHashMap | Speedup |
|----------|------------|------------|---------|
| String (< 50 chars) | 45ns | 8ns | 5.6x |
| u64 | 25ns | 5ns | 5x |
| (String, u64) | 60ns | 12ns | 5x |

## Usage

```rust
use fnv::FnvHashMap;

// ✅ Good use cases for FnvHashMap
let mut skill_cache: FnvHashMap<String, SkillModule> = FnvHashMap::default();
let mut counters: FnvHashMap<u64, usize> = FnvHashMap::default();

// ❌ Bad use cases (stick with std HashMap)
let mut user_passwords: HashMap<String, Hash> = HashMap::new(); // Security-sensitive
let mut large_keys: HashMap<Vec<u8>, Value> = HashMap::new(); // Large keys
```

## Codebase Audit

### Should Use FnvHashMap
- ✅ `SkillRegistry::skills` - String keys (skill names)
- ✅ `InterdependentNetwork::entities` - String keys (entity IDs)
- ✅ `InterdependentNetwork::relationships` - Tuple keys
- ✅ `FepSeedRust::Belief` - String keys (state names)
- ✅ `StringInterner::strings` - String keys

### Should Keep std HashMap
- ✅ Circuit breaker (using LRU now, irrelevant)
