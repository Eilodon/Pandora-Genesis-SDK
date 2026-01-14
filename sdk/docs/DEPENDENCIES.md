# Dependency Management

## Disabled Dependencies

### Machine Learning Crates
- `candle-core`, `candle-nn`: Disabled due to CUDA/MSRV requirements
  - Alternative: Use pure Rust implementations or external Python bridge
  - Status: Waiting for candle stable release
  
### Vector Databases
- `lance`: Disabled due to Arrow dependency conflicts
  - Alternative: Use Qdrant client or tantivy for search
  
- `tantivy`: Temporarily disabled
  - Alternative: Simple regex-based search for now

## Version Pinning

### nom 7.1.3
All parsing crates must use nom 7.x. The old meval crate (nom 1.x) has been replaced
with a custom arithmetic evaluator.

### tokio 1.47
Unified across all workspace crates for async runtime consistency.

