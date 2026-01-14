# Async Function Audit

## Rules
1. Function should ONLY be async if it:
   - Calls another async function (.await)
   - Performs I/O (network, disk, DB)
   - Needs to yield control (sleep, timeout)

2. CPU-bound work should use `spawn_blocking`

3. Pure computation should be synchronous

## Audit Results

### ✅ Should Stay Async
- `Orchestrator::process_request()` - calls async execute()
- `SkillModule::execute()` - may do I/O in future
- `RupaSkandha::process_event()` - trait requirement, potential I/O at ingress

### ❌ Converted To Sync
- `VedanaSkandha::feel()` - pure computation
- `SannaSkandha::perceive()` - pure computation  
- `SankharaSkandha::form_intent()` - pure computation
- `VinnanaSkandha::synthesize()` - pure computation
