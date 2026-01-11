#!/bin/bash

echo "🔍 Auditing panic! calls in library code..."
echo ""

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR" || exit 1

# Find panic-like patterns excluding tests/examples/benches
find */src -type f -name "*.rs" \
  ! -path "*/tests/*" \
  ! -path "*/examples/*" \
  ! -path "*/benches/*" \
  -exec grep -Hn "panic!\|unwrap()\|expect(" {} \; \
  > panic_audit.txt || true

echo "📊 Results saved to panic_audit.txt"
echo ""
echo "Summary:"
grep -c "panic!" panic_audit.txt || echo "0 panic! calls"
grep -c "unwrap()" panic_audit.txt || echo "0 unwrap() calls"
grep -c "expect(" panic_audit.txt || echo "0 expect() calls"


