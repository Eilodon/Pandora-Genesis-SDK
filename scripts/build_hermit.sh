#!/bin/bash
# Build AGOLOS for Hermit unikernel
#
# VAJRA-VOID: Unikernel Deployment
# Hermit provides <5ms boot time and minimal attack surface.
#
# Prerequisites:
#   1. Rust nightly: rustup install nightly
#   2. Hermit target: rustup target add x86_64-unknown-hermit
#   3. uhyve hypervisor: cargo install uhyve
#
# Usage:
#   ./scripts/build_hermit.sh         # Build
#   ./scripts/build_hermit.sh run     # Build and run in uhyve

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== AGOLOS Hermit Unikernel Build ==="
echo ""

# Check prerequisites
if ! rustup show | grep -q "nightly"; then
    echo "ERROR: Rust nightly not installed"
    echo "Run: rustup install nightly"
    exit 1
fi

# Add target if needed
if ! rustup target list --installed | grep -q "x86_64-unknown-hermit"; then
    echo "Installing Hermit target..."
    rustup target add x86_64-unknown-hermit --toolchain nightly
fi

# Build
echo "Building for x86_64-unknown-hermit..."
cargo +nightly build \
    --target x86_64-unknown-hermit \
    --release \
    -p zenb-core \
    --no-default-features

echo ""
echo "Build complete!"
echo "Binary: target/x86_64-unknown-hermit/release/zenb-core"

# Run if requested
if [ "$1" = "run" ]; then
    echo ""
    echo "Running in uhyve hypervisor..."
    
    if ! command -v uhyve &> /dev/null; then
        echo "ERROR: uhyve not installed"
        echo "Run: cargo install uhyve"
        exit 1
    fi
    
    uhyve target/x86_64-unknown-hermit/release/zenb-core
fi
