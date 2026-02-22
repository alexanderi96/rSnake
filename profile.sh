#!/bin/bash
# Profile snake-rs training and generate flamegraph
# Usage: ./profile.sh [duration_seconds]
# Default duration: 30 seconds

set -e

DURATION=${1:-30}
echo "🔥 Profiling snake-rs for ${DURATION} seconds..."
echo "   Compile with profiling feature..."

# Build with profiling feature
cargo build --release --features profiling

echo ""
echo "🚀 Starting profiled run for ${DURATION}s..."
echo "   Press 'T' in the game to enable turbo mode (no rendering) for best profiling"
echo ""

# Run with timeout
timeout ${DURATION} ./target/release/snake-rs --config config.toml 2>&1 || true

echo ""
echo "✅ Profiling complete!"
echo ""

# Find the latest profile file
LATEST_PROFILE=$(ls -t profile-*.svg 2>/dev/null | head -1)
if [ -n "$LATEST_PROFILE" ]; then
    echo "📊 Flamegraph generated: $LATEST_PROFILE"
    echo ""
    echo "   Open with:"
    echo "     firefox $LATEST_PROFILE"
    echo "     chrome $LATEST_PROFILE"
    echo "     xdg-open $LATEST_PROFILE"
    echo ""
    echo "   The flamegraph shows:"
    echo "     - Width = time spent in function"
    echo "     - Height = call stack depth"
    echo "     - Colors = random (just for visual separation)"
    echo ""
    echo "   Look for wide boxes to find bottlenecks!"
else
    echo "⚠️  No profile file generated (did the program run long enough?)"
fi
