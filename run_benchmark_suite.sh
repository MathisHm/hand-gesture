#!/bin/bash

################################################################################
# Automated Benchmark Suite Runner (minimal output)
# Usage: sudo ./run_benchmark_suite.sh [output_csv]
################################################################################

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Configuration
OUTPUT_CSV="${1:-benchmark_results.csv}"
BENCHMARK_SCRIPT="benchmark.py"
BENCHMARK_VIDEO="benchmark_video.mp4"

# Validate required files
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo "Error: $BENCHMARK_SCRIPT not found"
    exit 1
fi

if [ ! -f "$BENCHMARK_VIDEO" ]; then
    echo "Error: $BENCHMARK_VIDEO not found"
    exit 1
fi

# Cleanup function
cleanup() {
    if [ -f "./restore_system.sh" ]; then
        bash ./restore_system.sh
    fi
}
trap cleanup EXIT INT TERM

# Prepare system
if [ -f "./prepare_benchmark.sh" ]; then
    bash ./prepare_benchmark.sh
else
    echo "Error: prepare_benchmark.sh not found"
    exit 1
fi

# Wait for system to stabilize
sleep 5

# Run benchmark
nice -n -10 python3.9 "$BENCHMARK_SCRIPT"

echo "Benchmark complete. Results in $OUTPUT_CSV"
