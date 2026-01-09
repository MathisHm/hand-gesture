#!/bin/bash

################################################################################
# Automated Benchmark Suite Runner
# 
# Purpose: Run complete benchmark suite with proper preparation and cleanup
#
# Usage: sudo ./run_benchmark_suite.sh [output_csv]
#
# This script:
# 1. Prepares the system for benchmarking
# 2. Runs the benchmark
# 3. Restores normal system settings
# 4. Generates a summary report
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Configuration
OUTPUT_CSV="${1:-benchmark_results.csv}"
BENCHMARK_SCRIPT="benchmark.py"
BENCHMARK_VIDEO="benchmark_video.mp4"

# Validate required files exist
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo -e "${RED}Error: $BENCHMARK_SCRIPT not found${NC}"
    exit 1
fi

if [ ! -f "$BENCHMARK_VIDEO" ]; then
    echo -e "${RED}Error: $BENCHMARK_VIDEO not found${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Automated Benchmark Suite Runner     ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Benchmark script: ${BLUE}$BENCHMARK_SCRIPT${NC}"
echo -e "  Video source: ${BLUE}$BENCHMARK_VIDEO${NC}"
echo -e "  Output CSV: ${BLUE}$OUTPUT_CSV${NC}"
echo ""

# Trap to ensure cleanup happens even if script is interrupted
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    if [ -f "./restore_system.sh" ]; then
        bash ./restore_system.sh
    fi
}
trap cleanup EXIT INT TERM

################################################################################
# Step 1: System Preparation
################################################################################
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Step 1: Preparing system for benchmark${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

if [ -f "./prepare_benchmark.sh" ]; then
    bash ./prepare_benchmark.sh
else
    echo -e "${RED}Error: prepare_benchmark.sh not found${NC}"
    exit 1
fi

# Wait for system to stabilize
echo ""
echo -e "${YELLOW}Waiting 5 seconds for system to stabilize...${NC}"
sleep 5

################################################################################
# Step 2: Run Benchmark
################################################################################
echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Step 2: Running benchmark${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

START_TIME=$(date +%s)

# Run benchmark with high priority
echo -e "${BLUE}Starting benchmark with elevated priority...${NC}"
nice -n -10 python3 "$BENCHMARK_SCRIPT" 2>&1 | tee benchmark_run.log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}✓ Benchmark completed in ${DURATION} seconds${NC}"

################################################################################
# Step 3: System Restoration
################################################################################
echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Step 3: Restoring system settings${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

if [ -f "./restore_system.sh" ]; then
    bash ./restore_system.sh
fi

################################################################################
# Step 4: Generate Summary Report
################################################################################
echo ""
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Step 4: Generating summary report${NC}"
echo -e "${GREEN}═══════════════════════════════════════${NC}"
echo ""

REPORT_FILE="benchmark_report_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "Benchmark Summary Report"
    echo "Generated: $(date)"
    echo "========================================"
    echo ""
    echo "Benchmark Duration: ${DURATION} seconds"
    echo ""
    
    if [ -f "$OUTPUT_CSV" ]; then
        echo "Latest Results from $OUTPUT_CSV:"
        echo "----------------------------------------"
        # Show header
        head -n 1 "$OUTPUT_CSV"
        # Show last result
        tail -n 1 "$OUTPUT_CSV"
        echo ""
    fi
    
    if [ -f "benchmark_system_info.txt" ]; then
        echo ""
        echo "System Information:"
        echo "----------------------------------------"
        cat benchmark_system_info.txt
    fi
    
    if [ -f "benchmark_run.log" ]; then
        echo ""
        echo "Benchmark Log:"
        echo "----------------------------------------"
        cat benchmark_run.log
    fi
} > "$REPORT_FILE"

echo -e "${GREEN}✓ Report saved to: $REPORT_FILE${NC}"

################################################################################
# Final Summary
################################################################################
echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Benchmark Suite Complete!            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "Results:"
echo -e "  ${GREEN}✓${NC} CSV results: $OUTPUT_CSV"
echo -e "  ${GREEN}✓${NC} Full report: $REPORT_FILE"
echo -e "  ${GREEN}✓${NC} System info: benchmark_system_info.txt"
echo -e "  ${GREEN}✓${NC} Run log: benchmark_run.log"
echo ""

# Display quick stats if CSV exists
if [ -f "$OUTPUT_CSV" ]; then
    echo -e "${BLUE}Quick Stats (latest run):${NC}"
    echo "----------------------------------------"
    
    # Extract last line and parse
    LAST_LINE=$(tail -n 1 "$OUTPUT_CSV")
    HEADER=$(head -n 1 "$OUTPUT_CSV")
    
    # Simple column-based display
    paste <(echo "$HEADER" | tr ',' '\n') <(echo "$LAST_LINE" | tr ',' '\n') | column -t -s $'\t'
    echo ""
fi

echo -e "${GREEN}System has been restored to normal operation.${NC}"
echo ""
