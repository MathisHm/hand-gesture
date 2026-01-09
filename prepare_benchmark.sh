#!/bin/bash

################################################################################
# Raspberry Pi Benchmark Preparation Script
# 
# Purpose: Configure the Raspberry Pi for consistent, reproducible benchmarking
# by minimizing system interference and setting optimal performance settings.
#
# Usage: sudo ./prepare_benchmark.sh
#
# This script should be run before executing benchmark.py to ensure:
# - Consistent CPU performance (no throttling/frequency scaling)
# - Minimal background processes
# - Predictable thermal conditions
# - Reproducible system state
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

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Raspberry Pi Benchmark Preparation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

################################################################################
# 1. System Information Collection
################################################################################
echo -e "${GREEN}[1/9] Collecting system information...${NC}"

# Create benchmark info file
BENCHMARK_INFO="benchmark_system_info.txt"
{
    echo "Benchmark System Information"
    echo "Generated: $(date)"
    echo "======================================"
    echo ""
    echo "Hardware:"
    cat /proc/cpuinfo | grep -E "Model|Hardware|Revision" || echo "CPU info not available"
    echo ""
    echo "Memory:"
    free -h
    echo ""
    echo "Kernel:"
    uname -a
    echo ""
    echo "Temperature (before):"
    vcgencmd measure_temp 2>/dev/null || echo "Temperature monitoring not available"
    echo ""
    echo "CPU Frequency (before):"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "Frequency info not available"
    echo ""
} > "$BENCHMARK_INFO"

echo -e "  ${GREEN}✓${NC} System info saved to $BENCHMARK_INFO"

################################################################################
# 2. Stop Unnecessary Services
################################################################################
echo -e "${GREEN}[2/9] Stopping unnecessary background services...${NC}"

# List of services to stop (non-critical for benchmarking)
SERVICES_TO_STOP=(
    "bluetooth"
    "cups"
    "cups-browsed"
    "avahi-daemon"
    "triggerhappy"
    "ModemManager"
)

STOPPED_SERVICES=()
for service in "${SERVICES_TO_STOP[@]}"; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        systemctl stop "$service" 2>/dev/null && STOPPED_SERVICES+=("$service")
        echo -e "  ${GREEN}✓${NC} Stopped $service"
    fi
done

# Save list of stopped services for restoration
echo "${STOPPED_SERVICES[@]}" > /tmp/benchmark_stopped_services.txt

################################################################################
# 3. Set CPU Governor to Performance Mode
################################################################################
echo -e "${GREEN}[3/9] Setting CPU governor to performance mode...${NC}"

# Save current governor for restoration
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > /tmp/benchmark_original_governor.txt 2>/dev/null || echo "ondemand" > /tmp/benchmark_original_governor.txt

# Set all CPUs to performance mode
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        echo "performance" > "$cpu/cpufreq/scaling_governor"
        echo -e "  ${GREEN}✓${NC} Set $(basename $cpu) to performance mode"
    fi
done

################################################################################
# 4. Disable CPU Frequency Scaling
################################################################################
echo -e "${GREEN}[4/9] Locking CPU frequency to maximum...${NC}"

# Get max frequency and set it as min frequency (locks frequency)
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_max_freq" ]; then
        MAX_FREQ=$(cat "$cpu/cpufreq/scaling_max_freq")
        echo "$MAX_FREQ" > "$cpu/cpufreq/scaling_min_freq" 2>/dev/null || true
        CURRENT_FREQ=$(cat "$cpu/cpufreq/scaling_cur_freq")
        echo -e "  ${GREEN}✓${NC} $(basename $cpu): locked at ${CURRENT_FREQ} Hz (max: ${MAX_FREQ} Hz)"
    fi
done

################################################################################
# 5. Disable Swap (reduces I/O interference)
################################################################################
echo -e "${GREEN}[5/9] Disabling swap...${NC}"

# Check if swap is enabled
if [ "$(swapon --show | wc -l)" -gt 0 ]; then
    swapoff -a
    echo -e "  ${GREEN}✓${NC} Swap disabled"
else
    echo -e "  ${YELLOW}ℹ${NC} Swap already disabled"
fi

################################################################################
# 6. Clear System Caches
################################################################################
echo -e "${GREEN}[6/9] Clearing system caches...${NC}"

sync
echo 3 > /proc/sys/vm/drop_caches
echo -e "  ${GREEN}✓${NC} Page cache, dentries, and inodes cleared"

################################################################################
# 7. Set Process Priority
################################################################################
echo -e "${GREEN}[7/9] Configuring process scheduling...${NC}"

# Disable automatic nice adjustment for background processes
echo 0 > /proc/sys/kernel/sched_autogroup_enabled 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Disabled automatic process grouping"

################################################################################
# 8. Thermal Management
################################################################################
echo -e "${GREEN}[8/9] Checking thermal conditions...${NC}"

TEMP=$(vcgencmd measure_temp 2>/dev/null | grep -oP '\d+\.\d+' || echo "0")
TEMP_INT=${TEMP%.*}

echo -e "  Current temperature: ${TEMP}°C"

if [ "$TEMP_INT" -gt 70 ]; then
    echo -e "  ${YELLOW}⚠${NC}  WARNING: Temperature is high (${TEMP}°C)"
    echo -e "  ${YELLOW}⚠${NC}  Consider waiting for the system to cool down"
    echo -e "  ${YELLOW}⚠${NC}  High temperatures may cause thermal throttling and inconsistent results"
    read -p "  Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Benchmark preparation cancelled${NC}"
        exit 1
    fi
elif [ "$TEMP_INT" -gt 60 ]; then
    echo -e "  ${YELLOW}ℹ${NC}  Temperature is moderate. Results should be consistent."
else
    echo -e "  ${GREEN}✓${NC} Temperature is good for benchmarking"
fi

################################################################################
# 9. Network and I/O Optimization
################################################################################
echo -e "${GREEN}[9/9] Optimizing I/O and network...${NC}"

# Disable WiFi power management (can cause latency spikes)
if command -v iwconfig &> /dev/null; then
    for iface in $(ls /sys/class/net/ | grep -E '^wlan|^wlp'); do
        iwconfig "$iface" power off 2>/dev/null && echo -e "  ${GREEN}✓${NC} Disabled power management for $iface" || true
    done
fi

# Set I/O scheduler to 'noop' or 'none' for reduced latency
for device in /sys/block/mmcblk*/queue/scheduler; do
    if [ -f "$device" ]; then
        # Try 'none' first (newer kernels), fall back to 'noop'
        if grep -q "none" "$device"; then
            echo "none" > "$device" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Set I/O scheduler to 'none'"
        elif grep -q "noop" "$device"; then
            echo "noop" > "$device" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Set I/O scheduler to 'noop'"
        fi
    fi
done

################################################################################
# Final Status
################################################################################
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Benchmark preparation complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "System is now optimized for benchmarking:"
echo -e "  ${GREEN}✓${NC} CPU governor: performance"
echo -e "  ${GREEN}✓${NC} CPU frequency: locked to maximum"
echo -e "  ${GREEN}✓${NC} Background services: minimized"
echo -e "  ${GREEN}✓${NC} Swap: disabled"
echo -e "  ${GREEN}✓${NC} Caches: cleared"
echo -e "  ${GREEN}✓${NC} I/O scheduler: optimized"
echo ""

# Final system state
{
    echo ""
    echo "System State After Preparation:"
    echo "======================================"
    echo "Temperature (after):"
    vcgencmd measure_temp 2>/dev/null || echo "Temperature monitoring not available"
    echo ""
    echo "CPU Frequency (after):"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "Frequency info not available"
    echo ""
    echo "CPU Governor:"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "Governor info not available"
    echo ""
    echo "Memory (after cleanup):"
    free -h
    echo ""
} >> "$BENCHMARK_INFO"

echo -e "${YELLOW}Important:${NC}"
echo -e "  • Run your benchmark immediately for best results"
echo -e "  • Avoid running other programs during benchmarking"
echo -e "  • System info saved to: $BENCHMARK_INFO"
echo -e "  • Run ${BLUE}./restore_system.sh${NC} after benchmarking to restore normal settings"
echo ""
echo -e "${GREEN}Ready to run: python3 benchmark.py${NC}"
echo ""
