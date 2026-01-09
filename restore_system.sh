#!/bin/bash

################################################################################
# Raspberry Pi System Restoration Script
# 
# Purpose: Restore normal system settings after benchmarking
#
# Usage: sudo ./restore_system.sh
#
# This script reverses the changes made by prepare_benchmark.sh
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
echo -e "${BLUE}Restoring Normal System Settings${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

################################################################################
# 1. Restore CPU Governor
################################################################################
echo -e "${GREEN}[1/4] Restoring CPU governor...${NC}"

if [ -f /tmp/benchmark_original_governor.txt ]; then
    ORIGINAL_GOVERNOR=$(cat /tmp/benchmark_original_governor.txt)
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            echo "$ORIGINAL_GOVERNOR" > "$cpu/cpufreq/scaling_governor"
            echo -e "  ${GREEN}✓${NC} Restored $(basename $cpu) to $ORIGINAL_GOVERNOR mode"
        fi
    done
    rm /tmp/benchmark_original_governor.txt
else
    echo -e "  ${YELLOW}ℹ${NC} No saved governor state found, setting to 'ondemand'"
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            echo "ondemand" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
        fi
    done
fi

################################################################################
# 2. Restore CPU Frequency Scaling
################################################################################
echo -e "${GREEN}[2/4] Restoring CPU frequency scaling...${NC}"

# Reset min frequency to allow dynamic scaling
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/cpuinfo_min_freq" ] && [ -f "$cpu/cpufreq/scaling_min_freq" ]; then
        MIN_FREQ=$(cat "$cpu/cpufreq/cpuinfo_min_freq")
        echo "$MIN_FREQ" > "$cpu/cpufreq/scaling_min_freq" 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Restored frequency scaling for $(basename $cpu)"
    fi
done

################################################################################
# 3. Restart Services
################################################################################
echo -e "${GREEN}[3/4] Restarting stopped services...${NC}"

if [ -f /tmp/benchmark_stopped_services.txt ]; then
    SERVICES=$(cat /tmp/benchmark_stopped_services.txt)
    for service in $SERVICES; do
        systemctl start "$service" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Restarted $service" || echo -e "  ${YELLOW}⚠${NC}  Could not restart $service"
    done
    rm /tmp/benchmark_stopped_services.txt
else
    echo -e "  ${YELLOW}ℹ${NC} No stopped services to restore"
fi

################################################################################
# 4. Re-enable Swap
################################################################################
echo -e "${GREEN}[4/4] Re-enabling swap...${NC}"

swapon -a 2>/dev/null && echo -e "  ${GREEN}✓${NC} Swap re-enabled" || echo -e "  ${YELLOW}ℹ${NC} No swap to enable"

################################################################################
# Final Status
################################################################################
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}System restoration complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Normal system settings have been restored:"
echo -e "  ${GREEN}✓${NC} CPU governor: restored to dynamic mode"
echo -e "  ${GREEN}✓${NC} CPU frequency scaling: re-enabled"
echo -e "  ${GREEN}✓${NC} Background services: restarted"
echo -e "  ${GREEN}✓${NC} Swap: re-enabled"
echo ""
