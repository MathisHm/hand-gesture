#!/bin/bash

################################################################################
# System Restoration Script (minimal output)
# Restores normal system settings after benchmarking
################################################################################

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Restore CPU Governor
if [ -f /tmp/benchmark_original_governor.txt ]; then
    ORIGINAL_GOVERNOR=$(cat /tmp/benchmark_original_governor.txt)
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            echo "$ORIGINAL_GOVERNOR" > "$cpu/cpufreq/scaling_governor"
        fi
    done
    rm /tmp/benchmark_original_governor.txt
else
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            echo "ondemand" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
        fi
    done
fi

# Restore CPU Frequency Scaling
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/cpuinfo_min_freq" ] && [ -f "$cpu/cpufreq/scaling_min_freq" ]; then
        MIN_FREQ=$(cat "$cpu/cpufreq/cpuinfo_min_freq")
        echo "$MIN_FREQ" > "$cpu/cpufreq/scaling_min_freq" 2>/dev/null || true
    fi
done

# Restart Services
if [ -f /tmp/benchmark_stopped_services.txt ]; then
    SERVICES=$(cat /tmp/benchmark_stopped_services.txt)
    for service in $SERVICES; do
        systemctl start "$service" 2>/dev/null || true
    done
    rm /tmp/benchmark_stopped_services.txt
fi

# Re-enable Swap
swapon -a 2>/dev/null || true

echo "System restored to normal operation"
