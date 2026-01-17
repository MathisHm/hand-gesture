#!/bin/bash

################################################################################
# Raspberry Pi Benchmark Preparation Script
# Optimizes system for consistent benchmarking (minimal output)
################################################################################

set -e

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Stop unnecessary services
for service in bluetooth cups cups-browsed avahi-daemon triggerhappy ModemManager; do
    systemctl stop "$service" 2>/dev/null || true
done

# Save stopped services for restoration
STOPPED_SERVICES=()
for service in bluetooth cups cups-browsed avahi-daemon triggerhappy ModemManager; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        STOPPED_SERVICES+=("$service")
    fi
done
echo "${STOPPED_SERVICES[@]}" > /tmp/benchmark_stopped_services.txt

# Save current governor for restoration
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > /tmp/benchmark_original_governor.txt 2>/dev/null || echo "ondemand" > /tmp/benchmark_original_governor.txt

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        echo "performance" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
    fi
done

# Lock CPU frequency to maximum
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -f "$cpu/cpufreq/scaling_max_freq" ]; then
        MAX_FREQ=$(cat "$cpu/cpufreq/scaling_max_freq")
        echo "$MAX_FREQ" > "$cpu/cpufreq/scaling_min_freq" 2>/dev/null || true
    fi
done

# Disable swap
swapoff -a 2>/dev/null || true

# Clear system caches
sync
echo 3 > /proc/sys/vm/drop_caches

# Disable automatic process grouping
echo 0 > /proc/sys/kernel/sched_autogroup_enabled 2>/dev/null || true

# Disable WiFi power management
if command -v iwconfig &> /dev/null; then
    for iface in $(ls /sys/class/net/ | grep -E '^wlan|^wlp'); do
        iwconfig "$iface" power off 2>/dev/null || true
    done
fi

# Set I/O scheduler to 'none' or 'noop'
for device in /sys/block/mmcblk*/queue/scheduler; do
    if [ -f "$device" ]; then
        if grep -q "none" "$device"; then
            echo "none" > "$device" 2>/dev/null || true
        elif grep -q "noop" "$device"; then
            echo "noop" > "$device" 2>/dev/null || true
        fi
    fi
done

echo "System ready for benchmarking"
