"""
Temperature monitoring utility for Raspberry Pi benchmarking.

Uses vcgencmd to monitor CPU temperature and detect thermal throttling.

Usage:
    monitor = TemperatureMonitor()
    if monitor.is_available():
        temp = monitor.read()
        throttle_status = monitor.check_throttling()
"""

import subprocess


class TemperatureMonitor:
    """
    Temperature monitoring for Raspberry Pi using vcgencmd.
    """
    
    def __init__(self):
        """Initialize temperature monitor."""
        self.available = self._check_availability()
    
    def _check_availability(self):
        """Check if vcgencmd is available."""
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=1
            )
            return result.returncode == 0
        except (FileNotFoundError, Exception):
            return False
    
    def is_available(self):
        """Check if temperature monitoring is available."""
        return self.available
    
    def read(self):
        """
        Read current CPU temperature.
        
        Returns:
            float: Temperature in Celsius, or None if unavailable
        """
        if not self.available:
            return None
        
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                # Output format: "temp=42.8'C"
                temp_str = result.stdout.strip()
                temp = float(temp_str.split('=')[1].split("'")[0])
                return temp
            
        except (Exception, ValueError, IndexError):
            pass
        
        return None
    
    def check_throttling(self):
        """
        Check if the Pi is being throttled.
        
        Returns:
            dict: Throttling status with keys:
                - under_voltage: bool
                - freq_capped: bool
                - currently_throttled: bool
                - temp_limit: bool
                - raw_value: hex string
        """
        if not self.available:
            return None
        
        try:
            result = subprocess.run(
                ['vcgencmd', 'get_throttled'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                # Output format: "throttled=0x0" or "throttled=0x50000"
                throttled_hex = result.stdout.strip().split('=')[1]
                throttled_int = int(throttled_hex, 16)
                
                return {
                    'under_voltage': bool(throttled_int & 0x1),
                    'freq_capped': bool(throttled_int & 0x2),
                    'currently_throttled': bool(throttled_int & 0x4),
                    'temp_limit': bool(throttled_int & 0x8),
                    'raw_value': throttled_hex
                }
        
        except (Exception, ValueError, IndexError):
            pass
        
        return None


def test_temperature_monitor():
    """Test temperature monitoring functionality."""
    print("Testing Temperature Monitor")
    print("=" * 50)
    
    temp_mon = TemperatureMonitor()
    
    if temp_mon.is_available():
        print("✓ vcgencmd available")
        temp = temp_mon.read()
        print(f"  Temperature: {temp:.1f}°C")
        
        throttle = temp_mon.check_throttling()
        if throttle:
            print(f"  Throttling status:")
            print(f"    - Under voltage: {throttle['under_voltage']}")
            print(f"    - Frequency capped: {throttle['freq_capped']}")
            print(f"    - Currently throttled: {throttle['currently_throttled']}")
            print(f"    - Temperature limit: {throttle['temp_limit']}")
    else:
        print("✗ vcgencmd not available")
        print("  (This is expected if not running on Raspberry Pi)")
    
    print("=" * 50)


if __name__ == "__main__":
    test_temperature_monitor()
