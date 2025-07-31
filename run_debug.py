#!/usr/bin/env python3
"""
Simple wrapper to capture debug output properly.
"""

import subprocess
import sys

def run_debug():
    """Run the debug script and capture output."""
    try:
        # Run the debug script
        result = subprocess.run([
            sys.executable, 'debug_similarity_comprehensive.py'
        ], capture_output=True, text=True, timeout=60)
        
        # Print stdout (the actual debug info)
        if result.stdout:
            print("=== DEBUG OUTPUT ===")
            print(result.stdout)
        
        # Show stderr (warnings) separately
        if result.stderr and "resource_tracker" not in result.stderr:
            print("=== ERRORS ===")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Debug script timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run debug script: {e}")
        return False

if __name__ == "__main__":
    success = run_debug()
    if not success:
        print("Debug script failed to complete successfully")