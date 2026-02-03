#!/usr/bin/env python3
"""
SpaceMouse Test Script (Fixed Version)

This script tests if a 3Dconnexion SpaceMouse is properly connected and accessible.
It's designed to handle different versions of pyspacemouse and potential API inconsistencies.

Usage:
    python FixedTestSpaceMouse.py

Press Ctrl+C to exit the program.
"""

import time
import sys
import traceback
import threading

# Try importing the pyspacemouse library
try:
    import pyspacemouse
    print("Successfully imported pyspacemouse library")
    
    # Print the library version if available
    try:
        version = getattr(pyspacemouse, "__version__", "unknown")
        print(f"pyspacemouse version: {version}")
    except:
        print("Could not determine pyspacemouse version")
except ImportError as e:
    print(f"Failed to import pyspacemouse: {e}")
    print("Try installing it with: pip install pyspacemouse")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error importing pyspacemouse: {e}")
    traceback.print_exc()
    sys.exit(1)

def print_device_info(device):
    """Print detailed information about the SpaceMouse device."""
    print("\n===== DEVICE INFORMATION =====")
    
    # Check if device is a string or an object
    if isinstance(device, str):
        print(f"Device information: {device}")
        return
    
    # Try to access device attributes safely
    try:
        attrs = [
            ("Manufacturer", "manufacturer_string", "Unknown"),
            ("Product", "product_string", "Unknown"),
            ("Vendor ID", "vendor_id", "Unknown"),
            ("Product ID", "product_id", "Unknown"),
            ("Serial Number", "serial_number", "Unknown"),
            ("Release Number", "release_number", "Unknown"),
            ("Interface Number", "interface_number", "Unknown")
        ]
        
        for label, attr, default in attrs:
            value = getattr(device, attr, default)
            if attr in ["vendor_id", "product_id"] and value != "Unknown":
                print(f"{label}: 0x{value:04x}")
            else:
                print(f"{label}: {value}")
    except Exception as e:
        print(f"Error accessing device attributes: {e}")
        print(f"Raw device data: {device}")
        
    print("===============================\n")

def test_device_connection():
    """Test if the SpaceMouse device is connected and accessible."""
    print("Attempting to detect SpaceMouse devices...")
    
    try:
        # List all available devices
        try:
            devices = pyspacemouse.list_devices()
            print(f"list_devices() returned: {devices}")
            
            if not devices:
                print("No SpaceMouse devices found!")
                return None
            
            print(f"Found {len(devices)} devices.")
            
            # Safely print device information
            for i, device in enumerate(devices):
                print(f"Device {i+1}:")
                if isinstance(device, str):
                    print(f"  {device}")
                else:
                    try:
                        vendor_id = getattr(device, "vendor_id", "Unknown")
                        product_id = getattr(device, "product_id", "Unknown")
                        product_string = getattr(device, "product_string", "Unknown Device")
                        
                        if vendor_id != "Unknown" and product_id != "Unknown":
                            print(f"  {product_string} (Vendor ID: 0x{vendor_id:04x}, Product ID: 0x{product_id:04x})")
                        else:
                            print(f"  {product_string}")
                    except Exception as e:
                        print(f"  Error printing device {i+1} info: {e}")
                        print(f"  Raw device data: {device}")
            
            # Use the first device found
            return devices[0]
            
        except AttributeError:
            # Alternative approach if list_devices doesn't work as expected
            print("list_devices() method not working as expected, trying open() directly...")
            if pyspacemouse.open():
                print("Successfully opened a SpaceMouse device directly")
                return "SpaceMouse Device"
            else:
                print("Failed to open any SpaceMouse device directly")
                return None
    
    except Exception as e:
        print(f"Error detecting devices: {e}")
        traceback.print_exc()
        return None

def open_device(device):
    """Try to open the SpaceMouse device."""
    print("Attempting to open the SpaceMouse...")
    try:
        # Check if device is already open
        if hasattr(pyspacemouse, "is_open") and pyspacemouse.is_open():
            print("Device is already open")
            return True
            
        # Try to open the device
        if isinstance(device, str):
            # If device is a string, just try to open any device
            if pyspacemouse.open(callback=None):
                print("Successfully opened a SpaceMouse device")
                return True
        else:
            # Try to open the specific device
            try:
                if pyspacemouse.open(callback=None, device=device):
                    print("Successfully opened the SpaceMouse device")
                    return True
            except TypeError:
                # If device parameter isn't supported, try without it
                if pyspacemouse.open(callback=None):
                    print("Successfully opened a SpaceMouse device")
                    return True
        
        print("Failed to open the SpaceMouse device")
        print("Check if another program is already using it")
        return False
    
    except Exception as e:
        print(f"Error opening device: {e}")
        traceback.print_exc()
        return False

def safe_read():
    """Safely read from the device, handling potential API differences."""
    try:
        state = pyspacemouse.read()
        return state
    except Exception as e:
        print(f"Error reading from device: {e}")
        return None

def monitor_button_state(stop_event):
    """Monitor and print button state changes in a separate thread."""
    last_buttons = None
    
    while not stop_event.is_set():
        try:
            state = safe_read()
            if state and hasattr(state, "buttons") and state.buttons is not None:
                current_buttons = state.buttons
                if last_buttons != current_buttons:
                    buttons_pressed = [i for i, pressed in enumerate(current_buttons) if pressed]
                    if buttons_pressed:
                        print(f"Buttons pressed: {buttons_pressed}")
                    last_buttons = current_buttons.copy() if hasattr(current_buttons, "copy") else current_buttons
            time.sleep(0.05)  # Small sleep to prevent 100% CPU usage
        except Exception as e:
            print(f"Error in button monitoring thread: {e}")
            time.sleep(1)  # Wait a bit longer on error

def main():
    """Main function to test the SpaceMouse."""
    print("SpaceMouse Test Script (Fixed Version)")
    print("-------------------------------------")
    
    # First, check if we can detect any devices
    device = test_device_connection()
    if not device:
        print("\nNo SpaceMouse detected. Please ensure:")
        print("1. The device is connected to your computer")
        print("2. You have installed libhidapi (sudo apt-get install libhidapi-dev libhidapi-hidraw0)")
        print("3. You have proper permissions (sudo usermod -a -G plugdev $USER)")
        print("4. You've created proper udev rules if needed")
        sys.exit(1)
    
    # Print detailed device information
    print_device_info(device)
    
    # Try to open the device
    if not open_device(device):
        sys.exit(1)
    
    # Test basic functionality
    print("\nTesting basic device functionality...")
    test_state = safe_read()
    if test_state:
        print("Successfully read initial state from device:")
        try:
            attrs = ["x", "y", "z", "roll", "pitch", "yaw", "buttons"]
            for attr in attrs:
                if hasattr(test_state, attr):
                    value = getattr(test_state, attr)
                    print(f"  {attr}: {value}")
                else:
                    print(f"  {attr}: Not available")
        except Exception as e:
            print(f"Error reading attributes: {e}")
            print(f"Raw state: {test_state}")
    else:
        print("Could not read initial state from device!")
        print("The device might be connected but not functioning correctly.")
        sys.exit(1)
    
    # Create a thread to monitor button presses
    stop_event = threading.Event()
    button_thread = threading.Thread(target=monitor_button_state, args=(stop_event,))
    button_thread.daemon = True
    button_thread.start()
    
    # Monitor device movement
    print("\nMove your SpaceMouse to see the values")
    print("Press Ctrl+C to exit")
    print("\nReading SpaceMouse state...")
    
    try:
        last_print_time = time.time()
        while True:
            # Read the current state
            state = safe_read()
            
            if state:
                current_time = time.time()
                
                # Check if any movement data is available
                x = getattr(state, "x", 0) or 0
                y = getattr(state, "y", 0) or 0
                z = getattr(state, "z", 0) or 0
                roll = getattr(state, "roll", 0) or 0
                pitch = getattr(state, "pitch", 0) or 0
                yaw = getattr(state, "yaw", 0) or 0
                
                if any(abs(val) > 0.01 for val in [x, y, z, roll, pitch, yaw]):
                    # Print at most 10 times per second
                    if current_time - last_print_time >= 0.1:
                        print(f"\rPosition: X:{x:6.2f} Y:{y:6.2f} Z:{z:6.2f} | "
                              f"Rotation: Roll:{roll:6.2f} Pitch:{pitch:6.2f} Yaw:{yaw:6.2f}", 
                              end="", flush=True)
                        last_print_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent 100% CPU usage
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n\nError reading from device: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        stop_event.set()
        button_thread.join(timeout=1.0)
        try:
            pyspacemouse.close()
            print("\nClosed SpaceMouse connection")
        except Exception as e:
            print(f"\nError closing device: {e}")

if __name__ == "__main__":
    main()