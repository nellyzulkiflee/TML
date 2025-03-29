# Simple ArduCAM OV5642 Test Script
# For debugging camera communication issues

import serial
import time
import cv2
import numpy as np
import os
from serial.tools import list_ports

# Make sure this matches Arduino sketch
BAUD_RATE = 115200

# Create output directory
os.makedirs('camera_test', exist_ok=True)

def find_arduino_port():
    """Find Arduino port by listing all available ports"""
    print("\nAvailable serial ports:")
    ports = list(list_ports.comports())
    
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
    
    if not ports:
        print("No serial ports found!")
        return None
    
    choice = input("\nSelect Arduino port number (or Enter for first port): ")
    
    if choice.strip() == "":
        port = ports[0].device
        print(f"Using first port: {port}")
        return port
    
    try:
        index = int(choice) - 1
        if 0 <= index < len(ports):
            return ports[index].device
        else:
            print("Invalid selection, using first port")
            return ports[0].device
    except:
        print("Invalid input, using first port")
        return ports[0].device

def connect_to_arduino(port):
    """Connect to Arduino with verbose feedback"""
    print(f"Connecting to Arduino on {port} at {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for Arduino reset after connection
        
        # Flush input buffer
        ser.reset_input_buffer()
        print("Connection successful!")
        
        # Read and display initial messages from Arduino
        print("\nArduino startup messages:")
        timeout = time.time() + 5  # 5 second timeout
        while time.time() < timeout:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"  {line}")
                    # If we see the ready message, we can stop waiting
                    if "Camera ready" in line:
                        break
        
        return ser
    
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def capture_single_image(ser):
    """Capture and save a single image with verbose debugging"""
    print("\n--- Starting Image Capture Test ---")
    
    # Clear any pending data
    ser.reset_input_buffer()
    
    # Send capture command
    print("Sending capture command ('c')...")
    ser.write(b'c')
    
    # Wait for Arduino acknowledgment
    print("Waiting for Arduino response...")
    timeout = time.time() + 5  # 5 second timeout
    command_ack = False
    
    while time.time() < timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Arduino: {line}")
                if "Command received" in line:
                    command_ack = True
                    break
    
    if not command_ack:
        print("ERROR: Arduino did not acknowledge capture command")
        return False
    
    # Read debug messages until we find the image header
    print("Reading Arduino capture status messages...")
    timeout = time.time() + 10  # 10 second timeout
    img_size = None
    
    while time.time() < timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Arduino: {line}")
                
                # Look for image size header
                if line.startswith("IMG:"):
                    try:
                        img_size = int(line.split(':')[1])
                        print(f"Found image header! Size: {img_size} bytes")
                        break
                    except:
                        print(f"Error parsing image size from: {line}")
    
    if img_size is None:
        print("ERROR: Never received image header from Arduino")
        return False
    
    # Read start marker
    print("Reading start marker...")
    start_marker1 = ser.read(1)
    start_marker2 = ser.read(1)
    
    if start_marker1 != b'\xff' or start_marker2 != b'\xaa':
        print(f"ERROR: Invalid start marker: {start_marker1.hex()} {start_marker2.hex()}")
        return False
    
    print("Start marker correct (FF AA)")
    
    # Read image data with detailed feedback
    print(f"Reading {img_size} bytes of image data...")
    image_data = bytearray()
    timeout = time.time() + 30  # 30 second timeout
    
    while len(image_data) < img_size and time.time() < timeout:
        if ser.in_waiting:
            # Read in chunks for efficiency
            chunk = ser.read(min(ser.in_waiting, img_size - len(image_data)))
            image_data.extend(chunk)
            
            # Print progress every 10KB
            if len(image_data) % 10000 < len(chunk):
                print(f"Received {len(image_data)}/{img_size} bytes ({len(image_data)/img_size*100:.1f}%)")
    
    if len(image_data) < img_size:
        print(f"WARNING: Received fewer bytes than expected: {len(image_data)}/{img_size}")
    
    # Look for end marker
    print("Reading end marker...")
    found_end_marker = False
    timeout = time.time() + 5  # 5 second timeout
    
    while time.time() < timeout:
        if ser.in_waiting >= 2:
            marker1 = ser.read(1)
            marker2 = ser.read(1)
            
            if marker1 == b'\xff' and marker2 == b'\xbb':
                found_end_marker = True
                print("End marker found (FF BB)")
                break
            else:
                # Put markers back in the buffer if not end markers
                image_data.extend(marker1)
                image_data.append(marker2[0])
    
    if not found_end_marker:
        print("WARNING: End marker not found")
    
    # Check for JPEG structure
    if len(image_data) >= 2 and image_data[0] == 0xFF and image_data[1] == 0xD8:
        print("JPEG header detected (FF D8) - good sign!")
    else:
        print("WARNING: JPEG header not found at start of data")
        # Save the raw data for debugging
        with open('camera_test/raw_data.bin', 'wb') as f:
            f.write(image_data)
        print("Raw data saved to camera_test/raw_data.bin for debugging")
        
        # Try to print the first few bytes for debugging
        if len(image_data) > 0:
            print("First 16 bytes:")
            for i in range(min(16, len(image_data))):
                print(f"{image_data[i]:02X}", end=" ")
            print()
    
    # Try to decode as JPEG
    print("Attempting to decode image as JPEG...")
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            print("Failed to decode image - empty or corrupted")
            return False
            
        # Image decoded successfully!
        print(f"Successfully decoded image: {img.shape[1]}x{img.shape[0]}")
        
        # Save both raw and decoded image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_filename = f'camera_test/raw_image_{timestamp}.jpg'
        with open(raw_filename, 'wb') as f:
            f.write(image_data)
        
        decoded_filename = f'camera_test/decoded_image_{timestamp}.jpg'
        cv2.imwrite(decoded_filename, img)
        
        print(f"Raw image saved as {raw_filename}")
        print(f"Decoded image saved as {decoded_filename}")
        
        # Display the image
        cv2.imshow("Captured Image", img)
        print("Press any key to close image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"Error decoding image: {e}")
        
        # Save the raw data for further analysis
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'camera_test/failed_image_{timestamp}.bin'
        with open(filename, 'wb') as f:
            f.write(image_data)
        print(f"Raw data saved to {filename} for debugging")
        
        return False

def main():
    """Main test function"""
    print("ArduCAM OV5642 Debug Test")
    print("========================")
    
    # Find and connect to Arduino
    port = find_arduino_port()
    if not port:
        print("No port selected. Exiting.")
        return
    
    ser = connect_to_arduino(port)
    if not ser:
        print("Failed to connect to Arduino. Exiting.")
        return
    
    try:
        # Test capture
        input("\nPress Enter to capture an image...")
        success = capture_single_image(ser)
        
        if success:
            print("\nSUCCESS! Camera capture and image decoding worked!")
        else:
            print("\nFAILED: Camera capture or image decoding failed.")
        
    finally:
        # Clean up
        ser.close()
        print("Serial connection closed")

if __name__ == "__main__":
    main()