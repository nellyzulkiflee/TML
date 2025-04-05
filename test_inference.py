#!/usr/bin/env python3
# Advanced test_inference.py - Fixes JPEG issues and adds basic inference
# Works with the simplified Arduino sketch

import serial
import time
import cv2
import numpy as np
import os
import argparse
from serial.tools import list_ports

# Create directories for saving images
os.makedirs('test_images', exist_ok=True)
os.makedirs('test_images/processed', exist_ok=True)

def find_arduino_port():
    """Find Arduino port by listing all available ports"""
    ports = list(list_ports.comports())
    
    print("\nAvailable ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
    
    if not ports:
        print("No serial ports found!")
        return None
    
    choice = input("\nSelect Arduino port number (or Enter for first port): ")
    
    if choice.strip() == "":
        return ports[0].device
    
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

def connect_to_arduino(port=None, baud_rate=115200, timeout=10):
    """Connect to Arduino with detailed feedback"""
    if port is None:
        port = find_arduino_port()
        if port is None:
            print("No Arduino port found!")
            return None
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print(f"Connected to Arduino on {port}")
        time.sleep(2)  # Wait for Arduino to reset
        
        # Flush input buffer
        ser.reset_input_buffer()
        
        # Read any initial messages
        while ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"Arduino: {line}")
        
        return ser
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def advanced_jpeg_repair(data, debug=False):
    """
    Apply multiple repair strategies to fix corrupted JPEG data
    Returns a tuple (fixed_data, strategy_used)
    """
    # Make a copy to avoid modifying the original
    original_data = bytes(data)
    
    # Strategy 1: Check if it's already a valid JPEG
    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        if debug:
            print("Image already has a valid JPEG header (FF D8)")
        return data, "already_valid"
    
    # Strategy 2: Fix swapped header bytes
    if len(data) >= 2 and data[0] == 0xD8 and data[1] == 0xFF:
        print("Strategy 2: Fixing swapped JPEG header bytes (D8 FF -> FF D8)")
        fixed_data = bytearray(data)
        fixed_data[0] = 0xFF
        fixed_data[1] = 0xD8
        return bytes(fixed_data), "swap_fix"
    
    # Strategy 3: Check for JFIF/EXIF marker misplacement
    # Sometimes the FF D8 is present but buried a few bytes in
    for i in range(min(20, len(data) - 2)):
        if data[i] == 0xFF and data[i+1] == 0xD8:
            print(f"Strategy 3: Found JPEG SOI marker at offset {i}, trimming prefix")
            return data[i:], "trim_prefix"
    
    # Strategy 4: Rebuild with standard JPEG header
    print("Strategy 4: Rebuilding with standard JPEG header")
    # Standard JPEG header with JFIF APP0 segment
    header = bytearray([
        0xFF, 0xD8,             # SOI marker
        0xFF, 0xE0,             # APP0 marker
        0x00, 0x10,             # APP0 length (16 bytes)
        0x4A, 0x46, 0x49, 0x46, # "JFIF" identifier
        0x00,                   # JFIF version (1.0)
        0x01, 0x01,             # JFIF version (1.0)
        0x00,                   # Density units (0 = no units)
        0x00, 0x01,             # X density
        0x00, 0x01,             # Y density
        0x00, 0x00              # Thumbnail (none)
    ])
    
    # Look for the first FF marker in the file to determine where image data starts
    data_start = 0
    for i in range(min(50, len(data))):
        if data[i] == 0xFF and i+1 < len(data) and data[i+1] in [0xC0, 0xC2, 0xC4, 0xDB, 0xDD, 0xDA]:
            data_start = i
            print(f"Found likely JPEG marker at position {i}: FF {data[i+1]:02X}")
            break
    
    # Create new image with header + data
    if data_start > 0:
        return header + data[data_start:], "rebuild_partial"
    else:
        return header + data, "rebuild_full"

def capture_and_process_image(ser, debug=True):
    """
    Capture image from Arduino, apply fixes, and save
    Returns OpenCV image or None if failed
    """
    print("\n--- Starting Image Capture ---")
    
    # Clear any pending data
    ser.reset_input_buffer()
    
    # Send capture command
    print("Sending capture command ('c')...")
    ser.write(b'c')
    
    # Wait for Arduino response
    print("Waiting for Arduino response...")
    start_found = False
    image_size = 0
    
    timeout_time = time.time() + 10  # 10 second timeout
    while time.time() < timeout_time and not start_found:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if debug:
                print(f"Arduino: {line}")
            
            if line.startswith("RAW_IMAGE_START:"):
                start_found = True
                try:
                    image_size = int(line.split(":")[1])
                    print(f"Found raw image start marker, expecting {image_size} bytes")
                except:
                    print("Error parsing image size")
                    return None
    
    if not start_found:
        print("Failed to find raw image start marker")
        return None
    
    # Calculate timeout based on image size and baud rate
    bytes_per_sec = ser.baudrate / 10  # Account for start/stop bits
    timeout_seconds = max(10, (image_size / bytes_per_sec) * 2)  # Double for safety
    ser.timeout = timeout_seconds
    
    # Read the raw binary data directly
    print("Reading raw binary image data...")
    raw_data = ser.read(image_size)
    
    if len(raw_data) < image_size:
        print(f"Warning: Received fewer bytes than expected: {len(raw_data)}/{image_size}")
        if len(raw_data) < 100:  # Too small to be a valid image
            print("Not enough data received to be a valid image")
            return None
    
    print(f"Received {len(raw_data)} bytes of raw data")
    
    # Save the raw data for debugging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_filename = f'test_images/raw_image_{timestamp}.bin'
    with open(raw_filename, 'wb') as f:
        f.write(raw_data)
    print(f"Saved raw data to {raw_filename}")
    
    # Wait for end marker (but don't require it)
    timeout_time = time.time() + 5  # 5 second timeout
    while time.time() < timeout_time:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                if debug:
                    print(f"Arduino: {line}")
                if line == "RAW_IMAGE_END":
                    print("Found raw image end marker")
                    break
        else:
            # No more data to read
            break
    
    # Print hex dump of first few bytes for debugging
    if debug:
        print("\nFirst 16 bytes of raw data:")
        for i in range(min(16, len(raw_data))):
            print(f"{raw_data[i]:02X}", end=" ")
        print("\n")
    
    # Apply advanced JPEG repair
    fixed_data, strategy = advanced_jpeg_repair(raw_data, debug)
    
    # Save the fixed data
    fixed_filename = f'test_images/fixed_image_{timestamp}_{strategy}.jpg'
    with open(fixed_filename, 'wb') as f:
        f.write(fixed_data)
    print(f"Saved fixed data (strategy: {strategy}) to {fixed_filename}")
    
    # Try multiple decoding methods
    img = None
    
    # Method 1: Try direct decoding with OpenCV
    try:
        nparr = np.frombuffer(fixed_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None and img.size > 0:
            print(f"Successfully decoded image with cv2.imdecode: {img.shape}")
            
            # Save the successfully decoded image
            decoded_filename = f'test_images/processed/decoded_{timestamp}.jpg'
            cv2.imwrite(decoded_filename, img)
            print(f"Saved decoded image to {decoded_filename}")
            return img
    except Exception as e:
        print(f"Error with OpenCV decoding: {e}")
    
    # Method 2: Try with PIL if OpenCV fails
    if img is None:
        try:
            from PIL import Image
            import io
            
            img_pil = Image.open(io.BytesIO(fixed_data))
            print(f"Successfully opened image with PIL: {img_pil.format} {img_pil.size}")
            
            # Convert PIL to OpenCV format
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # Save the PIL decoded image
            pil_filename = f'test_images/processed/pil_decoded_{timestamp}.jpg'
            cv2.imwrite(pil_filename, img)
            print(f"Saved PIL-decoded image to {pil_filename}")
            return img
        except Exception as e:
            print(f"Error with PIL decoding: {e}")
    
    # If we get here, all decoding attempts failed
    print("Failed to decode image with all methods")
    return None

def perform_basic_inference(img):
    """
    Perform basic object detection on the image using OpenCV
    """
    if img is None:
        print("No valid image to perform inference on")
        return
    
    print("\n--- Performing Basic Inference ---")
    
    # Create a copy for drawing on
    result_img = img.copy()
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size to eliminate noise
    min_contour_area = img.shape[0] * img.shape[1] * 0.01  # 1% of image area
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    print(f"Detected {len(significant_contours)} significant objects in the image")
    
    # Draw contours on the image
    cv2.drawContours(result_img, significant_contours, -1, (0, 255, 0), 2)
    
    # Label objects with numbers
    for i, contour in enumerate(significant_contours):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate center of contour for text placement
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
        
        # Draw object number
        cv2.putText(result_img, f"#{i+1}", (cx-10, cy+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw bounding box
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Save result
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f'test_images/processed/inference_{timestamp}.jpg'
    cv2.imwrite(result_filename, result_img)
    print(f"Saved inference result to {result_filename}")
    
    # Display result
    cv2.imshow("Object Detection", result_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result_img

def main():
    parser = argparse.ArgumentParser(description='ArduCAM OV5642 Test with Inference')
    parser.add_argument('--port', help='Arduino serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--count', type=int, default=1, help='Number of images to capture')
    args = parser.parse_args()
    
    print("ArduCAM OV5642 Test with Inference")
    print("==================================")
    
    # Connect to Arduino
    ser = connect_to_arduino(args.port, args.baud)
    if ser is None:
        return
    
    try:
        for i in range(args.count):
            if args.count > 1:
                print(f"\nCapturing image {i+1} of {args.count}")
                
            # Capture and process image
            img = capture_and_process_image(ser, args.debug)
            
            if img is not None:
                # Display the successfully decoded image
                cv2.imshow("Captured Image", img)
                print("Press any key to continue to inference...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # Perform basic inference
                perform_basic_inference(img)
            else:
                print("Failed to decode image, skipping inference")
            
            if args.count > 1 and i < args.count - 1:
                print("Waiting 2 seconds before next capture...")
                time.sleep(2)
    
    finally:
        # Clean up
        ser.close()
        print("Serial connection closed")
        cv2.destroyAllWindows()
        print("Test complete")

if __name__ == "__main__":
    main()