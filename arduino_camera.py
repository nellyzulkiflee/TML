# File: arduino_camera.py
# Purpose: Receive images from Arduino connected OV5642 camera

import serial
import time
import cv2
import numpy as np
import os
from serial.tools import list_ports

def find_arduino_port():
    """Find Arduino port by listing all available ports"""
    ports = list(list_ports.comports())
    
    print("Available ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
    
    if not ports:
        return None
    
    choice = input("Select Arduino port number (or Enter for first port): ")
    
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

def connect_to_arduino(port=None, baud_rate=921600, timeout=10):
    """Connect to Arduino with OV5642 camera"""
    if port is None:
        port = find_arduino_port()
        if port is None:
            print("No Arduino port found!")
            return None
    
    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print(f"Connected to Arduino on {port}")
        time.sleep(2)  # Wait for Arduino to reset
        return ser
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def capture_image(ser):
    """Send capture command and receive image from Arduino"""
    # Flush any data in buffer
    ser.reset_input_buffer()
    
    # Send capture command
    ser.write(b'c')
    
    # Wait for "Capturing..." message
    response = ser.readline().decode('utf-8', errors='ignore').strip()
    if "Capturing" not in response:
        print(f"Unexpected response: {response}")
        return None
    
    # Wait for "Capture done!" message
    response = ser.readline().decode('utf-8', errors='ignore').strip()
    if "Capture done" not in response:
        print(f"Unexpected response: {response}")
        return None
    
    # Read image size
    size_line = ser.readline().decode('utf-8', errors='ignore').strip()
    try:
        size = int(size_line.split(': ')[1])
        print(f"Image size: {size} bytes")
    except:
        print(f"Invalid size format: {size_line}")
        return None
    
    # Read start marker
    start_marker1 = ser.read(1)
    start_marker2 = ser.read(1)
    if start_marker1 != b'\xff' or start_marker2 != b'\xaa':
        print(f"Invalid start marker: {start_marker1} {start_marker2}")
        return None
    
    # Read image data with timeout
    image_data = bytearray()
    escape_next = False
    
    start_time = time.time()
    while time.time() - start_time < 30:  # 30 second timeout
        if ser.in_waiting > 0:
            byte = ser.read(1)
            
            # Check for end marker
            if byte == b'\xff' and not escape_next:
                next_byte = ser.read(1)
                if next_byte == b'\xbb':  # End marker
                    break
                elif next_byte == b'\xff':  # Escaped FF
                    image_data.append(0xFF)
                else:
                    image_data.append(0xFF)
                    image_data.extend(next_byte)
            else:
                image_data.append(ord(byte))
    
    print(f"Received {len(image_data)} bytes")
    
    # Save raw data to file
    os.makedirs('capture', exist_ok=True)
    with open(f'capture/raw_image_{time.strftime("%Y%m%d_%H%M%S")}.jpg', 'wb') as f:
        f.write(image_data)
    
    # Try to decode as JPEG
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image")
            return None
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def change_resolution(ser, resolution_code):
    """Change the camera resolution"""
    ser.write(b'r')
    time.sleep(0.1)
    ser.write(resolution_code.encode())
    response = ser.readline().decode('utf-8', errors='ignore').strip()
    print(response)
    time.sleep(1)  # Give camera time to change resolution

def test_camera():
    """Test the Arduino OV5642 camera connection and capture"""
    ser = connect_to_arduino()
    if ser is None:
        return
    
    print("\nArduino OV5642 Camera Test")
    print("========================")
    print("1. Capture image")
    print("2. Change resolution")
    print("3. Capture multiple images")
    print("4. Exit")
    
    try:
        while True:
            choice = input("\nEnter choice (1-4): ")
            
            if choice == '1':
                print("Capturing image...")
                img = capture_image(ser)
                if img is not None:
                    print(f"Image captured successfully! Size: {img.shape}")
                    
                    # Save and display image
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"capture/image_{timestamp}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"Image saved as {filename}")
                    
                    cv2.imshow("Captured Image", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            elif choice == '2':
                print("\nAvailable resolutions:")
                print("0: 320x240")
                print("1: 640x480")
                print("2: 1024x768")
                print("3: 1280x960")
                print("4: 1600x1200")
                print("5: 2048x1536")
                print("6: 2592x1944")
                
                res_choice = input("Enter resolution choice (0-6): ")
                if res_choice in ['0', '1', '2', '3', '4', '5', '6']:
                    change_resolution(ser, res_choice)
                else:
                    print("Invalid resolution choice")
            
            elif choice == '3':
                count = input("Enter number of images to capture: ")
                try:
                    count = int(count)
                except:
                    print("Invalid input, using 5 images")
                    count = 5
                
                print(f"Capturing {count} images with 1 second interval...")
                for i in range(count):
                    print(f"Capturing image {i+1}/{count}...")
                    img = capture_image(ser)
                    if img is not None:
                        filename = f"capture/image_seq_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"Saved as {filename}")
                    time.sleep(1)
                
                print("Capture sequence completed")
            
            elif choice == '4':
                print("Exiting...")
                break
            
            else:
                print("Invalid choice")
    
    finally:
        ser.close()
        print("Serial connection closed")

if __name__ == "__main__":
    test_camera()