# Arduino OV5642 Camera Controller with OpenCV
# For hand gesture control project
# camera_test.py

import serial
import time
import cv2
import numpy as np
import os
import threading
import queue
from serial.tools import list_ports

# Create directories for saving images
os.makedirs('capture', exist_ok=True)
os.makedirs('capture/single', exist_ok=True)
os.makedirs('capture/sequence', exist_ok=True)

# Global variables
frame_queue = queue.Queue(maxsize=30)  # Frame buffer
latest_frame = None
stop_thread = False
exit_program = False
return_to_menu = False

def save_camera_settings(camera_index=0, width=320, height=240, fps=30):
    """
    Save camera settings to file for other scripts to use
    
    Parameters:
    camera_index: Camera device index (usually 0 for built-in or first connected camera)
    width: Camera capture width
    height: Camera capture height
    fps: Target frames per second
    """
    print(f"Saving camera settings: index={camera_index}, {width}x{height}, {fps}fps")
    
    with open('arduino_camera_settings.txt', 'w') as f:
        f.write(f"index={camera_index}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"fps={fps}\n")
    
    print("Camera settings saved to arduino_camera_settings.txt")

def save_camera_settings_menu():
    """Menu for saving camera settings"""
    print("\n== Save Camera Settings ==")
    print("This will save current camera settings for other scripts to use")
    
    # Get camera settings from user
    try:
        camera_index = int(input("Enter camera index (usually 0): "))
        width = int(input("Enter camera width (recommend 320): "))
        height = int(input("Enter camera height (recommend 240): "))
        fps = int(input("Enter target FPS (recommend 30): "))
        
        save_camera_settings(camera_index, width, height, fps)
    except ValueError:
        print("Invalid input. Using default settings.")
        save_camera_settings()

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

def connect_to_arduino(port=None, baud_rate=115200, timeout=5):
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
        
        # Flush any data
        ser.reset_input_buffer()
        
        return ser
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def read_image_from_serial(ser, debug=True):
    """
    Highly robust image reading function that handles various JPEG formats
    Returns the decoded image or None if failed
    """
    # Clear buffer first to ensure we're at the start of a new frame
    ser.reset_input_buffer()
    
    # Send capture command
    ser.write(b'c')
    
    # Wait for image data
    start_time = time.time()
    size = None
    
    # Step 1: Wait for IMG: marker
    while time.time() - start_time < 5:  # 5 second timeout
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if debug:
                print(f"Read line: {line}")
            
            if line.startswith("IMG:"):
                try:
                    size = int(line.split(':')[1])
                    if debug:
                        print(f"Found image size: {size} bytes")
                    break
                except:
                    if debug:
                        print("Failed to parse image size")
                    continue
        except Exception as e:
            if debug:
                print(f"Error reading line: {e}")
    
    if size is None:
        if debug:
            print("Timeout waiting for image header")
        return None
    
    # Step 2: Read start marker
    start_marker1 = ser.read(1)
    start_marker2 = ser.read(1)
    
    if not start_marker1 or not start_marker2:
        if debug:
            print("Timeout reading start marker")
        return None
        
    if start_marker1 != b'\xff' or start_marker2 != b'\xaa':
        if debug:
            print(f"Invalid start marker: {start_marker1.hex()} {start_marker2.hex()}")
        return None
    
    if debug:
        print("Found start marker (FF AA)")
    
    # Step 3: Read image data with careful handling of escape sequences
    image_data = bytearray()
    
    start_time = time.time()
    timeout = start_time + 15  # 15 second timeout
    
    # Read the raw data first
    raw_data = bytearray()
    bytes_read = 0
    end_marker_found = False
    
    while time.time() < timeout and bytes_read < size + 1000:  # Add some margin
        if ser.in_waiting > 0:
            chunk = ser.read(ser.in_waiting)
            raw_data.extend(chunk)
            bytes_read += len(chunk)
            
            # Check if we've received the end marker (0xFF 0xBB) in the data
            if len(raw_data) >= 2:
                for i in range(len(raw_data) - 1):
                    if raw_data[i] == 0xFF and raw_data[i + 1] == 0xBB:
                        end_marker_found = True
                        if debug:
                            print(f"Found end marker after {i} bytes in raw data")
                        break
                if end_marker_found:
                    break
        else:
            time.sleep(0.01)  # Short sleep to avoid tight loop
    
    if debug:
        print(f"Read {len(raw_data)} bytes total")
        
    if not end_marker_found:
        if debug:
            print("End marker not found in data")
            
        # Save the raw data for inspection
        with open('debug_raw_data.bin', 'wb') as f:
            f.write(raw_data)
        
        return None
    
    # Now process the raw data to handle the escape sequences
    i = 0
    while i < len(raw_data):
        # Check for end marker
        if i < len(raw_data) - 1 and raw_data[i] == 0xFF and raw_data[i + 1] == 0xBB:
            if debug:
                print(f"Processed {len(image_data)} bytes before end marker")
            break
        
        # Handle FF escape sequence
        if i < len(raw_data) - 1 and raw_data[i] == 0xFF and raw_data[i + 1] == 0xFF:
            image_data.append(0xFF)  # Add a single FF
            i += 2  # Skip both FF bytes
        else:
            image_data.append(raw_data[i])  # Add the current byte
            i += 1  # Move to next byte
    
    # Save both raw and processed data for debugging
    if debug:
        with open('debug_raw_image.bin', 'wb') as f:
            f.write(raw_data)
        with open('debug_processed_image.jpg', 'wb') as f:
            f.write(image_data)
        print(f"Saved raw and processed data files for debugging")
    
    # Check for valid JPEG header (FF D8)
    if len(image_data) >= 2:
        if image_data[0] == 0xFF and image_data[1] == 0xD8:
            if debug:
                print("JPEG header detected (FF D8)")
        else:
            if debug:
                print(f"WARNING: First bytes are not JPEG header: {image_data[0]:02X} {image_data[1]:02X}")
                
            # Try to find JPEG start anywhere in the first 100 bytes
            jpeg_start_idx = -1
            for i in range(len(image_data) - 2):
                if i > 100:  # Only check first 100 bytes
                    break
                if image_data[i] == 0xFF and image_data[i+1] == 0xD8:
                    jpeg_start_idx = i
                    if debug:
                        print(f"Found JPEG header at position {i}")
                    break
            
            if jpeg_start_idx >= 0:
                # Trim data to start from JPEG header
                image_data = image_data[jpeg_start_idx:]
                if debug:
                    print(f"Trimmed data to start from JPEG header, new size: {len(image_data)}")
            else:
                if debug:
                    print("No JPEG header found in data, trying to add one")
                # Add JPEG header if not found
                header = bytearray(b'\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00')
                new_data = header + image_data
                
                # Save with added header
                with open('debug_with_header.jpg', 'wb') as f:
                    f.write(new_data)
                
                image_data = new_data
    
    # Step 5: Try multiple methods to decode the image
    for attempt in range(3):
        try:
            if attempt == 0:
                # First attempt: standard decoding
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None and img.size > 0:
                    if debug:
                        print(f"Successfully decoded image on attempt {attempt+1}: {img.shape}")
                    return img
                else:
                    if debug:
                        print(f"Attempt {attempt+1} failed: Empty or corrupted image")
            
            elif attempt == 1:
                # Second attempt: Try to find and extract a valid JPEG segment
                # Look for FF D8 (SOI) followed by FF E0 (APP0) or FF DB (DQT)
                for i in range(len(image_data) - 4):
                    if (image_data[i] == 0xFF and image_data[i+1] == 0xD8 and 
                        image_data[i+2] == 0xFF and (image_data[i+3] == 0xE0 or image_data[i+3] == 0xDB)):
                        fixed_data = image_data[i:]
                        with open('debug_attempt2.jpg', 'wb') as f:
                            f.write(fixed_data)
                        
                        nparr = np.frombuffer(fixed_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None and img.size > 0:
                            if debug:
                                print(f"Successfully decoded image on attempt {attempt+1} after finding valid JPEG segment: {img.shape}")
                            return img
                
                if debug:
                    print(f"Attempt {attempt+1} failed: Could not find valid JPEG segment")
            
            elif attempt == 2:
                # Third attempt: Try adding a completely new header
                # This is a full JPEG header that works in many cases
                header = bytearray([
                    0xFF, 0xD8,                 # SOI marker
                    0xFF, 0xE0, 0x00, 0x10,     # APP0 marker
                    0x4A, 0x46, 0x49, 0x46, 0x00, # 'JFIF\0'
                    0x01, 0x01,                 # version
                    0x00,                       # units
                    0x00, 0x01, 0x00, 0x01,     # density
                    0x00, 0x00                  # thumbnail
                ])
                
                # Try to find a good place to inject it (after any existing headers)
                inject_point = 0
                for i in range(min(100, len(image_data) - 2)):
                    if image_data[i] == 0xFF and image_data[i+1] in [0xDB, 0xC0, 0xC4]:  # DQT, SOF, DHT markers
                        inject_point = i
                        break
                
                if inject_point > 0:
                    fixed_data = header + image_data[inject_point:]
                else:
                    fixed_data = header + image_data
                
                with open('debug_attempt3.jpg', 'wb') as f:
                    f.write(fixed_data)
                
                nparr = np.frombuffer(fixed_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None and img.size > 0:
                    if debug:
                        print(f"Successfully decoded image on attempt {attempt+1} with new header: {img.shape}")
                    return img
                else:
                    if debug:
                        print(f"Attempt {attempt+1} failed: Could not decode with new header")
        
        except Exception as e:
            if debug:
                print(f"Error in decoding attempt {attempt+1}: {e}")
    
    if debug:
        print("All decoding attempts failed")
    return None

def capture_single_image(ser):
    """Capture a single image with improved error handling"""
    print("Capturing single image...")
    
    # Use the robust read_image_from_serial function
    img = read_image_from_serial(ser, debug=True)
    
    if img is not None:
        # Save the decoded image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'capture/single/image_{timestamp}.jpg'
        cv2.imwrite(filename, img)
        print(f"Image saved as {filename}")
        return img
    
    print("Failed to capture image")
    return None

def continuous_capture_thread(ser):
    """Thread function to continuously receive images from Arduino"""
    global latest_frame, stop_thread
    
    # Tell Arduino to start continuous mode
    ser.write(b's')
    
    while not stop_thread:
        try:
            # Read image from serial
            img = read_image_from_serial(ser)
            
            if img is not None:
                # Update latest frame
                latest_frame = img
                
                # Put in queue for display (only if not too backed up)
                if frame_queue.qsize() < 25:
                    frame_queue.put(img)
        except Exception as e:
            print(f"Error in capture thread: {e}")
            time.sleep(0.1)

def terminal_input_thread():
    """Thread to handle terminal input during continuous mode"""
    global stop_thread, return_to_menu, exit_program
    
    print("\nCommands Available:")
    print("s      - Save current frame")
    print("m      - Return to main menu")
    print("q      - Quit program")
    
    while not stop_thread:
        try:
            cmd = input("Command: ").strip().lower()
            
            if not cmd:  # Skip empty input
                continue
                
            if cmd == 'q':
                print("Exiting program...")
                exit_program = True
                stop_thread = True
                break
                
            elif cmd == 'm':
                print("Returning to main menu...")
                return_to_menu = True
                stop_thread = True
                break
                
            elif cmd == 's' and latest_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture/sequence/frame_{timestamp}.jpg"
                cv2.imwrite(filename, latest_frame)
                print(f"Saved frame to {filename}")
                
            else:
                print("Unknown command")
                
        except Exception as e:
            pass

def run_continuous_mode(ser):
    """Run continuous capture mode with terminal input thread"""
    global stop_thread, latest_frame, return_to_menu, exit_program
    
    # Reset flags
    stop_thread = False
    return_to_menu = False
    exit_program = False
    
    # Start continuous capture thread
    capture_thread = threading.Thread(target=continuous_capture_thread, args=(ser,))
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread)
    input_thread.daemon = True
    input_thread.start()
    
    # Initialize display window
    cv2.namedWindow("Arduino Camera Stream", cv2.WINDOW_NORMAL)
    
    # Tracking variables
    frame_count = 0
    fps_start_time = time.time()
    fps_update_time = time.time()
    fps = 0
    
    print("\nWindow Controls:")
    print("- Press 's' to save current frame")
    print("- Press 'q' to quit")
    
    try:
        while not stop_thread:
            # Get frame from queue if available
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - fps_update_time >= 1.0:  # Update FPS display every second
                    fps = frame_count / (current_time - fps_start_time)
                    fps_update_time = current_time
                
                # Display information on frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Arduino Camera Stream", display_frame)
            elif latest_frame is not None:
                # If queue is empty but we have a latest frame
                cv2.waitKey(1)  # Just keep window responsive
            
            # Check for key presses (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            # q - quit program
            if key == ord('q'):
                print("Quitting program...")
                exit_program = True
                stop_thread = True
                break
                
            # s - save current frame
            elif key == ord('s') and latest_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture/sequence/frame_{timestamp}.jpg"
                cv2.imwrite(filename, latest_frame)
                print(f"Saved frame to {filename}")
    
    finally:
        # Clean up
        stop_thread = True  # Signal threads to stop
        ser.write(b'x')     # Stop continuous mode on Arduino
        time.sleep(0.5)
        cv2.destroyAllWindows()
        print("Continuous mode stopped")

def capture_multiple_images(ser, count):
    """Capture multiple images in sequence"""
    print(f"Capturing {count} images in sequence...")
    
    # Clear buffer first
    ser.reset_input_buffer()
    
    successful_captures = 0
    for i in range(count):
        print(f"Capturing image {i+1}/{count}...")
        img = capture_single_image(ser)
        
        if img is not None:
            filename = f"capture/sequence/seq_img_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, img)
            print(f"Saved as {filename}")
            successful_captures += 1
        else:
            print(f"Failed to capture image {i+1}")
        
        # Wait between captures
        time.sleep(1)
    
    print(f"Sequence capture complete. Successfully captured {successful_captures} of {count} images.")
    return successful_captures

def change_resolution(ser, resolution_code):
    """Change the camera resolution"""
    resolutions = {
        '0': "320x240",
        '1': "640x480",
        '2': "1024x768",
        '3': "1280x960",
        '4': "1600x1200",
        '5': "2048x1536",
        '6': "2592x1944"
    }
    
    if resolution_code in resolutions:
        print(f"Changing resolution to {resolutions[resolution_code]}...")
        
        # Send resolution change command
        ser.write(b'r')
        time.sleep(0.1)
        ser.write(resolution_code.encode())
        
        # Wait for confirmation
        time.sleep(1)
        
        # Clear any pending data
        ser.reset_input_buffer()
        
        return True
    else:
        print("Invalid resolution code")
        return False

def main_menu():
    """Display the main menu"""
    print("\n========= Arduino OV5642 Camera Control =========")
    print("1. Capture single image")
    print("2. Capture multiple images")
    print("3. Change resolution")
    print("4. Start continuous capture mode")
    print("5. Exit")
    print("6. Save camera settings for data collection")
    print("===============================================")
    return input("Select an option (1-6): ")

def display_resolution_menu():
    """Display available resolutions"""
    print("\nAvailable Resolutions:")
    print("0: 320x240")
    print("1: 640x480 (default)")
    print("2: 1024x768")
    print("3: 1280x960")
    print("4: 1600x1200")
    print("5: 2048x1536")
    print("6: 2592x1944")
    return input("\nEnter resolution code (0-6): ")

def main():
    """Main program"""
    global exit_program
    
    print("Arduino OV5642 Camera Control")
    print("============================")
    
    # Connect to Arduino
    ser = connect_to_arduino()
    if ser is None:
        print("Failed to connect to Arduino. Exiting.")
        return
    
    try:
        while not exit_program:
            choice = main_menu()
            
            if choice == '1':
                # Single image capture
                img = capture_single_image(ser)
                if img is not None:
                    # Display the image
                    cv2.imshow("Captured Image", img)
                    print("Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            elif choice == '2':
                # Multiple image capture
                try:
                    count = int(input("\nEnter number of images to capture: "))
                    if count > 0:
                        capture_multiple_images(ser, count)
                    else:
                        print("Please enter a positive number")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif choice == '3':
                # Change resolution
                res_code = display_resolution_menu()
                change_resolution(ser, res_code)
            
            elif choice == '4':
                # Continuous capture mode
                run_continuous_mode(ser)
                if exit_program:
                    break
            
            elif choice == '5':
                # Exit
                print("\nExiting program. Goodbye!")
                exit_program = True
                break
            
            elif choice == '6':
                # Save camera settings
                save_camera_settings_menu()
            
            else:
                print("\nInvalid option. Please try again.")
    
    finally:
        # Clean up
        if ser and ser.is_open:
            # Make sure continuous mode is stopped
            ser.write(b'x')
            time.sleep(0.1)
            ser.close()
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()