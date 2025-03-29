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
frame_queue = queue.Queue(maxsize=100)  # Frame buffer
latest_frame = None
stop_thread = False
current_resolution = 1  # Default: 640x480
exit_program = False
return_to_menu = False

# Resolution options
RESOLUTIONS = {
    0: "320x240",
    1: "640x480",
    2: "1024x768",
    3: "1280x960",
    4: "1600x1200",
    5: "2048x1536",
    6: "2592x1944"
}

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
        
        # Flush any data
        ser.reset_input_buffer()
        
        return ser
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return None

def capture_single_image(ser):
    """Capture a single image from the Arduino camera"""
    # Send capture command
    ser.write(b'c')
    
    # Wait for "Capturing..." message
    response = ser.readline().decode('utf-8', errors='ignore').strip()
    print(response)
    
    # Wait for "Capture done!" message
    response = ser.readline().decode('utf-8', errors='ignore').strip()
    print(response)
    
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
    
    # Read image data
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
    
    # Try to decode as JPEG
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode image")
            return None
        
        # Save raw data for debugging if needed
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(f'capture/single/raw_image_{timestamp}.jpg', 'wb') as f:
            f.write(image_data)
        
        # Save the decoded image
        filename = f'capture/single/image_{timestamp}.jpg'
        cv2.imwrite(filename, img)
        print(f"Image saved as {filename}")
        
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def continuous_capture_thread(ser):
    """Thread function to continuously receive images from Arduino"""
    global latest_frame, stop_thread
    
    while not stop_thread:
        try:
            # Read until we find "IMG:" marker
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line.startswith("IMG:"):
                # Parse image size
                try:
                    size = int(line.split(':')[1])
                    # print(f"Image size: {size} bytes")  # Comment out for less verbosity
                except:
                    continue
                
                # Read start marker
                start_marker1 = ser.read(1)
                start_marker2 = ser.read(1)
                if start_marker1 != b'\xff' or start_marker2 != b'\xaa':
                    continue
                
                # Read image data
                image_data = bytearray()
                escape_next = False
                
                timeout_time = time.time() + 5  # 5 second timeout
                
                while time.time() < timeout_time:
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
                
                # Decode image
                try:
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Put in queue for processing
                        if not frame_queue.full():
                            frame_queue.put(img)
                        
                        # Update latest frame
                        latest_frame = img
                except Exception as e:
                    print(f"Error decoding image: {e}")
                    
        except Exception as e:
            print(f"Error in capture thread: {e}")
            time.sleep(0.1)  # Prevent tight loop on error

# Fixed terminal_input_thread function with better command handling
def terminal_input_thread(ser):
    """Thread to handle terminal input during continuous mode"""
    global stop_thread, return_to_menu, exit_program, current_resolution
    
    print("\nTerminal Commands in Continuous Mode:")
    print("0-6    - Change resolution directly (e.g., 0 for 320x240)")
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
                filename = f"capture/sequence/frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, latest_frame)
                print(f"Saved frame to {filename}")
                
            # Check if it's a single digit for resolution (0-6)
            elif cmd.isdigit() and len(cmd) == 1:
                resolution = int(cmd)
                if resolution in RESOLUTIONS:
                    print(f"Changing resolution to {resolution} ({RESOLUTIONS[resolution]})...")
                    
                    # Stop continuous mode to change resolution
                    ser.write(b'x')
                    time.sleep(0.5)
                    
                    # Change resolution
                    change_resolution(ser, str(resolution))
                    current_resolution = resolution
                    
                    # Restart continuous mode
                    ser.write(b's')
                else:
                    print(f"Invalid resolution: {resolution}")
            
            # Also accept r0, r1, etc. format for backward compatibility
            elif cmd.startswith('r') and len(cmd) > 1 and cmd[1:].isdigit():
                resolution = int(cmd[1:])
                if resolution in RESOLUTIONS:
                    print(f"Changing resolution to {resolution} ({RESOLUTIONS[resolution]})...")
                    
                    # Stop continuous mode to change resolution
                    ser.write(b'x')
                    time.sleep(0.5)
                    
                    # Change resolution
                    change_resolution(ser, str(resolution))
                    current_resolution = resolution
                    
                    # Restart continuous mode
                    ser.write(b's')
                else:
                    print(f"Invalid resolution: {resolution}")
            
            else:
                print(f"Unknown command: '{cmd}'")
                print("Use 0-6 to change resolution, s to save frame, m for menu, q to quit")
                
        except Exception as e:
            print(f"Input error: {e}")
            
    print("Input thread stopped")

def change_resolution(ser, resolution_code):
    """Change the camera resolution"""
    global current_resolution
    
    if resolution_code in [str(i) for i in range(len(RESOLUTIONS))]:
        current_resolution = int(resolution_code)
        
        ser.write(b'r')
        time.sleep(0.1)
        ser.write(resolution_code.encode())
        
        # Wait for confirmation
        response = ser.readline().decode('utf-8', errors='ignore').strip()
        print(response)
        time.sleep(1)  # Give camera time to change resolution
        return True
    else:
        print("Invalid resolution code")
        return False

def display_resolution_menu():
    """Display available resolutions"""
    print("\nAvailable Resolutions:")
    for code, res in RESOLUTIONS.items():
        marker = " *" if code == current_resolution else ""
        print(f"{code}: {res}{marker}")

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
    input_thread = threading.Thread(target=terminal_input_thread, args=(ser,))
    input_thread.daemon = True
    input_thread.start()
    
    # Start continuous mode on Arduino
    print("Starting continuous capture mode...")
    ser.write(b's')
    
    # Initialize display window
    cv2.namedWindow("Arduino Camera Stream", cv2.WINDOW_NORMAL)
    
    # Tracking variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    print("\nContinuous Mode Active - Window Controls:")
    print("- Press '0'-'6' in window to change resolution")
    print("- Press 's' in window to save current frame")
    print("- Press 'q' in window to quit program")
    
    try:
        while not stop_thread:
            # Get frame from queue if available
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_count += 1
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Update FPS every second
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Display information on frame
                display_frame = frame.copy()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Resolution: {RESOLUTIONS[current_resolution]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Arduino Camera Stream", display_frame)
            elif latest_frame is not None:
                # If queue is empty but we have a latest frame, show it
                display_frame = latest_frame.copy()
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Resolution: {RESOLUTIONS[current_resolution]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Arduino Camera Stream", display_frame)
            
            # Check for key presses in window (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            
            # q - quit program from window
            if key == ord('q'):
                print("Quitting program...")
                exit_program = True
                stop_thread = True
                break
                
            # s - save current frame from window
            elif key == ord('s') and latest_frame is not None:
                filename = f"capture/sequence/frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, latest_frame)
                print(f"Saved frame to {filename}")
                
            # 0-6 - change resolution from window
            elif key >= ord('0') and key <= ord('6'):
                resolution = chr(key)
                print(f"Changing resolution to {resolution}...")
                
                # Stop continuous mode to change resolution
                ser.write(b'x')
                time.sleep(0.5)
                
                # Change resolution
                change_resolution(ser, resolution)
                
                # Restart continuous mode
                ser.write(b's')
    
    finally:
        # Clean up
        stop_thread = True  # Signal threads to stop
        ser.write(b'x')     # Stop continuous mode on Arduino
        time.sleep(0.5)
        cv2.destroyAllWindows()
        
        # Wait for input thread to finish
        if input_thread.is_alive():
            input_thread.join(timeout=1)
            
        print("Continuous mode stopped")

def capture_multiple_images(ser, count):
    """Capture multiple images in sequence"""
    print(f"Capturing {count} images in sequence...")
    
    for i in range(count):
        print(f"Capturing image {i+1}/{count}...")
        img = capture_single_image(ser)
        if img is not None:
            filename = f"capture/sequence/seq_img_{i+1}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, img)
            print(f"Saved as {filename}")
        time.sleep(1)  # Pause between captures
    
    print("Sequence capture complete")

def main_menu():
    """Display the main menu"""
    print("\n========= Arduino OV5642 Camera Control =========")
    print("1. Capture single image")
    print("2. Capture multiple images")
    print("3. Change resolution")
    print("4. Continuous capture mode")
    print("5. Exit")
    print("===============================================")
    return input("Select an option (1-5): ")

def main():
    """Main program"""
    global current_resolution, exit_program
    
    print("Arduino OV5642 Camera Control")
    print("=============================")
    
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
                print("\nCapturing single image...")
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
                display_resolution_menu()
                res_choice = input("\nEnter resolution code: ")
                change_resolution(ser, res_choice)
            
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
            
            else:
                print("\nInvalid option. Please try again.")
    
    finally:
        # Clean up
        ser.close()
        cv2.destroyAllWindows()
        print("Program terminated")

if __name__ == "__main__":
    main()