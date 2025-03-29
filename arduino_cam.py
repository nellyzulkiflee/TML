# arduino_camera.py - Reverted to compatible settings
# With multiple image capture and improved error handling

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

def connect_to_arduino(port=None, baud_rate=921600, timeout=5):
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

def read_image_from_serial(ser, debug=False):
    """
    Read and decode an image from serial with original protocol
    Returns the decoded image or None if failed
    """
    # Clear buffer first to ensure we're at the start of a new frame
    ser.reset_input_buffer()
    
    # Wait for image data
    start_time = time.time()
    size = None
    
    # Step 1: Wait for IMG: marker
    while time.time() - start_time < 5:  # 5 second timeout
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
    
    if size is None:
        if debug:
            print("Timeout waiting for image header")
        return None
    
    # Step 2: Read start marker
    start_marker1 = ser.read(1)
    start_marker2 = ser.read(1)
    if start_marker1 != b'\xff' or start_marker2 != b'\xaa':
        if debug:
            print(f"Invalid start marker: {start_marker1} {start_marker2}")
        return None
    
    # Step 3: Read image data with careful handling of escape sequences
    image_data = bytearray()
    
    start_time = time.time()
    while time.time() - start_time < 10:  # 10 second timeout
        if ser.in_waiting > 0:
            byte = ser.read(1)
            
            # Check for end marker
            if byte == b'\xff':
                next_byte = ser.read(1)
                if next_byte == b'\xbb':  # End marker
                    if debug:
                        print(f"Found end marker after {len(image_data)} bytes")
                    break
                elif next_byte == b'\xff':  # Escaped FF
                    image_data.append(0xFF)
                else:
                    # Normal FF followed by something else
                    image_data.append(0xFF)
                    image_data.append(ord(next_byte))
            else:
                # Normal byte
                image_data.append(ord(byte))
    
    # Step 4: Check if we got a reasonable amount of data
    if len(image_data) < 100:
        if debug:
            print(f"Image data too small: {len(image_data)} bytes")
        return None
    
    # Step 5: Try to decode as JPEG
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None or img.size == 0:
            if debug:
                print("Failed to decode image - empty or corrupted")
                # Save the raw data for debugging
                with open('debug_failed_image.jpg', 'wb') as f:
                    f.write(image_data)
            return None
            
        if debug:
            print(f"Successfully decoded image: {img.shape}")
        return img
        
    except Exception as e:
        if debug:
            print(f"Error decoding image: {e}")
        return None

def capture_single_image(ser):
    """Capture a single image with better error handling"""
    print("Capturing single image...")
    
    # Clear buffer first to ensure clean start
    ser.reset_input_buffer()
    
    # Send capture command
    ser.write(b'c')
    
    # Try multiple times if needed
    for attempt in range(3):
        if attempt > 0:
            print(f"Retrying capture (attempt {attempt+1})...")
        
        img = read_image_from_serial(ser, debug=(attempt > 0))
        
        if img is not None:
            # Save the decoded image
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f'capture/single/image_{timestamp}.jpg'
            cv2.imwrite(filename, img)
            print(f"Image saved as {filename}")
            return img
        
        # If failed, clear buffer and try again
        ser.reset_input_buffer()
        time.sleep(1)
        ser.write(b'c')
    
    print("Failed to capture image after multiple attempts")
    return None

def continuous_capture_thread(ser):
    """Thread function to continuously receive images from Arduino"""
    global latest_frame, stop_thread
    
    while not stop_thread:
        try:
            # Read image from serial
            img = read_image_from_serial(ser)
            
            if img is not None:
                # Update latest frame
                latest_frame = img
                
                # Put in queue for display (only if not too backed up)
                if frame_queue.qsize() < 5:
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
    
    # Clear any pending data
    ser.reset_input_buffer()
    
    # Start continuous capture thread
    capture_thread = threading.Thread(target=continuous_capture_thread, args=(ser,))
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start terminal input thread
    input_thread = threading.Thread(target=terminal_input_thread)
    input_thread.daemon = True
    input_thread.start()
    
    # Start continuous mode on Arduino
    print("Starting continuous capture mode...")
    ser.write(b's')
    
    # Initialize display window
    cv2.namedWindow("Arduino Camera Stream", cv2.WINDOW_NORMAL)
    
    # Tracking variables
    frame_count = 0
    fps_start_time = time.time()
    fps_update_time = time.time()
    fps = 0
    
    print("\nWindow Controls:")
    print("- Press 's' to save current frame")
    print("- Press 'q' to quit program")
    
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

def main_menu():
    """Display the main menu"""
    print("\n========= Arduino OV5642 Camera Control =========")
    print("1. Capture single image")
    print("2. Capture multiple images")
    print("3. Start continuous capture mode")
    print("4. Exit")
    print("===============================================")
    return input("Select an option (1-4): ")

def main():
    """Main program"""
    global exit_program
    
    print("Arduino OV5642 Camera Control")
    print("============================")
    print("Using 640x480 resolution")
    
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
                # Continuous capture mode
                run_continuous_mode(ser)
                if exit_program:
                    break
            
            elif choice == '4':
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