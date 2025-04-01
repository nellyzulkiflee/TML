# TinyML Gesture Data Collection Script

import serial
import time
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from serial.tools import list_ports

# Create necessary directories
def create_project_directories():
    """Create necessary directories for the project"""
    dirs = [
        'data/raw/images',
        'data/processed/features',
        'models'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created successfully")

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

def read_image_from_serial(ser, debug=False):
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

def collect_gesture_data(ser, num_samples=100, gesture_label=0, gesture_name="neutral"):
    """
    Collect image data for a specific gesture using Arduino camera
    
    Parameters:
    ser: Serial connection to Arduino
    num_samples: Number of images to collect
    gesture_label: Numeric label for the gesture
    gesture_name: Name of the gesture for display
    
    Returns:
    path to saved data directory
    """
    # Create session ID with timestamp and gesture name
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gesture_name}"
    
    # Create directory for this gesture
    data_dir = f"data/raw/images/{session_id}_label{gesture_label}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create metadata file
    metadata = []
    
    print(f"Starting data collection for gesture: {gesture_name} (label {gesture_label})")
    print("Get ready...")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("COLLECTING DATA - Perform the gesture continuously")
    
    # Flag for first frame debug
    first_frame = True
    
    # Collect images
    for i in tqdm(range(num_samples)):
        # Capture frame from Arduino - only debug first frame
        img = read_image_from_serial(ser, debug=first_frame)
        first_frame = False
        
        if img is not None:
            # Save image
            timestamp = time.time()
            filename = f"{data_dir}/img_{i:04d}.jpg"
            cv2.imwrite(filename, img)
            
            # Add to metadata
            metadata.append({
                "filename": filename,
                "timestamp": timestamp,
                "label": gesture_label,
                "gesture_name": gesture_name
            })
            
            # Display frame with label
            display_frame = img.copy()
            cv2.putText(display_frame, f"Recording: {gesture_name}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {i+1}/{num_samples}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add progress bar
            progress = int((i / num_samples) * display_frame.shape[1])
            cv2.rectangle(display_frame, (0, display_frame.shape[0] - 20), 
                         (progress, display_frame.shape[0]), (0, 255, 0), -1)
            
            cv2.imshow("Data Collection", display_frame)
            
            # Allow interruption with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection interrupted!")
                break
                
            # Small delay between captures for stability
            time.sleep(0.2)
        else:
            print(f"Failed to capture frame {i+1}, trying again")
            i -= 1  # Try this frame again
            time.sleep(0.5)  # Wait a bit before retrying
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{data_dir}/metadata.csv", index=False)
    
    print(f"Collected {len(metadata)} images for gesture '{gesture_name}'")
    cv2.destroyAllWindows()
    
    return data_dir

def collect_transition_data(ser, transition_name, from_gesture, to_gesture, num_samples=30):
    """
    Collect transition data between two gestures
    
    Parameters:
    ser: Serial connection to Arduino
    transition_name: Name for this transition
    from_gesture: Starting gesture (name, label)
    to_gesture: Ending gesture (name, label)
    num_samples: Number of transition frames to collect
    
    Returns:
    path to transition data directory
    """
    from_name, from_label = from_gesture
    to_name, to_label = to_gesture
    
    print(f"\nCollecting transition data: {from_name} → {to_name}")
    
    # Create transition directory
    transition_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_transition_{from_name}_to_{to_name}"
    transition_dir = f"data/raw/images/{transition_id}"
    os.makedirs(transition_dir, exist_ok=True)
    
    # Metadata for this transition
    metadata = []
    
    # Show instructions
    print(f"Start in '{from_name}' position, then smoothly transition to '{to_name}'")
    print("Press ENTER when ready...")
    input()
    
    print("COLLECTING TRANSITION - Move slowly and naturally")
    
    # Debug only first frame
    first_frame = True
    
    # Collect transition images
    for i in tqdm(range(num_samples)):
        # Calculate interpolation factor (0.0 to 1.0)
        transition_factor = i / (num_samples - 1)
        
        # Capture frame
        img = read_image_from_serial(ser, debug=first_frame)
        first_frame = False
        
        if img is not None:
            # Save image
            timestamp = time.time()
            filename = f"{transition_dir}/transition_{i:04d}.jpg"
            cv2.imwrite(filename, img)
            
            # Determine label based on transition factor
            if transition_factor < 0.3:
                # Beginning of transition
                current_label = from_label
                current_name = from_name
            elif transition_factor > 0.7:
                # End of transition
                current_label = to_label
                current_name = to_name
            else:
                # Middle of transition - try to use a pre_gesture label if available
                pre_name = f"pre_{to_name}"
                if pre_name in [g for g, _ in AVAILABLE_GESTURES]:
                    pre_label = [l for g, l in AVAILABLE_GESTURES if g == pre_name][0]
                    current_label = pre_label
                    current_name = pre_name
                else:
                    # Default to from label
                    current_label = from_label
                    current_name = f"transition_{from_name}_to_{to_name}"
            
            # Add to metadata
            metadata.append({
                "filename": filename,
                "timestamp": timestamp,
                "label": current_label,
                "gesture_name": current_name,
                "transition_factor": transition_factor,
                "from_gesture": from_name,
                "to_gesture": to_name
            })
            
            # Display frame with transition info
            display_frame = img.copy()
            cv2.putText(display_frame, f"Transition: {from_name} → {to_name}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Progress: {int(transition_factor * 100)}%", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Progress bar
            progress = int(transition_factor * display_frame.shape[1])
            cv2.rectangle(display_frame, (0, display_frame.shape[0] - 20), 
                         (progress, display_frame.shape[0]), (0, 255, 0), -1)
            
            cv2.imshow("Transition Collection", display_frame)
            
            # Allow interruption with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Transition collection interrupted!")
                break
            
            # Small delay for smoother collection
            time.sleep(0.2)
        else:
            print(f"Failed to capture transition frame {i+1}, trying again")
            i -= 1  # Try this frame again
            time.sleep(0.5)  # Wait before retrying
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{transition_dir}/metadata.csv", index=False)
    
    print(f"Collected {len(metadata)} transition frames from '{from_name}' to '{to_name}'")
    cv2.destroyAllWindows()
    
    return transition_dir

# Define available gestures
AVAILABLE_GESTURES = [
    ('neutral', 0),
    ('turn_right', 1),
    ('turn_left', 2),
    ('pre_turn_right', 3),
    ('pre_turn_left', 4)
]

# Define important transitions
IMPORTANT_TRANSITIONS = [
    ('neutral', 'pre_turn_right'),
    ('pre_turn_right', 'turn_right'),
    ('turn_right', 'neutral'),
    ('neutral', 'pre_turn_left'),
    ('pre_turn_left', 'turn_left'),
    ('turn_left', 'neutral')
]

def optimize_camera_settings(ser):
    """
    Optimize camera settings for best results
    """
    print("Optimizing camera settings...")
    
    # Set camera to 320x240 resolution for best performance
    ser.write(b'r')
    time.sleep(0.1)
    ser.write(b'0')  # 0 = 320x240
    time.sleep(1.0)
    
    # Clear buffer
    ser.reset_input_buffer()
    
    print("Camera set to 320x240 resolution")
    return True

def run_data_collection():
    """Run the complete data collection process"""
    create_project_directories()
    
    # Connect to Arduino
    ser = connect_to_arduino()
    if ser is None:
        print("Failed to connect to Arduino. Exiting.")
        return []
    
    # Optimize camera settings
    optimize_camera_settings(ser)
    
    # Test capture to make sure everything is working
    print("Taking test capture to verify camera...")
    test_img = read_image_from_serial(ser, debug=True)
    
    if test_img is None:
        print("Test capture failed! Please check camera connections.")
        return []
    else:
        print("Test capture successful! Camera is working properly.")
        cv2.imshow("Test Capture", test_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    # Collect data for each gesture
    data_dirs = []
    
    try:
        # First collect basic gestures
        for gesture_name, gesture_label in AVAILABLE_GESTURES:
            print(f"\nPreparing to collect data for gesture: {gesture_name} (label {gesture_label})")
            print("Position yourself and press Enter when ready...")
            input()
            
            # Collect data for this gesture
            data_dir = collect_gesture_data(
                ser, 
                num_samples=100,  # 100 samples per gesture is a good starting point
                gesture_label=gesture_label,
                gesture_name=gesture_name
            )
            
            data_dirs.append(data_dir)
            
            print(f"Completed collection for {gesture_name}. Take a short break...")
            time.sleep(3)
        
        # Then collect transition data (optional)
        do_transitions = input("\nCollect transition data? (y/n): ").strip().lower() == 'y'
        
        if do_transitions:
            for from_gesture_name, to_gesture_name in IMPORTANT_TRANSITIONS:
                # Get labels
                from_label = next((label for name, label in AVAILABLE_GESTURES if name == from_gesture_name), None)
                to_label = next((label for name, label in AVAILABLE_GESTURES if name == to_gesture_name), None)
                
                if from_label is None or to_label is None:
                    print(f"Warning: Could not find labels for {from_gesture_name} → {to_gesture_name}")
                    continue
                
                # Collect transition data
                transition_dir = collect_transition_data(
                    ser,
                    f"{from_gesture_name}_to_{to_gesture_name}",
                    (from_gesture_name, from_label),
                    (to_gesture_name, to_label),
                    num_samples=30  # 30 frames per transition
                )
                
                data_dirs.append(transition_dir)
                
                print(f"Completed transition. Take a short break...")
                time.sleep(3)
    
    finally:
        # Close serial connection
        ser.close()
        cv2.destroyAllWindows()
        print("Data collection complete.")
    
    # Save list of all data directories
    with open('data/gesture_data_dirs.txt', 'w') as f:
        for directory in data_dirs:
            f.write(f"{directory}\n")
    
    print(f"Data collection complete. Collected data in {len(data_dirs)} directories.")
    return data_dirs

if __name__ == "__main__":
    print("Starting TinyML gesture data collection...")
    
    try:
        data_dirs = run_data_collection()
        if data_dirs:
            print(f"Successfully collected data in {len(data_dirs)} directories.")
            print("\nNext step: Run model_training.py to train your gesture recognition model.")
        else:
            print("Data collection failed or was interrupted.")
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nError during data collection: {e}")