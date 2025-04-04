def preview_camera(ser, gesture_name, max_time=30):
    """
    Show a live preview from the camera to help with positioning
    
    Parameters:
    ser: Serial connection to Arduino
    gesture_name: Name of the gesture being prepared
    max_time: Maximum preview time in seconds
    
    Returns:
    True when user is ready to proceed
    """
    print(f"\n=== STARTING CAMERA PREVIEW FOR: {gesture_name} ===")
    print("Position yourself and adjust until you're ready")
    print("Press ENTER to start collection or ESC to cancel")
    print(f"Preview will timeout after {max_time} seconds")
    
    # Create window
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    
    # Make sure camera is ready
    reset_camera(ser)
    time.sleep(0.5)
    
    start_time = time.time()
    frame_count = 0
    last_capture_time = 0
    
    # Try to get at least one frame before continuing
    first_frame = None
    for _ in range(10):  # Try 10 times to get a frame
        first_frame = read_image_from_serial(ser, debug=True)
        if first_frame is not None:
            break
        time.sleep(0.5)
    
    if first_frame is None:
        print("WARNING: Could not get initial preview frame from camera")
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "Camera Error!", (80, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(blank, "Press any key to continue anyway", (40, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Camera Preview", blank)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        # Show the first frame
        display = first_frame.copy()
        cv2.putText(display, "Preview Started", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Preview", display)
        cv2.waitKey(1)
    
    while time.time() - start_time < max_time:
        # Limit capture rate to avoid overwhelming the camera
        current_time = time.time()
        if current_time - last_capture_time < 0.3:  # Max ~3 FPS for stability
            cv2.waitKey(1)  # Keep UI responsive
            time.sleep(0.05)
            continue
        
        # Capture frame
        img = read_image_from_serial(ser, debug=(frame_count == 0))
        last_capture_time = current_time
        
        if img is not None:
            frame_count += 1
            
            # Add guidance overlay
            display = img.copy()
            
            # Add gesture name
            cv2.putText(display, f"Preparing: {gesture_name}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(display, "Position yourself correctly", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add frame counter
            cv2.putText(display, f"Frame: {frame_count}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            
            # Add time remaining
            remaining = max_time - (current_time - start_time)
            cv2.putText(display, f"Preview ends in: {int(remaining)}s", (20, img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
            
            # Add bounding box to show optimal hand position area
            center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
            box_size = min(img.shape[1], img.shape[0]) // 2
            top_left = (center_x - box_size // 2, center_y - box_size // 2)
            bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
            cv2.rectangle(display, top_left, bottom_right, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("Camera Preview", display)
        
        # Check for key presses - make sure to handle this every loop
        key = cv2.waitKey(10) & 0xFF  # Wait longer (10ms) for key press
        
        if key == 13 or key == 10:  # ENTER key (13 on Windows, 10 on some systems)
            print("Preview ended - Starting collection")
            cv2.destroyAllWindows()
            return True
        elif key == 27:  # ESC key
            print("Preview cancelled")
            cv2.destroyAllWindows()
            return False
    
    # Timeout reached
    cv2.destroyAllWindows()
    print("Preview timeout reached")
    return True# TinyML Gesture Data Collection Script - Optimized Version

import serial
import time
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from serial.tools import list_ports
import threading

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
    Robust image reading function that handles various JPEG formats
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
    
    # Try multiple methods to decode the image
    for attempt in range(2):  # Reduced from 3 to 2 for speed
        try:
            if attempt == 0:
                # First attempt: standard decoding
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None and img.size > 0:
                    return img
            
            elif attempt == 1:
                # Second attempt: Try to find and extract a valid JPEG segment
                for i in range(len(image_data) - 4):
                    if (image_data[i] == 0xFF and image_data[i+1] == 0xD8 and 
                        image_data[i+2] == 0xFF and (image_data[i+3] == 0xE0 or image_data[i+3] == 0xDB)):
                        fixed_data = image_data[i:]
                        
                        nparr = np.frombuffer(fixed_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None and img.size > 0:
                            return img
        
        except Exception as e:
            if debug:
                print(f"Error in decoding attempt {attempt+1}: {e}")
    
    return None

def check_image_quality(img, strict_mode=False):
    """
    Check if the image is good enough for training
    
    Parameters:
    img: The image to check
    strict_mode: If False, uses more lenient quality standards
    
    Returns:
    True if image quality is acceptable, False otherwise
    """
    if img is None:
        return False
    
    # Check image size (very basic check)
    if img.shape[0] < 50 or img.shape[1] < 50:
        return False
    
    # In lenient mode, we only check if image exists with reasonable dimensions
    if not strict_mode:
        return True
    
    # More strict checks (only in strict mode)
    try:
        # Check for blurriness using Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Very lenient blur threshold
        if blur_value < 20:  # Image is extremely blurry
            return False
    except Exception:
        # If any error occurs during quality check, still accept the image
        pass
    
    return True

def reset_camera(ser):
    """Reset the camera to fix quality issues"""
    print("Resetting camera...")
    
    # Clear serial buffers
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    # Re-optimize camera settings
    ser.write(b'r')
    time.sleep(0.1)
    ser.write(b'0')  # 0 = 320x240
    time.sleep(1.0)
    
    # Take a dummy capture to reset internal state
    ser.write(b'c')
    time.sleep(1.0)
    
    # Clear buffers again
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    
    print("Camera reset complete")
    return True

def collect_gesture_data(ser, num_samples=50, gesture_label=0, gesture_name="neutral", skip_confirmation=False, strict_quality=False):
    """
    Collect image data for a specific gesture using Arduino camera - Optimized version
    
    Parameters:
    ser: Serial connection to Arduino
    num_samples: Number of images to collect
    gesture_label: Numeric label for the gesture
    gesture_name: Name of the gesture for display
    skip_confirmation: Skip the confirmation step if True
    strict_quality: Whether to use strict quality checks
    
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
    
    # Display gesture example/instructions
    display_instructions(gesture_name)
    
    if not skip_confirmation:
        print("Position yourself and press ENTER when ready...")
        input()
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(0.7)  # Slightly faster countdown
    
    print("COLLECTING DATA - Perform the gesture continuously")
    print(f"Need {num_samples} good quality samples")
    
    # Statistics tracking
    stats = {
        "good_frames": 0,
        "rejected_frames": 0,
        "total_attempts": 0
    }
    
    # Collect images
    good_frames_collected = 0
    pbar = tqdm(total=num_samples)
    
    # Flag for first frame debug
    first_frame = True
    
    max_attempts = num_samples * 2  # Limit total attempts to avoid endless loop
    attempts = 0
    
    while good_frames_collected < num_samples and attempts < max_attempts:
        attempts += 1
        stats["total_attempts"] += 1
        
        # Capture frame from Arduino - only debug first frame
        img = read_image_from_serial(ser, debug=first_frame)
        first_frame = False
        
        if img is not None:
            # Check image quality - using less strict mode by default
            quality_ok = check_image_quality(img, strict_mode=strict_quality)
            
            if quality_ok:
                # Save image
                timestamp = time.time()
                filename = f"{data_dir}/img_{good_frames_collected:04d}.jpg"
                cv2.imwrite(filename, img)
                
                # Add to metadata
                metadata.append({
                    "filename": filename,
                    "timestamp": timestamp,
                    "label": gesture_label,
                    "gesture_name": gesture_name
                })
                
                good_frames_collected += 1
                stats["good_frames"] += 1
                pbar.update(1)
                
                # Display frame with quality feedback
                display_frame = img.copy()
                cv2.putText(display_frame, f"Recording: {gesture_name}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Good frames: {good_frames_collected}/{num_samples}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add progress bar
                progress = int((good_frames_collected / num_samples) * display_frame.shape[1])
                cv2.rectangle(display_frame, (0, display_frame.shape[0] - 20), 
                             (progress, display_frame.shape[0]), (0, 255, 0), -1)
                
                cv2.imshow("Data Collection", display_frame)
            else:
                stats["rejected_frames"] += 1
                
                # Show rejected frame with reason
                if img is not None:
                    display_frame = img.copy()
                    cv2.putText(display_frame, "Low Quality - Trying Again", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Data Collection", display_frame)
            
            # Allow interruption with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection interrupted!")
                break
                
            # Small delay between captures for stability (reduced for speed)
            time.sleep(0.1)
        else:
            stats["rejected_frames"] += 1
            print(f"Failed to capture frame, trying again")
            time.sleep(0.3)  # Shorter wait before retrying
    
    pbar.close()
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{data_dir}/metadata.csv", index=False)
    
    # Print statistics
    print(f"\nCollection Statistics for '{gesture_name}':")
    print(f"- Good frames: {stats['good_frames']}")
    print(f"- Rejected frames: {stats['rejected_frames']}")
    print(f"- Total attempts: {stats['total_attempts']}")
    print(f"- Acceptance rate: {stats['good_frames'] / max(stats['total_attempts'], 1):.1%}")
    
    cv2.destroyAllWindows()
    
    return data_dir

def display_instructions(gesture_name):
    """Display example image or instructions for the gesture"""
    # Create a blank image with instructions
    instruction_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add text instructions based on gesture name
    if gesture_name == "neutral":
        cv2.putText(instruction_img, "NEUTRAL GESTURE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(instruction_img, "- Hand relaxed at side or in front", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_img, "- Palm facing camera", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    elif gesture_name == "turn_right":
        cv2.putText(instruction_img, "TURN RIGHT GESTURE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(instruction_img, "- Hand clearly visible", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_img, "- Full swipe to the right", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    elif gesture_name == "turn_left":
        cv2.putText(instruction_img, "TURN LEFT GESTURE", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(instruction_img, "- Hand clearly visible", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_img, "- Full swipe to the left", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    elif gesture_name == "pre_turn_right":
        cv2.putText(instruction_img, "PRE-TURN RIGHT", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(instruction_img, "- Hand positioned to start right movement", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_img, "- Ready position before turn right", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    elif gesture_name == "pre_turn_left":
        cv2.putText(instruction_img, "PRE-TURN LEFT", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(instruction_img, "- Hand positioned to start left movement", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(instruction_img, "- Ready position before turn left", (50, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Display the instructions
    cv2.putText(instruction_img, "Press any key to continue...", (150, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    
    cv2.imshow("Gesture Instructions", instruction_img)
    cv2.waitKey(1000)  # Display for at least 1 second
    
    # Wait for key press but also continue after 3 seconds
    start_time = time.time()
    key_pressed = False
    
    while time.time() - start_time < 3 and not key_pressed:
        if cv2.waitKey(100) != -1:
            key_pressed = True
    
    cv2.destroyWindow("Gesture Instructions")

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
    
    # Take a dummy capture to initialize settings
    ser.write(b'c')
    time.sleep(0.5)
    ser.reset_input_buffer()
    
    print("Camera set to 320x240 resolution and initialized")
    return True

# Define available gestures - full set
ALL_GESTURES = [
    ('neutral', 0),
    ('turn_right', 1),
    ('turn_left', 2),
    ('pre_turn_right', 3),
    ('pre_turn_left', 4)
]

# Define essential gestures (minimum required for functionality)
ESSENTIAL_GESTURES = [
    ('neutral', 0),
    ('turn_right', 1),
    ('turn_left', 2)
]

# Default to use essential gestures only
AVAILABLE_GESTURES = ESSENTIAL_GESTURES

def run_data_collection_fast():
    """Run an optimized data collection process focused on essential gestures"""
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
    
    # Quick collection setup
    print("\n=== OPTIMIZED DATA COLLECTION MODE ===")
    print("This optimized collection mode will help you finish faster by:")
    print("1. Focusing on only essential gestures (neutral, left, right)")
    print("2. Collecting high-quality samples with validation")
    print("3. Making transitions optional")
    
    # Let user choose gesture set
    print("\nWhich gesture set would you like to collect?")
    print("1: Essential gestures only (neutral, turn_right, turn_left) - RECOMMENDED")
    print("2: Full gesture set (includes pre_turn gestures)")
    
    gesture_choice = input("Enter choice (1 or 2, default is 1): ").strip()
    if gesture_choice == "2":
        global AVAILABLE_GESTURES
        AVAILABLE_GESTURES = ALL_GESTURES
        print("Using full gesture set (5 gestures)")
    else:
        AVAILABLE_GESTURES = ESSENTIAL_GESTURES
        print("Using essential gestures only (3 gestures)")
    
    # Let user choose sample count
    sample_count = 100  # Default to 100 samples per gesture
    try:
        user_count = input("\nHow many samples per gesture? (default: 100): ").strip()
        if user_count:
            sample_count = int(user_count)
    except:
        print("Invalid input, using 100 samples per gesture")
    
    # Enable preview mode option - default to YES
    print("\nCamera preview helps you position yourself correctly before data collection starts.")
    use_preview = input("Enable camera preview before collecting? (Y/n): ").strip().lower() != 'n'
    if use_preview:
        print("✓ Camera preview ENABLED - You'll see a live view before collection")
    else:
        print("✗ Camera preview DISABLED")
    
    # Batch mode option
    batch_mode = input("\nUse batch mode? This will collect all gestures in sequence (y/n): ").strip().lower() == 'y'
    
    # Collect data for each gesture
    data_dirs = []
    
    try:
        # First collect basic gestures
        for gesture_name, gesture_label in AVAILABLE_GESTURES:
            print(f"\nPreparing to collect data for gesture: {gesture_name} (label {gesture_label})")
            
            if not batch_mode:
                print("Position yourself and press Enter when ready...")
                input()
            
                # Reset camera before each gesture (except first)
            if gesture_name != AVAILABLE_GESTURES[0][0]:
                reset_camera(ser)
            
            # Collect data for this gesture with relaxed quality check
            data_dir = collect_gesture_data(
                ser, 
                num_samples=sample_count,
                gesture_label=gesture_label,
                gesture_name=gesture_name,
                skip_confirmation=batch_mode,
                strict_quality=False  # Use lenient quality checking
            )
            
            data_dirs.append(data_dir)
            
            if not batch_mode:
                print(f"Completed collection for {gesture_name}. Take a short break...")
                time.sleep(2)
            else:
                print(f"Next gesture coming up in 3 seconds...")
                time.sleep(3)
        
        # Optional transition data
        transitions = input("\nCollect transition data? (not required, takes extra time) (y/n): ").strip().lower() == 'y'
        
        if transitions:
            # Define important transitions - reduced set for speed
            important_transitions = [
                ('neutral', 'turn_right'),
                ('neutral', 'turn_left')
            ]
            
            for from_name, to_name in important_transitions:
                # Get labels
                from_label = next((label for name, label in AVAILABLE_GESTURES if name == from_name), None)
                to_label = next((label for name, label in AVAILABLE_GESTURES if name == to_name), None)
                
                if from_label is None or to_label is None:
                    continue
                
                # Create transition directory
                transition_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_transition_{from_name}_to_{to_name}"
                transition_dir = f"data/raw/images/{transition_id}"
                os.makedirs(transition_dir, exist_ok=True)
                
                # Show instructions
                print(f"\nCollecting transition: {from_name} → {to_name}")
                print("Start in first position, then transition to second when prompted")
                print("Press ENTER when ready...")
                input()
                
                # Simplified transition collection with 20 samples
                print("COLLECTING TRANSITION - Get ready...")
                
                # Create metadata list
                metadata = []
                
                # Countdown
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    time.sleep(0.7)
                
                # Collect 10 frames of starting position
                print(f"Showing {from_name} position...")
                for i in range(10):
                    img = read_image_from_serial(ser)
                    if img is not None:
                        filename = f"{transition_dir}/start_{i:02d}.jpg"
                        cv2.imwrite(filename, img)
                        
                        metadata.append({
                            "filename": filename,
                            "label": from_label,
                            "gesture_name": from_name,
                            "phase": "start"
                        })
                        
                        # Display
                        display_img = img.copy()
                        cv2.putText(display_img, f"Starting position: {from_name}", (20, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Transition", display_img)
                        cv2.waitKey(1)
                        
                        time.sleep(0.1)
                
                # Now transition
                print(f"\nTRANSITION NOW to {to_name}...")
                time.sleep(1)
                
                # Collect 10 frames of ending position
                for i in range(10):
                    img = read_image_from_serial(ser)
                    if img is not None:
                        filename = f"{transition_dir}/end_{i:02d}.jpg"
                        cv2.imwrite(filename, img)
                        
                        metadata.append({
                            "filename": filename,
                            "label": to_label,
                            "gesture_name": to_name,
                            "phase": "end"
                        })
                        
                        # Display
                        display_img = img.copy()
                        cv2.putText(display_img, f"Ending position: {to_name}", (20, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Transition", display_img)
                        cv2.waitKey(1)
                        
                        time.sleep(0.1)
                
                # Save metadata
                pd.DataFrame(metadata).to_csv(f"{transition_dir}/metadata.csv", index=False)
                data_dirs.append(transition_dir)
                
                cv2.destroyAllWindows()
                print(f"Transition collection complete")
                time.sleep(1)
    
    finally:
        # Close serial connection
        ser.close()
        cv2.destroyAllWindows()
        print("Data collection complete.")
    
    # Save list of all data directories
    with open('data/gesture_data_dirs.txt', 'w') as f:
        for directory in data_dirs:
            f.write(f"{directory}\n")
    
    print(f"\n===== DATA COLLECTION SUMMARY =====")
    print(f"Successfully collected data in {len(data_dirs)} directories")
    print(f"Total number of gestures: {len([d for d in data_dirs if 'transition' not in d])}")
    print(f"Total number of transitions: {len([d for d in data_dirs if 'transition' in d])}")
    print("\nNext step: Run model_training.py to train your gesture recognition model")
    
    return data_dirs

def force_preview_test():
    """Run a standalone test of the preview functionality"""
    print("\n=== TESTING CAMERA PREVIEW FUNCTIONALITY ===")
    
    # Connect to Arduino
    ser = connect_to_arduino()
    if ser is None:
        print("Failed to connect to Arduino. Exiting.")
        return
    
    # Initialize camera
    optimize_camera_settings(ser)
    
    try:
        # Test the preview function for each gesture
        for gesture_name, _ in ESSENTIAL_GESTURES:
            print(f"\nTesting preview for: {gesture_name}")
            preview_camera(ser, gesture_name, max_time=15)  # Shorter timeout for testing
            
            user_input = input("Did you see the preview working? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Preview test failed. Troubleshooting tips:")
                print("1. Make sure camera is properly connected")
                print("2. Check that you have OpenCV properly installed")
                print("3. Try restarting your Arduino")
                break
        
        print("\nPreview test complete.")
    finally:
        # Close serial connection
        ser.close()

if __name__ == "__main__":
    print("Starting TinyML gesture data collection (OPTIMIZED VERSION)...")
    
    # Add a special debug mode option to help diagnose camera issues
    print("\nSpecial options:")
    print("1: Normal collection mode")
    print("2: Debug mode (helps diagnose camera issues)")
    print("3: Test preview functionality")
    debug_choice = input("Enter choice (1-3, default is 1): ").strip()
    
    if debug_choice == "2":
        print("\n=== CAMERA DEBUG MODE ===")
        print("This will test camera stability by capturing multiple images in sequence")
        
        # Connect to Arduino
        ser = connect_to_arduino()
        if ser is not None:
            try:
                # Reset camera to start fresh
                reset_camera(ser)
                
                # Try to capture 20 consecutive images
                print("Testing camera stability with 20 consecutive captures...")
                success_count = 0
                
                for i in range(20):
                    print(f"Capture {i+1}/20...", end="")
                    img = read_image_from_serial(ser, debug=(i == 0))
                    
                    if img is not None:
                        success_count += 1
                        print(f" SUCCESS - Image size: {img.shape}")
                        
                        # Display the image
                        cv2.imshow(f"Capture {i+1}", img)
                        cv2.waitKey(500)
                        cv2.destroyAllWindows()
                    else:
                        print(" FAILED")
                    
                    # Reset camera every 5 frames
                    if (i+1) % 5 == 0:
                        reset_camera(ser)
                
                print(f"\nCamera stability test results: {success_count}/20 successful captures")
                print("If less than 15, your camera connection may have issues.")
                
                # Get more detailed information
                if success_count < 15:
                    print("\nTroubleshooting tips:")
                    print("1. Check physical connections (CS, MOSI, MISO, SCK pins)")
                    print("2. Try a lower camera resolution")
                    print("3. Ensure adequate power supply")
                    print("4. Try reducing data transfer speed")
                
                ser.close()
            except Exception as e:
                print(f"Error during debug: {e}")
                if ser:
                    ser.close()
        exit()
    elif debug_choice == "3":
        # Test just the preview functionality
        force_preview_test()
        exit()
    
    # Normal collection mode
    try:
        # Use the fast collection mode to save time
        data_dirs = run_data_collection_fast()
        
        if data_dirs:
            print(f"Successfully collected data in {len(data_dirs)} directories.")
            print("\nNext step: Run model_training.py to train your gesture recognition model.")
        else:
            print("Data collection failed or was interrupted.")
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nError during data collection: {e}")