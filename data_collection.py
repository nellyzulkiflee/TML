# TinyML Gesture Data Collection Script - Optimized Version
# Updated with distinct static poses AND SERIAL_TIMEOUT fix

import serial
import time
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from serial.tools import list_ports

# --- Constants ---
SERIAL_BAUDRATE = 115200
# --- <<< FIXED: Added SERIAL_TIMEOUT >>> ---
SERIAL_TIMEOUT = 1.0 # Timeout for general readline calls (in seconds)
# --- <<< END FIX >>> ---
IMAGE_READ_TIMEOUT = 10.0 # Longer timeout for reading image bytes
TARGET_IMG_SIZE = (32, 32) # Should match training input size

# --- <<< Gestures based on last discussion >>> ---
# IDLE: Hand out of view, PREV: Left Hand 'V', NEXT: Right Hand '1'
ESSENTIAL_GESTURES = [
    ('background', 0),      # IDLE state = Hand out of view
    ('right_one_finger', 1),  # NEXT Action: Right hand, 1 finger up
    ('left_two_fingers', 2)   # PREV Action: Left hand, 2 fingers up ('V')
]
AVAILABLE_GESTURES = ESSENTIAL_GESTURES # Default to essential for this final push
# --- <<< End Gestures >>> ---


# --- Directory Setup ---
def create_project_directories():
    """Create necessary directories"""
    dirs = [ 'data/raw/images', 'data/processed/simple_model_input', 'models' ]
    for directory in dirs: os.makedirs(directory, exist_ok=True)
    print("Project directories checked/created successfully")

# --- Serial Port Functions ---
def find_arduino_port():
    """Find Arduino port"""
    ports = list(list_ports.comports()); print("\nAvailable ports:")
    for i, port in enumerate(ports): print(f"{i+1}: {port.device} - {port.description}")
    if not ports: print("No serial ports found!"); return None
    choice = input("\nSelect Arduino port number (or Enter for first port): ")
    if choice.strip() == "": return ports[0].device if ports else None
    try:
        index = int(choice) - 1
        if 0 <= index < len(ports): return ports[index].device
        else: print("Invalid selection, using first port"); return ports[0].device if ports else None
    except: print("Invalid input, using first port"); return ports[0].device if ports else None

def connect_to_arduino(port=None, baud_rate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT): # Use SERIAL_TIMEOUT here
    """Connect to Arduino"""
    if port is None: port = find_arduino_port()
    if port is None: print("No Arduino port found!"); return None
    try:
        # Pass the defined SERIAL_TIMEOUT to the constructor
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print(f"Connected to Arduino on {port}"); time.sleep(2); ser.reset_input_buffer(); return ser
    except Exception as e: print(f"Error connecting to Arduino on {port}: {e}"); return None

# --- Image Reading Function (Should be correct now) ---
def read_image_from_serial(ser, debug=False):
    """Reads an image using IMAGE_START/SIZE/bytes/IMAGE_END protocol."""
    if debug: print("\n--- read_image_from_serial START ---")
    if ser is None or not ser.is_open:
        if debug: print("  read_image: Error - Serial port is not open.")
        return None
    try:
        ser.reset_input_buffer(); ser.write(b'c\n'); ser.flush()
    except Exception as e:
        if debug: print(f"  read_image: Error sending 'c': {e}"); return None

    start_marker_found = False; start_time = time.time(); read_timeout = 5.0; line = ""
    while time.time() - start_time < read_timeout:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "IMAGE_START": start_marker_found = True; break
                if line.startswith("DEBUG:") or line.startswith("ERROR:"): print(f"    Arduino MSG: {line}")
            else: time.sleep(0.01)
        except Exception as e: 
            if debug: print(f"  read_image: Error reading line for START: {e}"); return None
    if not start_marker_found: 
        if debug: print(f"  read_image: Timeout waiting for IMAGE_START. Last line: '{line}'"); return None

    size_line_found = False; size = None; start_time = time.time(); read_timeout = 2.0; line = ""
    while time.time() - start_time < read_timeout:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("SIZE:"):
                    try:
                        size = int(line[5:])
                        if 0 < size <= 500000: size_line_found = True; break
                        else: print(f"ERROR: Invalid image size: {size}"); break
                    except ValueError: print(f"ERROR: Could not parse size from '{line}'"); break
                if line.startswith("DEBUG:") or line.startswith("ERROR:"): print(f"    Arduino MSG: {line}")
            else: time.sleep(0.01)
        except Exception as e: 
            if debug: print(f"  read_image: Error reading line for SIZE: {e}"); return None
    if not size_line_found or size is None: 
        if debug: print(f"  read_image: Failed to get valid SIZE. Last line: '{line}'"); return None

    image_bytes = b''; bytes_received = 0; start_time = time.time()
    # Use shorter timeout for individual byte reads inside the loop
    # This timeout is handled by the outer IMAGE_READ_TIMEOUT check
    ser.timeout = 0.1
    while bytes_received < size and time.time() - start_time < IMAGE_READ_TIMEOUT:
        try:
            bytes_to_read = min(size - bytes_received, 4096)
            chunk = ser.read(bytes_to_read)
            if chunk: image_bytes += chunk; bytes_received += len(chunk)
            elif time.time() - start_time >= IMAGE_READ_TIMEOUT: # Explicitly check after read attempt
                 if debug: print(f"  read_image: Timeout reading image bytes. Got {bytes_received}/{size} bytes.")
                 ser.timeout = SERIAL_TIMEOUT # Restore default
                 return None
        except Exception as e: 
            if debug: print(f"  read_image: Error reading bytes: {e}"); ser.timeout = SERIAL_TIMEOUT; return None
    ser.timeout = SERIAL_TIMEOUT # Restore default timeout

    if bytes_received != size:
        if debug: print(f"  read_image: ERROR - Size mismatch. Expected {size}, got {bytes_received}"); return None

    end_marker_found = False; start_time = time.time(); read_timeout = 2.0; line = ""
    while time.time() - start_time < read_timeout:
        try:
             if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line == "IMAGE_END": end_marker_found = True; break
                if line.startswith("DEBUG:") or line.startswith("ERROR:"): print(f"    Arduino MSG: {line}")
             else: time.sleep(0.01)
        except Exception as e: 
            if debug: print(f"  read_image: Error reading line for END: {e}"); return None
    if not end_marker_found: 
        if debug: print(f"  read_image: Warning - Did not receive IMAGE_END. Last line: '{line}'.");

    try:
        jpg_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)
        if frame is None: 
            if debug: print("  read_image: ERROR - cv2.imdecode failed."); return None
        return frame
    except Exception as e:
        if debug: print(f"  read_image: ERROR - Exception during decode: {e}"); return None


# --- Other Helper Functions (check_image_quality, reset_camera, preview_camera, collect_gesture_data) ---
def check_image_quality(img):
    """ Basic check if the image is usable """
    if img is None: return False
    if img.shape[0] < 50 or img.shape[1] < 50: return False
    return True

def reset_camera(ser):
    """Send commands to Arduino to reset camera settings"""
    print("Resetting camera (via Arduino command r0)...")
    try:
        ser.reset_input_buffer(); ser.reset_output_buffer()
        ser.write(b'r\n'); time.sleep(0.1); ser.write(b'0\n'); ser.flush(); time.sleep(1.0)
        ser.write(b'c\n'); ser.flush(); time.sleep(0.5) # Dummy capture
        ser.reset_input_buffer(); print("Camera reset command sent.")
        return True
    except Exception as e: print(f"Error sending reset command: {e}"); return False

def preview_camera(ser, gesture_name, max_time=30):
    """Show a live preview from the camera"""
    print(f"\n=== PREVIEW: {gesture_name} ===")
    print("Position yourself. ENTER=Start Collect / ESC=Cancel.")
    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL); cv2.resizeWindow("Camera Preview", 640, 480)
    if not reset_camera(ser): print("Warning: Failed reset before preview.")
    time.sleep(0.5); start_time = time.time(); ready_to_collect = False

    while time.time() - start_time < max_time:
        img = read_image_from_serial(ser, debug=False)
        display_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Start with blank
        status_text = "Waiting for camera..."
        if img is not None: display_frame = img.copy(); status_text = f"Preview: {gesture_name}"

        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "ENTER=Start / ESC=Cancel", (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Camera Preview", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 13 or key == 10: print("Preview ended - User pressed Enter."); ready_to_collect = True; break
        elif key == 27: print("Preview cancelled by user."); ready_to_collect = False; break

    if not ready_to_collect and time.time() - start_time >= max_time:
         print("Preview timeout reached."); ready_to_collect = True
    cv2.destroyAllWindows()
    return ready_to_collect

def collect_gesture_data(ser, num_samples=50, gesture_label=0, gesture_name="neutral", skip_confirmation=False):
    """Collects and saves image data for a specific gesture."""
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{gesture_name}"
    data_dir = os.path.join("data", "raw", "images", f"{session_id}_label{gesture_label}") # Use os.path.join
    os.makedirs(data_dir, exist_ok=True); metadata = []
    print(f"\nStarting collection: {gesture_name} (Label {gesture_label})")
    display_instructions(gesture_name)

    if not skip_confirmation: print("Press ENTER for countdown..."); input()
    for i in range(3, 0, -1): print(f"{i}..."); time.sleep(0.8)
    print(f"GO! Hold '{gesture_name}' pose with slight wiggle for {num_samples} samples.")

    good_frames_collected = 0; pbar = tqdm(total=num_samples, desc=f"Collecting {gesture_name}")
    max_attempts = num_samples * 3; attempts = 0
    cv2.namedWindow("Data Collection Feed", cv2.WINDOW_NORMAL); cv2.resizeWindow("Data Collection Feed", 480, 360)

    while good_frames_collected < num_samples and attempts < max_attempts:
        attempts += 1
        img = read_image_from_serial(ser, debug=False)

        display_img = np.zeros((360, 480, 3), dtype=np.uint8) # Default blank
        status_color = (0, 0, 255) # Red for error/waiting

        if img is not None:
            if check_image_quality(img):
                timestamp = time.time()
                filename = os.path.join(data_dir, f"img_{good_frames_collected:04d}.jpg")
                cv2.imwrite(filename, img)
                metadata.append({"filename": filename, "timestamp": timestamp, "label": gesture_label, "gesture_name": gesture_name})
                good_frames_collected += 1; pbar.update(1)
                display_img = img.copy(); status_color = (0, 255, 0) # Green for success
            else:
                 print(" Low quality frame skipped.", end="")
                 display_img = img.copy(); status_color = (0, 165, 255) # Orange for low quality
                 time.sleep(0.1)
        else:
            print(" Frame capture failed.", end="")
            status_text = "Capture Fail"
            cv2.putText(display_img, status_text, (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            time.sleep(0.2)

        # Display feedback always, even on failure (shows blank screen then)
        text = f"{gesture_name}: {good_frames_collected}/{num_samples}"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Data Collection Feed", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'): print("\nCollection interrupted!"); break
        time.sleep(0.05) # Small delay between captures

    pbar.close(); cv2.destroyAllWindows()
    if good_frames_collected < num_samples: print(f"\nWarning: Only collected {good_frames_collected}/{num_samples} samples.")
    else: print(f"\nSuccessfully collected {good_frames_collected} samples for {gesture_name}.")
    meta_path = os.path.join(data_dir, "metadata.csv")
    pd.DataFrame(metadata).to_csv(meta_path, index=False); print(f"Metadata saved to {meta_path}")
    return data_dir

# --- <<< UPDATED Instructions for New Gestures >>> ---
def display_instructions(gesture_name):
    """Display instructions for the distinct static gestures."""
    instruction_img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    title_org=(30,40); line1_org=(30,100); line2_org=(30,140); line3_org=(30,180); continue_org=(150,350)
    font=cv2.FONT_HERSHEY_SIMPLEX; title_scale=0.8; line_scale=0.6; continue_scale=0.6; color=(0,0,0); thick=1; title_thick=2

    if gesture_name == "background":
        cv2.putText(instruction_img, "NEUTRAL: HAND OUT OF FRAME", title_org, font, title_scale, color, title_thick)
        cv2.putText(instruction_img, "- Keep BOTH hands completely OUTSIDE the camera view.", line1_org, font, line_scale, color, thick)
        cv2.putText(instruction_img, "- Let it record just the background.", line2_org, font, line_scale, color, thick)

    elif gesture_name == "right_one_finger": # For NEXT
        cv2.putText(instruction_img, "NEXT GESTURE: Right Hand - 1 Finger UP", title_org, font, title_scale, color, title_thick)
        cv2.putText(instruction_img, "- Show your RIGHT hand.", line1_org, font, line_scale, color, thick)
        cv2.putText(instruction_img, "- Point ONLY your INDEX finger straight up.", line2_org, font, line_scale, color, thick)
        cv2.putText(instruction_img, "- Hold pose relatively STILL (slight wiggle ok).", line3_org, font, line_scale, color, thick)

    elif gesture_name == "left_two_fingers": # For PREV
        cv2.putText(instruction_img, "PREV GESTURE: Left Hand - 2 Fingers UP ('V')", title_org, font, title_scale, color, title_thick)
        cv2.putText(instruction_img, "- Show your LEFT hand.", line1_org, font, line_scale, color, thick)
        cv2.putText(instruction_img, "- Point INDEX and MIDDLE fingers UP ('V' sign).", line2_org, font, line_scale, color, thick)
        cv2.putText(instruction_img, "- Hold pose relatively STILL (slight wiggle ok).", line3_org, font, line_scale, color, thick)

    else: # Default fallback
        cv2.putText(instruction_img, f"Gesture: {gesture_name.upper()}", title_org, font, title_scale, color, title_thick)

    cv2.putText(instruction_img, "Press any key when ready to position...", continue_org, font, continue_scale, (0, 0, 255), thick)
    cv2.imshow("Gesture Instructions", instruction_img); cv2.waitKey(0); cv2.destroyWindow("Gesture Instructions")

def optimize_camera_settings(ser):
    """Reset camera via Arduino command 'r0' """
    print("Sending camera reset/optimize command (r0)...")
    if not reset_camera(ser): print("Warning: Failed reset command.")
    else: print("Camera optimize command sent.")
    return True

# --- <<< UPDATED GESTURE LISTS >>> ---
ESSENTIAL_GESTURES = [
    ('background', 0),      # IDLE state = Hand out of view
    ('right_one_finger', 1),  # NEXT Action: Right hand, 1 finger up
    ('left_two_fingers', 2)   # PREV Action: Left hand, 2 fingers up ('V')
]
# --- <<< END UPDATED GESTURE LISTS >>> ---

AVAILABLE_GESTURES = ESSENTIAL_GESTURES # Default to essential for this final push

def run_data_collection_main():
    """Main function to run the data collection process."""
    create_project_directories(); ser = connect_to_arduino()
    if ser is None: return []
    optimize_camera_settings(ser)
    print("Taking test capture..."); test_img = read_image_from_serial(ser, debug=True)
    if test_img is None: print("\nERROR: Test capture failed! Check Arduino/Camera."); ser.close(); return []
    else: print("Test capture successful.")

    print("\n--- Data Collection Configuration ---")
    print("Collecting ESSENTIAL gestures only: background, right_one_finger, left_two_fingers")
    AVAILABLE_GESTURES = ESSENTIAL_GESTURES

    sample_count = 200 # Start with 200
    try:
        user_count = input(f"Samples per gesture? (default: {sample_count}): ").strip()
        if user_count: sample_count = max(50, int(user_count))
    except: print(f"Invalid input, using {sample_count} samples.")
    print(f"Collecting {sample_count} samples per gesture.")
    use_preview = input("Enable camera preview before each gesture? (Y/n): ").strip().lower() != 'n'
    print(f"Camera preview: {'ENABLED' if use_preview else 'DISABLED'}")

    data_dirs_collected = []
    try:
        for gesture_name, gesture_label in AVAILABLE_GESTURES:
            print("-" * 40)
            if use_preview:
                ready = preview_camera(ser, gesture_name)
                if not ready: print(f"Skipping gesture {gesture_name}."); continue
            data_dir = collect_gesture_data(ser, num_samples=sample_count, gesture_label=gesture_label, gesture_name=gesture_name, skip_confirmation=(use_preview))
            if data_dir: data_dirs_collected.append(data_dir)
            print("Pausing for 3 seconds..."); time.sleep(3.0)
    except KeyboardInterrupt: print("\nData collection interrupted.")
    except Exception as e: print(f"\nError during data collection loop: {e}")
    finally:
        if ser and ser.is_open: ser.close(); print("\nSerial port closed.")
        cv2.destroyAllWindows()

    if data_dirs_collected:
         with open('data/gesture_data_dirs.txt', 'w') as f:
             for directory in data_dirs_collected: f.write(f"{directory}\n")
         print(f"Collected data list saved: data/gesture_data_dirs.txt")
    else: print("No data collected.")
    print(f"\n===== DATA COLLECTION FINISHED ====="); print("\nNext step: Run model_training.py")
    return data_dirs_collected

# Function to run preview test (uses new gestures)
def force_preview_test():
    """Run a standalone test of the preview functionality"""
    print("\n=== TESTING CAMERA PREVIEW FUNCTIONALITY ===")
    ser = connect_to_arduino();
    if ser is None: return
    print("Optimizing camera for preview test...")
    optimize_camera_settings(ser)
    try:
        for gesture_name, _ in ESSENTIAL_GESTURES: # Use the new list
            print(f"\nTesting preview for: {gesture_name}")
            ready = preview_camera(ser, gesture_name, max_time=15)
            if not ready: print(f"Preview cancelled for {gesture_name}."); break
            time.sleep(0.5)
        print("\nPreview test sequence complete.")
    except Exception as e: print(f"An error occurred during preview test: {e}")
    finally:
        if ser and ser.is_open: ser.close(); print("Serial port closed (Preview Test).")
        cv2.destroyAllWindows()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting TinyML gesture data collection script...")
    print("Special options:")
    print("1: Normal data collection (Recommended)")
    print("2: Test preview functionality only")
    choice = input("Enter choice (1-2, default 1): ").strip()

    if choice == "2":
        force_preview_test() # Call the corrected function
    else: # Default to normal collection
        run_data_collection_main()