# File: data_collection.py
# Enhanced with better validation splitting and transition collection

import os
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import shutil

def get_arduino_camera_settings():
    """
    Get the saved Arduino camera settings
    Returns a dict with camera settings or None if file not found
    """
    if not os.path.exists('arduino_camera_settings.txt'):
        return None
    
    settings = {}
    with open('arduino_camera_settings.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            try:
                settings[key] = int(float(value))
            except:
                settings[key] = value
    
    return settings

def create_project_directories():
    """Create necessary directories for the project"""
    dirs = [
        'data/raw/images',
        'data/processed/images',
        'data/processed/features',
        'data/validation',  # Added validation directory
        'models',
        'arduino_deploy'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created successfully")

def setup_arduino_camera():
    """
    Initialize the Arduino OV5642 camera using saved settings
    """
    # Get saved settings
    settings = get_arduino_camera_settings()
    
    if settings is None:
        print("Arduino camera settings not found!")
        print("Please run camera_test.py first to detect and configure your Arduino camera.")
        return None
    
    camera_index = settings['index']
    
    # Initialize camera
    camera = cv2.VideoCapture(camera_index)
    
    if not camera.isOpened():
        print(f"Error: Could not open Arduino camera with index {camera_index}.")
        return None
    
    # Apply saved settings
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    # Some cameras support FPS setting, others don't
    try:
        camera.set(cv2.CAP_PROP_FPS, settings['fps'])
    except:
        pass
    
    # Read a test frame to confirm camera is working
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame from Arduino camera.")
        camera.release()
        return None
    
    print(f"Arduino camera initialized successfully:")
    print(f"- Resolution: {settings['width']}x{settings['height']}")
    print(f"- Target FPS: {settings['fps']}")
    return camera

def collect_gesture_data(camera, num_samples=200, gesture_label=0, gesture_name="neutral", 
                          show_countdown=True, preview_time=3):
    """
    Collect image data for a specific gesture using Arduino camera
    
    Parameters:
    camera: OpenCV camera object
    num_samples: Number of images to collect
    gesture_label: Numeric label for the gesture
    gesture_name: Name of the gesture for display
    show_countdown: Whether to show countdown before collection
    preview_time: How long to show preview before collecting data
    
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
    
    # Preview window to help user position correctly
    if preview_time > 0:
        print(f"Preview window for {gesture_name}. Position yourself correctly.")
        end_time = time.time() + preview_time
        while time.time() < end_time:
            ret, frame = camera.read()
            if ret:
                preview = frame.copy()
                cv2.putText(preview, f"PREVIEW: {gesture_name}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"Get ready... {int(end_time - time.time())}s", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Data Collection Preview", preview)
                cv2.waitKey(1)
    
    # Countdown
    if show_countdown:
        print(f"Starting data collection for gesture: {gesture_name} (label {gesture_label})")
        print("Get ready...")
        
        for i in range(3, 0, -1):
            print(f"{i}...")
            time.sleep(1)
    
    print("COLLECTING DATA - Perform the gesture continuously")
    
    # Collect images
    for i in tqdm(range(num_samples)):
        # Capture frame
        ret, frame = camera.read()
        
        if ret:
            # Save image
            timestamp = time.time()
            filename = f"{data_dir}/img_{i:04d}.jpg"
            cv2.imwrite(filename, frame)
            
            # Add to metadata
            metadata.append({
                "filename": filename,
                "timestamp": timestamp,
                "label": gesture_label,
                "gesture_name": gesture_name,
                "frame_index": i
            })
            
            # Optional: Display frame with label
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Recording: {gesture_name}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {i+1}/{num_samples}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add visual progress bar
            progress = int((i / num_samples) * display_frame.shape[1])
            cv2.rectangle(display_frame, (0, display_frame.shape[0] - 20), 
                         (progress, display_frame.shape[0]), (0, 255, 0), -1)
            
            cv2.imshow("Data Collection", display_frame)
            
            # Allow interruption with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection interrupted!")
                break
                
            # Adjust this delay if needed based on your camera's performance
            time.sleep(0.1)  
            
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{data_dir}/metadata.csv", index=False)
    
    print(f"Collected {len(metadata)} images for gesture '{gesture_name}'")
    cv2.destroyAllWindows()
    
    return data_dir

def collect_transition_data(camera, transition_name, from_gesture, to_gesture, num_samples=50):
    """
    Collect transition data between two gestures
    
    Parameters:
    camera: OpenCV camera object
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
    
    # Collect transition images
    for i in tqdm(range(num_samples)):
        # Calculate interpolation factor (0.0 to 1.0)
        transition_factor = i / (num_samples - 1)
        
        # Capture frame
        ret, frame = camera.read()
        
        if ret:
            # Save image
            timestamp = time.time()
            filename = f"{transition_dir}/transition_{i:04d}.jpg"
            cv2.imwrite(filename, frame)
            
            # Create interpolated label
            # For early frames, label as starting gesture
            # For middle frames, create an intermediate "pre_" label if available
            # For late frames, label as ending gesture
            if transition_factor < 0.3:
                # Beginning of transition - label as 'from' gesture
                current_label = from_label
                current_name = from_name
            elif transition_factor > 0.7:
                # End of transition - label as 'to' gesture
                current_label = to_label
                current_name = to_name
            else:
                # Middle of transition - if we have a 'pre_' label, use it
                pre_name = f"pre_{to_name}"
                if pre_name in [g for g, _ in AVAILABLE_GESTURES]:
                    pre_label = [l for g, l in AVAILABLE_GESTURES if g == pre_name][0]
                    current_label = pre_label
                    current_name = pre_name
                else:
                    # Otherwise, use a weighted label between from and to
                    # This is just for metadata - the actual training will use the image
                    current_label = from_label  # Simplified: use 'from' label
                    current_name = f"transition_{from_name}_to_{to_name}"
            
            # Add to metadata
            metadata.append({
                "filename": filename,
                "timestamp": timestamp,
                "label": current_label,
                "gesture_name": current_name,
                "transition_factor": transition_factor,
                "from_gesture": from_name,
                "to_gesture": to_name,
                "frame_index": i
            })
            
            # Display frame with transition info
            display_frame = frame.copy()
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
            
            # Small delay to make collection smoother
            time.sleep(0.1)
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{transition_dir}/metadata.csv", index=False)
    
    print(f"Collected {len(metadata)} transition frames from '{from_name}' to '{to_name}'")
    cv2.destroyAllWindows()
    
    return transition_dir

def create_validation_split(data_dirs, validation_percentage=20):
    """
    Create a validation split from collected data
    
    Parameters:
    data_dirs: List of data directories
    validation_percentage: Percentage of data to use for validation
    
    Returns:
    List of validation directories
    """
    print(f"\nCreating validation split ({validation_percentage}%)...")
    
    validation_dirs = []
    
    for data_dir in data_dirs:
        # Get metadata
        metadata_file = os.path.join(data_dir, 'metadata.csv')
        if not os.path.exists(metadata_file):
            print(f"Warning: No metadata found for {data_dir}, skipping")
            continue
        
        metadata = pd.read_csv(metadata_file)
        
        # Get gesture info from directory name
        dir_name = os.path.basename(data_dir)
        if 'transition' in dir_name:
            # For transitions, use the last part of directory name
            parts = dir_name.split('_')
            if len(parts) >= 4:
                gesture_name = parts[-1]  # Simplified - just use target gesture
            else:
                gesture_name = "transition"
        else:
            # For regular gestures, extract from directory name
            parts = dir_name.split('_')
            if len(parts) >= 2 and parts[-1].startswith('label'):
                gesture_name = parts[-2]
            else:
                gesture_name = "unknown"
        
        # Create validation directory
        validation_dir = f"data/validation/{gesture_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(validation_dir, exist_ok=True)
        
        # Randomly select validation samples
        total_samples = len(metadata)
        validation_count = int(total_samples * validation_percentage / 100)
        
        # Get random indices for validation
        validation_indices = np.random.choice(total_samples, validation_count, replace=False)
        
        # Copy validation files and create validation metadata
        validation_metadata = []
        
        for idx in validation_indices:
            # Get file info
            file_info = metadata.iloc[idx]
            src_file = file_info['filename']
            
            if not os.path.exists(src_file):
                print(f"Warning: File {src_file} not found, skipping")
                continue
            
            # Create destination filename
            base_name = os.path.basename(src_file)
            dst_file = os.path.join(validation_dir, base_name)
            
            # Copy file
            shutil.copy2(src_file, dst_file)
            
            # Update metadata with new filename
            file_info = file_info.copy()
            file_info['filename'] = dst_file
            validation_metadata.append(file_info)
        
        # Save validation metadata
        pd.DataFrame(validation_metadata).to_csv(f"{validation_dir}/metadata.csv", index=False)
        
        print(f"Created validation split for {gesture_name}: {len(validation_metadata)} samples")
        validation_dirs.append(validation_dir)
    
    return validation_dirs

# Define available gestures
AVAILABLE_GESTURES = [
    ('neutral', 0),
    ('turn_right', 1),
    ('turn_left', 2),
    ('swipe_next', 3),
    ('swipe_previous', 4),
    ('pre_turn_right', 5),  # Predictive states - beginning to turn
    ('pre_turn_left', 6)    # Predictive states - beginning to turn
]

# Define important transitions to collect
IMPORTANT_TRANSITIONS = [
    ('neutral', 'pre_turn_right'),
    ('pre_turn_right', 'turn_right'),
    ('turn_right', 'neutral'),
    ('neutral', 'pre_turn_left'),
    ('pre_turn_left', 'turn_left'),
    ('turn_left', 'neutral')
]

def run_data_collection():
    """Run the complete data collection process for all gestures using Arduino camera"""
    create_project_directories()
    
    # Initialize Arduino camera
    camera = setup_arduino_camera()
    if camera is None:
        print("Please run camera_test.py first to configure the Arduino camera.")
        return []
    
    # Collect data for each gesture
    data_dirs = []
    
    try:
        # First, collect data for each base gesture
        for gesture_name, gesture_label in AVAILABLE_GESTURES:
            print(f"\nPreparing to collect data for gesture: {gesture_name} (label {gesture_label})")
            print("Position yourself and press Enter when ready...")
            input()
            
            # Collect data for this gesture
            data_dir = collect_gesture_data(
                camera, 
                num_samples=200,
                gesture_label=gesture_label,
                gesture_name=gesture_name,
                preview_time=3  # Show preview window for 3 seconds
            )
            
            data_dirs.append(data_dir)
            
            print(f"Completed collection for {gesture_name}. Take a short break...")
            time.sleep(3)
        
        # Then collect transition data
        for from_gesture_name, to_gesture_name in IMPORTANT_TRANSITIONS:
            # Get labels for these gestures
            from_label = next((label for name, label in AVAILABLE_GESTURES if name == from_gesture_name), None)
            to_label = next((label for name, label in AVAILABLE_GESTURES if name == to_gesture_name), None)
            
            if from_label is None or to_label is None:
                print(f"Warning: Could not find labels for transition {from_gesture_name} → {to_gesture_name}")
                continue
            
            # Collect transition data
            transition_dir = collect_transition_data(
                camera,
                f"{from_gesture_name}_to_{to_gesture_name}",
                (from_gesture_name, from_label),
                (to_gesture_name, to_label),
                num_samples=50  # Collect 50 frames for each transition
            )
            
            data_dirs.append(transition_dir)
            
            print(f"Completed transition collection. Take a short break...")
            time.sleep(3)
        
        # Create validation split
        validation_dirs = create_validation_split(data_dirs, validation_percentage=20)
        print(f"Created {len(validation_dirs)} validation directories")
    
    finally:
        # Release camera
        camera.release()
        cv2.destroyAllWindows()
        print("Data collection complete. Camera released.")
    
    # Save list of all data directories
    all_dirs = data_dirs.copy()
    with open('data/gesture_data_dirs.txt', 'w') as f:
        for directory in all_dirs:
            f.write(f"{directory}\n")
    
    print(f"Data collection complete. Collected data in {len(all_dirs)} directories.")
    return all_dirs

if __name__ == "__main__":
    print("Starting data collection with Arduino camera...")
    data_dirs = run_data_collection()
    if data_dirs:
        print(f"Successfully collected data in {len(data_dirs)} directories.")
    else:
        print("Data collection failed or was interrupted.")