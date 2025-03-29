# File: data_collection.py
# Updated to use Arduino camera settings

import os
import cv2
import time
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

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

def collect_gesture_data(camera, num_samples=200, gesture_label=0, gesture_name="neutral"):
    """
    Collect image data for a specific gesture using Arduino camera
    
    Parameters:
    camera: OpenCV camera object
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
                "gesture_name": gesture_name
            })
            
            # Optional: Display frame with label
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Recording: {gesture_name}", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {i+1}/{num_samples}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Data Collection", display_frame)
            
            # Allow interruption with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Adjust this delay if needed based on your camera's performance
            # OV5642 might need a slightly longer delay
            time.sleep(0.1)  
            
    # Save metadata
    pd.DataFrame(metadata).to_csv(f"{data_dir}/metadata.csv", index=False)
    
    print(f"Collected {len(metadata)} images for gesture '{gesture_name}'")
    cv2.destroyAllWindows()
    
    return data_dir

def run_data_collection():
    """Run the complete data collection process for all gestures using Arduino camera"""
    create_project_directories()
    
    # Initialize Arduino camera
    camera = setup_arduino_camera()
    if camera is None:
        print("Please run camera_test.py first to configure the Arduino camera.")
        return []
    
    # Define gestures to collect
    gestures = {
        0: 'neutral',
        1: 'turn_right',
        2: 'turn_left',
        3: 'swipe_next',
        4: 'swipe_previous',
        5: 'pre_turn_right',  # Predictive states - beginning to turn
        6: 'pre_turn_left'    # Predictive states - beginning to turn
    }
    
    # Collect data for each gesture
    data_dirs = []
    
    try:
        for label, name in gestures.items():
            print(f"\nPreparing to collect data for gesture: {name} (label {label})")
            print("Position yourself and press Enter when ready...")
            input()
            
            # Collect data for this gesture
            data_dir = collect_gesture_data(
                camera, 
                num_samples=200,  # Adjust based on Arduino camera performance
                gesture_label=label,
                gesture_name=name
            )
            
            data_dirs.append(data_dir)
            
            print(f"Completed collection for {name}. Take a short break...")
            time.sleep(3)
    
    finally:
        # Release camera
        camera.release()
        cv2.destroyAllWindows()
        print("Data collection complete. Camera released.")
    
    # Save list of all data directories
    with open('data/gesture_data_dirs.txt', 'w') as f:
        for directory in data_dirs:
            f.write(f"{directory}\n")
    
    print(f"Data collection complete. Collected data for {len(gestures)} gestures.")
    return data_dirs

if __name__ == "__main__":
    print("Starting data collection with Arduino camera...")
    data_dirs = run_data_collection()
    if data_dirs:
        print(f"Successfully collected data in {len(data_dirs)} directories.")
    else:
        print("Data collection failed or was interrupted.")