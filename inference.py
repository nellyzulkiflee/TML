# File: inference.py
# Updated for Arduino camera inference

import cv2
import numpy as np
import tensorflow as tf
import time
import serial
from collections import deque
import pickle
import os

# Constants
IMG_SIZE = (96, 96)
SEQUENCE_LENGTH = 10  # Should match training sequence length
DETECTION_THRESHOLD = 0.7  # Confidence threshold
PREDICTION_THRESHOLD = 0.6  # Threshold for predictive actions

# Command definitions (match Arduino)
CMD_NEXT_SLIDE = 1
CMD_PREV_SLIDE = 2

# Class names
CLASS_NAMES = [
    'neutral', 
    'turn_right', 
    'turn_left', 
    'swipe_next', 
    'swipe_previous',
    'pre_turn_right',
    'pre_turn_left'
]

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

def create_feature_extractor():
    """Create MobileNetV2 feature extractor"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    
    return model

def load_model(model_path='models/gesture_model_best.h5'):
    """Load trained gesture model"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return None
        
    return tf.keras.models.load_model(model_path)

def preprocess_frame(frame):
    """Preprocess frame for feature extraction"""
    # Resize
    img = cv2.resize(frame, IMG_SIZE)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img = img / 255.0
    
    return img

def find_arduino_port():
    """Find the Arduino serial port"""
    import serial.tools.list_ports
    
    # Common port name patterns for Arduino
    arduino_patterns = [
        'arduino', 'ttyACM', 'ttyUSB', 'COM'
    ]
    
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("No serial ports found")
        return None
    
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"{i+1}. {port.device} - {port.description}")
        
        # Check if any pattern matches this port
        for pattern in arduino_patterns:
            if pattern.lower() in port.description.lower() or pattern.lower() in port.device.lower():
                print(f"  ↳ Possibly Arduino (matches pattern '{pattern}')")
    
    # Ask user to select port
    try:
        selection = int(input("\nSelect Arduino port number (or 0 to skip): "))
        if selection == 0:
            return None
        if 1 <= selection <= len(ports):
            selected_port = ports[selection-1].device
            print(f"Selected: {selected_port}")
            return selected_port
        else:
            print("Invalid selection")
            return None
    except ValueError:
        print("Invalid input")
        return None

def run_inference():
    """Run real-time inference and control with Arduino camera"""
    # Load model
    gesture_model = load_model()
    if gesture_model is None:
        return
        
    feature_extractor = create_feature_extractor()
    
    # Initialize Arduino camera
    camera = setup_arduino_camera()
    if camera is None:
        print("Arduino camera initialization failed.")
        print("Please run camera_test.py first to configure your camera.")
        return
    
    # Try to connect to Arduino for slide control
    arduino_port = find_arduino_port()
    if arduino_port:
        try:
            arduino = serial.Serial(arduino_port, 115200, timeout=1)
            print(f"Connected to Arduino on {arduino_port}")
            arduino_connected = True
            time.sleep(2)  # Allow time for connection to establish
        except:
            print(f"Could not connect to Arduino on {arduino_port}. Running in detection-only mode.")
            arduino_connected = False
    else:
        print("No Arduino port selected. Running in detection-only mode.")
        arduino_connected = False
    
    # Initialize sequence buffer
    feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize prediction trackers
    last_command_time = time.time()
    command_cooldown = 1.0  # Seconds between commands
    prediction_active = False
    
    # Track actual vs predicted actions
    detected_gestures = []
    
    # Initialize tracking variables
    consecutive_pre_turn_right = 0
    consecutive_pre_turn_left = 0
    
    print("Starting inference loop with Arduino camera. Press 'q' to quit.")
    
    # Performance tracking
    frame_times = deque(maxlen=30)  # Track last 30 frames for FPS calculation
    
    try:
        while True:
            frame_start_time = time.time()
            
            # Capture frame from Arduino camera
            ret, frame = camera.read()
            if not ret:
                print("Error reading from Arduino camera")
                # Wait a bit and try again
                time.sleep(0.1)
                continue
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Preprocess frame
            preprocessed = preprocess_frame(frame)
            
            # Extract features
            features = feature_extractor.predict(np.expand_dims(preprocessed, axis=0))[0]
            
            # Add to buffer
            feature_buffer.append(features)
            
            # Only run inference when we have enough frames
            if len(feature_buffer) == SEQUENCE_LENGTH:
                # Convert buffer to numpy array
                sequence = np.array([list(feature_buffer)])
                
                # Get prediction
                prediction = gesture_model.predict(sequence)[0]
                gesture_idx = np.argmax(prediction)
                confidence = prediction[gesture_idx]
                gesture_name = CLASS_NAMES[gesture_idx]
                
                # Display result on frame
                display_text = f"{gesture_name}: {confidence:.2f}"
                cv2.putText(display_frame, display_text, (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Track gesture for analysis
                detected_gestures.append((gesture_name, confidence))
                
                # Check for commands with cooldown
                current_time = time.time()
                if current_time - last_command_time > command_cooldown:
                    # If confidence threshold met
                    if confidence > DETECTION_THRESHOLD:
                        # Handle predictive states
                        if gesture_name == 'pre_turn_right':
                            consecutive_pre_turn_right += 1
                            consecutive_pre_turn_left = 0
                            
                            # If we've seen enough consecutive frames, predict next action
                            if consecutive_pre_turn_right >= 3 and not prediction_active:
                                print("PREDICTED: Turn Right (Next Slide)")
                                prediction_active = True
                                
                                # Display prediction on frame
                                cv2.putText(display_frame, "PREDICTED: Next Slide", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                                # Send command if Arduino connected
                                if arduino_connected:
                                    arduino.write(bytes([CMD_NEXT_SLIDE]))
                                    last_command_time = current_time
                        
                        elif gesture_name == 'pre_turn_left':
                            consecutive_pre_turn_left += 1
                            consecutive_pre_turn_right = 0
                            
                            # If we've seen enough consecutive frames, predict next action
                            if consecutive_pre_turn_left >= 3 and not prediction_active:
                                print("PREDICTED: Turn Left (Previous Slide)")
                                prediction_active = True
                                
                                # Display prediction on frame
                                cv2.putText(display_frame, "PREDICTED: Previous Slide", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                
                                # Send command if Arduino connected
                                if arduino_connected:
                                    arduino.write(bytes([CMD_PREV_SLIDE]))
                                    last_command_time = current_time
                        
                        # Handle actual gestures
                        elif gesture_name == 'turn_right' or gesture_name == 'swipe_next':
                            print("DETECTED: Next Slide")
                            prediction_active = False
                            consecutive_pre_turn_right = 0
                            consecutive_pre_turn_left = 0
                            
                            # Display action on frame
                            cv2.putText(display_frame, "NEXT SLIDE", (20, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            
                            # Send command if Arduino connected
                            if arduino_connected:
                                arduino.write(bytes([CMD_NEXT_SLIDE]))
                                last_command_time = current_time
                        
                        elif gesture_name == 'turn_left' or gesture_name == 'swipe_previous':
                            print("DETECTED: Previous Slide")
                            prediction_active = False
                            consecutive_pre_turn_right = 0
                            consecutive_pre_turn_left = 0
                            
                            # Display action on frame
                            cv2.putText(display_frame, "PREVIOUS SLIDE", (20, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            
                            # Send command if Arduino connected
                            if arduino_connected:
                                arduino.write(bytes([CMD_PREV_SLIDE]))
                                last_command_time = current_time
                        
                        elif gesture_name == 'neutral':
                            # Reset prediction state if we see neutral gesture
                            prediction_active = False
                            consecutive_pre_turn_right = 0
                            consecutive_pre_turn_left = 0
            
            # Track frame processing time for FPS calculation
            frame_times.append(time.time() - frame_start_time)
            avg_frame_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # Display FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display Arduino connection status
            status_color = (0, 255, 0) if arduino_connected else (0, 0, 255)
            status_text = "Arduino: Connected" if arduino_connected else "Arduino: Not Connected"
            cv2.putText(display_frame, status_text, (20, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display frame
            cv2.imshow("Arduino Camera Gesture Detection", display_frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Add a small delay if needed for Arduino camera stability
            # Adjust this based on your camera's performance
            if fps > 15:  # If running too fast, add a small delay
                time.sleep(0.01)
    
    finally:
        # Clean up
        camera.release()
        cv2.destroyAllWindows()
        
        if arduino_connected:
            arduino.close()
        
        # Save detected gestures for analysis
        with open('detected_gestures_log.pkl', 'wb') as f:
            pickle.dump(detected_gestures, f)
        
        print("Inference stopped")

if __name__ == "__main__":
    print("Starting real-time gesture detection with Arduino camera...")
    run_inference()