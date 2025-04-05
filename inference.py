import serial
import serial.tools.list_ports
import time
import cv2
import numpy as np
import pyautogui
import queue
import threading
from collections import deque

# Attempt to import TFLite runtime or full TensorFlow
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("Using full TensorFlow")
    except ImportError:
        print("ERROR: Failed to import tflite_runtime or tensorflow.")
        print("Please install one: pip install tflite-runtime OR pip install tensorflow")
        exit()

# --- Constants ---
# *** KEEP USING THE MODEL FROM THE ~98% ACCURACY RUN ***
# *** Use _float.tflite if it exists and you tested it, otherwise use _quantized.tflite ***
MODEL_PATH = 'models/gesture_simple_cnn_float.tflite' # Or _quantized.tflite
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0
IMAGE_READ_TIMEOUT = 10.0
TARGET_IMG_SIZE = (32, 32) # Should match training

# --- VERIFY THIS MAPPING against your class_map.pkl ---
# Based on last training log: {'neutral': 0, 'swipe_left': 1, 'swipe_right': 2}
GESTURE_MAP = { 0: "IDLE", 1: "PREV", 2: "NEXT" }
ACTION_MAP = { "PREV": "left", "NEXT": "right" }
try: IDLE_INDEX = [k for k, v in GESTURE_MAP.items() if v == "IDLE"][0]
except IndexError: IDLE_INDEX = 0

# --- Parameters for Tuning ---
GESTURE_COOLDOWN_S = 1.0
CONFIDENCE_THRESHOLD = 0.40 # Keep desired threshold for now
SMOOTHING_WINDOW = 3
IDLE_OVERRIDE_THRESHOLD = 0.30 # Keep override logic for now

# --- Global state variables ---
last_action_time = 0
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
debug_mode = False # Toggle with 'd' key

# --- Helper Functions (Keep as they were) ---
def find_arduino_port():
    """Find Arduino port"""
    # ... (code from previous version) ...
    print("Scanning for Arduino ports...")
    ports = list(serial.tools.list_ports.comports())
    available_ports = []
    for i, port in enumerate(ports):
        print(f"{i+1}: {port.device} - {port.description}")
        if "Arduino" in port.description or "USB Serial Device" in port.description or "CH340" in port.description or "CP210x" in port.description:
             available_ports.append(port)
    if not available_ports: print("Error: No potential Arduino ports found."); return None
    if len(available_ports) == 1: print(f"Auto-selected: {available_ports[0].device}"); return available_ports[0].device
    while True:
        try:
            choice = input(f"Select port number (1-{len(ports)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(ports): return ports[idx].device
            else: print("Invalid selection.")
        except (ValueError, IndexError): print("Invalid input.")

def connect_to_arduino(port=None, baud_rate=SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT):
    """Connect to Arduino"""
    # ... (code from previous version) ...
    if port is None: port = find_arduino_port()
    if port is None: print("No Arduino port found!"); return None
    try:
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print(f"Connecting to Arduino on {port}...")
        time.sleep(2)
        ser.reset_input_buffer()
        print("Connected successfully.")
        return ser
    except Exception as e:
        print(f"Error connecting to Arduino on {port}: {e}")
        return None

def read_image_from_serial(ser, debug=False):
    """Reads an image from the serial port"""
    # ... (Use the working version from previous steps - code omitted for brevity) ...
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

    image_bytes = b''; bytes_received = 0; start_time = time.time(); ser.timeout = 0.1
    while bytes_received < size and time.time() - start_time < IMAGE_READ_TIMEOUT:
        try:
            bytes_to_read = min(size - bytes_received, 4096)
            chunk = ser.read(bytes_to_read)
            if chunk: image_bytes += chunk; bytes_received += len(chunk)
            elif time.time() - start_time >= IMAGE_READ_TIMEOUT: break
        except Exception as e: 
            if debug: print(f"  read_image: Error reading bytes: {e}"); ser.timeout = SERIAL_TIMEOUT; return None
    ser.timeout = SERIAL_TIMEOUT
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
    # if not end_marker_found: if debug: print(f"  read_image: Warning - Did not receive IMAGE_END. Last line: '{line}'.");

    try:
        jpg_np = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)
        if frame is None: 
            if debug: print("  read_image: ERROR - cv2.imdecode failed."); return None
        return frame
    except Exception as e:
        if debug: print(f"  read_image: ERROR - Exception during decode: {e}"); return None


def preprocess_image_for_tflite(img, input_details):
    """Prepares the received image for the TFLite model with extra checks."""
    global debug_mode
    try:
        input_shape = input_details[0]['shape']; input_dtype = input_details[0]['dtype']
        height = input_shape[1]; width = input_shape[2]; channels = input_shape[3]
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        if channels == 1: gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        else: gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_float = gray_img.astype(np.float32) / 255.0
        # if debug_mode and np.random.rand() < 0.1: print(f"  DEBUG Preproc Float Range: min={np.min(img_float):.2f}, max={np.max(img_float):.2f}") # Reduce noise

        if channels == 1 and len(img_float.shape) == 2: img_float_ch = np.expand_dims(img_float, axis=-1)
        else: img_float_ch = img_float

        if input_dtype == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            if 'quant_params_printed' not in preprocess_image_for_tflite.__dict__:
                 print(f"  Quantizing INT8 Input using: scale={input_scale:.6f}, zero_point={input_zero_point}")
                 preprocess_image_for_tflite.quant_params_printed = True
            img_quantized_float = (img_float_ch / input_scale) + input_zero_point
            # if debug_mode and np.random.rand() < 0.1: print(f"  DEBUG Pre-Clip Quant Range: min={np.min(img_quantized_float):.2f}, max={np.max(img_quantized_float):.2f}")
            img_quantized_clipped = np.clip(img_quantized_float, -128, 127)
            input_data_final = img_quantized_clipped.astype(input_dtype)
            # if debug_mode and np.random.rand() < 0.1: print(f"  DEBUG Final INT8 Input Range: min={np.min(input_data_final)}, max={np.max(input_data_final)}")
        elif input_dtype == np.float32:
            input_data_final = img_float_ch
        else: print(f"ERROR: Unsupported model input dtype: {input_dtype}"); return None

        input_data = np.expand_dims(input_data_final, axis=0)
        assert input_data.shape == tuple(input_shape), f"Shape mismatch: {input_data.shape} vs {input_shape}"
        return input_data
    except Exception as e: print(f"Error during preprocessing: {e}"); return None

# --- Main Inference Loop ---
def main():
    global last_action_time, debug_mode, prediction_history

    # --- Setup ---
    ser = connect_to_arduino()
    if ser is None: return

    print(f"\nLoading TFLite model: {MODEL_PATH}")
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_input_type = input_details[0]['dtype']; model_output_type = output_details[0]['dtype']
        print("Model loaded successfully."); print(f"  Model Input Type: {model_input_type}"); print(f"  Model Output Type: {model_output_type}")
        if model_input_type == np.int8: input_scale, input_zero_point = input_details[0]['quantization'] # Get params now if needed later
        else: input_scale, input_zero_point = 1.0, 0 # Dummy values for float
        if model_input_type == np.int8 and (not input_scale or input_zero_point is None): print("\nERROR: INT8 Input requires quantization params!"); ser.close(); return
        if model_output_type == np.int8 and ('quantization' not in output_details[0] or not output_details[0]['quantization'] or not output_details[0]['quantization'][0]): print("\nERROR: INT8 Output requires quantization params!"); ser.close(); return
    except Exception as e: print(f"ERROR: Failed to load TFLite model: {e}"); ser.close(); return

    print("\nStarting inference loop... Press Ctrl+C to exit.")
    print("Ensure PowerPoint is the active window.")
    print(f"Actions trigger if confidence >= {CONFIDENCE_THRESHOLD:.2f} for {SMOOTHING_WINDOW} consecutive frames.")
    print(f"IDLE Override Threshold: {IDLE_OVERRIDE_THRESHOLD:.2f}")

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL); cv2.resizeWindow("Camera Feed", 480, 360)
    # --- <<< ADDED WINDOW FOR PREPROCESSED INPUT >>> ---
    cv2.namedWindow("Preprocessed Input to Model", cv2.WINDOW_NORMAL); cv2.resizeWindow("Preprocessed Input to Model", 256, 256)

    try:
        while True:
            current_frame = read_image_from_serial(ser, debug=False)

            if current_frame is not None:
                display_frame = current_frame.copy()
                input_data = preprocess_image_for_tflite(current_frame, input_details)

                if input_data is not None:
                    # --- <<< ADDED VISUAL & NUMERICAL DEBUG OF INPUT >>> ---
                    try:
                        vis_img = input_data[0] # Get the (32, 32, 1) image data
                        vis_dtype = vis_img.dtype
                        vis_min = np.min(vis_img)
                        vis_max = np.max(vis_img)
                        # Print check periodically or if debug mode is on
                        if debug_mode: print(f"  DEBUG Input Tensor Check: dtype={vis_dtype}, min={vis_min:.2f}, max={vis_max:.2f}")

                        # Convert back roughly to 0-255 range for viewing
                        vis_img_displayable = None
                        if vis_dtype == np.int8:
                             # Use scale/zero_point obtained during setup
                             vis_img_float = (vis_img.astype(np.float32) - input_zero_point) * input_scale
                             vis_img_uint8 = np.clip(vis_img_float * 255.0, 0, 255).astype(np.uint8)
                             vis_img_displayable = vis_img_uint8
                        elif vis_dtype == np.float32:
                             vis_img_uint8 = np.clip(vis_img * 255.0, 0, 255).astype(np.uint8)
                             vis_img_displayable = vis_img_uint8
                        else:
                             vis_img_displayable = vis_img.astype(np.uint8) # Fallback

                        # Resize for better visibility
                        if vis_img_displayable is not None:
                             vis_img_large = cv2.resize(vis_img_displayable, (256, 256), interpolation=cv2.INTER_NEAREST)
                             cv2.imshow("Preprocessed Input to Model", vis_img_large)
                        else: # Show blank if conversion failed
                             blank_preproc = np.zeros((256, 256, 1), dtype=np.uint8)
                             cv2.imshow("Preprocessed Input to Model", blank_preproc)

                    except Exception as e_vis:
                        print(f"Error in visualization/check: {e_vis}")
                    # --- <<< END VISUAL & NUMERICAL DEBUG >>> ---


                    # 3. Run Inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    inference_start_time = time.time()
                    interpreter.invoke()
                    inference_duration = time.time() - inference_start_time
                    output_data = interpreter.get_tensor(output_details[0]['index'])

                    predicted_label = "Unknown"; confidence = 0.0;

                    try:
                         # Process Output (Handles INT8 or FLOAT)
                         if model_output_type == np.int8:
                              output_scale, output_zero_point = output_details[0]['quantization']
                              output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
                         else: output_float = output_data
                         if len(output_float.shape) == 2 and output_float.shape[0] == 1: scores = output_float[0]
                         else: raise ValueError(f"Unexpected output shape: {output_float.shape}")
                         # Apply softmax if needed
                         if 'tf' in globals() and not np.isclose(np.sum(np.exp(scores)), 1.0, atol=0.1): scores = tf.nn.softmax(scores).numpy()
                         elif not np.isclose(np.sum(np.exp(scores)), 1.0, atol=0.1): exp_scores = np.exp(scores - np.max(scores)); scores = exp_scores / np.sum(exp_scores)

                         predicted_index = np.argmax(scores)
                         confidence = np.max(scores)
                         predicted_label = GESTURE_MAP.get(predicted_index, "Unknown")
                         idle_confidence = scores[IDLE_INDEX]

                         # Update History with IDLE Override Logic
                         final_label_for_history = "IDLE"
                         if predicted_label != "IDLE" and confidence >= CONFIDENCE_THRESHOLD:
                             final_label_for_history = predicted_label
                         elif predicted_label != "IDLE" and confidence < CONFIDENCE_THRESHOLD:
                             if idle_confidence >= IDLE_OVERRIDE_THRESHOLD: final_label_for_history = "IDLE"
                             else: final_label_for_history = "IDLE" # Treat as IDLE if low conf swipe AND low conf idle
                         else: final_label_for_history = "IDLE"
                         prediction_history.append(final_label_for_history)
                         # End History Update

                         # Display & Print
                         hist_str = "".join([p[0] if p != "IDLE" else "-" for p in prediction_history])
                         text = f"{predicted_label} ({confidence:.2f})"
                         cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                         print(f"Pred: {predicted_label:<6} | Conf: {confidence:.3f} | IdleConf: {idle_confidence:.3f} | Hist: {hist_str}")

                         # Action Logic (Smoothed + Cooldown)
                         perform_action = False; action_to_perform = "IDLE"
                         if len(prediction_history) == SMOOTHING_WINDOW:
                             first_pred_in_window = prediction_history[0]
                             if first_pred_in_window != "IDLE" and all(p == first_pred_in_window for p in prediction_history):
                                 action_to_perform = first_pred_in_window; perform_action = True
                         current_time = time.time(); cooldown_ok = (current_time - last_action_time > GESTURE_COOLDOWN_S)
                         if perform_action and cooldown_ok:
                             pyautogui_key = ACTION_MAP.get(action_to_perform)
                             if pyautogui_key:
                                 print(f"  >>> SMOOTHED ACTION (Hist based): {action_to_perform} ({pyautogui_key})")
                                 pyautogui.press(pyautogui_key)
                                 last_action_time = current_time
                                 prediction_history.clear()
                         # End Action Logic

                    except Exception as e: print(f"Error processing TFLite output: {e}")
                else: print("Preprocessing failed.")

                cv2.imshow("Camera Feed", display_frame)
            else: print("Failed to get frame from Arduino."); time.sleep(0.5)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): print("Exit requested."); break
            elif key == ord('d'): debug_mode = not debug_mode; print(f"Python debug mode: {'ON' if debug_mode else 'OFF'}")

    except KeyboardInterrupt: print("\nCtrl+C detected. Exiting...")
    except Exception as e: print(f"\nAn error occurred in the main loop: {e}"); traceback.print_exc()
    finally:
        if ser and ser.is_open: ser.close(); print("Serial port closed.")
        cv2.destroyAllWindows(); print("Application finished.")

if __name__ == "__main__":
    main()