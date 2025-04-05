# File: model_training.py
# Purpose: Process collected image data and train a SIMPLE gesture recognition model
#          using direct grayscale pixel input.
# Version: Reverted to Simple CNN (No Aug/BN) to replicate previous high-accuracy run.
#          Includes .keras saving format fix.

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import cv2
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.utils import class_weight
import random
import traceback # For detailed error printing

# --- Constants ---
TARGET_IMG_SIZE = (32, 32) # 32x32 pixels
TARGET_CHANNELS = 1 # Grayscale

# Use the updated gesture names
ESSENTIAL_GESTURES = ['neutral', 'swipe_left', 'swipe_right']

# --- Function to Preprocess Images Directly ---
def preprocess_images_for_simple_model(data_dirs=None, target_size=TARGET_IMG_SIZE):
    """Loads images, converts to grayscale, resizes, normalizes."""
    if data_dirs is None:
        try:
            with open('data/gesture_data_dirs.txt', 'r') as f: data_dirs = [line.strip() for line in f.readlines()]
        except Exception as e: print(f"ERROR reading data/gesture_data_dirs.txt: {e}"); return None, None, None, None

    print(f"Preprocessing images from {len(data_dirs)} directories for Simple Model...")
    print(f"Target image size: {target_size} (Grayscale)")

    all_metadata = []
    for directory in data_dirs:
        metadata_file = os.path.join(directory, 'metadata.csv')
        if os.path.exists(metadata_file):
            try:
                df = pd.read_csv(metadata_file); df['filename'] = df['filename'].apply(lambda f: os.path.join(directory, os.path.basename(f))); all_metadata.append(df)
            except Exception as e: print(f"Warning: Could not process metadata {metadata_file}: {e}")
        else: print(f"Warning: Metadata file not found in {directory}")

    if not all_metadata: print("ERROR: No valid metadata found."); return None, None, None, None
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    original_count = len(metadata_df)
    metadata_df = metadata_df[metadata_df['filename'].apply(os.path.exists)]; valid_count = len(metadata_df)
    if valid_count < original_count: print(f"Warning: {original_count - valid_count} files in metadata do not exist.")
    if valid_count == 0: print("ERROR: No existing image files found."); return None, None, None, None

    class_counts = metadata_df['label'].value_counts().sort_index()
    if 'gesture_name' in metadata_df.columns: class_names = sorted(list(metadata_df['gesture_name'].unique()))
    else: class_names = [f"class_{i}" for i in sorted(list(metadata_df['label'].unique()))]
    print(f"Found {valid_count} valid images."); print(f"Class distribution:\n{class_counts.to_dict()}"); print(f"Gesture classes found: {class_names}")
    if set(ESSENTIAL_GESTURES) != set(class_names): print("\nWARNING: Mismatch between ESSENTIAL_GESTURES and found class names!"); print(f"  ESSENTIAL = {sorted(ESSENTIAL_GESTURES)}"); print(f"  FOUND     = {sorted(class_names)}")

    X_processed = []; y_labels = []; skipped_count = 0
    print("Loading, resizing, and normalizing images...")
    for index, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        filepath = row['filename']; label = row['label']
        try:
            img = cv2.imread(filepath)
            if img is None: print(f"Warning: Failed load {filepath}. Skip."); skipped_count += 1; continue
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_final = np.expand_dims(img_normalized, axis=-1)
            X_processed.append(img_final); y_labels.append(label)
        except Exception as e: print(f"Warning: Error processing {filepath}: {e}. Skip."); skipped_count += 1

    if skipped_count > 0: print(f"Skipped {skipped_count} images.")
    if not X_processed: print("ERROR: No images successfully processed."); return None, None, None, None

    X = np.array(X_processed); y = np.array(y_labels)
    processed_dir = 'data/processed/simple_model_input'
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X_processed.npy'), X)
    np.save(os.path.join(processed_dir, 'y_labels.npy'), y)
    class_map = {name: i for i, name in enumerate(class_names)} # Map name to label index
    with open(os.path.join(processed_dir, 'class_map.pkl'), 'wb') as f: pickle.dump(class_map, f)
    with open(os.path.join(processed_dir, 'class_names.pkl'), 'wb') as f: pickle.dump(class_names, f) # Save ordered names list too
    print(f"\nImage preprocessing complete."); print(f"  Output X: {X.shape}, y: {y.shape}"); print(f"  Data saved: {processed_dir}"); print(f"  Class Map: {class_map}")
    return X, y, class_names, class_map

# --- Function to Build the Simple CNN Model (NO Augmentation / NO BN) ---
def build_simple_cnn_model(input_shape, num_classes):
    """
    Build the simple CNN model WITHOUT data augmentation or Batch Normalization.
    This architecture previously achieved high validation accuracy.
    """
    print(f"Building Simple CNN model (NO AUG/BN) with input shape {input_shape} for {num_classes} classes.")

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # --- Augmentation and BN REMOVED ---
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Keep Dropout
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25), # Keep Dropout
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.4), # Keep Dropout
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile model (Using default Adam LR)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Use 0.001 which likely worked before
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Function to Train the Model ---
def train_model(X_train, y_train, X_val, y_val, model, model_name='gesture_simple_cnn', batch_size=32, epochs=50):
    """Train the gesture recognition model, saving best as .keras"""
    print(f"\n========== Training {model_name} Model ==========")
    model.summary()

    try:
        unique_classes = np.unique(y_train); class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}; print(f"Applying class weights: {class_weight_dict}")
    except Exception as e: print(f"Warning: Could not compute class weights ({e})."); class_weight_dict = None

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=5, min_lr=0.00001, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    os.makedirs('models', exist_ok=True)
    # --- Save checkpoint in .keras format ---
    checkpoint_path = f'models/{model_name}_best.keras'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=False
    )

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=batch_size, epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        class_weight=class_weight_dict, verbose=1
    )

    # Load best model explicitly (even though restore_best_weights=True)
    if os.path.exists(checkpoint_path):
         print(f"Loading best weights from {checkpoint_path}...")
         try: model = tf.keras.models.load_model(checkpoint_path); print("Best model loaded.")
         except Exception as e: print(f"Warning: Could not reload best model: {e}")
    else: print("Warning: Best model checkpoint not found after training.")

    history_path = f'models/{model_name}_history.pkl'
    try:
        with open(history_path, 'wb') as f: pickle.dump(history.history, f); print(f"Training history saved: {history_path}")
    except Exception as e: print(f"Error saving history: {e}")
    plot_training_history(history, model_name)
    return model, history

# --- Function to Plot Training History ---
def plot_training_history(history, model_name):
    """ Plots training/validation accuracy and loss. """
    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(history.history['accuracy'], label='Train Acc'); plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f'{model_name} Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.legend(loc='lower right'); plt.grid(True)
        plt.subplot(1, 2, 2); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'{model_name} Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.legend(loc='upper right'); plt.grid(True)
        plt.tight_layout(); plot_path = f'models/{model_name}_training_history.png'; plt.savefig(plot_path); plt.close()
        print(f"Training plot saved: {plot_path}")
    except Exception as e: print(f"Warning: Could not plot history: {e}")

# --- Function to Load Keras Model and Convert to TFLite ---
def convert_to_tflite_int8(model_name, X_rep):
    """Loads the best saved Keras model (.keras) and converts to TFLite."""
    model_path = f'models/{model_name}_best.keras'
    if not os.path.exists(model_path):
        print(f"ERROR: Trained model file not found at {model_path}"); return None

    print(f"Loading model from {model_path} for TFLite conversion...")
    try: model = tf.keras.models.load_model(model_path)
    except Exception as e: print(f"Error loading Keras model: {e}"); return None

    print(f"\nConverting {model_name} to TFLite (int8 quantized)...")
    try:
        def representative_dataset_gen():
            # Use provided representative data subset (X_rep)
            for i in range(len(X_rep)): yield [np.expand_dims(X_rep[i], axis=0).astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model_quant = converter.convert()

        tflite_path = f'models/{model_name}_quantized.tflite'
        with open(tflite_path, 'wb') as f: f.write(tflite_model_quant)
        print(f"SUCCESS: Quantized TFLite model saved: {tflite_path} ({len(tflite_model_quant) / 1024:.2f} KB)")
        generate_c_header(tflite_model_quant, model_name)
        # --- Also save a Float32 fallback automatically ---
        print("\nSaving Float32 TFLite model as fallback...")
        try:
             converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
             converter_float.optimizations = [tf.lite.Optimize.DEFAULT]
             tflite_model_float = converter_float.convert()
             tflite_path_float = f'models/{model_name}_float.tflite' # Explicit _float name
             with open(tflite_path_float, 'wb') as f: f.write(tflite_model_float)
             print(f"SUCCESS: Float32 TFLite model saved: {tflite_path_float} ({len(tflite_model_float) / 1024:.2f} KB)")
        except Exception as e_fl: print(f"Warning: Could not save Float32 fallback model: {e_fl}")
        # --- End Float32 fallback ---
        return tflite_model_quant # Return the main quantized one

    except Exception as e:
        print(f"ERROR: Failed to convert model to quantized TFLite: {e}"); traceback.print_exc()
        return None

# --- Function to Generate C Header File ---
def generate_c_header(tflite_model, model_name):
    """ Generates a C header file from the TFLite model data. """
    try:
        # Define model name for C array (ensure it's valid)
        c_model_name = model_name.replace('-', '_').replace('.', '_') + "_model" # Add suffix
        header_path = f'models/{c_model_name}_data.h' # Use C-safe name for file too

        with open(header_path, 'w') as file:
            file.write(f'// TFLite model data for {model_name}\n// Converted Size: {len(tflite_model) / 1024:.2f} KB\n\n');
            file.write('#ifndef TENSORFLOW_LITE_MICRO_MODELS_{}_DATA_H_\n'.format(c_model_name.upper())) # Header guards
            file.write('#define TENSORFLOW_LITE_MICRO_MODELS_{}_DATA_H_\n\n'.format(c_model_name.upper()))
            file.write('// IMPORTANT: Align to 16 bytes for optimal performance\n')
            file.write(f'const unsigned char {c_model_name}_tflite[] __attribute__((aligned(16))) = {{\n  ') # Use _tflite suffix
            hex_array = [f'0x{byte:02x}' for byte in tflite_model]
            for i, hex_byte in enumerate(hex_array):
                file.write(hex_byte + ','); file.write('\n  ' if (i + 1) % 16 == 0 else ' ')
            file.write('\n};\n\n'); file.write(f'const unsigned int {c_model_name}_tflite_len = {len(tflite_model)};\n\n')
            file.write('#endif // TENSORFLOW_LITE_MICRO_MODELS_{}_DATA_H_\n'.format(c_model_name.upper()))
        print(f"SUCCESS: C header file generated: {header_path}")
    except Exception as e: print(f"ERROR: Failed to generate C header file: {e}")


# --- Function to Evaluate Model ---
def evaluate_model(model_name, X_test, y_test):
    """ Load the best model (.keras) and evaluate it. """
    print(f"\n--- Evaluating {model_name} Model ---")
    model_path = f'models/{model_name}_best.keras'
    if not os.path.exists(model_path): print(f"ERROR: No saved model found at {model_path}"); return None, None

    try:
        print(f"Loading best model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}"); print(f"Test Accuracy: {test_acc:.4f}")

        y_pred_probs = model.predict(X_test); y_pred_classes = np.argmax(y_pred_probs, axis=1)
        class_map_path = 'data/processed/simple_model_input/class_map.pkl'
        if os.path.exists(class_map_path):
             with open(class_map_path, 'rb') as f: class_map = pickle.load(f)
             class_names = [name for name, index in sorted(class_map.items(), key=lambda item: item[1])]
        else: num_classes = model.output_shape[1]; class_names = [f"Class_{i}" for i in range(num_classes)]

        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(y_test, y_pred_classes, target_names=class_names, digits=4, zero_division=0) # Added zero_division
        print("\nClassification Report:"); print(report)
        report_path = f'models/{model_name}_classification_report.txt'
        with open(report_path, 'w') as f: f.write(f"{model_name} Eval\nTest Acc: {test_acc:.4f}\n\n{report}")
        print(f"Report saved: {report_path}")

        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names))))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues); plt.title(f'{model_name} Confusion Matrix'); plt.colorbar()
        tick_marks = np.arange(len(class_names)); plt.xticks(tick_marks, class_names, rotation=45, ha="right"); plt.yticks(tick_marks, class_names)
        thresh = cm.max() / 2.;
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]): plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout(); plt.ylabel('True label'); plt.xlabel('Predicted label'); cm_path = f'models/{model_name}_confusion_matrix.png'; plt.savefig(cm_path); plt.close()
        print(f"Confusion matrix saved: {cm_path}")
        return test_loss, test_acc
    except Exception as e: print(f"Error during evaluation: {e}"); traceback.print_exc(); return None, None

# --- Main Processing and Training Function ---
def process_and_train_simple_cnn():
    """Main pipeline: preprocess, split, build, train, evaluate, convert."""
    print("Starting Simple CNN Model Training Pipeline...")
    processed_dir = 'data/processed/simple_model_input'
    X_path = os.path.join(processed_dir, 'X_processed.npy'); y_path = os.path.join(processed_dir, 'y_labels.npy')

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading preprocessed data from {processed_dir}..."); X = np.load(X_path); y = np.load(y_path)
        try:
            with open(os.path.join(processed_dir, 'class_map.pkl'), 'rb') as f: class_map = pickle.load(f)
            class_names = [name for name, index in sorted(class_map.items(), key=lambda item: item[1])]
        except: num_classes = len(np.unique(y)); class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"Loaded data shapes: X={X.shape}, y={y.shape}")
    else:
        print("Preprocessing images..."); X, y, class_names, class_map = preprocess_images_for_simple_model()

    if X is None or y is None or len(X) == 0: print("ERROR: Failed to load/process data."); return None
    num_classes = len(class_names); input_shape = X.shape[1:]

    print("\nSplitting data...");
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val)
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        test_dir = 'data/processed/simple_model_test'; os.makedirs(test_dir, exist_ok=True)
        np.save(os.path.join(test_dir, 'X_test.npy'), X_test); np.save(os.path.join(test_dir, 'y_test.npy'), y_test)
        print(f"Test data saved in {test_dir}")
    except Exception as e: print(f"Error splitting data: {e}."); return None

    model = build_simple_cnn_model(input_shape, num_classes)
    model_name = 'gesture_simple_cnn'
    trained_model, history = train_model(X_train, y_train, X_val, y_val, model, model_name=model_name)

    # Pass model_name, not the model object itself, to evaluation and conversion
    evaluate_model(model_name, X_test, y_test)
    num_rep_samples = min(200, len(X_train))
    representative_data = X_train[np.random.choice(X_train.shape[0], num_rep_samples, replace=False)]
    convert_to_tflite_int8(model_name, representative_data)

    print("\nSimple CNN Model training pipeline finished!")
    return model_name

# --- Main Execution Block ---
if __name__ == "__main__":
    print("*"*50); print("  Starting Arduino-Friendly Gesture Model Training "); print("*"*50)
    print(f"Using TensorFlow version: {tf.__version__}"); print(f"Target Image Size: {TARGET_IMG_SIZE} (Grayscale)")
    start_time = time.time()

    result_model_name = process_and_train_simple_cnn()

    end_time = time.time(); total_time = end_time - start_time
    print(f"\nTotal script execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if result_model_name:
        print("\nSUCCESS: Model training complete."); print("Check 'models' dir for saved models (.keras), plots (.png), reports (.txt), TFLite files (.tflite), and C header (.h).")
        print("\nNext steps (If accuracy looks good):"); print(" 1. Ensure the NEW '.tflite' file (float or quantized) is used by 'inference.py'."); print(" 2. Test 'inference.py', tuning CONFIDENCE_THRESHOLD and SMOOTHING_WINDOW.")
    else: print("\nERROR: Model training failed. Review logs.")