# File: model_training.py
# Purpose: Process collected data and train the gesture recognition model

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
import cv2
import joblib

# Constants
IMG_SIZE = (96, 96)
SEQUENCE_LENGTH = 10
STEP_SIZE = 5

def extract_image_features(data_dirs=None, img_size=IMG_SIZE, batch_size=32):
    """
    Extract features from collected images using MobileNetV2
    
    Parameters:
    data_dirs: List of data directories (if None, read from file)
    img_size: Size to resize images to before feature extraction
    batch_size: Batch size for processing
    
    Returns:
    X: Feature array
    y: Labels array
    """
    # If no data_dirs provided, read from file
    if data_dirs is None:
        try:
            with open('data/gesture_data_dirs.txt', 'r') as f:
                data_dirs = [line.strip() for line in f.readlines()]
        except:
            print("No data directories found. Please run data collection first.")
            return None, None
    
    print(f"Processing images from {len(data_dirs)} directories...")
    
    # Load all metadata files
    all_metadata = []
    for directory in data_dirs:
        metadata_file = os.path.join(directory, 'metadata.csv')
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            all_metadata.append(df)
    
    if not all_metadata:
        print("No metadata files found. Please run data collection first.")
        return None, None
    
    # Combine all metadata
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    # Filter to only include existing files
    metadata_df = metadata_df[metadata_df['filename'].apply(os.path.exists)]
    
    print(f"Found {len(metadata_df)} valid images for processing")
    
    # Create feature extractor from MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create model for feature extraction
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D()
    ])
    
    # Process images in batches
    features = []
    labels = []
    
    for i in tqdm(range(0, len(metadata_df), batch_size)):
        batch_files = metadata_df['filename'].iloc[i:i+batch_size].tolist()
        batch_labels = metadata_df['label'].iloc[i:i+batch_size].tolist()
        
        # Load and preprocess images
        batch_images = []
        for file in batch_files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            batch_images.append(img)
        
        # Convert to array
        batch_images = np.array(batch_images)
        
        # Extract features
        batch_features = model.predict(batch_images)
        
        # Save features and labels
        features.extend(batch_features)
        labels.extend(batch_labels)
    
    # Convert to arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Save processed data
    os.makedirs('data/processed/features', exist_ok=True)
    np.save('data/processed/features/X_features.npy', X)
    np.save('data/processed/features/y_labels.npy', y)
    
    print(f"Feature extraction complete. Shape of X: {X.shape}, y: {y.shape}")
    return X, y

def create_sequence_data(X, y, sequence_length=SEQUENCE_LENGTH, step=STEP_SIZE):
    """
    Create sequences from extracted features for temporal modeling
    
    Parameters:
    X: Feature array
    y: Labels array
    sequence_length: Number of frames in each sequence
    step: Step size for sliding window
    
    Returns:
    X_seq: Sequence features
    y_seq: Sequence labels (using the last frame's label)
    """
    X_seq = []
    y_seq = []
    
    # Create sequences with sliding window
    for i in range(0, len(X) - sequence_length, step):
        X_seq.append(X[i:i+sequence_length])
        # Use the label of the last frame in the sequence
        y_seq.append(y[i+sequence_length-1])
    
    # Convert to arrays
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Save sequence data
    os.makedirs('data/processed/features', exist_ok=True)
    np.save('data/processed/features/X_sequences.npy', X_seq)
    np.save('data/processed/features/y_sequences.npy', y_seq)
    
    print(f"Sequence creation complete. Shape of X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
    return X_seq, y_seq

def build_gesture_model(input_shape, num_classes):
    """
    Build a temporal model for gesture recognition
    
    Parameters:
    input_shape: Shape of input sequences
    num_classes: Number of gesture classes
    
    Returns:
    model: Compiled TensorFlow model
    """
    # Create model using LSTM for temporal information
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_gesture_model(X_train, y_train, X_val, y_val, batch_size=32, epochs=30):
    """
    Train the gesture recognition model
    
    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    batch_size: Batch size for training
    epochs: Number of epochs to train
    
    Returns:
    model: Trained model
    history: Training history
    """
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Build model
    model = build_gesture_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Model checkpoint to save best model
    os.makedirs('models', exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/gesture_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    # Save final model
    model.save('models/gesture_model_final.h5')
    
    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return model, history

def prepare_model_for_inference():
    """
    Prepare the model for inference by converting to TFLite
    
    Returns:
    tflite_model: Converted TensorFlow Lite model
    """
    # Check if model exists
    if not os.path.exists('models/gesture_model_best.h5'):
        print("No trained model found. Please train the model first.")
        return None
    
    # Load model
    model = tf.keras.models.load_model('models/gesture_model_best.h5')
    
    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model to file
    with open('models/gesture_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved. Size: {len(tflite_model) / 1024:.2f} KB")
    return tflite_model

def process_and_train():
    """Main function to process data and train model"""
    # Step 1: Check if processed features exist
    if (os.path.exists('data/processed/features/X_features.npy') and 
        os.path.exists('data/processed/features/y_labels.npy')):
        print("Loading pre-extracted features...")
        X = np.load('data/processed/features/X_features.npy')
        y = np.load('data/processed/features/y_labels.npy')
    else:
        print("Extracting features from images...")
        X, y = extract_image_features()
        
    if X is None or y is None:
        print("Error extracting features. Please check data collection.")
        return None, None
    
    # Step 2: Create sequences
    if (os.path.exists('data/processed/features/X_sequences.npy') and 
        os.path.exists('data/processed/features/y_sequences.npy')):
        print("Loading pre-created sequences...")
        X_seq = np.load('data/processed/features/X_sequences.npy')
        y_seq = np.load('data/processed/features/y_sequences.npy')
    else:
        print("Creating sequences from features...")
        X_seq, y_seq = create_sequence_data(X, y)
    
    # Step 3: Split data for training
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Data split complete. Training: {X_train.shape}, Validation: {X_val.shape}, Testing: {X_test.shape}")
    
    # Step 4: Train model
    print("Training gesture recognition model...")
    model, history = train_gesture_model(X_train, y_train, X_val, y_val)
    
    # Step 5: Save test data for evaluation
    np.save('data/processed/features/X_test.npy', X_test)
    np.save('data/processed/features/y_test.npy', y_test)
    
    print("Model training complete!")
    return model, history

if __name__ == "__main__":
    print("Starting image processing and model training...")
    model, history = process_and_train()
    
    if model is not None:
        print("Converting model to TFLite format...")
        prepare_model_for_inference()
        
        print("All processing and training complete!")
    else:
        print("Model training failed. Please check errors.")

