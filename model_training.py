# File: model_training.py
# Purpose: Process collected data and train the gesture recognition model (Optimized Version)

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
import matplotlib.pyplot as plt
import time
from sklearn.utils import class_weight
import random

# Constants
IMG_SIZE = (96, 96)  # Smaller size for faster processing
SEQUENCE_LENGTH = 5  # Reduced sequence length for quicker inference
STEP_SIZE = 2  # Smaller step size to get more sequences from limited data

# Essential gestures that match the updated data_collection.py
ESSENTIAL_GESTURES = ['neutral', 'turn_right', 'turn_left']
ALL_GESTURES = ['neutral', 'turn_right', 'turn_left', 'pre_turn_right', 'pre_turn_left']

def extract_image_features(data_dirs=None, img_size=IMG_SIZE, batch_size=32, use_augmentation=True):
    """
    Extract features from collected images using MobileNetV2
    
    Parameters:
    data_dirs: List of data directories (if None, read from file)
    img_size: Size to resize images to before feature extraction
    batch_size: Batch size for processing
    use_augmentation: Whether to apply data augmentation
    
    Returns:
    X: Feature array
    y: Labels array
    class_names: List of class names
    """
    # If no data_dirs provided, read from file
    if data_dirs is None:
        try:
            with open('data/gesture_data_dirs.txt', 'r') as f:
                data_dirs = [line.strip() for line in f.readlines()]
        except:
            print("No data directories found. Please run data collection first.")
            return None, None, None
    
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
        return None, None, None
    
    # Combine all metadata
    metadata_df = pd.concat(all_metadata, ignore_index=True)
    
    # Filter to only include existing files
    metadata_df = metadata_df[metadata_df['filename'].apply(os.path.exists)]
    
    # Get class information
    class_count = metadata_df['label'].value_counts().sort_index()
    class_names = metadata_df['gesture_name'].unique()
    
    print(f"Found {len(metadata_df)} valid images for processing")
    print(f"Class distribution: {class_count.to_dict()}")
    print(f"Gesture classes: {sorted(class_names)}")
    
    # Create feature extractor from MobileNetV2
    # Using alpha=0.5 for a smaller, faster model
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet',
        alpha=0.5  # Smaller model
    )
    
    # Extract features from the middle of the network for efficiency
    # Use the last "add" layer that exists in the alpha=0.5 model
    # Full list of layers can be seen in the error message
    layer_name = 'block_9_add'  # This layer exists in the alpha=0.5 model
    
    # Print all available layers for debugging
    print("Available layers in MobileNetV2:")
    for i, layer in enumerate(base_model.layers):
        if 'add' in layer.name:
            print(f"  - {layer.name} (layer {i})")
    
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(layer_name).output
    )
    
    # Freeze base model
    feature_extractor.trainable = False
    
    # Create model for feature extraction with global pooling
    model = tf.keras.Sequential([
        feature_extractor,
        layers.GlobalAveragePooling2D()
    ])
    
    # Process images in batches
    features = []
    labels = []
    gesture_names = []
    
    for i in tqdm(range(0, len(metadata_df), batch_size)):
        batch_files = metadata_df['filename'].iloc[i:i+batch_size].tolist()
        batch_labels = metadata_df['label'].iloc[i:i+batch_size].tolist()
        batch_gestures = metadata_df['gesture_name'].iloc[i:i+batch_size].tolist()
        
        # Load and preprocess images
        batch_images = []
        for file in batch_files:
            img = cv2.imread(file)
            if img is None:
                print(f"Warning: Could not read image {file}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            batch_images.append(img)
        
        # Convert to array
        batch_images = np.array(batch_images)
        
        # Extract features
        batch_features = model.predict(batch_images, verbose=0)
        
        # Save features and labels
        features.extend(batch_features)
        labels.extend(batch_labels)
        gesture_names.extend(batch_gestures)
    
    # Convert to arrays
    X = np.array(features)
    y = np.array(labels)
    gesture_names = np.array(gesture_names)
    
    # Apply data augmentation if requested
    if use_augmentation:
        print("Applying data augmentation to increase dataset size...")
        X_aug, y_aug, names_aug = augment_feature_data(X, y, gesture_names)
        
        # Combine original and augmented data
        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, y_aug], axis=0)
        gesture_names = np.concatenate([gesture_names, names_aug], axis=0)
        
        print(f"After augmentation: {len(X)} samples")
    
    # Save processed data
    os.makedirs('data/processed/features', exist_ok=True)
    np.save('data/processed/features/X_features.npy', X)
    np.save('data/processed/features/y_labels.npy', y)
    
    # Save class names for reference
    unique_class_names = np.unique(gesture_names)
    with open('data/processed/features/class_names.pkl', 'wb') as f:
        pickle.dump(unique_class_names, f)
    
    print(f"Feature extraction complete. Shape of X: {X.shape}, y: {y.shape}")
    print(f"Class names: {unique_class_names}")
    
    return X, y, unique_class_names

def augment_feature_data(X, y, gesture_names):
    """
    Augment feature data to increase dataset size and improve robustness
    
    Parameters:
    X: Original feature array
    y: Original labels
    gesture_names: Original gesture names
    
    Returns:
    X_aug: Augmented features
    y_aug: Augmented labels
    names_aug: Augmented gesture names
    """
    print("Generating augmented features from extracted data...")
    
    X_aug = []
    y_aug = []
    names_aug = []
    
    # Count samples per class for balanced augmentation
    class_counts = {}
    for label in np.unique(y):
        class_counts[label] = np.sum(y == label)
    
    # Target number of samples per class after augmentation
    target_count = max(class_counts.values()) * 2
    
    # Augment each class
    for label in np.unique(y):
        # Get all samples of this class
        indices = np.where(y == label)[0]
        
        # Get a gesture name for this label
        gesture_name = gesture_names[indices[0]]
        
        # Number of samples to generate
        num_to_generate = min(len(indices) * 3, target_count - class_counts[label])
        
        # Generate new samples
        for _ in range(num_to_generate):
            # Randomly select a sample
            idx = np.random.choice(indices)
            feature = X[idx].copy()
            
            # Apply feature-level augmentation
            # 1. Random noise
            noise_factor = np.random.uniform(0.92, 1.08)
            feature = feature * noise_factor
            
            # 2. Random dropout (set some features to 0)
            dropout_mask = np.random.binomial(1, 0.95, size=feature.shape)
            feature = feature * dropout_mask
            
            # 3. Smoothing (average with original)
            if np.random.random() > 0.5:
                aug_factor = np.random.uniform(0.7, 0.9)
                feature = feature * aug_factor + X[idx] * (1 - aug_factor)
            
            # Add augmented sample
            X_aug.append(feature)
            y_aug.append(label)
            names_aug.append(gesture_name)
    
    return np.array(X_aug), np.array(y_aug), np.array(names_aug)

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
    print(f"Creating sequences with length {sequence_length} and step size {step}...")
    
    X_seq = []
    y_seq = []
    
    # Group by class for more robust sequence creation
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        # Get indices of this class
        indices = np.where(y == cls)[0]
        
        # If we don't have enough samples, duplicate some
        if len(indices) < sequence_length:
            extra_needed = sequence_length - len(indices)
            indices = np.concatenate([indices, np.random.choice(indices, size=extra_needed)])
        
        # Sort indices to maintain temporal order if possible
        indices = np.sort(indices)
        
        # Create sequences with sliding window
        for i in range(0, len(indices) - sequence_length + 1, step):
            seq_indices = indices[i:i+sequence_length]
            X_seq.append(X[seq_indices])
            # Use the label of the last frame in the sequence
            y_seq.append(y[seq_indices[-1]])
    
    # Convert to arrays
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Save sequence data
    os.makedirs('data/processed/features', exist_ok=True)
    np.save('data/processed/features/X_sequences.npy', X_seq)
    np.save('data/processed/features/y_sequences.npy', y_seq)
    
    print(f"Sequence creation complete. Shape of X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
    return X_seq, y_seq

def build_lstm_model(input_shape, num_classes):
    """
    Build a temporal model for gesture recognition using LSTM
    
    Parameters:
    input_shape: Shape of input sequences
    num_classes: Number of gesture classes
    
    Returns:
    model: Compiled TensorFlow model
    """
    # Create model using LSTM for temporal information - simplified version
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(32, return_sequences=False),  # Single LSTM layer
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_simple_model(input_shape, num_classes):
    """
    Build a simple dense model for gesture recognition
    More suitable for microcontrollers with limited resources
    
    Parameters:
    input_shape: Shape of input features
    num_classes: Number of gesture classes
    
    Returns:
    model: Compiled TensorFlow model
    """
    # Get the flattened feature size
    feature_size = np.prod(input_shape[1:])
    
    # Create a simpler dense model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val, model_type='lstm', batch_size=32, epochs=30):
    """
    Train the gesture recognition model
    
    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    model_type: Type of model to train ('lstm' or 'simple')
    batch_size: Batch size for training
    epochs: Number of epochs to train
    
    Returns:
    model: Trained model
    history: Training history
    """
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    
    # Build the appropriate model
    if model_type == 'lstm':
        model = build_lstm_model(input_shape, num_classes)
        model_name = 'gesture_lstm_model'
    else:
        model = build_simple_model(input_shape, num_classes)
        model_name = 'gesture_simple_model'
    
    # Print model summary
    model.summary()
    
    # Calculate class weights for balanced training
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Reduce LR on plateau to improve training
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Model checkpoint to save best model
    os.makedirs('models', exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'models/{model_name}_best.h5',
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
        callbacks=[early_stopping, checkpoint, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Save final model
    model.save(f'models/{model_name}_final.h5')
    
    # Save training history
    with open(f'models/{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    # Plot training history
    plot_training_history(history, model_type)
    
    return model, history

def plot_training_history(history, model_type):
    """
    Plot training and validation accuracy/loss
    
    Parameters:
    history: Training history
    model_type: Type of model ('lstm' or 'simple')
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_type.upper()} Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_type.upper()} Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'models/{model_type}_training_history.png')
    plt.close()

# Replace your current prepare_model_for_inference function with this:

def prepare_model_for_inference(model_type='lstm'):
    """
    Prepare the model for inference by converting to TFLite
    
    Parameters:
    model_type: Type of model to convert ('lstm' or 'simple')
    
    Returns:
    tflite_model: Converted TensorFlow Lite model
    """
    # Check if model exists
    model_path = f'models/gesture_{model_type}_model_best.h5'
    if not os.path.exists(model_path):
        print(f"No trained {model_type} model found. Please train the model first.")
        return None
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    if model_type == 'lstm':
        print("\nConverting LSTM model to TFLite with SELECT_TF_OPS...")
        
        # For LSTM model, we need to use SELECT_TF_OPS and disable tensor list lowering
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Use recommended settings from error message
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        # Apply optimizations but avoid quantization for LSTM
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        try:
            # Convert the model
            tflite_model = converter.convert()
            
            # Save the model to file
            tflite_path = f'models/gesture_{model_type}_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved at {tflite_path}. Size: {len(tflite_model) / 1024:.2f} KB")
            print("Note: This model includes TF operations and will be larger than a fully optimized model")
            
            return tflite_model
        except Exception as e:
            print(f"Error converting LSTM to TFLite: {e}")
            print("\nFalling back to saving standard H5 model only")
            return None
    else:
        # For simple model, use standard TFLite conversion
        print(f"\nConverting Simple model to TFLite...")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Try quantization for simple model
        try:
            # Define representative dataset for quantization
            def representative_dataset():
                # Load test data if available
                if os.path.exists('data/processed/features/X_test_simple.npy'):
                    X_test = np.load('data/processed/features/X_test_simple.npy')
                    for i in range(min(100, len(X_test))):
                        yield [X_test[i:i+1].astype(np.float32)]
                else:
                    # Generate random data as fallback
                    shape = model.input_shape
                    for _ in range(100):
                        data = np.random.random((1,) + shape[1:]).astype(np.float32)
                        yield [data]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            # Save the quantized model
            tflite_path = f'models/gesture_{model_type}_model_quantized.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Quantized TFLite model saved at {tflite_path}. Size: {len(tflite_model) / 1024:.2f} KB")
        except Exception as e:
            print(f"Quantization failed: {e}")
            print("Trying standard float conversion...")
            
            # Try standard conversion without quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            tflite_path = f'models/gesture_{model_type}_model.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Float TFLite model saved at {tflite_path}. Size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model

def evaluate_model(model, X_test, y_test, model_type='lstm'):
    """
    Evaluate the model on test data
    
    Parameters:
    model: Trained model
    X_test, y_test: Test data
    model_type: Type of model ('lstm' or 'simple')
    
    Returns:
    test_loss, test_acc: Loss and accuracy on test data
    """
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"{model_type.upper()} Model Test Accuracy: {test_acc:.4f}")
    
    # Predict and get confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get class names
    try:
        with open('data/processed/features/class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
    except:
        class_names = [str(i) for i in range(len(np.unique(y_test)))]
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_type.upper()} Model Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'models/{model_type}_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    report = classification_report(y_test, y_pred_classes, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Save report to file
    with open(f'models/{model_type}_classification_report.txt', 'w') as f:
        f.write(f"{model_type.upper()} Model Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    return test_loss, test_acc

def process_and_train(model_type='both', use_augmentation=True):
    """
    Main function to process data and train model
    
    Parameters:
    model_type: Type of model to train ('lstm', 'simple', or 'both')
    use_augmentation: Whether to apply data augmentation
    
    Returns:
    models: Dictionary of trained models
    """
    models = {}
    
    # Step 1: Extract features
    if (os.path.exists('data/processed/features/X_features.npy') and 
        os.path.exists('data/processed/features/y_labels.npy')):
        print("Loading pre-extracted features...")
        X = np.load('data/processed/features/X_features.npy')
        y = np.load('data/processed/features/y_labels.npy')
        
        # Load class names
        try:
            with open('data/processed/features/class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
        except:
            class_names = [f"class_{i}" for i in range(len(np.unique(y)))]
    else:
        print("Extracting features from images...")
        X, y, class_names = extract_image_features(use_augmentation=use_augmentation)
        
    if X is None or y is None:
        print("Error extracting features. Please check data collection.")
        return None
    
    print(f"Working with {len(np.unique(y))} classes: {np.unique(y)}")
    
    # Step 2: Create sequences for LSTM model
    if model_type in ['lstm', 'both']:
        if (os.path.exists('data/processed/features/X_sequences.npy') and 
            os.path.exists('data/processed/features/y_sequences.npy')):
            print("Loading pre-created sequences...")
            X_seq = np.load('data/processed/features/X_sequences.npy')
            y_seq = np.load('data/processed/features/y_sequences.npy')
        else:
            print("Creating sequences from features...")
            X_seq, y_seq = create_sequence_data(X, y)
        
        # Split data for LSTM model
        X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
            X_train_seq, y_train_seq, test_size=0.2, random_state=42
        )
        
        print(f"LSTM data split complete. Training: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Testing: {X_test_seq.shape}")
    
    # Step 3: Prepare data for simple model (using individual frames)
    if model_type in ['simple', 'both']:
        # Split data for simple model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Reshape for simple model
        X_train_simple = X_train.reshape(X_train.shape[0], 1, -1)
        X_val_simple = X_val.reshape(X_val.shape[0], 1, -1)
        X_test_simple = X_test.reshape(X_test.shape[0], 1, -1)
        
        print(f"Simple model data split complete. Training: {X_train_simple.shape}, Validation: {X_val_simple.shape}, Testing: {X_test_simple.shape}")
    
    # Step 4: Train models
    if model_type in ['lstm', 'both']:
        print("\n========== Training LSTM Model ==========")
        lstm_model, lstm_history = train_model(
            X_train_seq, y_train_seq, 
            X_val_seq, y_val_seq,
            model_type='lstm',
            batch_size=32,
            epochs=50  # More epochs with early stopping
        )
        models['lstm'] = lstm_model
        
        # Save test data for LSTM model
        np.save('data/processed/features/X_test_seq.npy', X_test_seq)
        np.save('data/processed/features/y_test_seq.npy', y_test_seq)
        
        # Evaluate LSTM model
        evaluate_model(lstm_model, X_test_seq, y_test_seq, model_type='lstm')
    
    if model_type in ['simple', 'both']:
        print("\n========== Training Simple Model ==========")
        simple_model, simple_history = train_model(
            X_train_simple, y_train, 
            X_val_simple, y_val,
            model_type='simple',
            batch_size=32,
            epochs=50  # More epochs with early stopping
        )
        models['simple'] = simple_model
        
        # Save test data for simple model
        np.save('data/processed/features/X_test_simple.npy', X_test_simple)
        np.save('data/processed/features/y_test.npy', y_test)
        
        # Evaluate simple model
        evaluate_model(simple_model, X_test_simple, y_test, model_type='simple')
    
    # Step 5: Convert models to TFLite
    if model_type in ['lstm', 'both']:
        print("\nConverting LSTM model to TFLite...")
        prepare_model_for_inference(model_type='lstm')
    
    if model_type in ['simple', 'both']:
        print("\nConverting Simple model to TFLite...")
        prepare_model_for_inference(model_type='simple')
    
    print("\nAll processing and training complete!")
    
    return models

if __name__ == "__main__":
    print("Starting TinyML Gesture Recognition Model Training...")
    
    # Configuration options
    print("\n=== Training Configuration ===")
    print("1: Train LSTM model (better accuracy, more resource intensive)")
    print("2: Train Simple model (faster inference, better for Arduino)")
    print("3: Train both models and compare (recommended)")
    
    model_choice = input("Enter your choice (1-3, default is 3): ").strip()
    
    if model_choice == "1":
        model_type = "lstm"
        print("Training LSTM model only")
    elif model_choice == "2":
        model_type = "simple"
        print("Training Simple model only")
    else:
        model_type = "both"
        print("Training both models for comparison")
    
    # Data augmentation option
    aug_choice = input("Use data augmentation to improve model robustness? (Y/n): ").strip().lower()
    use_augmentation = aug_choice != "n"
    
    if use_augmentation:
        print("Data augmentation ENABLED")
    else:
        print("Data augmentation DISABLED")
    
    # Start training
    start_time = time.time()
    models = process_and_train(model_type=model_type, use_augmentation=use_augmentation)
    
    # Print training time
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    if models:
        print("\nTraining successful! Models are saved in the 'models' directory.")
        print("Next step: Run inference.py to test real-time gesture recognition.")
    else:
        print("\nTraining failed. Please check the error messages above.")