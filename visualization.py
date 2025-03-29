# File: visualization.py
# Purpose: Create visualizations of model performance

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Class names for gestures
CLASS_NAMES = [
    'neutral', 
    'turn_right', 
    'turn_left', 
    'swipe_next', 
    'swipe_previous',
    'pre_turn_right',
    'pre_turn_left'
]

def visualize_training_results(history=None, X_test=None, y_test=None, model=None):
    """
    Visualize model training results and performance
    
    Parameters:
    history: Training history (if None, load from file)
    X_test: Test features (if None, load from file)
    y_test: Test labels (if None, load from file)
    model: Trained model (if None, load from file)
    """
    # Create output directory
    os.makedirs('models', exist_ok=True)
    
    # Load history if not provided
    if history is None:
        try:
            with open('models/training_history.pkl', 'rb') as f:
                history = pickle.load(f)
        except:
            print("No training history found. Please train the model first.")
            return
    
    # Convert history to dictionary if it's not already
    if not isinstance(history, dict):
        history = history.history
    
    # Load test data if not provided
    if X_test is None or y_test is None:
        try:
            X_test = np.load('data/processed/features/X_test.npy')
            y_test = np.load('data/processed/features/y_test.npy')
        except:
            print("No test data found. Please train the model first.")
            return
    
    # Load model if not provided
    if model is None:
        try:
            model = tf.keras.models.load_model('models/gesture_model_best.h5')
        except:
            print("No trained model found. Please train the model first.")
            return
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("Training history visualization saved to 'models/training_history.png'")
    
    # Get predictions for confusion matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    # Create confusion matrix plots
    plt.figure(figsize=(12, 10))
    
    # Raw counts
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha="right")
    
    # Normalized
    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix visualization saved to 'models/confusion_matrix.png'")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Plot precision, recall, and f1-score
    plt.figure(figsize=(12, 8))
    metrics_df = report_df.iloc[:-3, :3]  # Skip averages
    
    sns.heatmap(metrics_df, annot=True, cmap='GnBu', fmt='.2f')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.tight_layout()
    plt.savefig('models/performance_metrics.png')
    print("Performance metrics visualization saved to 'models/performance_metrics.png'")
    
    # Save report to CSV
    report_df.to_csv('models/classification_report.csv')
    
    # Print summary
    print("\nModel Performance Summary:")
    print(f"Overall Accuracy: {report['accuracy']:.4f}")
    print("\nPer-class Performance:")
    for cls in CLASS_NAMES:
        print(f"{cls}: F1-Score={report[cls]['f1-score']:.2f}, Precision={report[cls]['precision']:.2f}, Recall={report[cls]['recall']:.2f}")
    
    # Special focus on predictive states
    pre_turn_right_index = CLASS_NAMES.index('pre_turn_right')
    pre_turn_left_index = CLASS_NAMES.index('pre_turn_left')
    turn_right_index = CLASS_NAMES.index('turn_right')
    turn_left_index = CLASS_NAMES.index('turn_left')
    
    print("\nPredictive Performance:")
    print(f"Pre-turn right → Turn right accuracy: {cm[pre_turn_right_index, turn_right_index] / max(1, cm[pre_turn_right_index].sum()):.2f}")
    print(f"Pre-turn left → Turn left accuracy: {cm[pre_turn_left_index, turn_left_index] / max(1, cm[pre_turn_left_index].sum()):.2f}")
    
    print("\nAll visualizations and reports saved to 'models/' directory")

if __name__ == "__main__":
    print("Generating visualizations of model performance...")
    visualize_training_results()
