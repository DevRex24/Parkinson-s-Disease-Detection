#!/usr/bin/env python3
"""
Quick Demo: MRI Model Performance Testing
This is a simplified version that trains fewer models for demonstration
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
dataset_dir = 'parkinsons_dataset'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_mri_dataset(dataset_dir):
    """Load and preprocess MRI dataset"""
    print(f"Loading MRI dataset from: {dataset_dir}")
    images, labels = [], []
    
    for label, class_name in enumerate(['normal', 'parkinson']):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"Processing {class_name} class...")
        file_count = 0
        
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, IMG_SIZE)
                        img = img / 255.0  # Normalize
                        images.append(img)
                        labels.append(label)
                        file_count += 1
                except:
                    continue
        
        print(f"Processed {file_count} files for {class_name} class")
    
    return np.array(images), np.array(labels)

def create_simple_cnn():
    """Simple CNN"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_improved_cnn():
    """Improved CNN with batch normalization"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""
    print(f"\nTraining {model_name}...")
    
    if hasattr(model, 'fit') and hasattr(model, 'predict_proba'):
        # Traditional ML model - flatten images
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Downsample to reduce dimensionality
        step = 500  # Take every 500th pixel
        X_train_sampled = X_train_flat[:, ::step]
        X_test_sampled = X_test_flat[:, ::step]
        
        model.fit(X_train_sampled, y_train)
        y_pred = model.predict(X_test_sampled)
        y_pred_prob = model.predict_proba(X_test_sampled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
    else:
        # CNN model
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def main():
    print("=" * 60)
    print("MRI MODEL QUICK COMPARISON DEMO")
    print("=" * 60)
    
    # Load dataset
    X, y = load_mri_dataset(dataset_dir)
    if X.size == 0:
        print("No data found!")
        return
    
    print(f"\nDataset: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}")
    
    # Test different models
    models_to_test = {
        'Simple_CNN': create_simple_cnn(),
        'Improved_CNN': create_improved_cnn(),
        'Random_Forest': RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        try:
            results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    # Create results table
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1 Score': f"{metrics['f1_score']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'TN': metrics['confusion_matrix'][0,0],
            'FP': metrics['confusion_matrix'][0,1],
            'FN': metrics['confusion_matrix'][1,0],
            'TP': metrics['confusion_matrix'][1,1]
        })
    
    df = pd.DataFrame(comparison_data)
    df_sorted = df.sort_values('F1 Score', ascending=False)
    
    print(df_sorted.to_string(index=False))
    
    # Save results
    df_sorted.to_csv('mri_model_quick_comparison.csv', index=False)
    
    # Best model
    best_model = df_sorted.iloc[0]['Model']
    best_f1 = df_sorted.iloc[0]['F1 Score']
    
    print(f"\nBest Model: {best_model} (F1 Score: {best_f1})")
    print(f"\nResults saved to: mri_model_quick_comparison.csv")
    
    # Plot confusion matrices
    try:
        n_models = len(results)
        cols = 2
        rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, metrics) in enumerate(results.items()):
            row = i // cols
            col = i % cols
            
            ax = axes[row, col]
            cm = metrics['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Parkinson'],
                       yticklabels=['Normal', 'Parkinson'])
            ax.set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('mri_model_quick_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices plot saved as: mri_model_quick_confusion_matrices.png")
    except Exception as e:
        print(f"Could not create plots: {str(e)}")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()