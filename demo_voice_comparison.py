#!/usr/bin/env python3
"""
Quick Demo: Voice Model Performance Testing
This is a simplified version that trains fewer models for demonstration
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuration
dataset_dir = 'Parkinsons_Voice'
N_MFCC = 13
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_voice_dataset(dataset_dir):
    """Load and extract features from voice dataset"""
    print(f"Loading voice dataset from: {dataset_dir}")
    features, labels = [], []
    
    for label, class_name in enumerate(['normal', 'parkinson']):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"Processing {class_name} class...")
        file_count = 0
        
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.wav'):
                audio_path = os.path.join(class_dir, fname)
                try:
                    y, sr = librosa.load(audio_path, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    
                    if len(mfcc_mean) == N_MFCC:
                        features.append(mfcc_mean)
                        labels.append(label)
                        file_count += 1
                except:
                    continue
        
        print(f"Processed {file_count} files for {class_name} class")
    
    return np.array(features), np.array(labels)

def create_simple_nn():
    """Simple neural network"""
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(N_MFCC,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_improved_nn():
    """Improved neural network with batch normalization"""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(N_MFCC,)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""
    print(f"\nTraining {model_name}...")
    
    # Scale data for neural networks
    if hasattr(model, 'fit') and hasattr(model, 'predict_proba'):
        # Traditional ML model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
    else:
        # Neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=16, verbose=0)
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
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
    print("VOICE MODEL QUICK COMPARISON DEMO")
    print("=" * 60)
    
    # Load dataset
    X, y = load_voice_dataset(dataset_dir)
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
        'Simple_NN': create_simple_nn(),
        'Improved_NN': create_improved_nn(),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
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
    df_sorted.to_csv('voice_model_quick_comparison.csv', index=False)
    
    # Best model
    best_model = df_sorted.iloc[0]['Model']
    best_f1 = df_sorted.iloc[0]['F1 Score']
    
    print(f"\nBest Model: {best_model} (F1 Score: {best_f1})")
    print(f"\nResults saved to: voice_model_quick_comparison.csv")
    
    # Plot confusion matrices
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
    plt.savefig('voice_model_quick_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Confusion matrices plot saved as: voice_model_quick_confusion_matrices.png")
    
    print(f"\n{'='*60}")
    print("DEMO COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()