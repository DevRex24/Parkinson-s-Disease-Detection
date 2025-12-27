#!/usr/bin/env python3
"""
Comprehensive MRI Model Test and Comparison Script
Tests multiple model architectures and compares their performance using:
- Accuracy Score
- F1 Score  
- Confusion Matrix
- Precision and Recall
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
IMG_SIZE = (224, 224)
dataset_dir = 'parkinsons_dataset'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

class MRIModelComparison:
    def __init__(self):
        self.results = {}
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_mri_dataset(self, dataset_dir):
        """Load and preprocess MRI dataset"""
        print(f"Loading MRI dataset from: {dataset_dir}")
        images, labels = [], []
        
        for label, class_name in enumerate(['normal', 'parkinson']):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
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
                    except Exception as e:
                        print(f"Error processing {fname}: {str(e)}")
                        continue
            
            print(f"Processed {file_count} files for {class_name} class")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Dataset loaded: {len(images)} samples, {len(np.unique(labels))} classes")
        print(f"Image shape: {images.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def prepare_data(self):
        """Load and split the dataset"""
        X, y = self.load_mri_dataset(dataset_dir)
        if X.size == 0 or y.size == 0:
            raise ValueError('No MRI images found in dataset. Please check the dataset directory.')
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def create_cnn_model_v1(self):
        """Original CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model_v2(self):
        """Enhanced CNN with batch normalization"""
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
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_cnn_model_v3(self):
        """Deep CNN with residual-like connections"""
        input_layer = layers.Input(shape=(224, 224, 3))
        
        # Block 1
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Classification head
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(input_layer, output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_transfer_learning_model(self):
        """Transfer learning with VGG16"""
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        base_model.trainable = False  # Freeze base model
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate_cnn(self, model_name, create_model_func, epochs=20):
        """Train and evaluate a CNN model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        model = create_model_func()
        model.summary()
        
        # Train model
        print(f"Training {model_name} for {epochs} epochs...")
        history = model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(self.X_test, self.y_test),
            verbose=1
        )
        
        # Make predictions
        y_pred_prob = model.predict(self.X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'history': history,
            'predictions': y_pred_prob.flatten()
        }
        
        self.models[model_name] = model
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return model
    
    def prepare_features_for_ml(self):
        """Extract features for traditional ML models"""
        print("Extracting features for traditional ML models...")
        
        # Flatten and downsample images for traditional ML
        X_train_flat = self.X_train.reshape(self.X_train.shape[0], -1)
        X_test_flat = self.X_test.reshape(self.X_test.shape[0], -1)
        
        # Downsample to reduce dimensionality
        step = 100  # Take every 100th pixel to reduce dimensionality
        X_train_sampled = X_train_flat[:, ::step]
        X_test_sampled = X_test_flat[:, ::step]
        
        print(f"Feature dimension reduced to: {X_train_sampled.shape[1]}")
        
        return X_train_sampled, X_test_sampled
    
    def train_and_evaluate_ml(self, model_name, model, X_train, X_test):
        """Train and evaluate traditional ML model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train, self.y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred_prob
        }
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return model
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, results) in enumerate(self.results.items()):
            row = i // cols
            col = i % cols
            
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Parkinson'],
                       yticklabels=['Normal', 'Parkinson'])
            ax.set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}, F1: {results["f1_score"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('mri_model_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self):
        """Create and display comparison table"""
        print(f"\n{'='*80}")
        print("MRI MODEL PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        # Create DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1 Score': f"{results['f1_score']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'TN': results['confusion_matrix'][0,0],
                'FP': results['confusion_matrix'][0,1],
                'FN': results['confusion_matrix'][1,0],
                'TP': results['confusion_matrix'][1,1]
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score (descending)
        df_sorted = df.sort_values('F1 Score', ascending=False)
        
        print(df_sorted.to_string(index=False))
        
        # Save to CSV
        df_sorted.to_csv('mri_model_comparison_results.csv', index=False)
        print(f"\nResults saved to: mri_model_comparison_results.csv")
        
        # Identify best model
        best_model = df_sorted.iloc[0]['Model']
        best_f1 = df_sorted.iloc[0]['F1 Score']
        print(f"\n🏆 Best Model: {best_model} (F1 Score: {best_f1})")
        
        return df_sorted
    
    def plot_performance_comparison(self):
        """Plot performance comparison chart"""
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        precisions = [self.results[model]['precision'] for model in models]
        recalls = [self.results[model]['recall'] for model in models]
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x - 0.5*width, f1_scores, width, label='F1 Score', alpha=0.8)
        bars3 = ax.bar(x + 0.5*width, precisions, width, label='Precision', alpha=0.8)
        bars4 = ax.bar(x + 1.5*width, recalls, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('MRI Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('mri_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_comparison(self):
        """Run complete model comparison"""
        print("=" * 80)
        print("MRI MODEL COMPARISON AND EVALUATION")
        print("=" * 80)
        
        # Prepare data
        self.prepare_data()
        
        # Train CNN models
        self.train_and_evaluate_cnn('CNN_v1_Original', self.create_cnn_model_v1, epochs=15)
        self.train_and_evaluate_cnn('CNN_v2_BatchNorm', self.create_cnn_model_v2, epochs=15)
        self.train_and_evaluate_cnn('CNN_v3_Deep', self.create_cnn_model_v3, epochs=15)
        
        # Note: VGG16 transfer learning commented out as it requires more resources
        # Uncomment the next line if you want to include it
        # self.train_and_evaluate_cnn('VGG16_Transfer', self.create_transfer_learning_model, epochs=10)
        
        # Prepare features for traditional ML models
        X_train_ml, X_test_ml = self.prepare_features_for_ml()
        
        # Train traditional ML models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        self.train_and_evaluate_ml('Random_Forest', rf_model, X_train_ml, X_test_ml)
        
        svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
        self.train_and_evaluate_ml('SVM_RBF', svm_model, X_train_ml, X_test_ml)
        
        # Generate comparison results
        self.create_comparison_table()
        self.plot_performance_comparison()
        self.plot_confusion_matrices()
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}")
        print("Generated files:")
        print("- mri_model_comparison_results.csv")
        print("- mri_model_performance_comparison.png")
        print("- mri_model_confusion_matrices.png")


def main():
    """Main function to run the comparison"""
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        print("Please ensure the MRI dataset is available in the specified directory.")
        return
    
    # Create comparison object and run
    comparison = MRIModelComparison()
    comparison.run_comparison()


if __name__ == '__main__':
    main()