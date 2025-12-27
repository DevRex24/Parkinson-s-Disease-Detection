#!/usr/bin/env python3
"""
Comprehensive Voice Model Test and Comparison Script
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
dataset_dir = 'Parkinsons_Voice'
N_MFCC = 13
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

class VoiceModelComparison:
    def __init__(self):
        self.results = {}
        self.models = {}
        self.scalers = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_voice_dataset(self, dataset_dir):
        """Load and extract features from voice dataset"""
        print(f"Loading voice dataset from: {dataset_dir}")
        features, labels = [], []
        
        for label, class_name in enumerate(['normal', 'parkinson']):
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            print(f"Processing {class_name} class...")
            file_count = 0
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.wav'):
                    audio_path = os.path.join(class_dir, fname)
                    try:
                        # Load audio
                        y, sr = librosa.load(audio_path, sr=None)
                        
                        # Extract MFCC features
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                        mfcc_mean = np.mean(mfcc.T, axis=0)
                        
                        # Ensure correct shape
                        if len(mfcc_mean) == N_MFCC:
                            features.append(mfcc_mean)
                            labels.append(label)
                            file_count += 1
                        else:
                            print(f"Warning: Skipping {fname} - incorrect feature shape")
                            
                    except Exception as e:
                        print(f"Error processing {fname}: {str(e)}")
                        continue
            
            print(f"Processed {file_count} files for {class_name} class")
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Dataset loaded: {len(features)} samples, {len(np.unique(labels))} classes")
        print(f"Feature shape: {features.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return features, labels
    
    def prepare_data(self):
        """Load and split the dataset"""
        X, y = self.load_voice_dataset(dataset_dir)
        if X.size == 0 or y.size == 0:
            raise ValueError('No audio files found in dataset. Please check the dataset directory.')
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def create_neural_model_v1(self):
        """Original neural network architecture"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(N_MFCC,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_neural_model_v2(self):
        """Enhanced neural network with batch normalization"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(N_MFCC,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_neural_model_v3(self):
        """Deep neural network with residual connections"""
        input_layer = layers.Input(shape=(N_MFCC,))
        
        # First block
        x = layers.Dense(256, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Second block with residual connection
        residual = layers.Dense(128)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Third block
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(input_layer, output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_neural_model_v4(self):
        """LSTM-inspired architecture (treating MFCC as sequence)"""
        input_layer = layers.Input(shape=(N_MFCC,))
        
        # Reshape for LSTM (treat as sequence of length 1)
        x = layers.Reshape((1, N_MFCC))(input_layer)
        
        # LSTM layers
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(input_layer, output)
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate_neural(self, model_name, create_model_func, epochs=50):
        """Train and evaluate a neural network model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Store scaler
        self.scalers[model_name] = scaler
        
        model = create_model_func()
        model.summary()
        
        # Train model
        print(f"Training {model_name} for {epochs} epochs...")
        history = model.fit(
            X_train_scaled, self.y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test_scaled, self.y_test),
            verbose=1
        )
        
        # Make predictions
        y_pred_prob = model.predict(X_test_scaled, verbose=0)
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
    
    def train_and_evaluate_ml(self, model_name, model):
        """Train and evaluate traditional ML model"""
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Create scaler for ML models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Store scaler
        self.scalers[model_name] = scaler
        
        # Train model
        print(f"Training {model_name}...")
        model.fit(X_train_scaled, self.y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Cross-validation for ML models
        cv_scores = cross_val_score(model, X_train_scaled, self.y_train, cv=CV_FOLDS, scoring='f1')
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred_prob,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
        
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"CV F1 Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Confusion Matrix:\n{cm}")
        
        return model
    
    def plot_training_histories(self):
        """Plot training histories for neural network models"""
        neural_models = {name: results for name, results in self.results.items() 
                        if 'history' in results}
        
        if not neural_models:
            return
        
        n_models = len(neural_models)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, results) in enumerate(neural_models.items()):
            history = results['history']
            
            # Plot accuracy
            axes[0, i].plot(history.history['accuracy'], label='Training')
            axes[0, i].plot(history.history['val_accuracy'], label='Validation')
            axes[0, i].set_title(f'{model_name} - Accuracy')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot loss
            axes[1, i].plot(history.history['loss'], label='Training')
            axes[1, i].plot(history.history['val_loss'], label='Validation')
            axes[1, i].set_title(f'{model_name} - Loss')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Loss')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('voice_model_training_histories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
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
        plt.savefig('voice_model_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_table(self):
        """Create and display comparison table"""
        print(f"\n{'='*80}")
        print("VOICE MODEL PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        # Create DataFrame
        comparison_data = []
        for model_name, results in self.results.items():
            row_data = {
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1 Score': f"{results['f1_score']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'TN': results['confusion_matrix'][0,0],
                'FP': results['confusion_matrix'][0,1],
                'FN': results['confusion_matrix'][1,0],
                'TP': results['confusion_matrix'][1,1]
            }
            
            # Add CV scores for ML models
            if 'cv_f1_mean' in results:
                row_data['CV_F1'] = f"{results['cv_f1_mean']:.4f}±{results['cv_f1_std']:.4f}"
            else:
                row_data['CV_F1'] = 'N/A'
            
            comparison_data.append(row_data)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score (descending)
        df_sorted = df.sort_values('F1 Score', ascending=False)
        
        print(df_sorted.to_string(index=False))
        
        # Save to CSV
        df_sorted.to_csv('voice_model_comparison_results.csv', index=False)
        print(f"\nResults saved to: voice_model_comparison_results.csv")
        
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
        ax.set_title('Voice Model Performance Comparison')
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
        plt.savefig('voice_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_best_models(self):
        """Save the best performing models"""
        # Find best neural network model
        neural_results = {name: results for name, results in self.results.items() 
                         if 'history' in results}
        
        if neural_results:
            best_neural = max(neural_results.items(), key=lambda x: x[1]['f1_score'])
            model_name, _ = best_neural
            
            # Save model and scaler
            os.makedirs('models/comparison', exist_ok=True)
            
            model_path = f'models/comparison/best_voice_model_{model_name.lower()}.h5'
            self.models[model_name].save(model_path)
            
            scaler_path = f'models/comparison/best_voice_scaler_{model_name.lower()}.pkl'
            joblib.dump(self.scalers[model_name], scaler_path)
            
            print(f"\nBest neural model saved:")
            print(f"Model: {model_path}")
            print(f"Scaler: {scaler_path}")
    
    def run_comparison(self):
        """Run complete model comparison"""
        print("=" * 80)
        print("VOICE MODEL COMPARISON AND EVALUATION")
        print("=" * 80)
        
        # Prepare data
        self.prepare_data()
        
        # Train neural network models
        self.train_and_evaluate_neural('NN_v1_Original', self.create_neural_model_v1, epochs=40)
        self.train_and_evaluate_neural('NN_v2_BatchNorm', self.create_neural_model_v2, epochs=40)
        self.train_and_evaluate_neural('NN_v3_Residual', self.create_neural_model_v3, epochs=40)
        self.train_and_evaluate_neural('NN_v4_LSTM', self.create_neural_model_v4, epochs=40)
        
        # Train traditional ML models
        rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        self.train_and_evaluate_ml('Random_Forest', rf_model)
        
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.train_and_evaluate_ml('Gradient_Boosting', gb_model)
        
        svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
        self.train_and_evaluate_ml('SVM_RBF', svm_model)
        
        lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        self.train_and_evaluate_ml('Logistic_Regression', lr_model)
        
        knn_model = KNeighborsClassifier(n_neighbors=5)
        self.train_and_evaluate_ml('KNN', knn_model)
        
        # Generate comparison results
        self.create_comparison_table()
        self.plot_performance_comparison()
        self.plot_confusion_matrices()
        self.plot_training_histories()
        self.save_best_models()
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}")
        print("Generated files:")
        print("- voice_model_comparison_results.csv")
        print("- voice_model_performance_comparison.png")
        print("- voice_model_confusion_matrices.png")
        print("- voice_model_training_histories.png")
        print("- Best models saved in models/comparison/")


def main():
    """Main function to run the comparison"""
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        print("Please ensure the voice dataset is available in the specified directory.")
        return
    
    # Create comparison object and run
    comparison = VoiceModelComparison()
    comparison.run_comparison()


if __name__ == '__main__':
    main()