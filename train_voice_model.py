import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configuration
dataset_dir = 'Parkinsons_Voice'
N_MFCC = 13
EPOCHS = 50
BATCH_SIZE = 16
RANDOM_STATE = 42

def load_voice_dataset(dataset_dir):
    """Load and extract features from voice dataset"""
    features, labels = [], []
    
    print(f"Loading dataset from: {dataset_dir}")
    
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

def create_voice_model():
    """Create a neural network for voice feature analysis"""
    model = models.Sequential([
        # Input layer for MFCC features
        layers.Dense(128, activation='relu', input_shape=(N_MFCC,)),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('voice_model_training_history.png')
    plt.show()

def main():
    print("=== Voice Model Training Script ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess the dataset
    X, y = load_voice_dataset(dataset_dir)
    if X.size == 0 or y.size == 0:
        print('Error: No audio files found in dataset. Please check the dataset directory.')
        return
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    print("\nCreating voice model...")
    model = create_voice_model()
    model.summary()
    
    print(f"\nTraining model for {EPOCHS} epochs...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test_scaled, y_test, verbose=0
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Save the model
    model_path = 'models/voice_model.h5'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save the scaler
    import joblib
    scaler_path = 'models/voice_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Plot training history
    try:
        plot_training_history(history)
        print("Training history plot saved as 'voice_model_training_history.png'")
    except Exception as e:
        print(f"Could not create training plot: {str(e)}")
    
    print("\n=== Training Complete ===")

if __name__ == '__main__':
    main() 