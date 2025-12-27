import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split

dataset_dir = 'Parkinsons_Voice'  # User-provided dataset directory
N_MFCC = 13

# Helper to load audio and extract MFCC features
def load_voice_dataset(dataset_dir):
    features, labels = [], []
    for label, class_name in enumerate(['normal', 'parkinson']):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.wav'):
                audio_path = os.path.join(class_dir, fname)
                y, sr = librosa.load(audio_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                labels.append(label)
    return np.array(features), np.array(labels)

def create_voice_model():
    """Create a neural network for voice feature analysis"""
    model = models.Sequential([
        # Input layer for MFCC features
        layers.Dense(64, activation='relu', input_shape=(N_MFCC,)),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess the dataset
    X, y = load_voice_dataset(dataset_dir)
    if X.size == 0 or y.size == 0:
        print('Error: No audio files found in dataset. Please check the dataset directory.')
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train the model
    model = create_voice_model()
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    model.save('models/voice_model.h5')
    print("Voice model trained and saved to models/voice_model.h5")
    
    # Note: In a real implementation, you would:
    # 1. Load and preprocess your voice recording dataset
    # 2. Extract MFCC features from audio files
    # 3. Split into training and validation sets
    # 4. Train the model using model.fit()
    # 5. Evaluate the model
    # 6. Save the trained model

if __name__ == '__main__':
    main() 