import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
dataset_dir = 'parkinsons_dataset'  # User-provided dataset directory

# Helper to load images and labels
def load_mri_dataset(dataset_dir):
    images, labels = [], []
    for label, class_name in enumerate(['normal', 'parkinson']):
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

def create_mri_model():
    """Create a CNN model for spiral drawing classification"""
    model = models.Sequential([
        # Input layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional layers
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
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

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    X, y = load_mri_dataset(dataset_dir)
    if X.size == 0 or y.size == 0:
        print('Error: No MRI images found in dataset. Please check the dataset directory.')
        return
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and train the model
    model = create_mri_model()
    model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))
    model.save('models/mri_model.h5')
    print("MRI model trained and saved to models/mri_model.h5")
    
    # Note: In a real implementation, you would:
    # 1. Load and preprocess your spiral drawing dataset
    # 2. Split into training and validation sets
    # 3. Train the model using model.fit()
    # 4. Evaluate the model
    # 5. Save the trained model

if __name__ == '__main__':
    main() 