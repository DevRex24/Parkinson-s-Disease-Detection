#!/usr/bin/env python3
"""
Test script for voice model integration
"""

import tensorflow as tf
import numpy as np
import os
import librosa

def test_voice_model_loading():
    """Test if the voice model can be loaded"""
    print("Testing voice model loading...")
    
    try:
        model = tf.keras.models.load_model('models/voice_model.h5')
        print("✓ Voice model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"✗ Failed to load voice model: {str(e)}")
        return None

def test_voice_model_prediction(model):
    """Test voice model prediction with dummy data"""
    print("\nTesting voice model prediction...")
    
    if model is None:
        print("✗ Cannot test prediction - model not loaded")
        return False
    
    try:
        # Create dummy MFCC features (13 features as expected by the model)
        dummy_features = np.random.randn(13)
        dummy_features = np.array([dummy_features])  # Add batch dimension
        
        print(f"  Input features shape: {dummy_features.shape}")
        
        # Make prediction
        prediction = model.predict(dummy_features, verbose=0)
        result = float(prediction.flatten()[0])
        
        print(f"✓ Prediction successful: {result:.4f}")
        
        # Determine risk level
        if result > 0.7:
            risk_level = "High"
        elif result > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        print(f"  Risk level: {risk_level}")
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return False

def test_audio_processing():
    """Test audio processing function"""
    print("\nTesting audio processing...")
    
    # Check if we have any audio files in the dataset
    dataset_dir = 'Parkinsons_Voice'
    audio_files = []
    
    for class_name in ['normal', 'parkinson']:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.wav'):
                    audio_files.append(os.path.join(class_dir, fname))
    
    if not audio_files:
        print("✗ No audio files found in dataset")
        return False
    
    print(f"✓ Found {len(audio_files)} audio files")
    
    # Test processing the first audio file
    try:
        test_file = audio_files[0]
        print(f"  Testing with: {os.path.basename(test_file)}")
        
        # Load and process audio
        y, sr = librosa.load(test_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        print(f"✓ Audio processing successful")
        print(f"  MFCC features shape: {mfcc_mean.shape}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Audio duration: {len(y)/sr:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio processing failed: {str(e)}")
        return False

def test_model_integration():
    """Test the complete integration"""
    print("\n=== Voice Model Integration Test ===\n")
    
    # Test 1: Model loading
    model = test_voice_model_loading()
    
    # Test 2: Model prediction
    prediction_ok = test_voice_model_prediction(model)
    
    # Test 3: Audio processing
    audio_ok = test_audio_processing()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Model loading: {'✓' if model else '✗'}")
    print(f"Model prediction: {'✓' if prediction_ok else '✗'}")
    print(f"Audio processing: {'✓' if audio_ok else '✗'}")
    
    if model and prediction_ok and audio_ok:
        print("\n🎉 All tests passed! Voice model is properly integrated.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    test_model_integration() 