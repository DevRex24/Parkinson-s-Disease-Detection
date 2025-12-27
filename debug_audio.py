#!/usr/bin/env python3
"""
Debug script for audio processing issues
"""

import requests
import json
import base64
import os

def test_audio_processing():
    """Test audio processing with a sample audio file"""
    
    # Test with a sample audio file from the dataset
    dataset_dir = 'Parkinsons_Voice'
    test_file = None
    
    # Find a test audio file
    for class_name in ['normal', 'parkinson']:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.wav'):
                    test_file = os.path.join(class_dir, fname)
                    break
            if test_file:
                break
    
    if not test_file:
        print("❌ No audio files found in dataset")
        return False
    
    print(f"Testing with audio file: {os.path.basename(test_file)}")
    
    try:
        # Read the audio file
        with open(test_file, 'rb') as f:
            audio_data = f.read()
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Test audio processing endpoint
        url = 'http://localhost:5000/api/test-audio'
        payload = {'audio': audio_base64}
        
        print("Sending request to /api/test-audio...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio processing test successful!")
            print(f"Features shape: {result.get('features_shape')}")
            print(f"Sample features: {result.get('features_sample')}")
            return True
        else:
            print(f"❌ Audio processing test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio test: {str(e)}")
        return False

def test_voice_model():
    """Test voice model with a sample audio file"""
    
    # Test with a sample audio file from the dataset
    dataset_dir = 'Parkinsons_Voice'
    test_file = None
    
    # Find a test audio file
    for class_name in ['normal', 'parkinson']:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.wav'):
                    test_file = os.path.join(class_dir, fname)
                    break
            if test_file:
                break
    
    if not test_file:
        print("❌ No audio files found in dataset")
        return False
    
    print(f"Testing voice model with: {os.path.basename(test_file)}")
    
    try:
        # Read the audio file
        with open(test_file, 'rb') as f:
            audio_data = f.read()
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Test voice model endpoint
        url = 'http://localhost:5000/api/test-voice'
        payload = {'audio': audio_base64}
        
        print("Sending request to /api/test-voice...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice model test successful!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Risk level: {result.get('risk_level')}")
            print(f"Model available: {result.get('model_available')}")
            return True
        else:
            print(f"❌ Voice model test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during voice model test: {str(e)}")
        return False

def check_server_status():
    """Check if the server is running and get model status"""
    try:
        # Check server status
        response = requests.get('http://localhost:5000/api/model-status', timeout=10)
        if response.status_code == 200:
            status = response.json()
            print("✅ Server is running")
            print(f"MRI model loaded: {status['mri_model']['loaded']}")
            print(f"Voice model loaded: {status['voice_model']['loaded']}")
            if status['voice_model']['loaded']:
                print(f"Voice model input shape: {status['voice_model']['input_shape']}")
            return True
        else:
            print(f"❌ Server responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        return False

def main():
    print("=== Audio Processing Debug Script ===\n")
    
    # Check server status
    print("1. Checking server status...")
    if not check_server_status():
        print("Please start the Flask server first: python app.py")
        return
    
    print("\n2. Testing audio processing...")
    audio_ok = test_audio_processing()
    
    print("\n3. Testing voice model...")
    voice_ok = test_voice_model()
    
    print("\n=== Debug Summary ===")
    print(f"Audio processing: {'✅' if audio_ok else '❌'}")
    print(f"Voice model: {'✅' if voice_ok else '❌'}")
    
    if audio_ok and voice_ok:
        print("\n🎉 All tests passed! The system should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        print("\nTroubleshooting tips:")
        print("- Ensure the Flask server is running")
        print("- Check that audio files are in WAV format")
        print("- Verify the voice model file exists")
        print("- Check console logs for detailed error messages")

if __name__ == '__main__':
    main() 