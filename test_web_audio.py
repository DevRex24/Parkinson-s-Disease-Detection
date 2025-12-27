#!/usr/bin/env python3
"""
Test script to simulate web interface audio recording
"""

import requests
import json
import base64
import os
import numpy as np

def create_test_audio_data():
    """Create a simple test audio data that simulates web recording"""
    
    # Create a simple sine wave as test audio (1 second, 440 Hz)
    sample_rate = 8000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file temporarily
    import wave
    temp_wav = 'test_audio.wav'
    with wave.open(temp_wav, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Read the WAV file and convert to base64
    with open(temp_wav, 'rb') as f:
        audio_bytes = f.read()
    
    # Clean up
    os.remove(temp_wav)
    
    # Convert to base64 data URL format (like web interface)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    data_url = f"data:audio/wav;base64,{audio_base64}"
    
    return data_url

def test_web_audio_processing():
    """Test audio processing with simulated web audio"""
    
    print("Creating test audio data...")
    test_audio = create_test_audio_data()
    print(f"Test audio data length: {len(test_audio)}")
    print(f"Data URL prefix: {test_audio[:50]}...")
    
    # Test audio processing endpoint
    url = 'http://localhost:5000/api/test-audio'
    payload = {'audio': test_audio}
    
    print("\nSending request to /api/test-audio...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio processing test successful!")
            print(f"Features shape: {result.get('features_shape')}")
            print(f"Sample features: {result.get('features_sample')}")
            print(f"Features stats: {result.get('features_stats')}")
            return True
        else:
            print(f"❌ Audio processing test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio test: {str(e)}")
        return False

def test_web_voice_model():
    """Test voice model with simulated web audio"""
    
    print("\nCreating test audio data for voice model...")
    test_audio = create_test_audio_data()
    
    # Test voice model endpoint
    url = 'http://localhost:5000/api/test-voice'
    payload = {'audio': test_audio}
    
    print("Sending request to /api/test-voice...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice model test successful!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Risk level: {result.get('risk_level')}")
            print(f"Model available: {result.get('model_available')}")
            print(f"Scaler available: {result.get('scaler_available')}")
            return True
        else:
            print(f"❌ Voice model test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during voice model test: {str(e)}")
        return False

def test_assessment_endpoint():
    """Test the full assessment endpoint with simulated data"""
    
    print("\nTesting full assessment endpoint...")
    
    # Create test data
    test_audio = create_test_audio_data()
    
    assessment_data = {
        'mode': 'voice',
        'audio': test_audio,
        'clinical': {
            'name': 'Test Patient',
            'age': 45,
            'gender': 'Male',
            'tremor': 5,
            'symptoms': 12,
            'medication': 'None',
            'history': 'No family history'
        }
    }
    
    url = 'http://localhost:5000/api/assess'
    
    try:
        response = requests.post(url, json=assessment_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Assessment test successful!")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Patient ID: {result.get('patient_id')}")
            print(f"Debug info: {result.get('debug_info')}")
            return True
        else:
            print(f"❌ Assessment test failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during assessment test: {str(e)}")
        return False

def main():
    print("=== Web Audio Processing Test ===\n")
    
    # Test 1: Audio processing
    print("1. Testing audio processing with simulated web audio...")
    audio_ok = test_web_audio_processing()
    
    # Test 2: Voice model
    print("\n2. Testing voice model with simulated web audio...")
    voice_ok = test_web_voice_model()
    
    # Test 3: Full assessment
    print("\n3. Testing full assessment endpoint...")
    assessment_ok = test_assessment_endpoint()
    
    print("\n=== Test Summary ===")
    print(f"Audio processing: {'✅' if audio_ok else '❌'}")
    print(f"Voice model: {'✅' if voice_ok else '❌'}")
    print(f"Full assessment: {'✅' if assessment_ok else '❌'}")
    
    if audio_ok and voice_ok and assessment_ok:
        print("\n🎉 All tests passed! The web interface should work correctly.")
    else:
        print("\n⚠️  Some tests failed. The issue might be:")
        print("- Audio format compatibility")
        print("- Data URL format issues")
        print("- Server configuration problems")

if __name__ == '__main__':
    main() 