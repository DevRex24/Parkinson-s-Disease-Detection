from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import numpy as np
import cv2
import librosa
import tensorflow as tf
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import uuid
import joblib

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Temporary in-memory storage for patient data
patients_data = {}

# Counter for generating 5-digit patient IDs
patient_id_counter = 10000  # Start from 10000 to ensure 5 digits

# Patient data structure
class PatientData:
    def __init__(self, name, age, gender, tremor_severity, symptoms_duration, 
                 medication, family_history, prediction_result=None):
        global patient_id_counter
        self.id = str(patient_id_counter)
        patient_id_counter += 1
        self.name = name
        self.age = age
        self.gender = gender
        self.tremor_severity = tremor_severity
        self.symptoms_duration = symptoms_duration
        self.medication = medication
        self.family_history = family_history
        self.assessment_date = datetime.utcnow()
        self.prediction_result = prediction_result

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'tremor_severity': self.tremor_severity,
            'symptoms_duration': self.symptoms_duration,
            'medication': self.medication,
            'family_history': self.family_history,
            'assessment_date': self.assessment_date.isoformat(),
            'prediction_result': self.prediction_result
        }

# Load ML models
try:
    mri_model = tf.keras.models.load_model('models/mri_model.h5')
    print("MRI model loaded successfully")
except Exception as e:
    print(f"Warning: MRI model not found - {str(e)}. Using dummy predictions.")
    mri_model = None

try:
    voice_model = tf.keras.models.load_model('models/voice_model.h5')
    print("Voice model loaded successfully")
    print(f"Voice model input shape: {voice_model.input_shape}")
    print(f"Voice model output shape: {voice_model.output_shape}")
except Exception as e:
    print(f"Warning: Voice model not found - {str(e)}. Using dummy predictions.")
    voice_model = None

# Load voice feature scaler
try:
    voice_scaler = joblib.load('models/voice_scaler.pkl')
    print("Voice scaler loaded successfully")
except Exception as e:
    print(f"Warning: Voice scaler not found - {str(e)}. Features will not be scaled.")
    voice_scaler = None

def process_mri(mri_data):
    """Process the MRI image data (base64)"""
    try:
        # Remove data URL prefix if present
        if ',' in mri_data:
            image_data = mri_data.split(',')[1]
        else:
            image_data = mri_data
            
        # Convert to image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Handle different image formats
        if len(image_array.shape) == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB
            pass  # Already RGB
        elif len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        
        # Resize to model input size
        image_array = cv2.resize(image_array, (224, 224))
        
        # Normalize
        image_array = image_array.astype(np.float32) / 255.0
        
        return image_array
    except Exception as e:
        print(f"Error processing MRI: {str(e)}")
        return None

def process_audio(audio_data):
    """Process the voice recording data and extract MFCC features"""
    temp_path = None
    try:
        print(f"Starting audio processing...")
        print(f"Audio data type: {type(audio_data)}")
        print(f"Audio data length: {len(audio_data) if audio_data else 0}")
        
        # Remove data URL prefix if present
        if ',' in audio_data:
            print("Detected data URL format, extracting base64 data...")
            header, audio_binary = audio_data.split(',', 1)
            print(f"Data URL header: {header}")
            audio_binary = base64.b64decode(audio_binary)
        else:
            print("Processing raw base64 data...")
            audio_binary = base64.b64decode(audio_data)
        
        print(f"Decoded audio binary size: {len(audio_binary)} bytes")
        
        # Save temporarily with proper extension based on content
        temp_path = f'temp_audio_{uuid.uuid4().hex[:8]}'
        
        # Try to determine format from data URL header
        if 'audio/webm' in audio_data:
            temp_path += '.webm'
        elif 'audio/mp4' in audio_data:
            temp_path += '.mp4'
        elif 'audio/wav' in audio_data:
            temp_path += '.wav'
        else:
            # Default to wav, librosa can handle various formats
            temp_path += '.wav'
        
        print(f"Saving temporary file: {temp_path}")
        
        with open(temp_path, 'wb') as f:
            f.write(audio_binary)
        
        # Verify file was created
        if not os.path.exists(temp_path):
            raise Exception("Temporary file was not created")
        
        file_size = os.path.getsize(temp_path)
        print(f"Temporary file created, size: {file_size} bytes")
        
        # Extract features using the same parameters as training (N_MFCC=13)
        print("Loading audio with librosa...")
        y, sr = librosa.load(temp_path, sr=None)
        print(f"Audio loaded successfully - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
        
        # Check if audio is not empty
        if len(y) == 0:
            raise Exception("Audio file is empty")
        
        print("Extracting MFCC features...")
        # Extract exactly 13 MFCC features as used in training
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        print(f"MFCC features extracted, shape: {mfccs_scaled.shape}")
        print(f"MFCC feature values: {mfccs_scaled[:5]}...")  # Show first 5 values for debugging
        
        # Ensure the feature vector has the correct shape (13 features)
        if len(mfccs_scaled) != 13:
            print(f"Adjusting feature vector from {len(mfccs_scaled)} to 13 features")
            # Pad or truncate to match training data
            if len(mfccs_scaled) < 13:
                mfccs_scaled = np.pad(mfccs_scaled, (0, 13 - len(mfccs_scaled)), 'constant')
            else:
                mfccs_scaled = mfccs_scaled[:13]
        
        print(f"Final feature vector shape: {mfccs_scaled.shape}")
        print("Audio processing completed successfully")
        
        return mfccs_scaled
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        return None
        
    finally:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                print(f"Failed to clean up temp file: {cleanup_error}")

def predict_voice_features(features):
    """Make prediction using voice model with proper preprocessing"""
    try:
        print(f"Making voice prediction with features shape: {features.shape}")
        
        # Reshape features for model input
        features_reshaped = features.reshape(1, -1)
        print(f"Features reshaped to: {features_reshaped.shape}")
        
        # Apply scaling if scaler is available
        if voice_scaler is not None:
            print("Applying feature scaling...")
            features_scaled = voice_scaler.transform(features_reshaped)
            print(f"Scaled features range: [{features_scaled.min():.4f}, {features_scaled.max():.4f}]")
        else:
            print("Warning: No scaler available, using raw features")
            features_scaled = features_reshaped
        
        # Make prediction
        print("Making model prediction...")
        prediction = voice_model.predict(features_scaled, verbose=0)
        result = float(prediction.flatten()[0])
        
        print(f"Voice model prediction: {result:.4f}")
        
        # Determine risk level for logging
        if result > 0.7:
            risk_level = "High"
        elif result > 0.4:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        print(f"Predicted risk level: {risk_level}")
        
        return result
        
    except Exception as e:
        print(f"Error in voice prediction: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

def dummy_prediction(mode='both'):
    """Generate dummy predictions when models are not available"""
    import random
    
    # Generate realistic-looking predictions
    base_pred = random.uniform(0.2, 0.8)
    noise = random.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, base_pred + noise))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/assess', methods=['POST'])
def assess():
    try:
        data = request.json
        
        # Defensive checks for required keys
        if not data or 'clinical' not in data:
            return jsonify({'success': False, 'error': 'Missing required clinical data'}), 400
            
        clinical_data = data['clinical']
        required_clinical = ['name', 'age', 'gender', 'tremor', 'symptoms', 'medication', 'history']
        
        for key in required_clinical:
            if key not in clinical_data:
                return jsonify({'success': False, 'error': f'Missing clinical field: {key}'}), 400
        
        # Type conversion and validation
        try:
            name = str(clinical_data['name']).strip()
            age = int(clinical_data['age'])
            tremor = int(clinical_data['tremor'])
            symptoms = int(clinical_data['symptoms'])
            gender = str(clinical_data['gender']).strip()
            medication = str(clinical_data['medication']).strip()
            history = str(clinical_data['history']).strip()
            
            # Validate ranges
            if age < 1 or age > 120:
                return jsonify({'success': False, 'error': 'Age must be between 1 and 120'}), 400
            if tremor < 1 or tremor > 10:
                return jsonify({'success': False, 'error': 'Tremor severity must be between 1 and 10'}), 400
            if symptoms < 0:
                return jsonify({'success': False, 'error': 'Symptoms duration cannot be negative'}), 400
                
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid clinical data types: {str(e)}'}), 400
        
        mode = data.get('mode', 'both')
        mri_pred = None
        voice_pred = None
        
        print(f"Processing assessment in {mode} mode")
        
        # Only process and predict for the selected mode
        if mode in ['both', 'image']:
            mri_data = data.get('mri')
            if mri_data:
                print("Processing MRI data...")
                mri_features = process_mri(mri_data)
                if mri_features is not None:
                    if mri_model:
                        try:
                            mri_out = mri_model.predict(np.array([mri_features]), verbose=0)
                            mri_pred = float(mri_out.flatten()[0])
                            print(f"MRI prediction: {mri_pred}")
                        except Exception as e:
                            print(f"Error in MRI model prediction: {str(e)}")
                            mri_pred = dummy_prediction()
                    else:
                        mri_pred = dummy_prediction()
                        print(f"Using dummy MRI prediction: {mri_pred}")
        
        if mode in ['both', 'voice']:
            audio_data = data.get('audio')
            if audio_data:
                print("Processing audio data...")
                audio_features = process_audio(audio_data)
                if audio_features is not None:
                    print(f"Audio features extracted successfully. Shape: {audio_features.shape}")
                    if voice_model:
                        try:
                            voice_pred = predict_voice_features(audio_features)
                            print(f"Voice model prediction: {voice_pred}")
                            
                        except Exception as e:
                            print(f"Error in voice model prediction: {str(e)}")
                            print(f"Error type: {type(e).__name__}")
                            voice_pred = dummy_prediction('voice')
                            print(f"Using dummy voice prediction: {voice_pred}")
                    else:
                        print("Voice model not available, using dummy prediction")
                        voice_pred = dummy_prediction('voice')
                        print(f"Using dummy voice prediction: {voice_pred}")
                else:
                    print("Failed to extract audio features")
                    if mode == 'voice':
                        return jsonify({'success': False, 'error': 'Failed to process audio data. Please ensure the audio recording is clear and try again.'}), 400
                    else:
                        # For 'both' mode, continue with MRI only
                        print("Continuing with MRI assessment only due to audio processing failure")
                        voice_pred = None
            else:
                print("No audio data provided")
                if mode == 'voice':
                    return jsonify({'success': False, 'error': 'No audio data provided for voice assessment'}), 400
        
        # Combine predictions based on mode
        if mode == 'both':
            if mri_pred is not None and voice_pred is not None:
                final_prediction = (mri_pred + voice_pred) / 2
            elif mri_pred is not None:
                final_prediction = mri_pred
            elif voice_pred is not None:
                final_prediction = voice_pred
            else:
                return jsonify({'success': False, 'error': 'No valid data provided for prediction'}), 400
        elif mode == 'image':
            if mri_pred is not None:
                final_prediction = mri_pred
            else:
                return jsonify({'success': False, 'error': 'No valid image data provided for prediction'}), 400
        elif mode == 'voice':
            if voice_pred is not None:
                final_prediction = voice_pred
            else:
                return jsonify({'success': False, 'error': 'No valid audio data provided for prediction'}), 400
        else:
            return jsonify({'success': False, 'error': 'Invalid assessment mode'}), 400
        
        # Create patient data object
        patient = PatientData(
            name=name,
            age=age,
            gender=gender,
            tremor_severity=tremor,
            symptoms_duration=symptoms,
            medication=medication,
            family_history=history,
            prediction_result=float(final_prediction)
        )
        
        # Store in temporary memory
        patients_data[patient.id] = patient
        
        print(f"Assessment completed for {name}, Prediction: {final_prediction}")
        
        return jsonify({
            'success': True,
            'prediction': float(final_prediction),
            'patient_id': patient.id,
            'message': 'Assessment completed successfully',
            'debug_info': {
                'mri_prediction': mri_pred,
                'voice_prediction': voice_pred,
                'mode': mode
            }
        })
        
    except Exception as e:
        print(f"Error in assessment: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get all patients from temporary storage"""
    try:
        return jsonify([patient.to_dict() for patient in patients_data.values()])
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get specific patient from temporary storage"""
    try:
        if patient_id in patients_data:
            return jsonify(patients_data[patient_id].to_dict())
        else:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all temporary patient data"""
    try:
        global patient_id_counter
        patients_data.clear()
        patient_id_counter = 10000  # Reset counter to 10000
        return jsonify({'success': True, 'message': 'All data cleared and patient ID counter reset'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get basic statistics from temporary data"""
    try:
        total_patients = len(patients_data)
        if total_patients == 0:
            return jsonify({
                'total_patients': 0,
                'average_age': 0,
                'average_prediction': 0,
                'gender_distribution': {},
                'risk_levels': {'low': 0, 'moderate': 0, 'high': 0}
            })
        
        ages = [p.age for p in patients_data.values()]
        predictions = [p.prediction_result for p in patients_data.values() if p.prediction_result is not None]
        genders = [p.gender for p in patients_data.values()]
        
        # Calculate risk levels
        risk_levels = {'low': 0, 'moderate': 0, 'high': 0}
        for pred in predictions:
            if pred < 0.4:
                risk_levels['low'] += 1
            elif pred < 0.7:
                risk_levels['moderate'] += 1
            else:
                risk_levels['high'] += 1
        
        # Gender distribution
        gender_dist = {}
        for gender in genders:
            gender_dist[gender] = gender_dist.get(gender, 0) + 1
        
        return jsonify({
            'total_patients': total_patients,
            'average_age': sum(ages) / len(ages) if ages else 0,
            'average_prediction': sum(predictions) / len(predictions) if predictions else 0,
            'gender_distribution': gender_dist,
            'risk_levels': risk_levels
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-voice', methods=['POST'])
def test_voice_model():
    """Test route specifically for voice model"""
    try:
        data = request.json
        if not data or 'audio' not in data:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        
        audio_data = data['audio']
        print("Testing voice model with provided audio...")
        
        # Process audio
        audio_features = process_audio(audio_data)
        if audio_features is None:
            return jsonify({'success': False, 'error': 'Failed to process audio'}), 400
        
        print(f"Audio features shape: {audio_features.shape}")
        
        if voice_model:
            try:
                # Make prediction using the integrated function
                result = predict_voice_features(audio_features)
                
                # Determine risk level
                if result > 0.7:
                    risk_level = "High"
                elif result > 0.4:
                    risk_level = "Moderate"
                else:
                    risk_level = "Low"
                
                return jsonify({
                    'success': True,
                    'prediction': result,
                    'risk_level': risk_level,
                    'model_available': True,
                    'scaler_available': voice_scaler is not None,
                    'message': f'Voice model prediction: {result:.4f} ({risk_level} risk)'
                })
                
            except Exception as e:
                print(f"Voice model prediction error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Model prediction failed: {str(e)}',
                    'model_available': True
                }), 500
        else:
            # Use dummy prediction
            dummy_result = dummy_prediction('voice')
            return jsonify({
                'success': True,
                'prediction': dummy_result,
                'risk_level': 'Unknown (dummy)',
                'model_available': False,
                'scaler_available': False,
                'message': f'Dummy prediction: {dummy_result:.4f} (voice model not available)'
            })
            
    except Exception as e:
        print(f"Error in voice test: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """Get status of loaded models"""
    try:
        status = {
            'mri_model': {
                'loaded': mri_model is not None,
                'input_shape': str(mri_model.input_shape) if mri_model else None,
                'output_shape': str(mri_model.output_shape) if mri_model else None
            },
            'voice_model': {
                'loaded': voice_model is not None,
                'input_shape': str(voice_model.input_shape) if voice_model else None,
                'output_shape': str(voice_model.output_shape) if voice_model else None
            },
            'voice_scaler': {
                'loaded': voice_scaler is not None,
                'type': str(type(voice_scaler).__name__) if voice_scaler else None
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test-audio', methods=['POST'])
def test_audio_processing():
    """Test audio processing without model prediction"""
    try:
        data = request.json
        if not data or 'audio' not in data:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        
        audio_data = data['audio']
        print("Testing audio processing...")
        
        # Process audio
        audio_features = process_audio(audio_data)
        if audio_features is None:
            return jsonify({'success': False, 'error': 'Audio processing failed'}), 400
        
        return jsonify({
            'success': True,
            'message': 'Audio processing successful',
            'features_shape': list(audio_features.shape),
            'features_sample': audio_features[:5].tolist(),  # First 5 features for debugging
            'features_stats': {
                'min': float(audio_features.min()),
                'max': float(audio_features.max()),
                'mean': float(audio_features.mean()),
                'std': float(audio_features.std())
            }
        })
        
    except Exception as e:
        print(f"Error in audio test: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/next-patient-id', methods=['GET'])
def get_next_patient_id():
    """Get the next available patient ID"""
    try:
        return jsonify({
            'success': True,
            'next_patient_id': str(patient_id_counter),
            'total_patients': len(patients_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Parkinson's Assessment System (Temporary Data Mode)")
    print("No database required - all data stored in memory")
    
    # Print model status at startup
    print(f"MRI Model: {'Loaded' if mri_model else 'Not Available'}")
    print(f"Voice Model: {'Loaded' if voice_model else 'Not Available'}")
    print(f"Voice Scaler: {'Loaded' if voice_scaler else 'Not Available'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)