<<<<<<< HEAD
# Parkinson's Disease Assessment System

A comprehensive multimodal system for Parkinson's Disease assessment using MRI images, voice recordings, and clinical data.

## Features

- **Multimodal Assessment**: Combines MRI analysis, voice analysis, and clinical data
- **Voice Model Integration**: Advanced voice analysis using MFCC features and neural networks
- **Real-time Processing**: Web-based interface for immediate assessment
- **Patient Management**: Store and retrieve patient assessment history
- **Risk Level Classification**: Automatic risk level determination (Low, Moderate, High)

## Voice Model Integration

The system now includes a sophisticated voice analysis component:

### Voice Model Architecture
- **Input**: 13 MFCC (Mel-frequency cepstral coefficients) features
- **Architecture**: 4-layer neural network with dropout regularization
- **Output**: Binary classification (0-1 probability score)
- **Training**: Uses the provided voice dataset with normal and Parkinson's samples

### Voice Processing Pipeline
1. **Audio Recording**: Capture voice samples through web interface
2. **Feature Extraction**: Extract MFCC features using librosa
3. **Model Prediction**: Use trained neural network for classification
4. **Risk Assessment**: Convert probability to risk level

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PD-MultiModal
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Structure

Ensure your dataset follows this structure:

```
Parkinsons_Voice/
├── normal/
│   ├── AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav
│   ├── AH_114S_A89F3548-0B61-4770-B800-2E26AB3908B6.wav
│   └── ...
└── parkinson/
    ├── AH_545616858-3A749CBC-3FEB-4D35-820E-E45C3E5B9B6A.wav
    ├── AH_545622717-461DFFFE-54AF-42AF-BA78-528BD505D624.wav
    └── ...
```

## Training the Voice Model

To train or retrain the voice model:

```bash
python train_voice_model.py
```

This will:
- Load and preprocess the voice dataset
- Extract MFCC features from audio files
- Train a neural network model
- Save the model to `models/voice_model.h5`
- Generate training history plots

## Testing the Integration

Run the test script to verify voice model integration:

```bash
python test_voice_model.py
```

This will test:
- Model loading
- Prediction functionality
- Audio processing pipeline

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## API Endpoints

### Assessment Endpoints
- `POST /api/assess` - Perform multimodal assessment
- `POST /api/test-voice` - Test voice model specifically

### Patient Management
- `GET /api/patients` - Get all patients
- `GET /api/patient/<id>` - Get specific patient
- `POST /api/clear` - Clear all data

### System Information
- `GET /api/model-status` - Get model loading status
- `GET /api/stats` - Get system statistics

## Usage

### Web Interface
1. **Patient Information**: Enter patient details
2. **Assessment Mode**: Choose between MRI, Voice, or Both
3. **Data Collection**: 
   - Upload MRI images or use camera
   - Record voice samples
   - Fill clinical questionnaire
4. **Results**: View assessment results with risk level

### Voice Recording Instructions
When recording voice samples:
- Speak clearly and naturally
- Record for 3-5 seconds
- Ensure quiet environment
- Follow the provided prompts

## Model Performance

The voice model provides:
- **Accuracy**: Based on training dataset
- **Risk Levels**: 
  - Low Risk: < 0.4
  - Moderate Risk: 0.4 - 0.7
  - High Risk: > 0.7

## Technical Details

### Voice Model Specifications
- **Input Features**: 13 MFCC coefficients
- **Architecture**: Dense neural network (128→64→32→1)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Regularization**: Dropout layers (0.3, 0.3, 0.2)
- **Optimizer**: Adam
- **Loss**: Binary crossentropy

### Audio Processing
- **Sample Rate**: Automatic detection
- **Feature Extraction**: librosa MFCC
- **Normalization**: Standard scaling
- **Temporary Files**: Automatic cleanup

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure `models/voice_model.h5` exists
   - Check TensorFlow version compatibility

2. **Audio Processing Errors**:
   - Verify audio file format (WAV)
   - Check librosa installation
   - Ensure sufficient disk space for temporary files

3. **Prediction Errors**:
   - Verify input feature shape (13 features)
   - Check model input/output compatibility

### Debugging
- Check console output for detailed error messages
- Use `/api/model-status` to verify model loading
- Run `test_voice_model.py` for component testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Voice dataset contributors
- TensorFlow and librosa communities
- Medical research community

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review console logs
3. Run test scripts
4. Create an issue with detailed information 
=======
# Parkinson-s-Disease-Detection
AI/ML Project
>>>>>>> 349b0c2e5ee146c09cc6055db90a40e5fcbed6f3
