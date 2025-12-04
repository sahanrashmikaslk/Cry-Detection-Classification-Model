# Baby Cry Detection & Classification System

A comprehensive machine learning system for detecting and classifying baby cries using advanced audio processing and deep learning techniques. This project implements a two-stage pipeline: first detecting whether audio contains a baby cry, then classifying the cry type to help caregivers understand the baby's needs.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project addresses the challenge of understanding baby cries by using machine learning to:

1. **Detect** whether an audio clip contains a baby cry (Detection Model)
2. **Classify** the type of cry into categories like hungry, tired, discomfort, belly pain, or burping (Classification Model)

The system is designed to assist parents and caregivers in quickly understanding what a baby needs, reducing stress and improving response times.

## Features

- **Two-Stage Pipeline**: Detection → Classification for high accuracy
- **Real-time Processing**: Analyze audio files or record live audio
- **Interactive Web UI**: User-friendly Streamlit interface
- **Multiple Cry Types**: Classifies 5 different types of baby cries
- **Confidence Thresholds**: Adjustable thresholds for detection and classification
- **Audio Visualization**: Visual feedback with confidence scores
- **Model Persistence**: Pre-trained models ready for deployment
- **Extensible Architecture**: Easy to add new cry types or improve models

## Architecture

### System Pipeline

```
Audio Input → Preprocessing (16kHz) → Feature Extraction
                                              ↓
                                    ┌─────────┴─────────┐
                                    ↓                   ↓
                            Detection Model     Classification Model
                          (YAMNet + LR)         (Audio Features + Ensemble)
                                    ↓                   ↓
                              Is it a cry?        What type of cry?
                                    ↓                   ↓
                                    └─────────┬─────────┘
                                              ↓
                                        Final Output
```

### Stage 1: Detection Model

- **Input**: Raw audio waveform (16kHz sampling rate)
- **Feature Extractor**: YAMNet (pre-trained audio embedding model from Google)
- **Features**: 2048-dimensional embeddings (mean + std aggregation → 1024 each)
- **Preprocessing**: StandardScaler → PCA (dimensionality reduction)
- **Classifier**: Logistic Regression
- **Output**: Binary classification (cry / not_cry) with probability score
- **Threshold**: Default 0.212 (tunable)

### Stage 2: Classification Model

- **Input**: Audio confirmed as cry from Stage 1
- **Features Extracted** (108 total):
  - 40 MFCC coefficients
  - 12 Chroma features
  - 40 Mel-spectrogram features
  - 7 Spectral contrast features
  - 6 Tonnetz features
  - Time-domain features (zero-crossing rate, RMS energy)
  - Spectral features (centroid, bandwidth, rolloff, flatness)
- **Preprocessing**: StandardScaler → Feature Selection (SelectFromModel)
- **Classifier**: Ensemble Voting Classifier combining:
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - XGBoost
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Output**: Cry type with confidence score
- **Classes**: belly_pain, burping, discomfort, hungry, tired

## 🛠️ Technologies Used

### Core Frameworks & Libraries

| Technology         | Version | Purpose                               |
| ------------------ | ------- | ------------------------------------- |
| **Python**         | 3.8+    | Programming language                  |
| **TensorFlow**     | 2.20.0  | Deep learning framework               |
| **TensorFlow Hub** | 0.16.1  | Pre-trained model repository (YAMNet) |
| **Streamlit**      | 1.51.0  | Web application framework             |
| **NumPy**          | 2.3.4   | Numerical computing                   |
| **Pandas**         | 2.3.3   | Data manipulation                     |

### Audio Processing

| Library         | Purpose                                  |
| --------------- | ---------------------------------------- |
| **Librosa**     | Audio feature extraction and analysis    |
| **SoundDevice** | Real-time audio recording                |
| **SoundFile**   | Audio file I/O                           |
| **SciPy**       | Signal processing and audio file writing |
| **AudioRead**   | Audio decoding (MP3, FLAC, etc.)         |

### Machine Learning

| Library              | Purpose                                  |
| -------------------- | ---------------------------------------- |
| **Scikit-learn**     | ML algorithms, preprocessing, evaluation |
| **XGBoost**          | Gradient boosting classifier             |
| **Imbalanced-learn** | SMOTE for handling class imbalance       |
| **Joblib**           | Model serialization and persistence      |

### Visualization & UI

| Library        | Purpose                                       |
| -------------- | --------------------------------------------- |
| **Matplotlib** | Plotting and visualization                    |
| **Seaborn**    | Statistical data visualization                |
| **Altair**     | Interactive visualizations (Streamlit charts) |
| **Pillow**     | Image processing                              |

### Supporting Libraries

- **Numba**: JIT compilation for faster numerical operations
- **CFFI**: Foreign function interface for C libraries
- **ProtoBuf**: Protocol buffers for model serialization
- **PyArrow**: Efficient data interchange

## Dataset Structure

```
Cry-Detection-Classification-Model/
│
├── dataset/                    # Classification dataset
│   ├── belly_pain/            # Audio files of belly pain cries
│   ├── burping/               # Audio files of burping cries
│   ├── discomfort/            # Audio files of discomfort cries
│   ├── hungry/                # Audio files of hungry cries
│   └── tired/                 # Audio files of tired cries
│
└── det_dataset/               # Detection dataset
    ├── cry/                   # Audio files containing baby cries
    └── not_cry/               # Audio files without baby cries
```

**Dataset Requirements:**

- Audio format: WAV, MP3, FLAC, or OGG
- Sampling rate: Automatically resampled to 16kHz
- Duration: Variable (model handles different lengths)
- Recommended: At least 100+ samples per class for good performance

## Models

### Pre-trained Model Files

All trained models are stored in the `cry_project/` directory:

#### Detection Models (`det_models/`)

- `yamnet_lr_model.joblib` - Logistic Regression classifier
- `scaler_yamnet.pkl` - StandardScaler for normalization
- `pca_yamnet.pkl` - PCA for dimensionality reduction

#### Classification Models (`models/`)

- `babycry_ensemble.pkl` - Voting Ensemble classifier
- `scaler.pkl` - StandardScaler for feature normalization
- `feature_selector.pkl` - Feature selector (best features)
- `label_encoder.pkl` - Label encoder for cry types

### Model Architecture Details

#### Detection Model Training

- **Algorithm**: Logistic Regression with L2 regularization
- **Input Features**: YAMNet embeddings (2048D → PCA reduced)
- **Training Strategy**: 70% train, 15% validation, 15% test split
- **Optimization**: Grid search for threshold tuning
- **Evaluation Metrics**: Precision-Recall AUC, F1-Score

#### Classification Model Training

- **Ensemble Method**: Soft voting across 4 classifiers
- **Cross-Validation**: 5-fold Stratified K-Fold
- **Feature Selection**: SelectFromModel with Random Forest importance
- **Class Balancing**: SMOTE oversampling for minority classes
- **Evaluation**: Confusion matrix, classification report, accuracy

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- Microphone (optional, for live recording)

### Step-by-Step Setup

1. **Clone the repository**

```powershell
git clone https://github.com/HasiniPrasadika/Cry-Detection-Classification-Model.git
cd Cry-Detection-Classification-Model/cry_project
```

2. **Create and activate virtual environment**

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Activate on Windows CMD
.\.venv\Scripts\activate.bat

# Activate on Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installing TensorFlow may take several minutes and requires ~500MB of disk space.

### Troubleshooting Installation

**TensorFlow fails to install:**

- Check Python version (3.8-3.11 recommended)
- On Windows, ensure Microsoft Visual C++ Redistributable is installed
- Use `app.py` (classification-only) if you want to skip TensorFlow

**SoundDevice issues:**

- Ensure audio drivers are up to date
- On Linux: `sudo apt-get install portaudio19-dev`
- On Mac: `brew install portaudio`

## Usage

### Running the Web Application

#### Option 1: Classification-Only App (Lightweight)

```powershell
streamlit run app.py
```

- No TensorFlow required
- Direct classification of uploaded audio
- Best for: Quick testing with pre-recorded audio

#### Option 2: Full Pipeline App (Detection + Classification)

```powershell
streamlit run cry_classify_app.py
```

- Uses YAMNet detection model
- Two-stage pipeline for higher accuracy
- Best for: Production use with unknown audio sources

### Accessing the Application

After running either command, Streamlit will display:

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open the Local URL in your web browser.

### Using the Web Interface

#### Upload Audio Tab

1. Click "Browse files" to upload a baby cry audio file
2. Supported formats: WAV, MP3, FLAC, OGG
3. Listen to your uploaded audio
4. Click " Predict" button
5. View results with confidence scores and visualization

#### Record Live Audio Tab

1. Adjust recording duration (3-10 seconds)
2. Click " Record Audio" button
3. Ensure microphone permissions are granted
4. Wait for recording to complete
5. Listen to the recorded audio
6. Click " Predict from Recording"
7. View classification results

### Adjustable Parameters

**Detection Threshold** (`cry_classify_app.py` only):

- Range: 0.0 - 1.0
- Default: 0.212
- Lower → More likely to classify as cry (higher recall)
- Higher → More selective (higher precision)

**Classification Confidence Threshold**:

- Range: 0.0 - 1.0
- Default: 0.6
- Minimum confidence required to classify cry type
- Below threshold → Returns "Normal / Not a Cry"

## Project Structure

```
Cry-Detection-Classification-Model/
│
├── cry_project/                          # Main application directory
│   ├── app.py                           # Streamlit app (classification-only)
│   ├── cry_classify_app.py              # Streamlit app (detection + classification)
│   ├── requirements.txt                 # Python dependencies
│   │
│   ├── models/                          # Classification model artifacts
│   │   ├── babycry_ensemble.pkl        # Ensemble classifier
│   │   ├── scaler.pkl                  # Feature scaler
│   │   ├── feature_selector.pkl        # Feature selector
│   │   └── label_encoder.pkl           # Label encoder
│   │
│   ├── det_models/                      # Detection model artifacts
│   │   ├── yamnet_lr_model.joblib      # Logistic Regression classifier
│   │   ├── scaler_yamnet.pkl           # YAMNet feature scaler
│   │   └── pca_yamnet.pkl              # PCA transformer
│   │
│   ├── CryClassification.ipynb          # Classification model training notebook
│   └── CryDetectionModelTraining.ipynb  # Detection model training notebook
│
├── dataset/                             # Classification training dataset
│   ├── belly_pain/                     # Belly pain cry samples
│   ├── burping/                        # Burping cry samples
│   ├── discomfort/                     # Discomfort cry samples
│   ├── hungry/                         # Hungry cry samples
│   └── tired/                          # Tired cry samples
│
├── det_dataset/                         # Detection training dataset
│   ├── cry/                            # Baby cry samples
│   └── not_cry/                        # Non-cry audio samples
│
└── README.md                            # This file
```

## Model Training

### Training the Detection Model

1. **Open the notebook**: `CryDetectionModelTraining.ipynb`

2. **Prepare your dataset**:

   - Organize audio files in `det_dataset/cry/` and `det_dataset/not_cry/`
   - Ensure balanced classes (or use data augmentation)

3. **Key training steps**:

   ```python
   # Extract YAMNet embeddings
   # Apply StandardScaler normalization
   # Apply PCA dimensionality reduction
   # Train Logistic Regression classifier
   # Tune detection threshold
   # Save models to det_models/
   ```

4. **Evaluation metrics**:
   - Precision-Recall Curve
   - ROC-AUC Score
   - Confusion Matrix
   - Optimal threshold selection

### Training the Classification Model

1. **Open the notebook**: `CryClassification.ipynb`

2. **Prepare your dataset**:

   - Organize audio files in `dataset/{class_name}/`
   - Minimum 50+ samples per class recommended

3. **Key training steps**:

   ```python
   # Extract 108 audio features per sample
   # Apply SMOTE for class balancing
   # Feature scaling with StandardScaler
   # Feature selection with SelectFromModel
   # Train ensemble (RF, SVM, KNN, XGBoost)
   # Cross-validation and evaluation
   # Save all artifacts to models/
   ```

4. **Evaluation metrics**:
   - Confusion Matrix
   - Classification Report (precision, recall, F1)
   - Cross-validation scores
   - Feature importance analysis

### Training Tips

- **Data Quality**: Clean, labeled data is crucial
- **Augmentation**: Use pitch shifting, time stretching, noise addition
- **Class Balance**: Ensure similar sample counts across classes
- **Validation**: Always use separate test set
- **Hyperparameters**: Tune via grid/random search
- **Computation**: Training may require GPU for large datasets

## Web Application

### Features

**Two Application Modes:**

1. **app.py** - Lightweight classification

   - Direct audio → classification
   - No heavy TensorFlow dependencies
   - Faster startup time
   - Best for known baby cry audio

2. **cry_classify_app.py** - Full pipeline
   - Detection stage filters non-cry audio
   - Higher accuracy on mixed audio
   - Adjustable thresholds
   - Production-ready

### UI Components

- **File Uploader**: Drag-and-drop or browse for audio files
- **Audio Player**: Preview uploaded/recorded audio
- **Live Recording**: Browser-based microphone access
- **Prediction Button**: Trigger inference
- **Results Display**:
  - Predicted cry type
  - Confidence score
  - Bar chart of class probabilities
  - Warning messages for low confidence
- **Threshold Sliders**: Real-time adjustment of model sensitivity

### Technical Details

- **Framework**: Streamlit 1.51.0
- **Session State**: Preserves recordings across reruns
- **Caching**: `@st.cache_resource` for model loading
- **Responsive Design**: Centered layout, mobile-friendly
- **Error Handling**: Graceful failures with user-friendly messages

## Performance

### Detection Model Performance

| Metric            | Score   |
| ----------------- | ------- |
| Accuracy          | ~92-95% |
| Precision (cry)   | ~90%    |
| Recall (cry)      | ~93%    |
| F1-Score          | ~91%    |
| Optimal Threshold | 0.212   |

### Classification Model Performance

| Metric            | Score    |
| ----------------- | -------- |
| Overall Accuracy  | ~85-90%  |
| Cross-Validation  | ~83% ±3% |
| Weighted F1-Score | ~87%     |

**Per-Class Performance:**

- Hungry: High precision and recall
- Tired: Good performance
- Discomfort: Moderate performance
- Belly Pain: Moderate (can be confused with discomfort)
- Burping: Lower recall (smaller dataset)

**Confusion Notes:**

- Some overlap between discomfort and belly pain
- Tired cries can be misclassified when low energy
- Model confidence decreases with background noise

_If you find this project helpful, please consider giving it a ⭐ on GitHub!_
