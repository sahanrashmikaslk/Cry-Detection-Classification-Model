# app.py
import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import tempfile
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import os

# ---------------------------
# === Configuration ===
# ---------------------------
# Update these paths if your files are stored elsewhere (e.g., drive paths)
# Detection (YAMNet-based) artifacts
DETECTOR_MODEL_PATH = "det_models/yamnet_lr_model.joblib"       # joblib saved detection classifier (lr)
DETECTOR_SCALER_PATH = "det_models/scaler_yamnet.pkl"          # StandardScaler used on yamnet embeddings
DETECTOR_PCA_PATH = "det_models/pca_yamnet.pkl"                # PCA used on yamnet embeddings

# Classification artifacts (your existing cry classifier ensemble)
CLASS_ENSEMBLE_PATH = "models/babycry_ensemble.pkl"
CLASS_SCALER_PATH = "models/scaler.pkl"
CLASS_SELECTOR_PATH = "models/feature_selector.pkl"
CLASS_LE_PATH = "models/label_encoder.pkl"

# audio settings
SR = 16000  # YAMNet expects 16kHz

# ---------------------------
# === Helpers & Loading ===
# ---------------------------
@st.cache_resource
def load_tf_yamnet():
    """Load YAMNet model from TF Hub. Cached for speed."""
    yamnet_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet = hub.load(yamnet_handle)
    return yamnet

@st.cache_resource
def load_detection_components():
    """Load detection classifier + preprocessing (scaler, pca)."""
    missing = []
    for p in [DETECTOR_MODEL_PATH, DETECTOR_SCALER_PATH, DETECTOR_PCA_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing detection files: {missing}")
    det_model = joblib.load(DETECTOR_MODEL_PATH)
    det_scaler = joblib.load(DETECTOR_SCALER_PATH)
    det_pca = joblib.load(DETECTOR_PCA_PATH)
    return det_model, det_scaler, det_pca

@st.cache_resource
def load_classification_components():
    """Load your cry classification ensemble and preprocessors."""
    missing = []
    for p in [CLASS_ENSEMBLE_PATH, CLASS_SCALER_PATH, CLASS_SELECTOR_PATH, CLASS_LE_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError(f"Missing classification files: {missing}")
    ensemble = joblib.load(CLASS_ENSEMBLE_PATH)
    scaler = joblib.load(CLASS_SCALER_PATH)
    selector = joblib.load(CLASS_SELECTOR_PATH)
    le = joblib.load(CLASS_LE_PATH)
    return ensemble, scaler, selector, le

# load models (will raise early if missing)
yamnet = load_tf_yamnet()
det_model, det_scaler, det_pca = load_detection_components()
ensemble, cls_scaler, feature_selector, label_encoder = load_classification_components()

# ---------------------------
# === Feature extraction (YAMNet embedding) ===
# ---------------------------
def extract_yamnet_embedding_for_file(path, yamnet_model=yamnet, sr=SR):
    """Return aggregated YAMNet features for a file: mean + std of embeddings -> shape (2048,)"""
    # load at 16 kHz
    wav, _ = librosa.load(path, sr=sr, mono=True)
    waveform = wav.astype(np.float32)
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # Run YAMNet - returns (scores, embeddings, spectrogram)
    scores, embeddings, spectrogram = yamnet_model(waveform_tf)
    embeddings_np = embeddings.numpy()  # (frames, 1024)
    # aggregate
    mean_emb = np.mean(embeddings_np, axis=0)
    std_emb  = np.std(embeddings_np, axis=0)
    feat = np.concatenate([mean_emb, std_emb]).astype(np.float32)
    return feat

# ---------------------------
# === Detection wrapper ===
# ---------------------------
def detect_is_cry(audio_path, detector_model=det_model, scaler=det_scaler, pca=det_pca, threshold=0.212):
    """
    Returns (is_cry: bool, cry_prob: float)
    cry_prob is probability of class 'cry' (class index 1 used during training)
    """
    try:
        emb = extract_yamnet_embedding_for_file(audio_path)
    except Exception as e:
        st.error(f"YAMNet embedding extraction failed: {e}")
        return False, 0.0

    emb = emb.reshape(1, -1)
    emb_s = scaler.transform(emb)
    emb_p = pca.transform(emb_s)
    probs = detector_model.predict_proba(emb_p)[0]
    # assuming during training label 1 == cry
    cry_prob = float(probs[1])
    is_cry = cry_prob >= threshold
    return is_cry, cry_prob

# ---------------------------
# === Cry classification (your pre-existing pipeline) ===
# ---------------------------
def extract_features_for_classification(audio_input):
    """
    same feature extraction as your classification training pipeline
    Accepts path (str) or raw audio array (numpy).
    """
    try:
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=SR)
        else:
            # if audio_input is numpy array, assume it's 1-D float32 at SR
            y = audio_input
            sr = SR

        stft = np.abs(librosa.stft(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
        energy = np.mean(librosa.feature.rms(y=y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        combined_features = np.concatenate([
            mfcc[:40],
            chroma[:12],
            mel[:40],
            contrast[:7],
            tonnetz[:6],
            [zero_crossing],
            [energy],
            [spec_centroid],
            [spec_bandwidth],
            [spec_rolloff],
            [spec_flatness]
        ])
        return combined_features
    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

def classify_cry(audio_path, ensemble_model=ensemble, scaler=cls_scaler, selector=feature_selector, le=label_encoder, conf_threshold=0.6):
    """Return (predicted_label (str or 'Normal / Not a Cry'), probs array, max_prob)"""
    feats = extract_features_for_classification(audio_path)
    if feats is None:
        return "Normal / Not a Cry", None, 0.0
    feats = feats.reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    feats_selected = selector.transform(feats_scaled)
    probs = ensemble_model.predict_proba(feats_selected)[0]
    max_prob = float(np.max(probs))
    predicted_label = le.inverse_transform([np.argmax(probs)])[0]
    if max_prob < conf_threshold:
        return "Normal / Not a Cry", probs, max_prob
    return predicted_label, probs, max_prob

# ---------------------------
# === Streamlit UI ===
# ---------------------------

st.set_page_config(page_title="Baby Cry Detection + Classification", page_icon="👶", layout="centered")
st.title("👶 Baby Cry — Detection then Classification")
st.write("First the app determines whether the audio contains a baby cry (detection). If it is a cry, the app runs your cry classification ensemble to label the cry type.")

col1, col2 = st.columns([1, 1])
with col1:
    det_threshold = st.slider("Detection threshold (probability for 'cry')", min_value=0.0, max_value=1.0, value=0.212, step=0.001, help="Lower → more likely to classify as cry. Training-chosen threshold ~0.212.")
with col2:
    cls_threshold = st.slider("Classification confidence threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="Minimum confidence required for the cry-type classifier to return a label; otherwise it will be 'Normal / Not a Cry'.")

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Live"])

# --------------------------
# Upload Audio Tab
# --------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac", "ogg"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path)
        if st.button("🔍 Detect & Classify (Uploaded)"):
            with st.spinner("Running detection..."):
                try:
                    is_cry, cry_prob = detect_is_cry(tmp_path, threshold=det_threshold)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    is_cry, cry_prob = False, 0.0

            st.write(f"Detection - Cry probability: **{cry_prob:.3f}**")
            if not is_cry:
                st.warning(f"⚠️ This sound was detected as **NOT a cry** (prob={cry_prob:.3f}).")
            else:
                st.success(f"✅ Detected as a **cry** (prob={cry_prob:.3f}). Now running classification...")
                with st.spinner("Running cry-type classification..."):
                    label, probs, max_prob = classify_cry(tmp_path, conf_threshold=cls_threshold)
                if label == "Normal / Not a Cry":
                    st.warning(f"Classifier is not confident (max_prob={max_prob:.3f}) — returning Normal / Not a Cry.")
                else:
                    st.success(f"**Predicted Cry Type:** {label}  \n(Confidence: {max_prob:.3f})")
                # show class probabilities (if available)
                if probs is not None:
                    fig, ax = plt.subplots()
                    try:
                        classes = label_encoder.classes_
                    except Exception:
                        classes = [f"class_{i}" for i in range(len(probs))]
                    ax.bar(classes, probs)
                    ax.set_ylabel("Confidence")
                    ax.set_title("Prediction Confidence per Class")
                    st.pyplot(fig)

# --------------------------
# Record Audio Tab
# --------------------------
with tab2:
    st.write("Press **Record** to capture audio (samples are saved at 16 kHz).")
    duration = st.slider("Recording duration (seconds):", 2, 10, 5)
    if st.button("🎙️ Record Audio"):
        st.info("Recording... make sure microphone can hear the sound clearly.")
        fs = SR
        try:
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
        except Exception as e:
            st.error(f"Recording failed: {e}")
            audio_data = None

        if audio_data is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_rec:
                # audio_data is shape (N,1)
                write(tmp_rec.name, fs, (audio_data * 32767).astype(np.int16))
                recorded_path = tmp_rec.name
                # persist the recorded file path so it survives the next rerun
                st.session_state["recorded_path"] = recorded_path

    # Read persisted recorded path (if any)
    recorded_path = st.session_state.get("recorded_path", None)
    if recorded_path:
        st.audio(recorded_path)

        if st.button("🔍 Detect & Classify (Recording)"):
            with st.spinner("Running detection..."):
                try:
                    is_cry, cry_prob = detect_is_cry(recorded_path, threshold=det_threshold)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    is_cry, cry_prob = False, 0.0

            st.write(f"Detection - Cry probability: **{cry_prob:.3f}**")
            if not is_cry:
                st.warning(f"⚠️ This sound was detected as **NOT a cry** (prob={cry_prob:.3f}).")
            else:
                st.success(f"✅ Detected as a **cry** (prob={cry_prob:.3f}). Now running classification...")
                with st.spinner("Running cry-type classification..."):
                    label, probs, max_prob = classify_cry(recorded_path, conf_threshold=cls_threshold)
                if label == "Normal / Not a Cry":
                    st.warning(f"Classifier is not confident (max_prob={max_prob:.3f}) — returning Normal / Not a Cry.")
                else:
                    st.success(f"**Predicted Cry Type:** {label}  \n(Confidence: {max_prob:.3f})")
                if probs is not None:
                    fig, ax = plt.subplots()
                    try:
                        classes = label_encoder.classes_
                    except Exception:
                        classes = [f"class_{i}" for i in range(len(probs))]
                    ax.bar(classes, probs)
                    ax.set_ylabel("Confidence")
                    ax.set_title("Prediction Confidence per Class")
                    st.pyplot(fig)
    else:
        st.info("No recording yet — press ▶️ Record Audio to make a recording.")


# ---------------------------
# Footer / quick tips
# ---------------------------
st.markdown("---")
st.info("Notes: The detection model uses YAMNet embeddings + scaler + PCA + LR (as in your training). If you stored your artifacts in different locations (e.g. Google Drive), update the path constants at the top of this file.")
