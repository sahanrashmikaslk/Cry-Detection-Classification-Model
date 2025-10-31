import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import tempfile
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ Load Saved Model and Preprocessing Tools
# ------------------------------------------------------------
@st.cache_resource
def load_model_components():
    ensemble = joblib.load("models/babycry_ensemble.pkl")
    scaler = joblib.load("models/scaler.pkl")
    selector = joblib.load("models/feature_selector.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return ensemble, scaler, selector, le

ensemble, scaler, selector, le = load_model_components()


# ------------------------------------------------------------
# 2️⃣ Feature Extraction Function (same as training)
# ------------------------------------------------------------
def extract_features(audio_input):
    try:
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=16000)
        else:
            y = audio_input
            sr = 16000

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


# ------------------------------------------------------------
# 3️⃣ Prediction Function
# ------------------------------------------------------------
def predict(audio, threshold=0.6):
    features = extract_features(audio)
    if features is None:
        predicted_label = "Normal / Not a Cry"
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_selected = selector.transform(features_scaled)
    probs = ensemble.predict_proba(features_selected)[0]
    max_prob = np.max(probs)
    predicted_label = le.inverse_transform([np.argmax(probs)])[0]

    # If below confidence threshold → Normal
    if max_prob < threshold:
        predicted_label = "Normal / Not a Cry"
    return predicted_label, probs, max_prob


# ------------------------------------------------------------
# 4️⃣ Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Baby Cry Classification", page_icon="👶", layout="centered")
st.title("👶 Baby Cry Classification App")
st.write("Upload or record a baby cry sound to identify the reason (e.g., hungry, tired, discomfort, etc.)")

tab1, tab2 = st.tabs(["📁 Upload Audio", "🎤 Record Live"])

# --------------------------
# 📁 Upload Audio Tab
# --------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload a baby cry audio file", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(tmp_path)
        if st.button("🔍 Predict from Uploaded File"):
            label, probs, max_prob = predict(tmp_path)
            if label:
                if label == "Normal / Not a Cry":
                    st.warning(f"⚠️ This sound is likely **not a baby cry** (confidence: {max_prob:.2f})")
                else:
                    st.success(f"**Predicted Cry Type:** {label}  \n(Confidence: {max_prob:.2f})")

                # Show probabilities
                fig, ax = plt.subplots()
                ax.bar(le.classes_, probs)
                ax.set_ylabel("Confidence")
                ax.set_title("Prediction Confidence per Class")
                st.pyplot(fig)

# --------------------------
# 🎤 Record Audio Tab
# --------------------------
with tab2:
    st.write("Press **Record** to capture live audio (5 seconds). Ensure the baby’s cry can be heard clearly.")
    duration = st.slider("Recording duration (seconds):", 3, 10, 5)

    if st.button("🎙️ Record Audio"):
        st.info("Recording... Please wait.")
        fs = 16000
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        st.success("Recording complete!")

        # Save recorded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_rec:
            write(tmp_rec.name, fs, (audio_data * 32767).astype(np.int16))
            recorded_path = tmp_rec.name

        st.audio(recorded_path)

        if st.button("🔍 Predict from Recording"):
            
            label, probs, max_prob = predict(recorded_path)
            
            if label:
                if label == "Normal / Not a Cry":
                    st.warning(f"⚠️ This sound is likely **not a baby cry** (confidence: {max_prob:.2f})")
                else:
                    st.success(f"**Predicted Cry Type:** {label}  \n(Confidence: {max_prob:.2f})")

                # Show probabilities
                fig, ax = plt.subplots()
                ax.bar(le.classes_, probs)
                ax.set_ylabel("Confidence")
                ax.set_title("Prediction Confidence per Class")
                st.pyplot(fig)
