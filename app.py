import streamlit as st
import librosa
import numpy as np
import joblib
from tempfile import NamedTemporaryFile
import os
import base64
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time

# Set page config
st.set_page_config(
    page_title="Speech Emotion Detector",
    page_icon="🎵",
    layout="centered"
)

# Function to set background image
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64_encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Add overlay for better readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: -1;
        }}
        
        /* Style main content areas */
        .main .block-container {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        /* Header styling */
        .main-header {{
            font-size: 3rem;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            font-weight: bold;
        }}
        
        /* Success box styling */
        .success-box {{
            padding: 1.5rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            border: 2px solid #28a745;
            color: #155724;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }}
        
        /* Recording box styling */
        .recording-box {{
            padding: 1.5rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #ffeaa7, #fab1a0);
            border: 2px solid #e17055;
            color: #2d3436;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            text-align: center;
        }}
        
        /* File info styling */
        .file-info {{
            padding: 1.5rem;
            background: rgba(248, 249, 250, 0.9);
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }}
        
        /* Button styling */
        .stButton>button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 25px;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }}
        
        /* Record button styling */
        .record-button {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        }}
        
        /* Stop button styling */
        .stop-button {{
            background: linear-gradient(135deg, #00b894 0%, #00a085 100%) !important;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            font-weight: bold;
        }}
        
        /* Footer styling */
        .footer {{
            text-align: center;
            color: white;
            margin-top: 2rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set the background image
set_background("Party vinyl.jpeg")

# Initialize session state for recording
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recorded_file' not in st.session_state:
    st.session_state.recorded_file = None

# Title with enhanced styling
st.markdown('<h1 class="main-header">🎵 Speech Emotion Detector</h1>', unsafe_allow_html=True)

# About section
with st.expander("ℹ️ About this App", expanded=True):
    st.markdown("""
    **This app detects emotions from speech using machine learning.**
    
    **Features:**
    - 🎤 **Live Voice Recording** - Record and analyze your voice in real-time
    - 📁 **Audio File Upload** - Upload existing audio files
    - 🎯 **8 Emotions Detected** - Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised
    
    **Model Accuracy:** 56.6% on RAVDESS dataset
    
    *Built with Streamlit and Scikit-learn*
    """)

# ===== FIXED FEATURE EXTRACTION =====
def extract_streamlit_features(file_path):
    """
    EXACTLY the same function used during training
    This ensures consistent 189 features
    """
    try:
        # Load audio - consistent parameters
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
        target_length = 66150  # 3 seconds
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        features = []
        
        # 1. MFCCs - 40 coefficients with derivatives
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        mfccs_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        
        features.extend(mfccs_mean)    # 40
        features.extend(mfccs_std)     # 40
        features.extend(mfccs_delta)   # 40
        features.extend(mfccs_delta2)  # 40
        
        # 2. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512)
        features.extend(np.mean(chroma, axis=1))  # 12
        features.extend(np.std(chroma, axis=1))   # 12
        
        # 3. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
        
        # 4. Temporal features
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        features.extend([zcr, rms])
        
        # Ensure exactly 189 features
        if len(features) < 189:
            features.extend([0.0] * (189 - len(features)))
        else:
            features = features[:189]
            
        return np.array(features)
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# ===== FIXED PREDICTION FUNCTION =====
def predict_emotion(audio_file_path):
    """
    Fixed prediction function that uses the same feature extraction
    """
    try:
        # Extract features using the SAME function as training
        features = extract_streamlit_features(audio_file_path)
        
        if features is None:
            return "Error: Could not extract features from audio", 0
        
        # Ensure correct dimensions
        features = features.reshape(1, -1)
        
        # Load the Streamlit-compatible model
        model = joblib.load('streamlit_emotion_model.pkl')
        scaler = joblib.load('streamlit_scaler.pkl')
        encoder = joblib.load('streamlit_encoder.pkl')
        
        # Transform and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        emotion = encoder.inverse_transform(prediction)[0]
        
        # Get confidence score
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities) * 100
        
        return emotion, confidence
        
    except Exception as e:
        return f"Prediction error: {str(e)}", 0

# ===== LIVE RECORDING FUNCTIONS =====
def record_audio(duration=5, sample_rate=22050):
    """Record audio for specified duration"""
    st.info(f"🎤 Recording for {duration} seconds... Speak now!")
    
    # Record audio
    audio_data = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1, 
                       dtype='float64')
    sd.wait()  # Wait until recording is finished
    
    # Save to temporary file
    temp_file = "temp_recording.wav"
    sf.write(temp_file, audio_data, sample_rate)
    
    return temp_file

def start_recording():
    """Start recording in a separate thread"""
    st.session_state.recording = True
    st.session_state.audio_data = None
    
    # Record for 5 seconds
    recorded_file = record_audio(duration=5)
    st.session_state.recorded_file = recorded_file
    st.session_state.recording = False
    
    st.success("✅ Recording completed! Click 'Analyze Recording' to detect emotion.")

# ===== MAIN APP INTERFACE =====
st.markdown("## 🎯 Choose Detection Method")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["🎤 Live Recording", "📁 Upload Audio"])

with tab1:
    st.markdown("### 🎙️ Live Voice Recording")
    st.markdown("Record your voice and analyze emotions in real-time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 Start Recording", key="record", use_container_width=True):
            start_recording()
    
    with col2:
        if st.session_state.recorded_file and not st.session_state.recording:
            if st.button("🔍 Analyze Recording", key="analyze", use_container_width=True):
                with st.spinner("🔍 Analyzing your recording..."):
                    emotion, confidence = predict_emotion(st.session_state.recorded_file)
                    
                    # Display results
                    if "error" not in emotion.lower():
                        emotion_icons = {
                            'angry': '😠', 'calm': '😌', 'disgust': '🤢',
                            'fearful': '😨', 'happy': '😊', 'neutral': '😐',
                            'sad': '😢', 'surprised': '😲'
                        }
                        
                        icon = emotion_icons.get(emotion, '🎵')
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>{icon} Emotion Detected!</h3>
                            <p><strong>Emotion:</strong> {emotion.capitalize()}</p>
                            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show emotion description
                        emotion_descriptions = {
                            'angry': 'Characterized by loud, high-energy speech with tense vocal cords.',
                            'calm': 'Soft, steady speech with balanced pitch and moderate pace.',
                            'disgust': 'Often features nasal tones and contemptuous inflections.',
                            'fearful': 'Higher pitch, faster speech, and sometimes trembling voice.',
                            'happy': 'Bright, energetic tone with varied pitch and fast pace.',
                            'neutral': 'Flat, monotone speech with little emotional variation.',
                            'sad': 'Slow, low-pitched speech with less energy and variation.',
                            'surprised': 'Sudden pitch changes, faster speech, and higher energy.'
                        }
                        
                        description = emotion_descriptions.get(emotion, 'Emotion characteristics vary by individual.')
                        st.info(f"**About {emotion.capitalize()} speech:** {description}")
                        
                        # Play recorded audio
                        st.audio(st.session_state.recorded_file, format='audio/wav')
    
    # Show recording status
    if st.session_state.recording:
        st.markdown("""
        <div class="recording-box">
            <h3>🔴 Recording in Progress...</h3>
            <p>Speak clearly into your microphone</p>
            <p>⏰ 5 seconds remaining</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📁 Upload Audio File")
    st.markdown("Upload an existing audio file for emotion analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a'],
        help="Limit 200MB per file - WAV, MP3, M4A",
        key="file_uploader"
    )

    if uploaded_file is not None:
        # Display file info
        file_size = uploaded_file.size / 1024  # KB
        st.markdown(f"""
        <div class="file-info">
            <strong>File:</strong> {uploaded_file.name}<br>
            <strong>Size:</strong> {file_size:.1f} KB<br>
            <strong>Type:</strong> {uploaded_file.type}
        </div>
        """, unsafe_allow_html=True)
        
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Save uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Predict button
        if st.button("🎭 Detect Emotion", type="primary", use_container_width=True, key="detect_file"):
            with st.spinner("🔍 Analyzing audio features..."):
                # Make prediction
                emotion, confidence = predict_emotion(tmp_path)
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                # Display results
                if "error" in emotion.lower():
                    st.error(f"❌ {emotion}")
                else:
                    emotion_icons = {
                        'angry': '😠', 'calm': '😌', 'disgust': '🤢',
                        'fearful': '😨', 'happy': '😊', 'neutral': '😐',
                        'sad': '😢', 'surprised': '😲'
                    }
                    
                    icon = emotion_icons.get(emotion, '🎵')
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>{icon} Emotion Detected!</h3>
                        <p><strong>Emotion:</strong> {emotion.capitalize()}</p>
                        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show emotion description
                    emotion_descriptions = {
                        'angry': 'Characterized by loud, high-energy speech with tense vocal cords.',
                        'calm': 'Soft, steady speech with balanced pitch and moderate pace.',
                        'disgust': 'Often features nasal tones and contemptuous inflections.',
                        'fearful': 'Higher pitch, faster speech, and sometimes trembling voice.',
                        'happy': 'Bright, energetic tone with varied pitch and fast pace.',
                        'neutral': 'Flat, monotone speech with little emotional variation.',
                        'sad': 'Slow, low-pitched speech with less energy and variation.',
                        'surprised': 'Sudden pitch changes, faster speech, and higher energy.'
                    }
                    
                    description = emotion_descriptions.get(emotion, 'Emotion characteristics vary by individual.')
                    st.info(f"**About {emotion.capitalize()} speech:** {description}")

# Instructions
st.markdown("---")
st.markdown("### 📝 Instructions")
st.markdown("""
**For Live Recording:**
1. Click **"Start Recording"** 
2. Speak clearly for 5 seconds
3. Click **"Analyze Recording"** to get results

**For File Upload:**
1. Click **"Browse files"** to upload audio
2. Click **"Detect Emotion"** to analyze

**Tips for better accuracy:**
- Speak clearly and at normal volume
- Record in a quiet environment
- Minimum 2-3 seconds of speech recommended
""")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Built with ❤️ using Streamlit and Scikit-learn | Model Accuracy: 56.6%</div>', unsafe_allow_html=True)