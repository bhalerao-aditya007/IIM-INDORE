import streamlit as st
import os
import tempfile
import gdown
import pickle
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torchaudio
from tts_models.models.glow_tts import Glow_TTS
from tts_models.models.tacotron2 import Tacotron2
import logging
from typing import Tuple, Optional
import hashlib

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive Model URLs (replace with your actual shareable links)
MODEL_CONFIG = {
    "scenery": {
        "drive_url": "https://drive.google.com/uc?id=YOUR_SCENERY_MODEL_ID",
        "filename": "scenery_vision_model.pkl",
        "description": "Scenery and static environment analysis model"
    },
    "walking": {
        "drive_url": "https://drive.google.com/uc?id=YOUR_WALKING_MODEL_ID",
        "filename": "walking_video_model.pkl",
        "description": "Dynamic movement and navigation analysis model"
    }
}

# Text Content (Bilingual - No Machine Translation)
CONTENT_EN = {
    "app_title": "AccessibleVision",
    "app_subtitle": "Your Digital Eyes — Independent Navigation, Audible Descriptions",
    "mode_selection": "Select Operation Mode",
    "scenery_mode": "📸 Scenery Mode",
    "scenery_desc": "Analyze static images of your surroundings",
    "walking_mode": "🎥 Walking Mode",
    "walking_desc": "Real-time video analysis for navigation",
    "language": "Language / भाषा",
    "upload_image": "Upload Image",
    "upload_video": "Upload Video",
    "supported_formats_image": "Supported: JPG, PNG (Max 50MB)",
    "supported_formats_video": "Supported: MP4, WebM (Max 200MB)",
    "processing": "Processing your media...",
    "analyzing": "Analyzing with vision model...",
    "generating_audio": "Generating audio description...",
    "success": "✅ Description Generated Successfully",
    "error_title": "⚠️ Processing Error",
    "error_no_model": "Model files could not be downloaded. Check your internet connection and model URLs.",
    "error_invalid_file": "Invalid file format or corrupted file.",
    "error_file_size": "File exceeds size limit.",
    "error_processing": "An error occurred during processing.",
    "playback_controls": "Audio Playback",
    "metadata": "Description Metadata",
    "mode_used": "Mode Used",
    "timestamp": "Generated At",
    "file_name": "Original File",
    "download_audio": "Download Audio Description",
    "new_analysis": "Perform New Analysis",
    "loading_model": "Loading vision model...",
    "accessibility_note": "This platform is designed for independent navigation. Audio descriptions will help you understand your surroundings.",
}

CONTENT_HI = {
    "app_title": "AccessibleVision",
    "app_subtitle": "आपकी डिजिटल आंखें — स्वतंत्र नेविगेशन, सुनने योग्य विवरण",
    "mode_selection": "ऑपरेशन मोड चुनें",
    "scenery_mode": "📸 दृश्य मोड",
    "scenery_desc": "अपने आसपास के स्थिर चित्रों का विश्लेषण करें",
    "walking_mode": "🎥 चलने की मोड",
    "walking_desc": "नेविगेशन के लिए वास्तविक समय वीडियो विश्लेषण",
    "language": "Language / भाषा",
    "upload_image": "छवि अपलोड करें",
    "upload_video": "वीडियो अपलोड करें",
    "supported_formats_image": "समर्थित: JPG, PNG (अधिकतम 50MB)",
    "supported_formats_video": "समर्थित: MP4, WebM (अधिकतम 200MB)",
    "processing": "आपके मीडिया को संसाधित किया जा रहा है...",
    "analyzing": "विजन मॉडल के साथ विश्लेषण किया जा रहा है...",
    "generating_audio": "ऑडियो विवरण उत्पन्न किया जा रहा है...",
    "success": "✅ विवरण सफलतापूर्वक उत्पन्न हुआ",
    "error_title": "⚠️ प्रसंस्करण त्रुटि",
    "error_no_model": "मॉडल फ़ाइलें डाउनलोड नहीं की जा सकीं। अपने इंटरनेट कनेक्शन और मॉडल URLs की जांच करें।",
    "error_invalid_file": "अमान्य फ़ाइल प्रारूप या क्षतिग्रस्त फ़ाइल।",
    "error_file_size": "फ़ाइल आकार सीमा से अधिक है।",
    "error_processing": "प्रसंस्करण के दौरान त्रुटि हुई।",
    "playback_controls": "ऑडियो प्लेबैक",
    "metadata": "विवरण मेटाडेटा",
    "mode_used": "उपयोग की गई मोड",
    "timestamp": "द्वारा उत्पन्न",
    "file_name": "मूल फ़ाइल",
    "download_audio": "ऑडियो विवरण डाउनलोड करें",
    "new_analysis": "नया विश्लेषण करें",
    "loading_model": "विजन मॉडल लोड किया जा रहा है...",
    "accessibility_note": "यह प्लेटफॉर्म स्वतंत्र नेविगेशन के लिए डिज़ाइन किया गया है। ऑडियो विवरण आपको अपने आसपास को समझने में मदद करेगा।",
}

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="AccessibleVision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Accessible CSS with High Contrast
st.markdown("""
    <style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }
    
    body {
        background-color: #0a0e27;
        color: #f0f0f0;
    }
    
    .main {
        background-color: #0a0e27;
        padding: 2rem;
    }
    
    .stButton > button {
        background-color: #00d4ff;
        color: #000;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #00a3cc;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:focus {
        outline: 3px solid #ffff00;
        outline-offset: 2px;
    }
    
    .mode-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #2a2f5a 100%);
        border: 3px solid #00d4ff;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .mode-card:hover {
        border-color: #00ffff;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
        transform: translateY(-4px);
    }
    
    .mode-card:focus-within {
        outline: 3px solid #ffff00;
        outline-offset: 2px;
    }
    
    h1, h2, h3 {
        color: #00ffff;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 3px solid #00d4ff;
    }
    
    .success-message {
        background-color: #0d3d1e;
        border-left: 5px solid #00ff00;
        color: #00ff00;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .error-message {
        background-color: #3d0d0d;
        border-left: 5px solid #ff3333;
        color: #ff6666;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .info-message {
        background-color: #0d1f3d;
        border-left: 5px solid #00d4ff;
        color: #00d4ff;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .metadata-table {
        background-color: #1a1f3a;
        border: 2px solid #00d4ff;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
    }
    
    .stSlider > label {
        font-weight: 700;
        font-size: 16px;
    }
    
    .stSelectbox > label, .stRadio > label {
        font-weight: 700;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "language" not in st.session_state:
    st.session_state.language = "EN"
if "mode" not in st.session_state:
    st.session_state.mode = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "metadata" not in st.session_state:
    st.session_state.metadata = {}

def get_text(key: str) -> str:
    """Retrieve text in selected language"""
    content = CONTENT_HI if st.session_state.language == "HI" else CONTENT_EN
    return content.get(key, key)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

@st.cache_resource
def download_model(mode: str) -> Optional[object]:
    """Download model from Google Drive and cache it"""
    try:
        config = MODEL_CONFIG[mode]
        temp_dir = tempfile.gettempdir()
        model_path = os.path.join(temp_dir, config["filename"])
        
        # Check if model already cached locally
        if not os.path.exists(model_path):
            logger.info(f"Downloading {mode} model from Google Drive...")
            with st.spinner(get_text("loading_model")):
                gdown.download(config["drive_url"], model_path, quiet=False)
        
        logger.info(f"Loading {mode} model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        logger.error(f"Model download/load error for {mode}: {str(e)}")
        st.error(f"{get_text('error_no_model')}\n\nDebug: {str(e)}")
        return None

# ============================================================================
# MEDIA PROCESSING FUNCTIONS
# ============================================================================

def validate_file(uploaded_file, mode: str) -> Tuple[bool, str]:
    """Validate uploaded file format and size"""
    if mode == "scenery":
        valid_formats = ['jpg', 'jpeg', 'png']
        max_size = 50 * 1024 * 1024  # 50MB
    else:  # walking
        valid_formats = ['mp4', 'webm']
        max_size = 200 * 1024 * 1024  # 200MB
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    if file_ext not in valid_formats:
        return False, get_text("error_invalid_file")
    
    if uploaded_file.size > max_size:
        return False, get_text("error_file_size")
    
    return True, "OK"

def extract_frames_from_video(video_path: str, sample_rate: int = 2) -> list:
    """Extract frames from video file at specified sample rate"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    return frames

def process_scenery_image(image_path: str, model) -> str:
    """Process image through scenery model and generate description"""
    try:
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for model consistency
        image_resized = cv2.resize(image_rgb, (512, 512))
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Model inference
        with torch.no_grad():
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).permute(0, 3, 1, 2)
            description = model.predict(image_tensor)
        
        return description
    except Exception as e:
        logger.error(f"Scenery processing error: {str(e)}")
        raise

def process_walking_video(video_path: str, model) -> str:
    """Process video through walking model and generate description"""
    try:
        # Extract frames
        frames = extract_frames_from_video(video_path, sample_rate=5)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Resize frames
        processed_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (512, 512))
            normalized = resized.astype(np.float32) / 255.0
            processed_frames.append(normalized)
        
        # Model inference
        with torch.no_grad():
            frames_tensor = torch.from_numpy(np.array(processed_frames)).permute(0, 3, 1, 2)
            description = model.predict(frames_tensor)
        
        return description
    except Exception as e:
        logger.error(f"Walking processing error: {str(e)}")
        raise

def generate_audio_from_text(text: str, lang: str = "en") -> Tuple[torch.Tensor, int]:
    """Generate speech audio from text description"""
    try:
        # Initialize TTS model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if lang == "en":
            model = Glow_TTS(device=device)
        else:  # Hindi
            model = Glow_TTS(device=device)
        
        # Generate mel-spectrogram and convert to waveform
        mel = model.text_to_mel(text)
        
        # Use a vocoder (MelGAN or similar) to convert mel to audio
        # For this example, using a placeholder approach
        wav = torch.randn(1, 16000 * 5)  # 5-second placeholder
        sample_rate = 16000
        
        return wav, sample_rate
    except Exception as e:
        logger.error(f"TTS generation error: {str(e)}")
        raise

def save_audio_file(waveform: torch.Tensor, sample_rate: int, filename: str) -> bytes:
    """Save waveform to audio file and return bytes"""
    try:
        temp_audio_path = os.path.join(tempfile.gettempdir(), filename)
        torchaudio.save(temp_audio_path, waveform, sample_rate)
        
        with open(temp_audio_path, 'rb') as f:
            audio_bytes = f.read()
        
        os.remove(temp_audio_path)
        return audio_bytes
    except Exception as e:
        logger.error(f"Audio save error: {str(e)}")
        raise

# ============================================================================
# HEADER & LANGUAGE SELECTION
# ============================================================================

col_title, col_lang = st.columns([4, 1])

with col_title:
    st.markdown(f"<h1 style='color: #00ffff; font-size: 42px;'>{get_text('app_title')}</h1>", 
                unsafe_allow_html=True)
    st.markdown(f"<p style='color: #00d4ff; font-size: 18px; font-weight: 600;'>"
                f"{get_text('app_subtitle')}</p>", unsafe_allow_html=True)

with col_lang:
    lang_choice = st.radio(
        get_text("language"),
        options=["English", "हिन्दी"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state.language = "HI" if lang_choice == "हिन्दी" else "EN"

st.markdown("---")

# ============================================================================
# ACCESSIBILITY NOTE
# ============================================================================

st.markdown(f"<div class='info-message'>{get_text('accessibility_note')}</div>", 
            unsafe_allow_html=True)

# ============================================================================
# MODE SELECTION
# ============================================================================

st.markdown(f"<h2 class='section-title'>{get_text('mode_selection')}</h2>", 
            unsafe_allow_html=True)

mode_col1, mode_col2 = st.columns(2)

with mode_col1:
    if st.button(
        f"{get_text('scenery_mode')}\n\n{get_text('scenery_desc')}",
        key="btn_scenery",
        use_container_width=True,
        help="Static image analysis"
    ):
        st.session_state.mode = "scenery"
        st.rerun()

with mode_col2:
    if st.button(
        f"{get_text('walking_mode')}\n\n{get_text('walking_desc')}",
        key="btn_walking",
        use_container_width=True,
        help="Dynamic video analysis"
    ):
        st.session_state.mode = "walking"
        st.rerun()

st.markdown("---")

# ============================================================================
# PROCESSING SECTION
# ============================================================================

if st.session_state.mode:
    mode_title = get_text("scenery_mode") if st.session_state.mode == "scenery" else get_text("walking_mode")
    st.markdown(f"<h2 class='section-title'>{mode_title}</h2>", unsafe_allow_html=True)
    
    # File Upload
    if st.session_state.mode == "scenery":
        uploaded_file = st.file_uploader(
            get_text("upload_image"),
            type=['jpg', 'jpeg', 'png'],
            help=get_text("supported_formats_image")
        )
    else:
        uploaded_file = st.file_uploader(
            get_text("upload_video"),
            type=['mp4', 'webm'],
            help=get_text("supported_formats_video")
        )
    
    if uploaded_file:
        # Validate file
        is_valid, validation_msg = validate_file(uploaded_file, st.session_state.mode)
        
        if not is_valid:
            st.markdown(f"<div class='error-message'>{validation_msg}</div>", 
                       unsafe_allow_html=True)
        else:
            # Save uploaded file temporarily
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process button
            if st.button(get_text("processing"), use_container_width=True):
                try:
                    # Load model
                    st.info(get_text("analyzing"))
                    model = download_model(st.session_state.mode)
                    
                    if model is None:
                        raise ValueError("Model loading failed")
                    
                    # Process media
                    if st.session_state.mode == "scenery":
                        description = process_scenery_image(temp_file_path, model)
                    else:
                        description = process_walking_video(temp_file_path, model)
                    
                    # Generate audio
                    st.info(get_text("generating_audio"))
                    lang_code = "hi" if st.session_state.language == "HI" else "en"
                    waveform, sample_rate = generate_audio_from_text(description, lang=lang_code)
                    
                    # Save audio
                    audio_filename = f"description_{hashlib.md5(description.encode()).hexdigest()[:8]}.wav"
                    audio_bytes = save_audio_file(waveform, sample_rate, audio_filename)
                    
                    # Store in session state
                    st.session_state.audio_data = audio_bytes
                    st.session_state.metadata = {
                        "mode": st.session_state.mode,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_name": uploaded_file.name,
                        "description": description
                    }
                    
                    # Success message
                    st.success(get_text("success"))
                    
                except Exception as e:
                    logger.error(f"Processing failed: {str(e)}")
                    st.markdown(f"<div class='error-message'>{get_text('error_processing')}\n\n{str(e)}</div>", 
                               unsafe_allow_html=True)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# ============================================================================
# RESULTS SECTION
# ============================================================================

if st.session_state.audio_data and st.session_state.metadata:
    st.markdown("---")
    st.markdown(f"<h2 class='section-title'>{get_text('playback_controls')}</h2>", 
                unsafe_allow_html=True)
    
    # Audio player
    st.audio(st.session_state.audio_data, format="audio/wav")
    
    # Download button
    st.download_button(
        label=get_text("download_audio"),
        data=st.session_state.audio_data,
        file_name=f"description_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        mime="audio/wav",
        use_container_width=True
    )
    
    # Metadata
    st.markdown("---")
    st.markdown(f"<h3 style='color: #00ffff;'>{get_text('metadata')}</h3>", 
                unsafe_allow_html=True)
    
    metadata_html = f"""
    <div class='metadata-table'>
        <p><strong>{get_text('mode_used')}:</strong> {get_text('scenery_mode') if st.session_state.metadata['mode'] == 'scenery' else get_text('walking_mode')}</p>
        <p><strong>{get_text('timestamp')}:</strong> {st.session_state.metadata['timestamp']}</p>
        <p><strong>{get_text('file_name')}:</strong> {st.session_state.metadata['file_name']}</p>
        <hr style='border: 1px solid #00d4ff; margin: 12px 0;'>
        <p><strong>{get_text('scenery_desc') if st.session_state.metadata['mode'] == 'scenery' else get_text('walking_desc')}:</strong></p>
        <p style='color: #00ffff; font-style: italic;'>{st.session_state.metadata['description']}</p>
    </div>
    """
    st.markdown(metadata_html, unsafe_allow_html=True)
    
    # New analysis button
    if st.button(get_text("new_analysis"), use_container_width=True):
        st.session_state.audio_data = None
        st.session_state.metadata = {}
        st.session_state.mode = None
        st.rerun()
