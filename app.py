import streamlit as st
import os
import tempfile
import gdown
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple, Optional
import hashlib
from PIL import Image
import numpy as np
import io
import base64

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Drive Model URLs - YOUR MODELS THAT OUTPUT AUDIO
MODEL_CONFIG = {
    "scenery": {
        "drive_url": "https://drive.google.com/uc?id=13OzMZ2SYI1pZIFEDs-frk9WcW-D_ndyG",
        "filename": "scenery_vision_model.pkl",
        "description": "Image → Audio Model",
        "input_type": "image",
        "output_type": "audio"  # RETURNS AUDIO BYTES
    },
    "walking": {
        "drive_url": "https://drive.google.com/uc?id=1wE1W33tQ0WUqGBK8ISb2eeSrFUG3vJJm",
        "filename": "walking_video_model.pkl",
        "description": "Video → Audio Model",
        "input_type": "video",
        "output_type": "audio"  # RETURNS AUDIO BYTES
    }
}

# Text Content (Bilingual - No Machine Translation)
CONTENT_EN = {
    "app_title": "AccessibleVision",
    "app_subtitle": "Your Digital Eyes — Independent Navigation, Audible Descriptions",
    "mode_selection": "Select Operation Mode",
    "scenery_mode": "📸 Scenery Mode",
    "scenery_desc": "Get audio description of static images",
    "walking_mode": "🎥 Walking Mode",
    "walking_desc": "Get audio guidance from video",
    "language": "Language / भाषा",
    "upload_image": "Upload Image",
    "upload_video": "Upload Video",
    "supported_formats_image": "Supported: JPG, PNG (Max 50MB)",
    "supported_formats_video": "Supported: MP4, WebM (Max 200MB)",
    "processing": "🎬 Generate Audio Description",
    "analyzing": "🔍 Analyzing media with vision model...",
    "generating_audio": "🔊 Generating audio description...",
    "success": "✅ Audio Description Ready!",
    "error_title": "⚠️ Processing Error",
    "error_no_model": "Model files could not be downloaded. Check your internet connection and verify model URLs are correct.",
    "error_invalid_file": "Invalid file format. Please upload JPG/PNG for images or MP4/WebM for videos.",
    "error_file_size": "File exceeds size limit. Images: max 50MB, Videos: max 200MB.",
    "error_processing": "An error occurred during audio generation.",
    "playback_controls": "Audio Description Results",
    "metadata": "File Information",
    "mode_used": "Mode Used",
    "timestamp": "Generated At",
    "file_name": "Original File",
    "download_audio": "⬇️ Download Audio Description",
    "new_analysis": "Generate New Description",
    "loading_model": "Loading vision model...",
    "accessibility_note": "This platform converts your images and videos into audio descriptions. Simply upload your media and listen to the generated audio.",
    "playback_label": "Listen to Description",
}

CONTENT_HI = {
    "app_title": "AccessibleVision",
    "app_subtitle": "आपकी डिजिटल आंखें — स्वतंत्र नेविगेशन, सुनने योग्य विवरण",
    "mode_selection": "ऑपरेशन मोड चुनें",
    "scenery_mode": "📸 दृश्य मोड",
    "scenery_desc": "स्थिर छवियों का ऑडियो विवरण प्राप्त करें",
    "walking_mode": "🎥 चलने की मोड",
    "walking_desc": "वीडियो से ऑडियो मार्गदर्शन प्राप्त करें",
    "language": "Language / भाषा",
    "upload_image": "छवि अपलोड करें",
    "upload_video": "वीडियो अपलोड करें",
    "supported_formats_image": "समर्थित: JPG, PNG (अधिकतम 50MB)",
    "supported_formats_video": "समर्थित: MP4, WebM (अधिकतम 200MB)",
    "processing": "🎬 ऑडियो विवरण उत्पन्न करें",
    "analyzing": "🔍 माध्यम का विश्लेषण किया जा रहा है...",
    "generating_audio": "🔊 ऑडियो विवरण उत्पन्न किया जा रहा है...",
    "success": "✅ ऑडियो विवरण तैयार है!",
    "error_title": "⚠️ प्रसंस्करण त्रुटि",
    "error_no_model": "मॉडल फ़ाइलें डाउनलोड नहीं की जा सकीं। इंटरनेट कनेक्शन जांचें और मॉडल URLs सत्यापित करें।",
    "error_invalid_file": "अमान्य फ़ाइल प्रारूप। छवियों के लिए JPG/PNG या वीडियो के लिए MP4/WebM अपलोड करें।",
    "error_file_size": "फ़ाइल आकार सीमा से अधिक है। छवियां: अधिकतम 50MB, वीडियो: अधिकतम 200MB।",
    "error_processing": "ऑडियो उत्पन्न करने में त्रुटि हुई।",
    "playback_controls": "ऑडियो विवरण परिणाम",
    "metadata": "फ़ाइल जानकारी",
    "mode_used": "उपयोग की गई मोड",
    "timestamp": "द्वारा उत्पन्न",
    "file_name": "मूल फ़ाइल",
    "download_audio": "⬇️ ऑडियो विवरण डाउनलोड करें",
    "new_analysis": "नया विवरण उत्पन्न करें",
    "loading_model": "विजन मॉडल लोड किया जा रहा है...",
    "accessibility_note": "यह प्लेटफॉर्म आपकी छवियों और वीडियो को ऑडियो विवरण में परिवर्तित करता है। बस अपना माध्यम अपलोड करें और उत्पन्न ऑडियो सुनें।",
    "playback_label": "विवरण सुनें",
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
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0e27 !important;
        color: #f0f0f0 !important;
    }
    
    .stButton > button {
        background-color: #00d4ff !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #00a3cc !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0, 212, 255, 0.3) !important;
    }
    
    .stButton > button:focus {
        outline: 3px solid #ffff00 !important;
        outline-offset: 2px !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00ffff !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
    }
    
    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 24px;
        padding-bottom: 12px;
        border-bottom: 3px solid #00d4ff;
        color: #00ffff;
    }
    
    .success-message {
        background-color: #0d3d1e !important;
        border-left: 5px solid #00ff00 !important;
        color: #00ff00 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .error-message {
        background-color: #3d0d0d !important;
        border-left: 5px solid #ff3333 !important;
        color: #ff6666 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .info-message {
        background-color: #0d1f3d !important;
        border-left: 5px solid #00d4ff !important;
        color: #00d4ff !important;
        padding: 16px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    .metadata-box {
        background-color: #1a1f3a !important;
        border: 2px solid #00d4ff !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin: 16px 0 !important;
    }
    
    .stRadio > label {
        font-weight: 700 !important;
        font-size: 16px !important;
        color: #f0f0f0 !important;
    }
    
    .audio-player {
        background-color: #1a1f3a !important;
        border: 2px solid #00d4ff !important;
        border-radius: 8px !important;
        padding: 20px !important;
        margin: 16px 0 !important;
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
        
        # Check if model already cached
        if not os.path.exists(model_path):
            logger.info(f"Downloading {mode} model from Google Drive...")
            gdown.download(config["drive_url"], model_path, quiet=False)
        
        logger.info(f"Loading {mode} model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except Exception as e:
        logger.error(f"Model download/load error for {mode}: {str(e)}")
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

def read_image_as_array(image_path: str) -> np.ndarray:
    """Read image and convert to numpy array"""
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image, dtype=np.uint8)
    except Exception as e:
        logger.error(f"Image reading error: {str(e)}")
        raise

def read_video_file(video_path: str) -> bytes:
    """Read video file as bytes"""
    try:
        with open(video_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Video reading error: {str(e)}")
        raise

def process_scenery_image(image_path: str, model) -> bytes:
    """
    Process image through scenery model.
    Model takes image as input and RETURNS AUDIO BYTES.
    """
    try:
        # Read image as numpy array
        image_array = read_image_as_array(image_path)
        
        logger.info(f"Image shape: {image_array.shape}")
        
        # Call model - expects image array, returns AUDIO BYTES
        audio_bytes = model.predict(image_array)
        
        if not isinstance(audio_bytes, bytes):
            raise TypeError(f"Model returned {type(audio_bytes)}, expected bytes")
        
        logger.info(f"Audio generated: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Scenery processing error: {str(e)}")
        raise

def process_walking_video(video_path: str, model) -> bytes:
    """
    Process video through walking model.
    Model takes video as input and RETURNS AUDIO BYTES.
    """
    try:
        # Read video as bytes
        video_bytes = read_video_file(video_path)
        
        logger.info(f"Video size: {len(video_bytes)} bytes")
        
        # Call model - expects video bytes, returns AUDIO BYTES
        audio_bytes = model.predict(video_bytes)
        
        if not isinstance(audio_bytes, bytes):
            raise TypeError(f"Model returned {type(audio_bytes)}, expected bytes")
        
        logger.info(f"Audio generated: {len(audio_bytes)} bytes")
        return audio_bytes
        
    except Exception as e:
        logger.error(f"Walking processing error: {str(e)}")
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
    ):
        st.session_state.mode = "scenery"
        st.rerun()

with mode_col2:
    if st.button(
        f"{get_text('walking_mode')}\n\n{get_text('walking_desc')}",
        key="btn_walking",
        use_container_width=True,
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
            type=['jpg', 'jpeg', 'png']
        )
    else:
        uploaded_file = st.file_uploader(
            get_text("upload_video"),
            type=['mp4', 'webm']
        )
    
    if uploaded_file:
        # Validate file
        is_valid, validation_msg = validate_file(uploaded_file, st.session_state.mode)
        
        if not is_valid:
            st.error(validation_msg)
        else:
            # Show file info
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.info(f"📄 {uploaded_file.name} ({file_size_mb:.2f} MB)")
            
            # Save uploaded file temporarily
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process button
            if st.button(get_text("processing"), use_container_width=True, key="process_btn"):
                try:
                    progress_bar = st.progress(0)
                    
                    # Load model
                    with st.spinner(get_text("analyzing")):
                        progress_bar.progress(25)
                        model = download_model(st.session_state.mode)
                        
                        if model is None:
                            st.error(get_text("error_no_model"))
                            st.stop()
                        
                        progress_bar.progress(50)
                        
                        # Process media - model outputs AUDIO BYTES
                        with st.spinner(get_text("generating_audio")):
                            if st.session_state.mode == "scenery":
                                audio_bytes = process_scenery_image(temp_file_path, model)
                            else:
                                audio_bytes = process_walking_video(temp_file_path, model)
                        
                        progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.audio_data = audio_bytes
                    st.session_state.metadata = {
                        "mode": st.session_state.mode,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_name": uploaded_file.name,
                        "file_size": f"{file_size_mb:.2f} MB",
                        "audio_size": f"{len(audio_bytes) / 1024 / 1024:.2f} MB"
                    }
                    
                    progress_bar.progress(100)
                    st.success(get_text("success"))
                    
                except Exception as e:
                    logger.error(f"Processing failed: {str(e)}")
                    st.error(f"{get_text('error_processing')}\n\nDetails: {str(e)}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# ============================================================================
# RESULTS SECTION - AUDIO PLAYBACK
# ============================================================================

if st.session_state.audio_data and st.session_state.metadata:
    st.markdown("---")
    st.markdown(f"<h2 class='section-title'>{get_text('playback_controls')}</h2>", 
                unsafe_allow_html=True)
    
    # Audio Player
    st.markdown(f"### 🎵 {get_text('playback_label')}")
    st.audio(st.session_state.audio_data, format="audio/wav")
    
    # Download Button
    col_download = st.columns(1)[0]
    with col_download:
        st.download_button(
            label=get_text("download_audio"),
            data=st.session_state.audio_data,
            file_name=f"description_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            mime="audio/wav",
            use_container_width=True
        )
    
    # Metadata
    st.markdown("---")
    st.markdown(f"### 📋 {get_text('metadata')}")
    
    metadata_html = f"""
    <div class='metadata-box'>
        <p><strong>{get_text('mode_used')}:</strong> {get_text('scenery_mode') if st.session_state.metadata['mode'] == 'scenery' else get_text('walking_mode')}</p>
        <p><strong>{get_text('timestamp')}:</strong> {st.session_state.metadata['timestamp']}</p>
        <p><strong>{get_text('file_name')}:</strong> {st.session_state.metadata['file_name']}</p>
        <p><strong>📁 Input Size:</strong> {st.session_state.metadata['file_size']}</p>
        <p><strong>🔊 Audio Size:</strong> {st.session_state.metadata['audio_size']}</p>
    </div>
    """
    st.markdown(metadata_html, unsafe_allow_html=True)
    
    # New analysis button
    if st.button(get_text("new_analysis"), use_container_width=True):
        st.session_state.audio_data = None
        st.session_state.metadata = {}
        st.session_state.mode = None
        st.rerun()
