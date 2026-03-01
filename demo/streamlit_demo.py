"""Streamlit demo for multilingual ASR system."""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import WhisperProcessor

from src.models.whisper_model import WhisperASRModel
from src.eval.evaluator import ASREvaluator
from src.utils.device import get_device, set_seed
from src.utils.audio import load_audio, normalize_audio, trim_silence, get_audio_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multilingual Speech Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Privacy & Ethics Disclaimer</h4>
    <p><strong>This is a research demonstration system only.</strong></p>
    <ul>
        <li>This system is designed for educational and research purposes</li>
        <li>Audio data is processed locally and not stored permanently</li>
        <li>Do not use for biometric identification or voice cloning in production</li>
        <li>Misuse of this technology for deceptive purposes is prohibited</li>
        <li>Please respect privacy and obtain consent before processing others' speech</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🎤 Multilingual Speech Recognition</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model configuration
model_name = st.sidebar.selectbox(
    "Model",
    ["openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"],
    help="Select the Whisper model variant"
)

device = st.sidebar.selectbox(
    "Device",
    ["auto", "cpu", "cuda", "mps"],
    help="Select computation device"
)

language = st.sidebar.selectbox(
    "Language",
    ["auto", "en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi", "pt"],
    help="Select language (auto for detection)"
)

enable_timestamps = st.sidebar.checkbox(
    "Enable Timestamps",
    value=False,
    help="Include word-level timestamps in output"
)

# Initialize model
@st.cache_resource
def load_model(model_name: str, device: str) -> WhisperASRModel:
    """Load and cache the ASR model."""
    try:
        set_seed(42)
        model = WhisperASRModel(
            model_name=model_name,
            device=device,
            torch_dtype="float16" if device != "cpu" else "float32",
            enable_language_detection=True,
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = load_model(model_name, device)

if model is None:
    st.error("Failed to load model. Please check your configuration.")
    st.stop()

# Model info
with st.sidebar.expander("Model Information"):
    model_info = model.get_model_info()
    st.write(f"**Model:** {model_info['model_name']}")
    st.write(f"**Device:** {model_info['device']}")
    st.write(f"**Parameters:** {model_info['parameters']:,}")
    st.write(f"**Language Detection:** {model_info['language_detection']}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎤 Live Transcription", "📁 File Upload", "📊 Evaluation", "ℹ️ About"])

# Tab 1: Live Transcription
with tab1:
    st.header("Live Audio Recording")
    
    # Audio recording
    audio_data = st.audio(
        label="Record audio",
        format="audio/wav",
        sample_rate=16000,
        help="Click to start recording, then click again to stop"
    )
    
    if audio_data is not None:
        # Process recorded audio
        with st.spinner("Processing audio..."):
            try:
                # For demo purposes, we'll simulate processing
                # In a real implementation, you'd process the recorded audio
                st.success("Audio recorded successfully!")
                st.info("In a real implementation, this would process the recorded audio and display the transcription.")
                
                # Simulate transcription result
                st.markdown("### Transcription Result")
                st.text_area(
                    "Transcribed Text",
                    value="This is a simulated transcription result. In a real implementation, the recorded audio would be processed by the ASR model.",
                    height=100
                )
                
            except Exception as e:
                st.error(f"Processing failed: {e}")

# Tab 2: File Upload
with tab2:
    st.header("Audio File Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        help="Upload an audio file for transcription"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.markdown("### File Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Size", f"{len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        with col2:
            st.metric("File Type", uploaded_file.type)
        
        with col3:
            st.metric("File Name", uploaded_file.name)
        
        # Process uploaded file
        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Processing audio file..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Transcribe audio
                    language_code = None if language == "auto" else language
                    
                    result = model.transcribe(
                        tmp_file_path,
                        language=language_code,
                        return_timestamps=enable_timestamps
                    )
                    
                    # Display results
                    st.markdown("### Transcription Results")
                    
                    if isinstance(result, dict):
                        transcription = result['text']
                        detected_language = result.get('language', 'unknown')
                        timestamps = result.get('timestamps', [])
                    else:
                        transcription = result
                        detected_language = language_code or 'auto-detected'
                        timestamps = []
                    
                    # Main transcription
                    st.text_area(
                        "Transcribed Text",
                        value=transcription,
                        height=150
                    )
                    
                    # Language detection
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Detected Language", detected_language)
                    
                    with col2:
                        st.metric("Audio Duration", f"{len(transcription.split()) * 0.5:.1f}s (estimated)")
                    
                    # Timestamps if enabled
                    if enable_timestamps and timestamps:
                        st.markdown("### Word-Level Timestamps")
                        timestamps_df = pd.DataFrame(timestamps)
                        st.dataframe(timestamps_df, use_container_width=True)
                    
                    # Clean up temporary file
                    Path(tmp_file_path).unlink()
                    
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    logger.error(f"Transcription error: {e}")

# Tab 3: Evaluation
with tab3:
    st.header("Model Evaluation")
    
    st.markdown("""
    This section demonstrates the evaluation capabilities of the ASR system.
    You can evaluate the model on sample data or upload your own test files.
    """)
    
    # Evaluation options
    eval_option = st.radio(
        "Evaluation Mode",
        ["Sample Dataset", "Custom Files"],
        help="Choose evaluation mode"
    )
    
    if eval_option == "Sample Dataset":
        st.markdown("### Sample Dataset Evaluation")
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation on sample dataset..."):
                try:
                    # Create sample dataset
                    from src.data.multilingual_dataset import MultilingualASRDataset
                    
                    dataset = MultilingualASRDataset(
                        data_dir="data",
                        split="test",
                        max_duration=10.0,
                        min_duration=1.0,
                    )
                    
                    # Create evaluator
                    evaluator = ASREvaluator(model, device=device)
                    
                    # Run evaluation
                    results = evaluator.evaluate_dataset(
                        dataset,
                        batch_size=1,
                        return_predictions=True
                    )
                    
                    # Display results
                    st.markdown("### Evaluation Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("WER", f"{results['wer']:.3f}")
                    
                    with col2:
                        st.metric("CER", f"{results['cer']:.3f}")
                    
                    with col3:
                        st.metric("Token Accuracy", f"{results['token_accuracy']:.3f}")
                    
                    with col4:
                        st.metric("Mean Latency", f"{results['latency']['mean_latency']:.3f}s")
                    
                    # RTF
                    st.metric("Mean RTF", f"{results['rtf']['mean_rtf']:.3f}")
                    
                    # Detailed results
                    with st.expander("Detailed Results"):
                        st.json(results)
                    
                    # Create leaderboard
                    leaderboard = evaluator.create_leaderboard(results)
                    st.markdown("### Performance Leaderboard")
                    st.dataframe(leaderboard, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    logger.error(f"Evaluation error: {e}")
    
    else:  # Custom Files
        st.markdown("### Custom File Evaluation")
        
        # Upload multiple files
        uploaded_files = st.file_uploader(
            "Upload test files",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            accept_multiple_files=True,
            help="Upload multiple audio files for evaluation"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            if st.button("Evaluate Files", type="primary"):
                with st.spinner("Evaluating files..."):
                    try:
                        evaluator = ASREvaluator(model, device=device)
                        
                        results = []
                        for uploaded_file in uploaded_files:
                            # Save file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_file_path = tmp_file.name
                            
                            # Evaluate file
                            result = evaluator.evaluate_file(
                                tmp_file_path,
                                language=language if language != "auto" else None
                            )
                            
                            result['filename'] = uploaded_file.name
                            results.append(result)
                            
                            # Clean up
                            Path(tmp_file_path).unlink()
                        
                        # Display results
                        st.markdown("### Evaluation Results")
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary metrics
                        if len(results) > 1:
                            avg_latency = np.mean([r['latency'] for r in results])
                            st.metric("Average Latency", f"{avg_latency:.3f}s")
                        
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                        logger.error(f"Evaluation error: {e}")

# Tab 4: About
with tab4:
    st.header("About This System")
    
    st.markdown("""
    ## Multilingual Speech Recognition System
    
    This is a research demonstration of a multilingual speech recognition system
    built using OpenAI's Whisper model. The system supports multiple languages
    and provides various evaluation metrics.
    
    ### Features
    
    - **Multilingual Support**: Automatic language detection and transcription
    - **Multiple Models**: Support for different Whisper model variants
    - **Device Flexibility**: CPU, CUDA, and MPS (Apple Silicon) support
    - **Comprehensive Evaluation**: WER, CER, token accuracy, and latency metrics
    - **Privacy-Focused**: Local processing with privacy safeguards
    
    ### Supported Languages
    
    The system supports over 100 languages including:
    - English (en)
    - Spanish (es)
    - French (fr)
    - German (de)
    - Chinese (zh)
    - Japanese (ja)
    - Korean (ko)
    - Arabic (ar)
    - Hindi (hi)
    - Portuguese (pt)
    - And many more...
    
    ### Technical Details
    
    - **Model**: OpenAI Whisper (base/small/medium variants)
    - **Framework**: PyTorch with Transformers
    - **Audio Processing**: 16kHz sample rate, mono channel
    - **Evaluation Metrics**: WER, CER, token accuracy, latency, RTF
    
    ### Usage Guidelines
    
    1. **Research Only**: This system is designed for research and educational purposes
    2. **Privacy**: Audio data is processed locally and not stored
    3. **Consent**: Obtain proper consent before processing others' speech
    4. **Ethics**: Do not use for deceptive or harmful purposes
    
    ### Limitations
    
    - Performance may vary depending on audio quality and language
    - Real-time processing depends on hardware capabilities
    - Some languages may have lower accuracy than others
    - Background noise can affect transcription quality
    
    ### Contact
    
    For questions or issues, please refer to the project documentation
    or contact the research team.
    """)
    
    # System information
    st.markdown("### System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("PyTorch Version", torch.__version__)
        st.metric("Device", str(get_device(device)))
    
    with col2:
        st.metric("Model Parameters", f"{model_info['parameters']:,}")
        st.metric("Model Size", f"{model_info['parameters'] * 4 / 1024 / 1024:.1f} MB")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Multilingual Speech Recognition System - Research Demo</p>
        <p><strong>Privacy Notice:</strong> This system processes audio locally and does not store or transmit your data.</p>
    </div>
    """,
    unsafe_allow_html=True
)
