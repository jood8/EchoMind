import streamlit as st
import librosa
import numpy as np
import plotly.graph_objects as go
import pickle
import torch
import os
import tempfile
from transformers import AutoModelForAudioClassifications, AutoFeatureExtractor
import matplotlib.pyplot as plt

# ==========================================
# Page Settings & Title
# ==========================================
st.set_page_config(page_title="EchoMind", layout="wide")
st.title("EchoMind")

# ==========================================
# Dictionary for Instrument Images
# ==========================================
INSTRUMENT_IMAGES = {
    "piano": "https://s.mc-doualiya.com/media/display/81517b06-fa57-11e9-b415-005056a964fe/w:1280/p:4x3/piano_musique.jpg",
    "pia": "https://s.mc-doualiya.com/media/display/81517b06-fa57-11e9-b415-005056a964fe/w:1280/p:4x3/piano_musique.jpg",
    
    "guitar": "https://guitarworks.ca/cdn/shop/products/qxdgeaeabfijj8ienidh_1600x.jpg?v=1569011154",
    "gac": "https://images.unsplash.com/photo-1510915361894-db8b60106cb1?q=80&w=400&auto=format&fit=crop", 
    "gel": "https://media.guitarcenter.com/is/image/MMGS7/L90659000002000-00-2000x2000.jpg", 
    
    "violin": "https://lazermusic.com.ph/cdn/shop/files/1_490f37a2-d8ad-4b5b-aa35-ac3b2286ab81.png?v=1767316420",
    "vio": "https://lazermusic.com.ph/cdn/shop/files/1_490f37a2-d8ad-4b5b-aa35-ac3b2286ab81.png?v=1767316420",
    
    "drums": "https://royal.az/cdn/shop/files/vad516_angle_gal.jpg?v=1770994149&width=600",
    
    "saxophone": "https://www.axiommusic.com.au/assets/full/SP1001G.jpg?20250707212350",
    "sax": "https://www.axiommusic.com.au/assets/full/SP1001G.jpg?20250707212350",
    
    "flute": "https://storage.googleapis.com/stateless-blog-g4m-co-uk/2023/10/Featured-image-The-Different-Types-of-Flutes-from-Around-the-World.jpg",
    "flu": "https://storage.googleapis.com/stateless-blog-g4m-co-uk/2023/10/Featured-image-The-Different-Types-of-Flutes-from-Around-the-World.jpg",
    
    "cello": "https://www.smithsonianchambermusic.org/sites/default/files/styles/large/public/2024-03/ken2_servais.jpg?itok=TzgwyDh4",
    "cel": "https://www.smithsonianchambermusic.org/sites/default/files/styles/large/public/2024-03/ken2_servais.jpg?itok=TzgwyDh4",
    
    "trumpet": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQan_IzP0D71Eg7hahQUrNm2Da5UCGoYsULa-b9HimGObNknF4eOFHKfo0&s=10",
    "tru": "https://uk.yamaha.com/en/files/ycl-631ii_540x540_tcm117-1737467.jpg?impolicy=large&imwid=735&imhei=735",
    
    "clarinet": "https://assets-sb.thomann.de/f/49568/2000x2000/d0bf5c6f04/tps_1335_100448_oktuber125817-100448_crop.jpg/m/1600x0/filters:quality(90)",
    "cla": "https://www.axiommusic.com.au/assets/full/AXCLEFLAT.jpg?20250707212350",
    
    "organ": "https://upload.wikimedia.org/wikipedia/commons/8/87/OrgueSaintThomasStrasbourg_a.jpg",
    "org": "https://upload.wikimedia.org/wikipedia/commons/8/87/OrgueSaintThomasStrasbourg_a.jpg",
    
    "voice": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=400&auto=format&fit=crop",
    "voi": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=400&auto=format&fit=crop",
    
    "default": "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=400&auto=format&fit=crop"
}

# ==========================================
# Dictionary for Instrument Frequency Ranges
# ==========================================
INSTRUMENT_RANGES = {
    "piano": {"min": 27, "max": 4186, "desc": "Very wide range covering deep bass to bright crystalline treble frequencies."},
    "pia": {"min": 27, "max": 4186, "desc": "Very wide range covering deep bass to bright crystalline treble frequencies."},
    "gac": {"min": 82, "max": 1000, "desc": "Warm mid-range frequencies with prominent acoustic string resonances."},
    "guitar": {"min": 82, "max": 1000, "desc": "Warm mid-range frequencies with prominent acoustic string resonances."},
    "gel": {"min": 82, "max": 1200, "desc": "Mid-range focused, highly dependent on electronic amplification and effects."},
    "violin": {"min": 196, "max": 3520, "desc": "High-pitched soprano register with piercing and expressive frequencies."},
    "vio": {"min": 196, "max": 3520, "desc": "High-pitched soprano register with piercing and expressive frequencies."},
    "cel": {"min": 65, "max": 1000, "desc": "Deep, sonorous bass and baritone frequencies with a rich melodic tone."},
    "cello": {"min": 65, "max": 1000, "desc": "Deep, sonorous bass and baritone frequencies with a rich melodic tone."},
    "sax": {"min": 138, "max": 880, "desc": "Rich mid-range woodwind frequency spectrum with dense harmonics."},
    "saxophone": {"min": 138, "max": 880, "desc": "Rich mid-range woodwind frequency spectrum with dense harmonics."},
    "flu": {"min": 261, "max": 2349, "desc": "High, pure wind frequencies resembling whistles or birdsongs."},
    "flute": {"min": 261, "max": 2349, "desc": "High, pure wind frequencies resembling whistles or birdsongs."},
    "tru": {"min": 165, "max": 987, "desc": "Powerful, piercing brass frequencies with loud projections."},
    "trumpet": {"min": 165, "max": 987, "desc": "Powerful, piercing brass frequencies with loud projections."},
    "cla": {"min": 147, "max": 1568, "desc": "Extremely versatile range fluidly jumping from deep low tones to high altissimo."},
    "clarinet": {"min": 147, "max": 1568, "desc": "Extremely versatile range fluidly jumping from deep low tones to high altissimo."},
    "org": {"min": 16, "max": 8372, "desc": "The widest frequency range possible, extending below and far above human voice."},
    "organ": {"min": 16, "max": 8372, "desc": "The widest frequency range possible, extending below and far above human voice."},
    "voi": {"min": 85, "max": 1100, "desc": "The natural frequency of human speech and singing vocals."},
    "voice": {"min": 85, "max": 1100, "desc": "The natural frequency of human speech and singing vocals."},
    "default": {"min": 20, "max": 20000, "desc": "The standard full spectrum of human hearing capabilities."}
}

# ==========================================
# Load Models (Cached for speed)
# ==========================================
@st.cache_resource
def load_models():
    genre_model = AutoModelForAudioClassification.from_pretrained("genre_model")
    genre_feature_extractor = AutoFeatureExtractor.from_pretrained("genre_model")
    with open("label_encoder_genre.pkl", "rb") as f:
        genre_encoder = pickle.load(f)

    instrument_model = AutoModelForAudioClassification.from_pretrained("irmas_model")
    instrument_feature_extractor = AutoFeatureExtractor.from_pretrained("irmas_model")
    with open("label_encoder_IRMAS.pkl", "rb") as f:
        instrument_encoder = pickle.load(f)
        
    return (genre_model, genre_feature_extractor, genre_encoder, 
            instrument_model, instrument_feature_extractor, instrument_encoder)

(genre_model, genre_feature_extractor, genre_encoder, 
 instrument_model, instrument_feature_extractor, instrument_encoder) = load_models()

# ==========================================
# Sidebar 
# ==========================================
with st.sidebar:
    st.header("Framework Benchmarks")
    st.markdown("### Genre Classification")
    st.caption("Baseline CNN (From Scratch): **66.12%**")
    st.markdown(" **AST (Transfer Learning): 87.5%**")
    st.success(" Improvement: +21.38%")
    
    st.markdown("---")
    
    st.markdown("###  Instrument Classification")
    st.caption("Baseline CNN (From Scratch): **57.02%**")
    st.markdown("**AST (Transfer Learning): 89.70%**")
    st.success(" Improvement: +32.68%")

# ==========================================
# Main Application Tabs Structure
# ==========================================
main_tab1, main_tab2 = st.tabs(["🎵 Audio Analysis", "📊 Model Evaluation"])

# ==========================================
# TAB 1: Audio Analysis Process
# ==========================================
with main_tab1:
    # Prediction Helper Functions
    def predict_genre(audio_path):
        audio, sr = librosa.load(audio_path, sr=16000, duration=5)
        inputs = genre_feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = genre_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        genre_name = genre_encoder.inverse_transform([pred_idx])[0]
        return genre_name, confidence

    def predict_instrument(audio_path):
        audio, sr = librosa.load(audio_path, sr=16000, duration=3)
        inputs = instrument_feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            outputs = instrument_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
        instrument_name = instrument_encoder.inverse_transform([pred_idx])[0]
        return instrument_name, confidence

    # File Uploader
    active_file = st.file_uploader("Upload an audio file to analyze", type=["wav", "mp3"])

    if active_file is not None:
        st.audio(active_file)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(active_file.getbuffer())
            temp_filename = temp_file.name
            
        try:
            y, sr = librosa.load(temp_filename, sr=16000)
            # Waveform Plot
            step = max(1, len(y) // 5000) 
            y_sampled = y[::step]
            times = (np.arange(len(y)) / sr)[::step]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(times, y_sampled, color="#2dd4bf")
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # Action Button
            if st.button("Analyze Audio", type="primary"):
                with st.spinner("Analyzing audio.... Please wait..."):
                    
                    genre_name, genre_conf = predict_genre(temp_filename)
                    instrument_name, instrument_conf = predict_instrument(temp_filename)
                    
                    # تعديل مهم: تحويل الـ tempo إلى قيمة مفردة آمنة باستخدام .item()
                    tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
                    tempo = float(tempo_raw.item()) if hasattr(tempo_raw, "item") else float(tempo_raw)
                    
                    clean_inst_name = str(instrument_name).lower().strip()
                    instrument_img_url = INSTRUMENT_IMAGES.get(clean_inst_name, INSTRUMENT_IMAGES["default"])
                    
                    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    mean_cent = float(np.mean(cent))
                    inst_range = INSTRUMENT_RANGES.get(clean_inst_name, INSTRUMENT_RANGES["default"])

                    st.write("---")
                    st.subheader("Analysis Results")
                    
                    dash_col1, dash_col2, dash_col3, dash_col4 = st.columns([1, 1, 1, 1.2])
                    
                    with dash_col1:
                        st.metric("Music Genre", genre_name.upper(), f"{genre_conf*100:.2f}% Confidence")
                        
                    with dash_col2:
                        st.metric("Instrument", instrument_name.upper(), f"{instrument_conf*100:.2f}% Confidence")
                    with dash_col3:
                        st.metric("Estimated Tempo", f"{int(tempo)} BPM")
                    with dash_col4:
                        st.image(instrument_img_url, caption=f"Identified Class Architecture: {instrument_name.upper()}", use_container_width=True)
                    
                    # تعديل المسافات (Indentation): إدخال الـ Tabs لتظهر فقط بعد الضغط على الزر بنجاح
                    info_tab, freq_tab = st.tabs(["Audio Information", "Frequency Analysis"])
                    with info_tab:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Sampling Rate", f"{sr} Hz")
                        col2.metric("Duration", f"{len(y)/sr:.2f} sec")
                        col3.metric("Max Amplitude", f"{np.max(np.abs(y)):.4f}")
                    with freq_tab:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("Average Frequency", f"{mean_cent:.0f} Hz")
                            st.metric("Frequency Range", f"{inst_range['min']} - {inst_range['max']} Hz")
                            st.info(inst_range["desc"])
                        with col2:
                            stft_data = np.abs(librosa.stft(y))
                            stft_db = librosa.amplitude_to_db(stft_data, ref=np.max)
                            fig_spec, ax_spec = plt.subplots(figsize=(10, 4.5))
                            img = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='linear', ax=ax_spec, cmap='magma')                        
                            ax_spec.set_ylim(0, 8000)
                            ax_spec.set_title("Spectrogram")
                            ax_spec.set_xlabel("Time (s)")
                            ax_spec.set_ylabel("Hz")                        
                            st.pyplot(fig_spec)
                                
        except Exception as e:
            st.error(f"Execution Exception encountered during audio parsing: {e}")
            
        finally:
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                try:
                    import gc
                    gc.collect() 
                    os.remove(temp_filename)
                except Exception:
                    pass

# ==========================================
# TAB 2: Model Scientific Diagnostics
# ==========================================
with main_tab2:    
    sub_tab1, sub_tab2 = st.tabs(["Confusion Matrices (Errors Validation)", "Learning Curves (Loss Records)"])
    
    with sub_tab1:
        st.markdown("### Confusion Matrices")
        col1, col2 = st.columns(2)
        with col1:
            st.image("confusion_matrix_gtzan.png", caption="Genre Classification Matrix", use_container_width=True)
        with col2:
            st.image("confusion_matrix_irmas.png", caption="Instrument Classification Matrix", use_container_width=True)
            
    with sub_tab2:
        st.markdown("### Convergence Curves")
        col3, col4 = st.columns(2)
        with col3:
            st.image("loss_gtzan.png", caption="Genre Learning Convergence Curve", use_container_width=True)
        with col4:
            st.image("loss_irmas.png", caption="Instrument Learning Curve", use_container_width=True)
