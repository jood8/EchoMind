# 🎵 EchoMind: AI-Powered Music Analysis Platform

**EchoMind** is a Deep Learning web application for audio analysis. It can classify music genres and identify musical instruments from audio files using Audio Spectrogram Transformer (AST) models. The platform also provides waveform visualization, tempo estimation, frequency analysis, and model evaluation through an interactive Streamlit interface.

🚀 **Live Demo:** https://echomind-smgxz62w9aud37jd9knlpw.streamlit.app/

📺 **Demo Video:** https://drive.google.com/file/d/1HzPng8Xr-vEvgqgjxLt6gijnMkBwn-Zq/view?usp=sharing

---

## 🚀 Model Performance

The project compares traditional CNN models with AST-based Transfer Learning models.

| Task                      | Dataset | CNN Baseline | AST Model | Improvement |
| :------------------------ | :-----: | :----------: | :-------: | :---------: |
| Genre Classification      |  GTZAN  |    66.12%    |   87.50%  |   +21.38%   |
| Instrument Classification |  IRMAS  |    57.02%    |   89.70%  |   +32.68%   |

---

## ✨ Features

### 🎼 Genre Classification

* Predicts the music genre from an audio file.
* Supports genres such as Classical, Jazz, HipHop, Metal, Pop, Rock, and more.
* Uses a fine-tuned AST model trained on the GTZAN dataset.

### 🎻 Instrument Classification

* Detects the dominant musical instrument in the audio.
* Can identify instruments such as Piano, Guitar, Violin, Flute, Saxophone, Trumpet, and Human Voice.
* Uses a fine-tuned AST model trained on the IRMAS dataset.

### 📊 Audio Analysis

* Interactive audio waveform visualization.
* Tempo estimation (BPM).
* Frequency analysis and spectral centroid calculation.
* Instrument frequency range comparison.

### 📈 Model Evaluation

* Confusion matrices for both models.
* Training and validation loss curves.
* Performance comparison between CNN and AST models.

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Audio Spectrogram Transformer (AST)
* Librosa
* NumPy
* Pandas
* Scikit-Learn
* Streamlit
* Plotly
* Matplotlib

---

## 📁 Project Structure

```text
├── genre_model/
├── irmas_model/
├── label_encoder_genre.pkl
├── label_encoder_IRMAS.pkl
├── app.py
├── gtzan.py
├── irmas.py
├── confusion_matrix_gtzan.png
├── confusion_matrix_irmas.png
├── loss_gtzan.png
├── loss_irmas.png
└── requirements.txt
```

---

## 🎯 Project Goal

The goal of EchoMind is to demonstrate how Transfer Learning with Audio Spectrogram Transformers can significantly improve audio classification performance compared to traditional CNN-based approaches while providing an easy-to-use interface for music analysis.
