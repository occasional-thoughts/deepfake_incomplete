import modal
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import io
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path

# ---------------- Modal Setup ----------------
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")   
    .pip_install("torch", "librosa", "fastapi[all]", "numpy", "soundfile")
)

app = modal.App("deepfake-detector", image=image)


# ---------------- Model Definition ----------------
class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectorCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# ---------------- Feature Extractor ----------------
class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mels=128, max_len=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len
    
    def extract_mel_spectrogram(self, audio_bytes: bytes):
        # Load audio (librosa can decode MP3, WAV, FLAC, etc. via audioread/ffmpeg)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True, duration=5.0)

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or trim along time axis
        if mel_spec_db.shape[1] < self.max_len:
            mel_spec_db = np.pad(
                mel_spec_db,
                ((0, 0), (0, self.max_len - mel_spec_db.shape[1])),
                mode="constant"
            )
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]

        return mel_spec_db.astype(np.float32)



# ---------------- Modal Inference Class ----------------
@app.cls(gpu="T4", image=image, volumes={MODEL_DIR: volume})
class DeepfakeAudioAPI:
    @modal.enter()
    def load_model(self):
        print("🔄 Loading Deepfake Audio Detector model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Define architecture + load weights
        self.model = DeepfakeDetectorCNN()
        state_dict = torch.load(MODEL_DIR/"sound_deepfake_detector.pth", map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

        # Init feature extractor
        self.extractor = AudioFeatureExtractor()
        print("✅ Model loaded successfully")

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    async def predict(self, file: UploadFile):
        audio_bytes = await file.read()

        # Extract features directly from memory
        features = self.extractor.extract_mel_spectrogram(audio_bytes)
        features_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        real_prob, fake_prob = probabilities
        result = {
            "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
            "confidence": float(max(fake_prob, real_prob)),
            "probabilities": {
                "real": float(real_prob),
                "fake": float(fake_prob)
            }
        }
        return JSONResponse(result)
