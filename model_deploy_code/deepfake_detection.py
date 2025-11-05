import modal
import torch
import torch.nn as nn
import numpy as np
import io
import shutil
from pathlib import Path
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
import cv2
import tempfile
from moviepy import VideoFileClip,AudioFileClip
import os
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1
from efficientnet_pytorch import EfficientNet
import subprocess



class SpatialTemporalDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, max_frames=32, feature_dim=1408):
        super().__init__()

        # Spatial feature extractor
        self.spatial_encoder = EfficientNet.from_pretrained('efficientnet-b2')
        self.spatial_encoder._fc = nn.Identity()

        # Spatial feature projection
        self.spatial_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Temporal modeling with LSTM + Transformer
        self.temporal_lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=512,  # bidirectional LSTM output
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Positional encoding for temporal sequences
        self.pos_encoding = nn.Parameter(torch.randn(max_frames, 512))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, valid_frames=None):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Extract spatial features for each frame
        x_reshaped = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder.extract_features(x_reshaped)  # (B*T, 1408, h, w)

        # Project spatial features
        spatial_features = self.spatial_projection(spatial_features)  # (B*T, 512)
        spatial_features = spatial_features.view(B, T, -1)  # (B, T, 512)

        # Add positional encoding
        spatial_features = spatial_features + self.pos_encoding[:T].unsqueeze(0)

        # Temporal modeling with LSTM
        lstm_out, _ = self.temporal_lstm(spatial_features)  # (B, T, 512)

        # Create attention mask for padded frames (optional)
        if valid_frames is not None:
            # Create mask for attention
            mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            for i, vf in enumerate(valid_frames):
                mask[i, :vf] = True
            attn_mask = ~mask
        else:
            attn_mask = None

        # Temporal attention
        attended_features, attention_weights = self.temporal_attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
        )  # (B, T, 512)

        # Global temporal pooling (weighted by valid frames)
        if valid_frames is not None:
            # Weighted average based on valid frames
            temporal_representation = []
            for i, vf in enumerate(valid_frames):
                temporal_representation.append(
                    torch.mean(attended_features[i, :vf], dim=0)
                )
            temporal_representation = torch.stack(temporal_representation)
        else:
            temporal_representation = torch.mean(attended_features, dim=1)  # (B, 512)

        # Classification
        output = self.classifier(temporal_representation)

        return output

class FaceExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.detector = MTCNN(keep_all=True, device=device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Parameters
        self.max_cosine_dist = 0.5
        self.max_age = 30  # max frame age to retain a track
        self.sequence_length = 32  # Match model's max_frames
        self.max_gap_sec = 0.5
        self.sample_interval_sec = 0.1

    def extract_faces_from_video(self, video_path, output_dir=None):
        """Extract face sequences from a video"""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='face_extraction_')

        os.makedirs(output_dir, exist_ok=True)
        print(f"🎬 Processing video: {video_path}")
        print(f"📁 Output directory: {output_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Failed to open {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25.0
        frame_interval = int(round(fps * self.sample_interval_sec))

        # Reset state
        tracks = {}
        track_frames = {}
        next_track_id = 1
        frame_num = 0
        processed_frame_count = 0

        print(f"🔍 Extracting faces at {1/self.sample_interval_sec:.1f} FPS...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % frame_interval != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # seconds

            # Detect faces
            boxes, probs = self.detector.detect(rgb)
            if boxes is None:
                continue

            new_faces = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.99:
                    continue
                x1, y1, x2, y2 = map(int, box)
                margin = 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(frame.shape[1], x2 + margin)
                y2 = min(frame.shape[0], y2 + margin)

                face_crop = rgb[y1:y2, x1:x2]
                if face_crop.size == 0 or face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
                    continue

                resized = cv2.resize(face_crop, (160, 160))
                face_tensor = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_tensor = face_tensor.to(self.device)

                with torch.no_grad():
                    embedding = self.embedder(face_tensor).cpu().numpy()[0]

                new_faces.append((embedding, face_crop))

            # Track assignment
            assigned = []
            for emb, crop in new_faces:
                best_match = None
                best_dist = float('inf')
                for track_id, data in tracks.items():
                    dist = cosine(emb, data['embedding'])
                    if dist < best_dist and dist < self.max_cosine_dist:
                        best_dist = dist
                        best_match = track_id

                if best_match is not None:
                    tracks[best_match]['embedding'] = emb
                    tracks[best_match]['last_frame'] = frame_num
                    assigned.append((best_match, crop))
                else:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = {'embedding': emb, 'last_frame': frame_num}
                    track_frames[track_id] = []
                    assigned.append((track_id, crop))

            # Save frames
            for track_id, crop in assigned:
                save_path = os.path.join(output_dir, f'person_{track_id:03d}')
                os.makedirs(save_path, exist_ok=True)
                processed_frame_count += 1
                filename = f'frame_{processed_frame_count:04d}_{frame_time:.2f}s.jpg'
                filepath = os.path.join(save_path, filename)
                cv2.imwrite(filepath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                track_frames[track_id].append((frame_time, filepath))

            # Clean old tracks
            tracks = {k: v for k, v in tracks.items() if frame_num - v['last_frame'] <= self.max_age}

        cap.release()

        # Post-process sequences
        self._post_process_sequences(track_frames)

        print(f"✅ Face extraction complete!")
        print(f"📊 Found {len(track_frames)} face tracks")
        for track_id, frames in track_frames.items():
            print(f"   Person {track_id:03d}: {len(frames)} frames")

        return output_dir, track_frames

    def _post_process_sequences(self, track_frames):
        """Pad sequences for RNN/LSTM processing"""
        print("🔧 Post-processing sequences...")

        for track_id, frames in track_frames.items():
            if not frames:
                continue

            # Sort original frames by timestamp
            frames.sort(key=lambda x: x[0])
            save_dir = os.path.dirname(frames[0][1])
            valid_frames = [frames[0]]  # Start with first frame

            # Fill time gaps with duplicates of previous frame
            for i in range(1, len(frames)):
                prev_time, prev_path = valid_frames[-1]
                curr_time, curr_path = frames[i]
                time_diff = curr_time - prev_time

                if time_diff <= self.max_gap_sec:
                    # Compute how many frames are missing based on sample interval
                    n_missing = int(round(time_diff / self.sample_interval_sec)) - 1
                    for m in range(n_missing):
                        pad_time = prev_time + (m + 1) * self.sample_interval_sec
                        pad_path = os.path.join(save_dir, f'frame_fill_tmp_{pad_time:.2f}s.jpg')
                        shutil.copy(prev_path, pad_path)
                        valid_frames.append((pad_time, pad_path))

                    valid_frames.append((curr_time, curr_path))
                else:
                    break  # Stop this track if time jump too large

            if valid_frames:  # Only proceed if there are frames
                last_time, last_path = valid_frames[-1]
                if len(valid_frames) < self.sequence_length:
                    for i in range(len(valid_frames), self.sequence_length):
                        pad_time = last_time + (i - len(valid_frames) + 1) * self.sample_interval_sec
                        pad_path = os.path.join(save_dir, f'frame_pad_tmp_{pad_time:.2f}s.jpg')
                        try:
                            shutil.copy2(last_path, pad_path)
                            valid_frames.append((pad_time, pad_path))
                        except Exception as e:
                            print(f"⚠️ Failed to copy {last_path} to {pad_path}: {e}")
            else:
                # Skip padding if no frames in this track
                print(f"⚠️ Track {track_id} has no valid frames, skipping padding.")


            # Final sort and renaming to guarantee temporal order
            valid_frames.sort(key=lambda x: x[0])
            for i, (t, old_path) in enumerate(valid_frames):
                new_filename = f'frame_{i+1:04d}_{t:.2f}s.jpg'
                new_path = os.path.join(save_dir, new_filename)
                if os.path.exists(old_path) and old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                    except Exception as e:
                        print(f"⚠️ Failed to rename {old_path} to {new_path}: {e}")



class DeepfakeInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SpatialTemporalDeepfakeDetector().to(device)

        # Load trained model
        print(f"📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"✅ Model loaded successfully on {device}")

    def predict_sequence_batches(self, sequence_dir, max_frames=32, min_frames=16):
        """Predict on a face sequence directory using batches like training"""
        try:
            # Get all frame files
            frame_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                frame_files.extend(glob.glob(os.path.join(sequence_dir, ext)))

            frame_files = sorted(frame_files)

            if len(frame_files) == 0:
                print(f"⚠️ No frames found in {sequence_dir}")
                return []

            print(f"📸 Found {len(frame_files)} frames in {sequence_dir}")

            # Create batches like in training
            batches_results = []
            total_frames = len(frame_files)

            if total_frames < min_frames:
                print(f"⚠️ Skipping {sequence_dir}: only {total_frames} frames (< {min_frames})")
                return []

            # Create non-overlapping batches (same as training)
            stride = max_frames  # Non-overlapping batches
            start_idx = 0
            batch_num = 1

            while start_idx < total_frames:
                end_idx = min(start_idx + max_frames, total_frames)
                batch_frame_files = frame_files[start_idx:end_idx]

                # Only process batches that meet minimum frame requirement
                if len(batch_frame_files) >= min_frames:
                    print(f"  🔄 Processing batch {batch_num}: frames {start_idx+1} to {end_idx} ({len(batch_frame_files)} frames)")

                    # Load and preprocess frames for this batch
                    images = []
                    successful_loads = 0

                    for frame_path in batch_frame_files:
                        try:
                            img = Image.open(frame_path).convert('RGB')
                            img = self.transform(img)
                            images.append(img)
                            successful_loads += 1
                        except Exception as e:
                            print(f"⚠️ Error loading {frame_path}: {e}")
                            # Use black image as fallback
                            fallback_img = Image.new('RGB', (224, 224), (0, 0, 0))
                            img = self.transform(fallback_img)
                            images.append(img)

                    if len(images) == 0:
                        print(f"⚠️ No valid frames loaded for batch {batch_num}")
                        start_idx += stride
                        continue

                    # Pad sequence to max_frames if needed (same as training)
                    while len(images) < max_frames:
                        images.append(images[-1].clone())  # Duplicate last frame

                    # Convert to tensor and add batch dimension
                    sequence = torch.stack(images).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
                    valid_frames = torch.tensor([successful_loads]).to(self.device)

                    # Predict
                    with torch.no_grad():
                        outputs = self.model(sequence, valid_frames)
                        probabilities = torch.softmax(outputs, dim=1)
                        prediction = torch.argmax(outputs, dim=1)
                        confidence = torch.max(probabilities, dim=1)[0]

                    batch_result = {
                        'batch_number': batch_num,
                        'frame_range': f"{start_idx+1}-{end_idx}",
                        'prediction': 'FAKE' if prediction.item() == 1 else 'REAL',
                        'confidence': confidence.item(),
                        'fake_probability': probabilities[0, 1].item(),
                        'real_probability': probabilities[0, 0].item(),
                        'frames_in_batch': len(batch_frame_files),
                        'valid_frames': successful_loads,
                        'start_frame': start_idx + 1,
                        'end_frame': end_idx
                    }

                    batches_results.append(batch_result)

                    # Print batch result immediately
                    print(f"    📊 Batch {batch_num} Result: {batch_result['prediction']} "
                          f"(confidence: {batch_result['confidence']:.3f}, "
                          f"fake_prob: {batch_result['fake_probability']:.3f})")

                    batch_num += 1

                start_idx += stride

                # If remaining frames are less than min_frames, break
                if total_frames - start_idx < min_frames:
                    break

            return batches_results

        except Exception as e:
            print(f"❌ Error in predict_sequence_batches: {str(e)}")
            import traceback
            traceback.print_exc()
            return []


class DeepfakeDetectionPipeline:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.face_extractor = FaceExtractor(device)
        self.inference = DeepfakeInference(model_path, device)

    def process_video(self, video_path, cleanup_temp=True):
        """Complete pipeline: extract faces and run inference"""
        print(f"🚀 Starting deepfake detection pipeline...")
        print(f"📹 Video: {video_path}")

        # Extract faces
        temp_dir, track_frames = self.face_extractor.extract_faces_from_video(video_path)

        if not track_frames:
            print("❌ No faces found in video")
            return None

        # Run inference on each face track
        results = {}
        print(f"\n🔮 Running batch inference on {len(track_frames)} face tracks...")

        for track_id, frames in track_frames.items():
            person_dir = os.path.join(temp_dir, f'person_{track_id:03d}')

            if os.path.exists(person_dir):
                print(f"\n👤 Processing Person {track_id:03d}:")
                batch_results = self.inference.predict_sequence_batches(person_dir)

                if batch_results:
                    results[f'person_{track_id:03d}'] = {
                        'batches': batch_results,
                        'total_batches': len(batch_results)
                    }

                    # Print summary for this person
                    fake_count = sum(1 for batch in batch_results if batch['prediction'] == 'FAKE')
                    real_count = sum(1 for batch in batch_results if batch['prediction'] == 'REAL')
                    avg_confidence = sum(batch['confidence'] for batch in batch_results) / len(batch_results)
                    avg_fake_prob = sum(batch['fake_probability'] for batch in batch_results) / len(batch_results)

                    print(f"  📈 Person {track_id:03d} Summary:")
                    print(f"     Total batches: {len(batch_results)}")
                    print(f"     FAKE batches: {fake_count}")
                    print(f"     REAL batches: {real_count}")
                    print(f"     Average confidence: {avg_confidence:.3f}")
                    print(f"     Average fake probability: {avg_fake_prob:.3f}")

                    # Determine overall prediction for this person
                    person_prediction = "FAKE" if fake_count > real_count else "REAL"
                    print(f"     🎯 Person prediction: {person_prediction}")

                else:
                    print(f"   ❌ Person {track_id:03d}: No valid batches processed")

        # Cleanup temporary directory
        if cleanup_temp:
            try:
                shutil.rmtree(temp_dir)
                print(f"\n🧹 Cleaned up temporary directory")
            except Exception as e:
                print(f"⚠️ Could not cleanup temp directory: {e}")
        else:
            print(f"\n📁 Temporary files kept in: {temp_dir}")

        return results

    def get_overall_prediction(self, results):
        """Get overall video prediction based on all face tracks and their batches"""
        if not results:
            return "UNKNOWN", 0.0

        total_fake_batches = 0
        total_real_batches = 0
        total_confidence = 0
        total_batches = 0

        for person, person_data in results.items():
            if 'batches' in person_data:
                for batch in person_data['batches']:
                    if batch['prediction'] == 'FAKE':
                        total_fake_batches += 1
                    else:
                        total_real_batches += 1

                    total_confidence += batch['confidence']
                    total_batches += 1

        if total_batches == 0:
            return "UNKNOWN", 0.0

        avg_confidence = total_confidence / total_batches

        # If majority of batches are fake, classify as fake
        if total_fake_batches > total_real_batches:
            return "FAKE", avg_confidence
        else:
            return "REAL", avg_confidence

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libsm6", "libxext6")
    .pip_install("torch", "torchvision", "facenet-pytorch", "efficientnet_pytorch",
                 "librosa", "fastapi[all]", "numpy", "moviepy", "soundfile", "scipy", "Pillow","opencv-python-headless")
)

app = modal.App("og-deepfake-detector", image=image)

class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.4),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.4),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256*8*8,512), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        x=self.fc_layers(x)
        return x
    
class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mels=128, max_len=128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_len = max_len
    
    def extract_mel_spectrogram(self, audio_bytes: bytes):
        import librosa
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True, duration=5.0)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_spec_db.shape[1] < self.max_len:
            mel_spec_db = np.pad(mel_spec_db, ((0,0),(0,self.max_len-mel_spec_db.shape[1])), mode="constant")
        else:
            mel_spec_db = mel_spec_db[:, :self.max_len]
        return mel_spec_db.astype(np.float32)

@app.cls(gpu="T4", image=image, volumes={MODEL_DIR: volume},timeout=180)
class OriginalDeepfakeAPI:
    @modal.enter()
    def load_models(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # ----- Load Audio Model -----
        self.audio_model = DeepfakeDetectorCNN()
        state_dict = torch.load(MODEL_DIR/"sound_deepfake_detector.pth", map_location=device)
        self.audio_model.load_state_dict(state_dict)
        self.audio_model.eval().to(device)
        self.audio_extractor = AudioFeatureExtractor()

        # ----- Load Video Model -----
        
        video_model_path = MODEL_DIR/"main_deepfake_model.pth"
        
        self.video_pipeline = DeepfakeDetectionPipeline(video_model_path, device)
        print("✅ Audio and Video models loaded successfully")

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    async def predict(self, file: UploadFile):
        # ----- Save temporary video file -----
        temp_video = tempfile.mktemp(suffix=".mp4")
        with open(temp_video,"wb") as f:
            f.write(await file.read())

        results = {}

        # ----- Video Prediction -----
        video_results = self.video_pipeline.process_video(temp_video, cleanup_temp=True)
        overall_video_pred, overall_video_conf = self.video_pipeline.get_overall_prediction(video_results)
        results['video'] = {
            "overall_prediction": overall_video_pred,
            "average_confidence": overall_video_conf,
            "detailed_results": video_results
        }

        # ----- Audio Prediction -----
        try:
            clip = VideoFileClip(temp_video)
            
            # Fallback: extract audio using ffmpeg if clip.audio is None
            if clip.audio is None:
                audio_path = tempfile.mktemp(suffix=".wav")
                cmd = f"ffmpeg -y -i {temp_video} -vn -acodec pcm_s16le -ar 22050 -ac 1 {audio_path}"
                subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    results['audio'] = {"prediction": "NO AUDIO FOUND", "fake_probability": None, "real_probability": None}
                else:
                    with open(audio_path,"rb") as af:
                        audio_bytes = af.read()
                    os.remove(audio_path)
            else:
                audio_path = tempfile.mktemp(suffix=".wav")
                clip.audio.write_audiofile(audio_path)
                with open(audio_path,"rb") as af:
                    audio_bytes = af.read()
                os.remove(audio_path)
            
            # Extract features and predict
            if 'audio_bytes' in locals():
                features = self.audio_extractor.extract_mel_spectrogram(audio_bytes)
                features_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.audio_model(features_tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                results['audio'] = {
                    "prediction": "FAKE" if probs[1]>0.5 else "REAL",
                    "fake_probability": float(probs[1]),
                    "real_probability": float(probs[0])
                }

        except Exception as e:
            results['audio'] = {"prediction": "NO AUDIO FOUND", "error": str(e)}
        os.remove(temp_video)
        return JSONResponse(results)