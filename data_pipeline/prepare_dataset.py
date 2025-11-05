#!/usr/bin/env python3
"""
prepare_dataset.py

Usage examples:
# single video (label 'fake'):
python data_pipeline/prepare_dataset.py --videos data/videos/obama.mp4 --out data/dataset --label fake --fps 2

# multiple videos (one or more labels):
python data_pipeline/prepare_dataset.py --videos data/videos/*.mp4 --out data/dataset --label fake --fps 2

# create train/val split after processing:
python data_pipeline/prepare_dataset.py --videos data/videos/*.mp4 --out data/dataset --label fake --fps 2 --val-split 0.2
"""
import argparse
import subprocess
import shlex
from pathlib import Path
import tempfile
import csv
import random
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as T
import os
import sys

def extract_frames_ffmpeg(video_path: Path, out_dir: Path, fps=2):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", pattern]
    subprocess.run(cmd, check=True)

def detect_and_save_faces(frames_dir: Path, out_label_dir: Path, mtcnn: MTCNN, size=(160,160)):
    transform = T.Compose([T.Resize(size), T.CenterCrop(size)])
    saved = 0
    for frame in sorted(frames_dir.glob("*.jpg")):
        try:
            img = Image.open(frame).convert("RGB")
        except Exception:
            continue
        boxes, _ = mtcnn.detect(img)
        if boxes is None: 
            continue
        # keep all faces
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            crop = img.crop((x1, y1, x2, y2))
            crop = transform(crop)
            out_name = out_label_dir / f"{frame.stem}_face{i:02d}.jpg"
            crop.save(out_name, quality=95)
            saved += 1
    return saved

def prepare(videos, out_dir, label, fps, min_face_size, val_split):
    out_dir = Path(out_dir)
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, device='cpu', min_face_size=min_face_size)

    manifest = []
    for vid in videos:
        vid = Path(vid)
        if not vid.exists():
            print("Skipping missing:", vid)
            continue
        # temp folder for frames for this video
        with tempfile.TemporaryDirectory() as tmpd:
            frames_dir = Path(tmpd)
            print("Extracting frames from", vid.name)
            try:
                extract_frames_ffmpeg(vid, frames_dir, fps=fps)
            except subprocess.CalledProcessError:
                print("ffmpeg failed for", vid)
                continue
            print("Detecting faces...")
            saved = detect_and_save_faces(frames_dir, label_dir, mtcnn, size=(160,160))
            print(f"Saved {saved} face crops for {vid.name}")
            # add to manifest
            for f in sorted(label_dir.glob("*.jpg")):
                manifest.append((str(f.resolve()), label))
    # write manifest CSV
    manifest_file = out_dir / "manifest.csv"
    print("Writing manifest to", manifest_file)
    with open(manifest_file, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label"])
        for p,l in manifest:
            writer.writerow([p,l])
    # optional train/val split
    if val_split and 0.0 < val_split < 1.0:
        random.shuffle(manifest)
        cut = int(len(manifest)*(1.0-val_split))
        train = manifest[:cut]
        val = manifest[cut:]
        def write_list(lst, fname):
            with open(out_dir/fname, "w") as fh:
                for p,l in lst:
                    fh.write(f"{p},{l}\n")
        write_list(train, "train_list.csv")
        write_list(val, "val_list.csv")
        print(f"Train/val split written: train={len(train)} val={len(val)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True, help="one or more video paths (supports glob if quoted)")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--label", required=True, help="label for these videos (e.g. fake or real)")
    ap.add_argument("--fps", type=int, default=2, help="frames per second to extract")
    ap.add_argument("--min-face-size", type=int, default=40, help="minimum face size for detector")
    ap.add_argument("--val-split", type=float, default=0.0, help="fraction for validation split (0.0 disables)")
    args = ap.parse_args()
    prepare(args.videos, args.out, args.label, args.fps, args.min_face_size, args.val_split)
