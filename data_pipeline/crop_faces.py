#!/usr/bin/env python3
from pathlib import Path
import argparse
from facenet_pytorch import MTCNN
from PIL import Image

def crop_faces(frames_dir, out_dir, min_size=20):
    frames_dir = Path(frames_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(keep_all=True, device='cpu', min_face_size=min_size)

    frames = sorted(frames_dir.glob("*.jpg"))
    saved = 0
    for i, f in enumerate(frames):
        img = Image.open(f).convert('RGB')
        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            continue
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            crop = img.crop((x1, y1, x2, y2))
            out_path = out_dir / f"{f.stem}_face{j:02d}.jpg"
            crop.save(out_path)
            saved += 1
    print(f"Saved {saved} face crops to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir with frames (jpg)")
    ap.add_argument("--out", required=True, help="output dir for face crops")
    ap.add_argument("--min-size", type=int, default=20, help="min face size for detector")
    args = ap.parse_args()
    crop_faces(args.frames, args.out, args.min_size)
