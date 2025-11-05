#!/usr/bin/env python3
# save as data_pipeline/video_infer_fast.py
import argparse, subprocess, tempfile, csv, torch
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F

def load_model(path, device):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

tf = transforms.Compose([transforms.Resize((160,160)),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="video_infer_fast.csv")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--sample", type=int, default=3, help="process every Nth extracted frame")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    mt = MTCNN(keep_all=True, device=device, min_face_size=40)

    out_rows=[]
    with tempfile.TemporaryDirectory() as td:
        pattern = f"{td}/frame_%06d.jpg"
        subprocess.run(['ffmpeg','-y','-i', args.video, '-vf', f'fps={args.fps}', pattern], check=True)
        from pathlib import Path
        frames = sorted(Path(td).glob('*.jpg'))
        for idx,f in enumerate(frames):
            if idx % args.sample != 0:
                continue
            img = Image.open(f).convert('RGB')
            boxes, probs = mt.detect(img)
            if boxes is None:
                continue
            for i,box in enumerate(boxes):
                x1,y1,x2,y2 = [int(max(0,v)) for v in box]
                crop = img.crop((x1,y1,x2,y2)).resize((160,160))
                x = tf(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(x)
                    p = F.softmax(out, dim=1).cpu().numpy()[0]
                out_rows.append([str(f.name), i, x1,y1,x2,y2, float(p[0]), float(p[1])])
    with open(args.out,'w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['frame','face_id','x1','y1','x2','y2','prob_fake','prob_real']); w.writerows(out_rows)
    print("Wrote", args.out)
