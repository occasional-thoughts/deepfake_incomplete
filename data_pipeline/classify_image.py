#!/usr/bin/env python3
import argparse, torch
from pathlib import Path
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

def load_model(path, device):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)   # two classes: fake/real
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def preprocess(img_path):
    tf = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return tf(img).unsqueeze(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    x = preprocess(args.image).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    print("pred_class_index:", int(pred), "probs:", probs)
