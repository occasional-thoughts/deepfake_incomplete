#!/usr/bin/env python3
import argparse, torch, csv
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from torchvision import transforms, models
import numpy as np
import torch.nn.functional as F

def load_model(path, device):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

def preprocess_pil(img):
    tf = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tf(img).unsqueeze(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val_list", required=False, default="data/dataset/val_list.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    rows = []
    p = Path(args.val_list)
    if not p.exists():
        raise SystemExit("val_list.csv not found at " + str(p))
    with open(p) as fh:
        for line in fh:
            path,label = line.strip().split(",")
            rows.append((path,label))

    y_true = []
    y_prob = []
    y_pred = []
    for path,label in rows:
        img = Image.open(path).convert('RGB')
        x = preprocess_pil(img).to(device)
        with torch.no_grad():
            out = model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(probs.argmax())
        y_true.append(1 if label=='real' else 0)   # change if your labels reversed
        y_prob.append(float(probs[1]))             # probability for class index 1 (real)
        y_pred.append(pred)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_true, y_prob))
    except Exception as e:
        print("ROC AUC error:", e)
