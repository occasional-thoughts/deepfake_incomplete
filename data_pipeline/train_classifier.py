#!/usr/bin/env python3
"""
train_classifier.py

Usage:
python data_pipeline/train_classifier.py --data_dir data/dataset --epochs 6 --batch 32 --out models/resnet18.pt
"""
import argparse
from pathlib import Path
import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

def train(data_dir, epochs, batch_size, lr, out_path, device):
    data_dir = Path(data_dir)
    # simple transforms
    train_tf = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((160,160)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_dir = data_dir  # expects subfolders per class: data_dir/fake, data_dir/real
    dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    # split small val automatically 80/20
    n = len(dataset)
    n_val = max(1, int(0.2*n))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.resnet18(pretrained=True)
    # adapt final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb,yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()*xb.size(0)
        train_loss = running / len(train_loader.dataset)

        # val
        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                correct += (preds==yb).sum().item()
                total += xb.size(0)
        val_acc = correct/total if total>0 else 0.0
        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_path)
            print("Saved best model to", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dataset dir containing class subfolders")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=str, default="models/resnet18.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train(args.data_dir, args.epochs, args.batch, args.lr, args.out, device)
