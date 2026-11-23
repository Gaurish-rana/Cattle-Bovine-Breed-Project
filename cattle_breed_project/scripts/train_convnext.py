#!/usr/bin/env python3
"""
Experiment: ConvNeXt-Tiny for 17-Class Cattle & Buffalo Breed Classification.
Follows identical dataset split (train.csv / val.csv) and evaluation metrics
as the EfficientNet-B2 baseline.
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CattleDataset(Dataset):
    def __init__(self, df, project_dir, transform=None, label2idx=None):
        self.df = df.reset_index(drop=True)
        self.project_dir = project_dir
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.project_dir, row["image_path"])
        label_name = str(row["breed_name"]).strip().lower()

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label_name]
        return image, label_idx


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=17, dropout_rate=0.3):
        super(ConvNeXtClassifier, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.model = models.convnext_tiny(weights=weights)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "all_preds": all_preds,
        "all_labels": all_labels
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    annotations_dir = os.path.join(project_dir, "annotations")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df_train = pd.read_csv(os.path.join(annotations_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(annotations_dir, "val.csv"))

    breeds = sorted(df_train["breed_name"].unique())
    num_classes = len(breeds)
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    idx2label = {i: breed for breed, i in label2idx.items()}

    CONFIG = {
        "architecture": "ConvNeXt-Tiny",
        "num_classes": num_classes,
        "seed": 42,
        "batch_size": 32,
        "warmup_epochs": 2,
        "finetune_epochs": 8,
        "warmup_lr": 1e-3,
        "backbone_lr": 5e-5,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "dropout_rate": 0.3,
        "img_size": 224,
        "train_samples": len(df_train),
        "val_samples": len(df_val)
    }

    set_seed(CONFIG["seed"])

    device = torch.device("cpu")
    cpu_cores = os.cpu_count() or 8
    torch.set_num_threads(min(cpu_cores, 12))
    print(f"Running ConvNeXt-Tiny on CPU with {torch.get_num_threads()} threads")

    # Class weighting
    class_counts = df_train["breed_name"].value_counts().sort_index()
    weights = [len(df_train) / (num_classes * class_counts.get(b, 1)) for b in breeds]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=CONFIG["label_smoothing"])

    train_transform, val_transform = get_transforms()
    train_dataset = CattleDataset(df_train, project_dir, transform=train_transform, label2idx=label2idx)
    val_dataset = CattleDataset(df_val, project_dir, transform=val_transform, label2idx=label2idx)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = ConvNeXtClassifier(num_classes=num_classes, dropout_rate=CONFIG["dropout_rate"]).to(device)

    # Stage 1: Warmup
    print("Stage 1: Warming up head (backbone frozen)...")
    for param in model.model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.model.classifier.parameters(), lr=CONFIG["warmup_lr"], weight_decay=CONFIG["weight_decay"])

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "macro_f1": []}
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    t_start = time.time()

    for ep in range(1, CONFIG["warmup_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vm = validate_one_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0
        history["epoch"].append(ep)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(vm["val_loss"], 4))
        history["val_acc"].append(round(vm["val_acc"], 4))
        history["macro_f1"].append(round(vm["macro_f1"], 4))

        print(f"Epoch [{ep:02d}/{CONFIG['warmup_epochs']+CONFIG['finetune_epochs']:02d}] (Warmup) | {dt:.1f}s | "
              f"Tr Loss: {tr_loss:.4f}, Acc: {tr_acc:.2%} | Val Loss: {vm['val_loss']:.4f}, Acc: {vm['val_acc']:.2%}, Macro F1: {vm['macro_f1']:.4f}")
        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            best_val_acc = vm["val_acc"]
            best_epoch = ep

    # Stage 2: Fine-tuning
    print("\nStage 2: Unfreezing for End-to-End Fine-Tuning...")
    for param in model.model.features.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.model.features.parameters(), "lr": CONFIG["backbone_lr"]},
        {"params": model.model.classifier.parameters(), "lr": CONFIG["head_lr"]}
    ], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["finetune_epochs"], eta_min=1e-6)

    ckpt_path = os.path.join(models_dir, "convnext_tiny_best.pth")

    for ep_rel in range(1, CONFIG["finetune_epochs"] + 1):
        ep = CONFIG["warmup_epochs"] + ep_rel
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vm = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["epoch"].append(ep)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(vm["val_loss"], 4))
        history["val_acc"].append(round(vm["val_acc"], 4))
        history["macro_f1"].append(round(vm["macro_f1"], 4))

        print(f"Epoch [{ep:02d}/{CONFIG['warmup_epochs']+CONFIG['finetune_epochs']:02d}] | {dt:.1f}s | "
              f"Tr Loss: {tr_loss:.4f}, Acc: {tr_acc:.2%} | Val Loss: {vm['val_loss']:.4f}, Acc: {vm['val_acc']:.2%}, Macro F1: {vm['macro_f1']:.4f}")

        if vm["macro_f1"] > best_val_f1:
            best_val_f1 = vm["macro_f1"]
            best_val_acc = vm["val_acc"]
            best_epoch = ep
            payload = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "breeds": breeds,
                "label2idx": label2idx,
                "idx2label": idx2label,
                "architecture": "ConvNeXt-Tiny",
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "config": CONFIG
            }
            torch.save(payload, ckpt_path)
            print(f"   --> Saved best ConvNeXt-Tiny checkpoint (Val F1: {best_val_f1:.4f}, Acc: {best_val_acc:.2%})")

    total_time = time.time() - t_start
    CONFIG["total_training_time_minutes"] = round(total_time / 60, 2)
    CONFIG["best_epoch"] = best_epoch
    CONFIG["best_val_f1"] = round(best_val_f1, 4)
    CONFIG["best_val_acc"] = round(best_val_acc, 4)

    # Save artifacts
    with open(os.path.join(models_dir, "convnext_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(os.path.join(models_dir, "convnext_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on best model
    if os.path.exists(ckpt_path):
        chk = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
    val_final = validate_one_epoch(model, val_loader, criterion, device)
    rep = classification_report(val_final["all_labels"], val_final["all_preds"], target_names=breeds, digits=4)
    print("\nFINAL CONVNEXT-TINY VALIDATION REPORT:")
    print(rep)
    with open(os.path.join(models_dir, "convnext_classification_report.txt"), "w") as f:
        f.write(f"ConvNeXt-Tiny Validation Report (Best Ep {best_epoch})\nBest Val Acc: {best_val_acc:.4f}\nBest Val Macro F1: {best_val_f1:.4f}\n\n" + rep)


if __name__ == "__main__":
    main()
