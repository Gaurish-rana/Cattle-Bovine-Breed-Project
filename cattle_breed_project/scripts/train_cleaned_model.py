#!/usr/bin/env python3
"""
Production Training Pipeline for Cattle & Buffalo Breed Classification (17 Classes)
Using EfficientNet-B2 with Custom Classifier Head.

Features:
1. 17 indigenous breeds (9 Buffalo + 8 Cattle).
2. Reads from stratified split files: annotations/train.csv and annotations/val.csv.
3. Strict isolation: annotations/test.csv is NEVER touched during training or selection.
4. Pretrained weights: EfficientNet_B2_Weights.DEFAULT (ImageNet-1K).
5. Two-Stage Fine-Tuning:
   - Stage 1: Warmup with frozen backbone (train classifier head).
   - Stage 2: End-to-end fine-tuning with differential learning rates.
6. Balanced Class-Weighted Cross-Entropy Loss with Label Smoothing (alpha=0.1).
7. Realistic data augmentations: RandomResizedCrop, RandomFlip, Rotation, ColorJitter, RandomErasing.
8. AdamW optimizer with Cosine Annealing learning rate schedule.
9. Early stopping based on Validation Macro F1 score.
10. Saves complete checkpoint (state dicts, label2idx, breeds, config, metrics).
11. Saves training history, config JSON, and training curves plot.
"""

import os
import sys
import json
import random
import time
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

# Safe UTF-8 console output for Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ----------------------------------------------------
# 1. Reproducibility & Device Configuration
# ----------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------------------
# 2. Dataset Class
# ----------------------------------------------------
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
        rel_path = row["image_path"]
        img_path = os.path.join(self.project_dir, rel_path)
        label_name = str(row["breed_name"]).strip().lower()

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback to white canvas if reading fails
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label_name]
        return image, label_idx


# ----------------------------------------------------
# 3. Model Architecture (EfficientNet-B2 + Custom Head)
# ----------------------------------------------------
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes=17, dropout_rate=0.3):
        super(ImprovedCattleClassifier, self).__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)
        in_features = self.backbone.classifier[1].in_features  # 1408
        self.backbone.classifier = nn.Identity()

        # Custom classifier head matching project checkpoint design
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ----------------------------------------------------
# 4. Data Augmentation & Transforms
# ----------------------------------------------------
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


# ----------------------------------------------------
# 5. Training & Evaluation Functions
# ----------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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

    metrics = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "all_preds": all_preds,
        "all_labels": all_labels
    }
    return metrics


# ----------------------------------------------------
# 6. Main Training Pipeline
# ----------------------------------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    annotations_dir = os.path.join(project_dir, "annotations")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    train_csv = os.path.join(annotations_dir, "train.csv")
    val_csv = os.path.join(annotations_dir, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("Split CSVs not found. Please run prepare_dataset_splits.py first.")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    # Class mappings
    breeds = sorted(df_train["breed_name"].unique())
    num_classes = len(breeds)
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    idx2label = {i: breed for breed, i in label2idx.items()}

    # Hyperparameters & Config
    CONFIG = {
        "architecture": "EfficientNet-B2",
        "num_classes": num_classes,
        "seed": 42,
        "batch_size": 32,
        "warmup_epochs": 3,
        "finetune_epochs": 12,
        "warmup_lr": 1e-3,
        "backbone_lr": 1e-4,
        "head_lr": 5e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "early_stopping_patience": 5,
        "dropout_rate": 0.3,
        "img_size": 224,
        "train_samples": len(df_train),
        "val_samples": len(df_val)
    }

    set_seed(CONFIG["seed"])

    # Device & Threads
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True
        num_workers = 4
        print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)} (Mixed Precision Enabled)")
    else:
        device = torch.device("cpu")
        use_amp = False
        num_workers = 0  # 0 is safest on Windows without CUDA
        cpu_cores = os.cpu_count() or 8
        torch.set_num_threads(min(cpu_cores, 12))
        print(f"Running on CPU with {torch.get_num_threads()} threads (Mixed Precision Disabled)")

    print("=" * 85)
    print("STARTING EFFICIENTNET-B2 17-CLASS TRAINING PIPELINE")
    print("=" * 85)
    print(f"Target Classes ({num_classes}): {', '.join(breeds)}")
    print(f"Training Samples:   {len(df_train)}")
    print(f"Validation Samples: {len(df_val)}")
    print(f"Batch Size:         {CONFIG['batch_size']}")
    print(f"Warmup Epochs:      {CONFIG['warmup_epochs']} (Classifier head only)")
    print(f"Finetune Epochs:    {CONFIG['finetune_epochs']} (End-to-end with differential LR)")
    print("-" * 85)

    # Calculate class weights for imbalanced loss handling
    class_counts = df_train["breed_name"].value_counts().sort_index()
    total_samples = len(df_train)
    weights = []
    for breed in breeds:
        cnt = class_counts.get(breed, 1)
        w = total_samples / (num_classes * cnt)
        weights.append(w)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=CONFIG["label_smoothing"])

    # Transforms and Loaders
    train_transform, val_transform = get_transforms()
    train_dataset = CattleDataset(df_train, project_dir, transform=train_transform, label2idx=label2idx)
    val_dataset = CattleDataset(df_val, project_dir, transform=val_transform, label2idx=label2idx)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    # Initialize Model
    model = ImprovedCattleClassifier(num_classes=num_classes, dropout_rate=CONFIG["dropout_rate"])
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # History tracker
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "macro_f1": [],
        "weighted_f1": [], "macro_prec": [], "macro_rec": [],
        "lr": []
    }

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    checkpoint_save_path = os.path.join(models_dir, "Best_model_final.pth")
    root_model_path = os.path.join(project_dir, "best_model.pth")
    history_json_path = os.path.join(models_dir, "training_history.json")
    config_json_path = os.path.join(models_dir, "training_config.json")
    curves_plot_path = os.path.join(models_dir, "training_curves.png")

    total_epochs = CONFIG["warmup_epochs"] + CONFIG["finetune_epochs"]
    global_epoch = 0

    # ====================================================
    # STAGE 1: Warmup with Frozen Backbone
    # ====================================================
    print("\n>>> STAGE 1: Classifier Head Warmup (Pretrained Backbone Frozen)...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        model.classifier.parameters(),
        lr=CONFIG["warmup_lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    t_start = time.time()
    for ep in range(CONFIG["warmup_epochs"]):
        global_epoch += 1
        ep_t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        v_metrics = validate_one_epoch(model, val_loader, criterion, device)
        ep_time = time.time() - ep_t0

        history["epoch"].append(global_epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(v_metrics["val_loss"], 4))
        history["val_acc"].append(round(v_metrics["val_acc"], 4))
        history["macro_f1"].append(round(v_metrics["macro_f1"], 4))
        history["weighted_f1"].append(round(v_metrics["weighted_f1"], 4))
        history["macro_prec"].append(round(v_metrics["macro_prec"], 4))
        history["macro_rec"].append(round(v_metrics["macro_rec"], 4))
        history["lr"].append(CONFIG["warmup_lr"])

        print(f"Epoch [{global_epoch:02d}/{total_epochs:02d}] (Warmup) | Time: {ep_time:.1f}s | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2%} | "
              f"Val Loss: {v_metrics['val_loss']:.4f}, Acc: {v_metrics['val_acc']:.2%}, Macro F1: {v_metrics['macro_f1']:.4f}")

        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            best_val_acc = v_metrics["val_acc"]
            best_epoch = global_epoch
            # Save checkpoint
            save_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": global_epoch,
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "val_loss": v_metrics["val_loss"],
                "num_classes": num_classes,
                "breeds": breeds,
                "label2idx": label2idx,
                "idx2label": idx2label,
                "model_architecture": "EfficientNet-B2",
                "config": CONFIG
            }
            torch.save(save_payload, checkpoint_save_path)
            torch.save(save_payload, root_model_path)
            print(f"   --> Saved new best checkpoint (Val F1: {best_val_f1:.4f}, Val Acc: {best_val_acc:.2%})")

    # ====================================================
    # STAGE 2: End-to-End Fine-Tuning
    # ====================================================
    print("\n>>> STAGE 2: Unfreezing Backbone for End-to-End Fine-Tuning...")
    for param in model.backbone.parameters():
        param.requires_grad = True

    # Differential learning rates
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": CONFIG["backbone_lr"]},
        {"params": model.classifier.parameters(), "lr": CONFIG["head_lr"]}
    ], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["finetune_epochs"], eta_min=1e-6
    )

    for ep in range(CONFIG["finetune_epochs"]):
        global_epoch += 1
        ep_t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        v_metrics = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - ep_t0

        history["epoch"].append(global_epoch)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(v_metrics["val_loss"], 4))
        history["val_acc"].append(round(v_metrics["val_acc"], 4))
        history["macro_f1"].append(round(v_metrics["macro_f1"], 4))
        history["weighted_f1"].append(round(v_metrics["weighted_f1"], 4))
        history["macro_prec"].append(round(v_metrics["macro_prec"], 4))
        history["macro_rec"].append(round(v_metrics["macro_rec"], 4))
        history["lr"].append(curr_lr)

        print(f"Epoch [{global_epoch:02d}/{total_epochs:02d}] | Time: {ep_time:.1f}s | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.2%} | "
              f"Val Loss: {v_metrics['val_loss']:.4f}, Acc: {v_metrics['val_acc']:.2%}, Macro F1: {v_metrics['macro_f1']:.4f}")

        # Early stopping and checkpointing
        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            best_val_acc = v_metrics["val_acc"]
            best_epoch = global_epoch
            patience_counter = 0

            save_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": global_epoch,
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "val_loss": v_metrics["val_loss"],
                "num_classes": num_classes,
                "breeds": breeds,
                "label2idx": label2idx,
                "idx2label": idx2label,
                "model_architecture": "EfficientNet-B2",
                "config": CONFIG
            }
            torch.save(save_payload, checkpoint_save_path)
            torch.save(save_payload, root_model_path)
            print(f"   --> Saved new best checkpoint (Val F1: {best_val_f1:.4f}, Val Acc: {best_val_acc:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"\n[EARLY STOPPING TRIGGERED] No validation improvement for {CONFIG['early_stopping_patience']} epochs.")
                break

    total_training_time = time.time() - t_start
    print("-" * 85)
    print(f"TRAINING COMPLETED in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Macro F1:  {best_val_f1:.4f} (at Epoch {best_epoch})")
    print(f"Best Validation Accuracy:  {best_val_acc:.2%}")
    print(f"Checkpoint saved to:       {checkpoint_save_path}")
    print(f"Root model checkpoint at:  {root_model_path}")

    # Save Config and History JSON
    with open(config_json_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved configuration to:    {config_json_path}")
    print(f"Saved history log to:      {history_json_path}")

    # Save Training Curves Plot
    plt.figure(figsize=(14, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], "b-o", label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], "r-o", label="Val Loss")
    plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best Model (Ep {best_epoch})")
    plt.title("Loss Curves over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Accuracy & Macro F1 Curve
    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(history["epoch"], history["val_acc"], "r-o", label="Val Accuracy")
    plt.plot(history["epoch"], history["macro_f1"], "g--s", label="Val Macro F1")
    plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best Model (Ep {best_epoch})")
    plt.title("Accuracy and Macro F1 Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(curves_plot_path, dpi=150)
    plt.close()
    print(f"Saved training curves to:  {curves_plot_path}")
    print("=" * 85)

    # Generate Detailed Validation Report with Best Model
    print("\nEVALUATING BEST MODEL ON VALIDATION SET FOR FINAL METRICS...")
    best_checkpoint = torch.load(checkpoint_save_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_final_metrics = validate_one_epoch(model, val_loader, criterion, device)

    val_report = classification_report(
        val_final_metrics["all_labels"],
        val_final_metrics["all_preds"],
        target_names=breeds,
        digits=4
    )
    print("\nFINAL VALIDATION CLASSIFICATION REPORT (BEST CHECKPOINT):")
    print(val_report)

    # Save detailed report to text
    report_file = os.path.join(models_dir, "validation_classification_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Validation Classification Report - Epoch {best_epoch}\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best Val Macro F1: {best_val_f1:.4f}\n\n")
        f.write(val_report)
    print(f"Detailed classification report saved to: {report_file}")


if __name__ == "__main__":
    main()
