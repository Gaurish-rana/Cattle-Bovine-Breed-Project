#!/usr/bin/env python3
"""
Hierarchical Classification Pipeline for Cattle & Buffalo Breed Classification.

Structure:
- Stage 1: Species Classifier (Binary: Cattle vs Buffalo)
- Stage 2A: Cattle Breed Specialist (8 Breeds)
- Stage 2B: Buffalo Breed Specialist (9 Breeds)
- End-to-End Hierarchical Evaluator: Joint probability & Hard routing against 17-class baseline.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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


class CattleDataset(Dataset):
    def __init__(self, df, project_dir, target_col="breed_name", transform=None, label2idx=None):
        self.df = df.reset_index(drop=True)
        self.project_dir = project_dir
        self.target_col = target_col
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.project_dir, row["image_path"])
        label_name = str(row[self.target_col]).strip().lower()

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label_name]
        return image, label_idx


class BaseClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(BaseClassifier, self).__init__()
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


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


def train_model(name, model, train_loader, val_loader, criterion, device, epochs=6, warmup_epochs=2, lr=1e-3, ft_lr=2e-4):
    print(f"\n[{name}] Starting Training (Total Epochs: {epochs})...")
    # Warmup
    for param in model.backbone.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_weights = None

    for ep in range(1, epochs + 1):
        if ep == warmup_epochs + 1:
            print(f"[{name}] Unfreezing backbone for fine-tuning...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": ft_lr * 0.2},
                {"params": model.classifier.parameters(), "lr": ft_lr}
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

        model.train()
        tr_loss, tr_corr, tr_tot = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            _, p = torch.max(out, 1)
            tr_corr += (p == lbls).sum().item()
            tr_tot += lbls.size(0)

        if ep > warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        preds, labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, p = torch.max(out, 1)
                v_corr += (p == lbls).sum().item()
                v_tot += lbls.size(0)
                preds.extend(p.cpu().numpy())
                labels.extend(lbls.cpu().numpy())

        val_acc = v_corr / v_tot
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        print(f"[{name}] Epoch {ep:02d}/{epochs:02d} | Tr Acc: {tr_corr/tr_tot:.2%} | Val Acc: {val_acc:.2%}, Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"[{name}] Best Validation Macro F1: {best_val_f1:.4f}")
    return model, best_val_f1


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    annotations_dir = os.path.join(project_dir, "annotations")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df_train = pd.read_csv(os.path.join(annotations_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(annotations_dir, "val.csv"))

    set_seed(42)
    device = torch.device("cpu")
    torch.set_num_threads(12)
    train_transform, val_transform = get_transforms()

    print("=" * 80)
    print("HIERARCHICAL CLASSIFICATION EXPERIMENT")
    print("=" * 80)

    # ----------------------------------------------------
    # STAGE 1: Species Classifier (Cattle vs Buffalo)
    # ----------------------------------------------------
    species_list = ["buffalo", "cattle"]
    species2idx = {s: i for i, s in enumerate(species_list)}
    idx2species = {i: s for s, i in species2idx.items()}

    sp_train_loader = DataLoader(
        CattleDataset(df_train, project_dir, "species", train_transform, species2idx),
        batch_size=32, shuffle=True
    )
    sp_val_loader = DataLoader(
        CattleDataset(df_val, project_dir, "species", val_transform, species2idx),
        batch_size=32, shuffle=False
    )

    species_model = BaseClassifier(num_classes=2).to(device)
    sp_criterion = nn.CrossEntropyLoss()
    species_model, sp_best_f1 = train_model("Species_Stage1", species_model, sp_train_loader, sp_val_loader, sp_criterion, device, epochs=5, warmup_epochs=2)

    # ----------------------------------------------------
    # STAGE 2A: Cattle Specialist (8 Breeds)
    # ----------------------------------------------------
    df_tr_cattle = df_train[df_train["species"] == "cattle"].reset_index(drop=True)
    df_val_cattle = df_val[df_val["species"] == "cattle"].reset_index(drop=True)
    cattle_breeds = sorted(df_tr_cattle["breed_name"].unique())
    cattle2idx = {b: i for i, b in enumerate(cattle_breeds)}
    idx2cattle = {i: b for b, i in cattle2idx.items()}

    c_train_loader = DataLoader(
        CattleDataset(df_tr_cattle, project_dir, "breed_name", train_transform, cattle2idx),
        batch_size=32, shuffle=True
    )
    c_val_loader = DataLoader(
        CattleDataset(df_val_cattle, project_dir, "breed_name", val_transform, cattle2idx),
        batch_size=32, shuffle=False
    )

    # Class weights for cattle
    c_counts = df_tr_cattle["breed_name"].value_counts().sort_index()
    c_w = torch.tensor([len(df_tr_cattle)/(len(cattle_breeds)*c_counts.get(b,1)) for b in cattle_breeds], dtype=torch.float32).to(device)
    c_criterion = nn.CrossEntropyLoss(weight=c_w, label_smoothing=0.1)

    cattle_model = BaseClassifier(num_classes=len(cattle_breeds)).to(device)
    cattle_model, c_best_f1 = train_model("Cattle_Stage2A", cattle_model, c_train_loader, c_val_loader, c_criterion, device, epochs=8, warmup_epochs=2)

    # ----------------------------------------------------
    # STAGE 2B: Buffalo Specialist (9 Breeds)
    # ----------------------------------------------------
    df_tr_buffalo = df_train[df_train["species"] == "buffalo"].reset_index(drop=True)
    df_val_buffalo = df_val[df_val["species"] == "buffalo"].reset_index(drop=True)
    buffalo_breeds = sorted(df_tr_buffalo["breed_name"].unique())
    buffalo2idx = {b: i for i, b in enumerate(buffalo_breeds)}
    idx2buffalo = {i: b for b, i in buffalo2idx.items()}

    b_train_loader = DataLoader(
        CattleDataset(df_tr_buffalo, project_dir, "breed_name", train_transform, buffalo2idx),
        batch_size=32, shuffle=True
    )
    b_val_loader = DataLoader(
        CattleDataset(df_val_buffalo, project_dir, "breed_name", val_transform, buffalo2idx),
        batch_size=32, shuffle=False
    )

    # Class weights for buffalo
    b_counts = df_tr_buffalo["breed_name"].value_counts().sort_index()
    b_w = torch.tensor([len(df_tr_buffalo)/(len(buffalo_breeds)*b_counts.get(b,1)) for b in buffalo_breeds], dtype=torch.float32).to(device)
    b_criterion = nn.CrossEntropyLoss(weight=b_w, label_smoothing=0.1)

    buffalo_model = BaseClassifier(num_classes=len(buffalo_breeds)).to(device)
    buffalo_model, b_best_f1 = train_model("Buffalo_Stage2B", buffalo_model, b_train_loader, b_val_loader, b_criterion, device, epochs=8, warmup_epochs=2)

    # ----------------------------------------------------
    # END-TO-END HIERARCHICAL EVALUATION ON 271 VAL IMAGES
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("END-TO-END HIERARCHICAL EVALUATION ACROSS ALL 271 VALIDATION SAMPLES")
    print("=" * 80)

    species_model.eval()
    cattle_model.eval()
    buffalo_model.eval()

    all_17_breeds = sorted(df_train["breed_name"].unique())
    label2idx_17 = {b: i for i, b in enumerate(all_17_breeds)}

    # Direct data loader for all 271 validation images
    full_val_dataset = CattleDataset(df_val, project_dir, "breed_name", val_transform, label2idx_17)
    full_val_loader = DataLoader(full_val_dataset, batch_size=32, shuffle=False)

    hard_preds_17 = []
    soft_preds_17 = []
    sp_preds = []
    true_17 = []
    true_species = []

    # Map sub-model class indices to 17-class global indices
    cattle_sub2global = [label2idx_17[b] for b in cattle_breeds]
    buffalo_sub2global = [label2idx_17[b] for b in buffalo_breeds]

    with torch.no_grad():
        for i in range(len(df_val)):
            row = df_val.iloc[i]
            img_path = os.path.join(project_dir, row["image_path"])
            true_b = row["breed_name"].strip().lower()
            true_s = row["species"].strip().lower()

            true_17.append(label2idx_17[true_b])
            true_species.append(species2idx[true_s])

            img = Image.open(img_path).convert("RGB")
            tensor = val_transform(img).unsqueeze(0).to(device)

            # Stage 1: Species prediction
            sp_logits = species_model(tensor)
            sp_probs = torch.softmax(sp_logits, dim=1).cpu().numpy()[0]  # [p(buffalo), p(cattle)]
            pred_sp_idx = np.argmax(sp_probs)
            sp_preds.append(pred_sp_idx)

            # Stage 2: Hard routing
            if pred_sp_idx == species2idx["cattle"]:
                c_logits = cattle_model(tensor)
                c_pred_sub = torch.argmax(c_logits, dim=1).item()
                hard_pred_global = cattle_sub2global[c_pred_sub]
            else:
                b_logits = buffalo_model(tensor)
                b_pred_sub = torch.argmax(b_logits, dim=1).item()
                hard_pred_global = buffalo_sub2global[b_pred_sub]
            hard_preds_17.append(hard_pred_global)

            # Soft hierarchical joint probability
            c_logits = cattle_model(tensor)
            c_sub_probs = torch.softmax(c_logits, dim=1).cpu().numpy()[0]
            b_logits = buffalo_model(tensor)
            b_sub_probs = torch.softmax(b_logits, dim=1).cpu().numpy()[0]

            joint_17 = np.zeros(17)
            # Joint: P(cattle) * P(cattle_breed_i | cattle)
            for sub_i, glob_i in enumerate(cattle_sub2global):
                joint_17[glob_i] = sp_probs[species2idx["cattle"]] * c_sub_probs[sub_i]
            # Joint: P(buffalo) * P(buffalo_breed_j | buffalo)
            for sub_j, glob_j in enumerate(buffalo_sub2global):
                joint_17[glob_j] = sp_probs[species2idx["buffalo"]] * b_sub_probs[sub_j]

            soft_preds_17.append(np.argmax(joint_17))

    # Compute Metrics
    sp_acc = accuracy_score(true_species, sp_preds)
    sp_f1 = f1_score(true_species, sp_preds, average="macro")

    hard_acc = accuracy_score(true_17, hard_preds_17)
    hard_macro_f1 = f1_score(true_17, hard_preds_17, average="macro", zero_division=0)
    hard_weighted_f1 = f1_score(true_17, hard_preds_17, average="weighted", zero_division=0)

    soft_acc = accuracy_score(true_17, soft_preds_17)
    soft_macro_f1 = f1_score(true_17, soft_preds_17, average="macro", zero_division=0)
    soft_weighted_f1 = f1_score(true_17, soft_preds_17, average="weighted", zero_division=0)

    # Cross-species error analysis
    sp_errors = sum(1 for ts, ps in zip(true_species, sp_preds) if ts != ps)

    print(f"\n--- STAGE 1 (Species: Cattle vs Buffalo) ---")
    print(f"Validation Accuracy: {sp_acc:.2%}")
    print(f"Validation Macro F1: {sp_f1:.4f}")
    print(f"Stage 1 Species Errors: {sp_errors}/{len(true_species)} ({sp_errors/len(true_species):.2%})")

    print(f"\n--- STAGE 2 SPECIALISTS ---")
    print(f"Cattle Specialist (8 classes) Val Macro F1:  {c_best_f1:.4f}")
    print(f"Buffalo Specialist (9 classes) Val Macro F1: {b_best_f1:.4f}")

    print(f"\n--- END-TO-END 17-CLASS HIERARCHICAL RESULTS ---")
    print(f"Hard Routing Accuracy:   {hard_acc:.2%}")
    print(f"Hard Routing Macro F1:   {hard_macro_f1:.4f}")
    print(f"Hard Routing Weighted F1:{hard_weighted_f1:.4f}")
    print(f"Soft Joint Accuracy:     {soft_acc:.2%}")
    print(f"Soft Joint Macro F1:     {soft_macro_f1:.4f}")
    print(f"Soft Joint Weighted F1:  {soft_weighted_f1:.4f}")

    rep_hard = classification_report(true_17, hard_preds_17, target_names=all_17_breeds, digits=4)
    print("\nHIERARCHICAL 17-CLASS CLASSIFICATION REPORT (HARD ROUTING):")
    print(rep_hard)

    # Save summary & reports
    summary = {
        "species_stage1": {
            "val_acc": round(sp_acc, 4),
            "val_macro_f1": round(sp_f1, 4),
            "species_errors": sp_errors
        },
        "stage2_specialists": {
            "cattle_8_classes_macro_f1": round(c_best_f1, 4),
            "buffalo_9_classes_macro_f1": round(b_best_f1, 4)
        },
        "end_to_end_17_classes": {
            "hard_routing_val_acc": round(hard_acc, 4),
            "hard_routing_val_macro_f1": round(hard_macro_f1, 4),
            "hard_routing_val_weighted_f1": round(hard_weighted_f1, 4),
            "soft_joint_val_acc": round(soft_acc, 4),
            "soft_joint_val_macro_f1": round(soft_macro_f1, 4),
            "soft_joint_val_weighted_f1": round(soft_weighted_f1, 4)
        }
    }

    with open(os.path.join(models_dir, "hierarchical_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(models_dir, "hierarchical_classification_report.txt"), "w") as f:
        f.write("Hierarchical Classification Report (Hard Routing)\n")
        f.write(f"End-to-End Val Accuracy: {hard_acc:.4f}\n")
        f.write(f"End-to-End Val Macro F1: {hard_macro_f1:.4f}\n\n")
        f.write(rep_hard)

    # Save checkpoints
    torch.save(species_model.state_dict(), os.path.join(models_dir, "hierarchical_stage1_species.pth"))
    torch.save(cattle_model.state_dict(), os.path.join(models_dir, "hierarchical_stage2a_cattle.pth"))
    torch.save(buffalo_model.state_dict(), os.path.join(models_dir, "hierarchical_stage2b_buffalo.pth"))
    print("\nSaved all hierarchical checkpoints and reports to models/")


if __name__ == "__main__":
    main()
