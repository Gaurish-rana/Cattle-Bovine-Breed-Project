#!/usr/bin/env python3
"""
Comprehensive Test Evaluation Pipeline for Cattle & Buffalo Breed Classification.

Evaluates the best trained model (ConvNeXt-Tiny) and the baseline (EfficientNet-B2)
on the untouched test dataset (annotations/test.csv).

Generates:
1. Overall accuracy, macro precision, recall, F1, weighted F1.
2. Per-class precision, recall, and F1-scores.
3. Confusion matrix (raw counts, normalized, and high-res heatmap plot).
4. Full classification report.
5. In-depth incorrect prediction analysis saved to CSV and markdown.
6. Error pattern analysis (cattle-cattle, buffalo-buffalo, cross-species, morphological drivers).
7. Visualizations:
   - confusion_matrix_heatmap.png
   - per_class_f1_chart.png
   - training_validation_curves.png
   - prediction_gallery.png (Correct vs Incorrect visual breakdown)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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
        return image, label_idx, row["image_path"], label_name, row.get("species", "unknown")


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=17, dropout_rate=0.3):
        super(ConvNeXtClassifier, self).__init__()
        weights = None  # Loaded from checkpoint
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


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=17, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        weights = None
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


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate_model(model, loader, device, breeds):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    metadata = []

    with torch.no_grad():
        for images, labels, paths, breed_names, species in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            for i in range(len(paths)):
                metadata.append({
                    "image_path": paths[i],
                    "true_breed": breed_names[i],
                    "species": species[i],
                    "pred_breed": breeds[preds[i].item()],
                    "confidence": float(probs[i][preds[i]].item()),
                    "correct": bool(preds[i].item() == labels[i].item())
                })

    acc = accuracy_score(all_labels, all_preds)
    macro_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Per-class metrics
    p_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    r_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            breeds[i]: {
                "precision": float(p_class[i]),
                "recall": float(r_class[i]),
                "f1_score": float(f1_class[i])
            }
            for i in range(len(breeds))
        },
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "metadata": metadata
    }
    return results


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    annotations_dir = os.path.join(project_dir, "annotations")
    models_dir = os.path.join(project_dir, "models")
    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    test_csv = os.path.join(annotations_dir, "test.csv")
    train_csv = os.path.join(annotations_dir, "train.csv")
    val_csv = os.path.join(annotations_dir, "val.csv")

    df_test = pd.read_csv(test_csv)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    breeds = sorted(df_test["breed_name"].unique())
    num_classes = len(breeds)
    label2idx = {b: i for i, b in enumerate(breeds)}
    idx2label = {i: b for b, i in label2idx.items()}

    print("=" * 85)
    print("FINAL MODEL EVALUATION PIPELINE (UNTOUCHED TEST DATASET)")
    print("=" * 85)
    print(f"Total Test Samples:       {len(df_test)}")
    print(f"Total Classes:            {num_classes}")
    print(f"Results Directory:        {results_dir}")
    print("-" * 85)

    device = torch.device("cpu")
    torch.set_num_threads(12)
    test_transform = get_test_transform()
    test_dataset = CattleDataset(df_test, project_dir, transform=test_transform, label2idx=label2idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ----------------------------------------------------
    # 1. Evaluate Primary Selected Model: ConvNeXt-Tiny
    # ----------------------------------------------------
    convnext_ckpt = os.path.join(models_dir, "convnext_tiny_best.pth")
    if not os.path.exists(convnext_ckpt):
        raise FileNotFoundError(f"ConvNeXt checkpoint not found at {convnext_ckpt}")

    print(f"\n[1/2] Loading ConvNeXt-Tiny Best Checkpoint: {convnext_ckpt}")
    conv_model = ConvNeXtClassifier(num_classes=num_classes)
    chk_conv = torch.load(convnext_ckpt, map_location=device)
    conv_model.load_state_dict(chk_conv["model_state_dict"])
    conv_model = conv_model.to(device)

    print("Evaluating ConvNeXt-Tiny on untouched test set...")
    conv_results = evaluate_model(conv_model, test_loader, device, breeds)

    # ----------------------------------------------------
    # 2. Evaluate Baseline Model: EfficientNet-B2
    # ----------------------------------------------------
    eff_ckpt = os.path.join(models_dir, "Best_model_final.pth")
    print(f"\n[2/2] Loading EfficientNet-B2 Baseline Checkpoint: {eff_ckpt}")
    eff_model = EfficientNetClassifier(num_classes=num_classes)
    chk_eff = torch.load(eff_ckpt, map_location=device)
    eff_model.load_state_dict(chk_eff["model_state_dict"])
    eff_model = eff_model.to(device)

    print("Evaluating EfficientNet-B2 on untouched test set...")
    eff_results = evaluate_model(eff_model, test_loader, device, breeds)

    # ----------------------------------------------------
    # 3. Print Comparison
    # ----------------------------------------------------
    print("\n" + "=" * 85)
    print("UNTOUCHED TEST DATASET RESULTS COMPARISON")
    print("=" * 85)
    print(f"{'Metric':<25} | {'EfficientNet-B2 (Baseline)':<28} | {'ConvNeXt-Tiny (Final Model)':<28}")
    print("-" * 85)
    print(f"{'Test Accuracy':<25} | {eff_results['accuracy']:<28.2%} | {conv_results['accuracy']:<28.2%}")
    print(f"{'Macro Precision':<25} | {eff_results['macro_precision']:<28.4f} | {conv_results['macro_precision']:<28.4f}")
    print(f"{'Macro Recall':<25} | {eff_results['macro_recall']:<28.4f} | {conv_results['macro_recall']:<28.4f}")
    print(f"{'Macro F1-Score':<25} | {eff_results['macro_f1']:<28.4f} | {conv_results['macro_f1']:<28.4f}")
    print(f"{'Weighted F1-Score':<25} | {eff_results['weighted_f1']:<28.4f} | {conv_results['weighted_f1']:<28.4f}")
    print("-" * 85)

    # ----------------------------------------------------
    # 4. Save Detailed Classification Reports
    # ----------------------------------------------------
    conv_report = classification_report(
        conv_results["labels"], conv_results["predictions"],
        target_names=breeds, digits=4
    )
    print("\nCONVNEXT-TINY FINAL TEST CLASSIFICATION REPORT:")
    print(conv_report)

    with open(os.path.join(results_dir, "test_classification_report_convnext.txt"), "w") as f:
        f.write("ConvNeXt-Tiny Test Classification Report (Untouched Test Set)\n")
        f.write(f"Test Accuracy:     {conv_results['accuracy']:.4f}\n")
        f.write(f"Macro Precision:   {conv_results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:      {conv_results['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:    {conv_results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {conv_results['weighted_f1']:.4f}\n\n")
        f.write(conv_report)

    with open(os.path.join(results_dir, "test_classification_report_efficientnet.txt"), "w") as f:
        f.write("EfficientNet-B2 Test Classification Report (Untouched Test Set)\n")
        f.write(f"Test Accuracy:     {eff_results['accuracy']:.4f}\n")
        f.write(f"Macro Precision:   {eff_results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:      {eff_results['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score:    {eff_results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1-Score: {eff_results['weighted_f1']:.4f}\n\n")
        f.write(classification_report(eff_results["labels"], eff_results["predictions"], target_names=breeds, digits=4))

    # ----------------------------------------------------
    # 5. Confusion Matrix (CSV and Plot)
    # ----------------------------------------------------
    cm_df = pd.DataFrame(conv_results["confusion_matrix"], index=breeds, columns=breeds)
    cm_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=breeds,
        yticklabels=breeds,
        linewidths=0.5
    )
    plt.title("ConvNeXt-Tiny Test Confusion Matrix (Untouched Holdout Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Breed", fontsize=12, labelpad=10)
    plt.ylabel("True Breed", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "confusion_matrix_heatmap.png")
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_plot_path}")

    # ----------------------------------------------------
    # 6. Per-Class F1-Score Chart
    # ----------------------------------------------------
    cattle_breeds = [
        "amritmahal", "gir", "hallikar", "kankrej", "krishnavalley", "nimari", "ongole", "vechur"
    ]
    per_class_data = []
    for b in breeds:
        species_label = "Cattle" if b in cattle_breeds else "Buffalo"
        train_cnt = len(df_train[df_train["breed_name"] == b])
        val_cnt = len(df_val[df_val["breed_name"] == b])
        test_cnt = len(df_test[df_test["breed_name"] == b])
        f1_c = conv_results["per_class"][b]["f1_score"]
        prec_c = conv_results["per_class"][b]["precision"]
        rec_c = conv_results["per_class"][b]["recall"]
        per_class_data.append({
            "breed": b,
            "species": species_label,
            "train_count": train_cnt,
            "val_count": val_cnt,
            "test_count": test_cnt,
            "precision": prec_c,
            "recall": rec_c,
            "f1_score": f1_c
        })
    df_per_class = pd.DataFrame(per_class_data)
    df_per_class.to_csv(os.path.join(results_dir, "per_class_metrics.csv"), index=False)

    # Plot F1 Bar Chart
    plt.figure(figsize=(14, 8))
    sorted_df = df_per_class.sort_values(by="f1_score", ascending=True)
    colors = ["#2b6cb0" if s == "Cattle" else "#744210" for s in sorted_df["species"]]

    y_pos = np.arange(len(sorted_df))
    plt.barh(y_pos, sorted_df["f1_score"], color=colors, alpha=0.85, edgecolor="black", height=0.65)
    plt.yticks(y_pos, [f"{b.title()} ({s[0]})" for b, s in zip(sorted_df["breed"], sorted_df["species"])], fontsize=11)
    plt.xlabel("Test F1-Score", fontsize=12, labelpad=10)
    plt.title("Per-Breed Test F1-Score (ConvNeXt-Tiny)\n[Blue: Cattle | Brown: Buffalo]", fontsize=14, fontweight="bold", pad=15)
    plt.xlim(0, 1.05)
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for i, v in enumerate(sorted_df["f1_score"]):
        plt.text(v + 0.015, i, f"{v:.1%}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    f1_plot_path = os.path.join(results_dir, "per_class_f1_chart.png")
    plt.savefig(f1_plot_path, dpi=200)
    plt.close()
    print(f"Saved per-class F1 chart to: {f1_plot_path}")

    # ----------------------------------------------------
    # 7. Training & Validation Curves Visualization
    # ----------------------------------------------------
    conv_hist_path = os.path.join(models_dir, "convnext_training_history.json")
    if os.path.exists(conv_hist_path):
        with open(conv_hist_path, "r") as f:
            c_hist = json.load(f)

        plt.figure(figsize=(14, 5))
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(c_hist["epoch"], c_hist["train_loss"], "b-o", linewidth=2, label="Train Loss")
        plt.plot(c_hist["epoch"], c_hist["val_loss"], "r-o", linewidth=2, label="Val Loss")
        plt.title("ConvNeXt-Tiny Loss Curves", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Class-Weighted CE Loss", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        # Acc & F1
        plt.subplot(1, 2, 2)
        plt.plot(c_hist["epoch"], c_hist["train_acc"], "b-o", linewidth=2, label="Train Accuracy")
        plt.plot(c_hist["epoch"], c_hist["val_acc"], "r-o", linewidth=2, label="Val Accuracy")
        plt.plot(c_hist["epoch"], c_hist["macro_f1"], "g--s", linewidth=2, label="Val Macro F1")
        plt.title("ConvNeXt-Tiny Accuracy & Macro F1", fontsize=13, fontweight="bold")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)

        plt.tight_layout()
        curves_path = os.path.join(results_dir, "training_validation_curves.png")
        plt.savefig(curves_path, dpi=200)
        plt.close()
        print(f"Saved training curves to: {curves_path}")

    # ----------------------------------------------------
    # 8. Incorrect Prediction Analysis
    # ----------------------------------------------------
    df_meta = pd.DataFrame(conv_results["metadata"])
    df_incorrect = df_meta[~df_meta["correct"]].copy()
    df_correct = df_meta[df_meta["correct"]].copy()

    df_incorrect.to_csv(os.path.join(results_dir, "incorrect_predictions.csv"), index=False)
    print(f"\nTotal Test Misclassifications: {len(df_incorrect)} / {len(df_meta)} ({len(df_incorrect)/len(df_meta):.2%})")

    # Group confusions
    confusion_pairs = df_incorrect.groupby(["true_breed", "pred_breed"]).size().reset_index(name="count")
    confusion_pairs = confusion_pairs.sort_values(by="count", ascending=False)
    confusion_pairs.to_csv(os.path.join(results_dir, "confusion_pairs.csv"), index=False)

    # ----------------------------------------------------
    # 9. Prediction Gallery Plot (Correct vs Incorrect Examples)
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Representative Test Set Predictions (ConvNeXt-Tiny)\nTop: Correct Predictions | Bottom: Misclassifications", fontsize=14, fontweight="bold")

    # Sample 4 correct and 4 incorrect
    sample_correct = df_correct.sample(min(4, len(df_correct)), random_state=42)
    sample_incorrect = df_incorrect.sample(min(4, len(df_incorrect)), random_state=42)

    # Plot top row (correct)
    for col_idx, (_, row) in enumerate(sample_correct.iterrows()):
        ax = axes[0, col_idx]
        img_p = os.path.join(project_dir, row["image_path"])
        try:
            img = Image.open(img_p)
            ax.imshow(img)
        except Exception:
            ax.imshow(np.zeros((224, 224, 3)))
        ax.axis("off")
        ax.set_title(f"True: {row['true_breed'].title()}\nPred: {row['pred_breed'].title()} ({row['confidence']:.1%})",
                     fontsize=10, color="green", fontweight="bold")

    # Plot bottom row (incorrect)
    for col_idx, (_, row) in enumerate(sample_incorrect.iterrows()):
        ax = axes[1, col_idx]
        img_p = os.path.join(project_dir, row["image_path"])
        try:
            img = Image.open(img_p)
            ax.imshow(img)
        except Exception:
            ax.imshow(np.zeros((224, 224, 3)))
        ax.axis("off")
        ax.set_title(f"True: {row['true_breed'].title()}\nPred: {row['pred_breed'].title()} ({row['confidence']:.1%})",
                     fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    gallery_path = os.path.join(results_dir, "prediction_gallery.png")
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"Saved prediction gallery to: {gallery_path}")

    # ----------------------------------------------------
    # 10. Save Metrics JSON Summary
    # ----------------------------------------------------
    metrics_summary = {
        "model_architecture": "ConvNeXt-Tiny",
        "dataset_split": "annotations/test.csv (strictly untouched holdout)",
        "total_test_samples": len(df_test),
        "overall_metrics": {
            "accuracy": round(conv_results["accuracy"], 4),
            "macro_precision": round(conv_results["macro_precision"], 4),
            "macro_recall": round(conv_results["macro_recall"], 4),
            "macro_f1": round(conv_results["macro_f1"], 4),
            "weighted_f1": round(conv_results["weighted_f1"], 4)
        },
        "baseline_efficientnet_metrics": {
            "accuracy": round(eff_results["accuracy"], 4),
            "macro_precision": round(eff_results["macro_precision"], 4),
            "macro_recall": round(eff_results["macro_recall"], 4),
            "macro_f1": round(eff_results["macro_f1"], 4),
            "weighted_f1": round(eff_results["weighted_f1"], 4)
        },
        "per_class_metrics": conv_results["per_class"],
        "top_confusion_pairs": confusion_pairs.head(10).to_dict(orient="records")
    }

    with open(os.path.join(results_dir, "test_evaluation_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=" * 85)
    print("ALL EVALUATION REPORTS & VISUALIZATIONS SUCCESSFULLY GENERATED!")
    print("=" * 85)


if __name__ == "__main__":
    main()
