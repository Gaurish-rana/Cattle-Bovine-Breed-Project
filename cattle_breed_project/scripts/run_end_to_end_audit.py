#!/usr/bin/env python3
"""
Comprehensive End-to-End System Audit for Cattle & Buffalo Breed Classification System.
Verifies Dataset, Model Checkpoints, Backend API, and Frontend Build.
"""

import os
import sys
import io
import json
import torch
import pandas as pd
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
import app as backend_app

def run_audit():
    print("=" * 85)
    print("END-TO-END SYSTEM AUDIT: CATTLE BREED CLASSIFICATION PROJECT")
    print("=" * 85)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    root_dir = os.path.abspath(os.path.join(project_dir, ".."))

    passed_checks = 0
    total_checks = 0

    def check(name, condition, details=""):
        nonlocal passed_checks, total_checks
        total_checks += 1
        if condition:
            passed_checks += 1
            print(f"  [PASS] {name} {details}")
        else:
            print(f"  [FAIL] {name} {details}")

    # =========================================================================
    # 1. DATASET INTEGRITY AUDIT
    # =========================================================================
    print("\n--- 1. DATASET INTEGRITY AUDIT ---")
    raw_dir = os.path.join(project_dir, "data", "raw")
    cleaned_dir = os.path.join(project_dir, "data", "cleaned")
    buffalo_clean = os.path.join(cleaned_dir, "BUFFALO")
    cattle_clean = os.path.join(cleaned_dir, "CATTLE")

    check("Raw dataset preserved untouched", os.path.exists(raw_dir))
    check("Cleaned dataset directory exists", os.path.exists(cleaned_dir))

    buffalo_folders = os.listdir(buffalo_clean) if os.path.exists(buffalo_clean) else []
    cattle_folders = os.listdir(cattle_clean) if os.path.exists(cattle_clean) else []
    check("9 Buffalo breed folders exist", len(buffalo_folders) == 9, f"Found {len(buffalo_folders)}")
    check("8 Cattle breed folders exist", len(cattle_folders) == 8, f"Found {len(cattle_folders)}")

    # Total cleaned images
    clean_images = []
    for root, _, files in os.walk(cleaned_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                clean_images.append(os.path.join(root, f))
    check("Cleaned dataset contains exactly 1,820 valid images", len(clean_images) == 1820, f"Found {len(clean_images)}")

    # Check Splits
    train_csv = os.path.join(project_dir, "annotations", "train.csv")
    val_csv = os.path.join(project_dir, "annotations", "val.csv")
    test_csv = os.path.join(project_dir, "annotations", "test.csv")
    labels_csv = os.path.join(project_dir, "annotations", "breed_labels.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    df_labels = pd.read_csv(labels_csv)

    check("Train split exists with 1,278 images (70.2%)", len(df_train) == 1278, f"Found {len(df_train)}")
    check("Val split exists with 271 images (14.9%)", len(df_val) == 271, f"Found {len(df_val)}")
    check("Test split exists with 271 images (14.9%)", len(df_test) == 271, f"Found {len(df_test)}")
    check("Total split images equals 1,820", len(df_train) + len(df_val) + len(df_test) == 1820)
    check("All 17 breeds present in labels CSV", len(df_labels) == 17)

    # Cross-split leakage check
    set_train = set(df_train["image_path"].str.lower())
    set_val = set(df_val["image_path"].str.lower())
    set_test = set(df_test["image_path"].str.lower())

    train_val_leak = len(set_train.intersection(set_val))
    train_test_leak = len(set_train.intersection(set_test))
    val_test_leak = len(set_val.intersection(set_test))
    check("Zero train-val data leakage", train_val_leak == 0, f"Overlaps: {train_val_leak}")
    check("Zero train-test data leakage", train_test_leak == 0, f"Overlaps: {train_test_leak}")
    check("Zero val-test data leakage", val_test_leak == 0, f"Overlaps: {val_test_leak}")

    # =========================================================================
    # 2. MODEL ARCHITECTURE & CHECKPOINT AUDIT
    # =========================================================================
    print("\n--- 2. MODEL & CHECKPOINT AUDIT ---")
    convnext_ckpt = os.path.join(project_dir, "models", "convnext_tiny_best.pth")
    effnet_ckpt = os.path.join(project_dir, "models", "Best_model_final.pth")

    check("ConvNeXt-Tiny checkpoint exists", os.path.exists(convnext_ckpt), f"Size: {os.path.getsize(convnext_ckpt)/1e6:.1f} MB")
    check("EfficientNet-B2 checkpoint exists", os.path.exists(effnet_ckpt), f"Size: {os.path.getsize(effnet_ckpt)/1e6:.1f} MB")

    chk_conv = torch.load(convnext_ckpt, map_location="cpu")
    check("Checkpoint specifies ConvNeXt-Tiny", "convnext" in chk_conv.get("architecture", "").lower())
    check("Checkpoint configured for 17 classes", chk_conv.get("num_classes") == 17)
    check("Checkpoint contains complete label2idx", len(chk_conv.get("label2idx", {})) == 17)

    # Initialize model instance and test forward pass
    conv_model = backend_app.ConvNeXtClassifier(num_classes=17)
    conv_model.load_state_dict(chk_conv["model_state_dict"])
    conv_model.eval()
    with torch.no_grad():
        dummy_out = conv_model(torch.randn(1, 3, 224, 224))
    check("ConvNeXt forward pass produces 17-class logits", dummy_out.shape == (1, 17))

    # =========================================================================
    # 3. BACKEND API AUDIT
    # =========================================================================
    print("\n--- 3. BACKEND API AUDIT ---")
    client = backend_app.app.test_client()

    # /health
    res_h = client.get("/health")
    data_h = res_h.get_json()
    check("/health endpoint returns 200", res_h.status_code == 200)
    check("/health reports model_loaded=True", data_h.get("model_loaded") is True)
    check("/health reports active architecture", data_h.get("architecture") == "ConvNeXt-Tiny")

    # /breeds
    res_b = client.get("/breeds")
    data_b = res_b.get_json()
    check("/breeds endpoint returns 200", res_b.status_code == 200)
    check("/breeds returns all 17 breeds", data_b.get("count") == 17)
    first_b = data_b.get("breeds", [])[0]
    check("Breeds contain verified accession numbers", "accession_number" in first_b and bool(first_b["accession_number"]))

    # /predict with actual image (Gir)
    gir_dir = os.path.join(cattle_clean, "INDIA_CATTLE_0400_GIR_03007")
    gir_img = os.path.join(gir_dir, os.listdir(gir_dir)[0])
    with open(gir_img, "rb") as f:
        res_p = client.post("/predict", data={"image": (io.BytesIO(f.read()), "gir.jpg")}, content_type="multipart/form-data")
    data_p = res_p.get_json()
    check("/predict returns 200 on valid image", res_p.status_code == 200)
    check("/predict returns model_loaded=True", data_p.get("model_loaded") is True)
    check("/predict top match is non-empty", len(data_p.get("predictions", [])) > 0)
    check("/predict returns full ranked probabilities (17 classes)", len(data_p.get("all_probabilities", [])) == 17)
    check("Prediction includes species disambiguation", "species" in data_p["predictions"][0])
    check("Prediction includes confidence score", 0.0 <= data_p["predictions"][0]["confidence"] <= 1.0)

    # Error handling
    res_err1 = client.post("/predict", data={}, content_type="multipart/form-data")
    check("Missing file returns HTTP 400", res_err1.status_code == 400)

    res_err2 = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "fake.jpg")}, content_type="multipart/form-data")
    check("Corrupted file returns HTTP 400", res_err2.status_code == 400)

    res_err3 = client.post("/predict", data={"image": (io.BytesIO(b"some text"), "file.exe")}, content_type="multipart/form-data")
    check("Unsupported extension returns HTTP 400", res_err3.status_code == 400)

    # =========================================================================
    # 4. FRONTEND BUILD AUDIT
    # =========================================================================
    print("\n--- 4. FRONTEND BUILD AUDIT ---")
    frontend_dist = os.path.join(project_dir, "cattle-sense-main", "dist")
    dist_index = os.path.join(frontend_dist, "index.html")
    check("Frontend production bundle exists (dist/)", os.path.exists(frontend_dist))
    check("Frontend index.html built successfully", os.path.exists(dist_index))

    # Check gitignore
    gitignore_file = os.path.join(root_dir, ".gitignore")
    check(".gitignore exists in root directory", os.path.exists(gitignore_file))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 85)
    print(f"AUDIT SUMMARY: {passed_checks} / {total_checks} checks passed ({(passed_checks/total_checks):.1%})")
    print("=" * 85)

    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_audit()
    sys.exit(0 if success else 1)
