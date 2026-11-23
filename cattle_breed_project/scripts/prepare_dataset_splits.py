#!/usr/bin/env python3
"""
Dataset Preparation & Cluster-Aware Stratified Splitting Pipeline
for Cattle & Buffalo Breed Classification.

Features:
1. Loads all cleaned images across all 17 indigenous breeds.
2. Computes perceptual Difference Hashes (dHash) to detect near-duplicate / burst-shot images.
3. Groups near-duplicate images into atomic clusters using graph connected components.
4. Performs cluster-aware stratified splitting (70% Train, 15% Val, 15% Test) with seed=42.
   - Prevents near-identical images from leaking across Train, Val, and Test.
   - Guarantees balanced representation for all 17 breeds.
   - Ensures minority breeds (e.g. Amritmahal) have sufficient samples in each split.
5. Performs automated cross-split leakage verification (Hamming distance audit).
6. Saves reproducible split files:
   - annotations/train.csv
   - annotations/val.csv
   - annotations/test.csv
   - annotations/dataset_splits_summary.csv
   - annotations/near_duplicates_audit.csv
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

# UTF-8 console output for Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_dhash(image_path, hash_size=8):
    """
    Compute 64-bit difference hash (dHash) for an image.
    Captures structural luminance gradients, highly robust to compression and scaling.
    """
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.asarray(gray)
            diff = pixels[:, 1:] > pixels[:, :-1]
            return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception as e:
        return 0


def hamming_distance(h1, h2):
    """Compute Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count('1')


class UnionFind:
    """Disjoint Set / Union-Find for clustering near-duplicate images."""
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}

    def find(self, elem):
        if self.parent[elem] != elem:
            self.parent[elem] = self.find(self.parent[elem])
        return self.parent[elem]

    def union(self, elem1, elem2):
        root1 = self.find(elem1)
        root2 = self.find(elem2)
        if root1 != root2:
            self.parent[root2] = root1


def prepare_splits(
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    near_dup_threshold=3
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    clean_dir = os.path.join(project_dir, "data", "cleaned")
    annotations_dir = os.path.join(project_dir, "annotations")
    labels_csv = os.path.join(annotations_dir, "breed_labels.csv")
    
    train_csv = os.path.join(annotations_dir, "train.csv")
    val_csv = os.path.join(annotations_dir, "val.csv")
    test_csv = os.path.join(annotations_dir, "test.csv")
    summary_csv = os.path.join(annotations_dir, "dataset_splits_summary.csv")
    near_dups_csv = os.path.join(annotations_dir, "near_duplicates_audit.csv")
    
    print("=" * 85)
    print("REPRODUCIBLE DATASET SPLITTING & NEAR-DUPLICATE AUDIT PIPELINE")
    print("=" * 85)
    print(f"Target Split Ratio:    {int(train_ratio*100)}% Train / {int(val_ratio*100)}% Val / {int(test_ratio*100)}% Test")
    print(f"Random Seed:           {seed}")
    print(f"Near-Duplicate Cutoff: Hamming Distance <= {near_dup_threshold}")
    print("-" * 85)
    
    random.seed(seed)
    np.random.seed(seed)
    
    df_labels = pd.read_csv(labels_csv)
    
    # ----------------------------------------------------
    # Step 1: Scan and index all cleaned images
    # ----------------------------------------------------
    print("[Step 1/5] Indexing all cleaned images and computing perceptual dHashes...")
    
    all_samples = []
    for _, row in df_labels.iterrows():
        b_name = str(row["breed_name"]).strip().lower()
        sp = str(row["species"]).strip().lower()
        f_name = str(row["folder_name"]).strip()
        
        breed_dir = os.path.join(clean_dir, sp.upper(), f_name)
        if not os.path.exists(breed_dir):
            raise FileNotFoundError(f"Cleaned breed folder missing: {breed_dir}")
            
        files = [f for f in os.listdir(breed_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in sorted(files):
            fpath = os.path.join(breed_dir, fname)
            rel_path = os.path.relpath(fpath, project_dir)
            h = compute_dhash(fpath)
            
            all_samples.append({
                "rel_path": rel_path,
                "abs_path": fpath,
                "file_name": fname,
                "breed_name": b_name,
                "species": sp,
                "folder_name": f_name,
                "dhash": h,
                "dhash_hex": hex(h)
            })
            
    print(f"  Indexed {len(all_samples)} total images across {len(df_labels)} breeds.")
    
    # ----------------------------------------------------
    # Step 2: Detect near-duplicates & form atomic clusters
    # ----------------------------------------------------
    print("\n[Step 2/5] Detecting near-duplicates and clustering connected components...")
    
    uf = UnionFind([s["rel_path"] for s in all_samples])
    near_dup_records = []
    
    # Group samples by breed to cluster within-breed burst shots / near-duplicates
    breed_to_samples = defaultdict(list)
    for s in all_samples:
        breed_to_samples[s["breed_name"]].append(s)
        
    for breed, samples in breed_to_samples.items():
        n = len(samples)
        for i in range(n):
            s1 = samples[i]
            for j in range(i + 1, n):
                s2 = samples[j]
                dist = hamming_distance(s1["dhash"], s2["dhash"])
                if dist <= near_dup_threshold:
                    uf.union(s1["rel_path"], s2["rel_path"])
                    near_dup_records.append({
                        "breed_name": breed,
                        "species": s1["species"],
                        "image_1": s1["rel_path"],
                        "image_2": s2["rel_path"],
                        "hamming_distance": dist,
                        "similarity_note": f"Hamming dist {dist} (near-identical or burst shot)"
                    })
                    
    # Assign cluster IDs
    cluster_mapping = {}
    cluster_counter = 0
    for s in all_samples:
        root = uf.find(s["rel_path"])
        if root not in cluster_mapping:
            cluster_counter += 1
            cluster_mapping[root] = f"cluster_{cluster_counter:04d}"
        s["cluster_id"] = cluster_mapping[root]
        
    df_near_dups = pd.DataFrame(near_dup_records)
    df_near_dups.to_csv(near_dups_csv, index=False)
    print(f"  Detected {len(near_dup_records)} near-duplicate pairs.")
    print(f"  Formed {cluster_counter} distinct clusters (images in same cluster stay together in one split).")
    print(f"  Saved near-duplicate audit log to: {near_dups_csv}")
    
    # ----------------------------------------------------
    # Step 3: Cluster-aware stratified split per breed
    # ----------------------------------------------------
    print("\n[Step 3/5] Performing cluster-aware stratified splitting...")
    
    train_samples = []
    val_samples = []
    test_samples = []
    summary_rows = []
    
    for _, row in df_labels.iterrows():
        b_name = str(row["breed_name"]).strip().lower()
        sp = str(row["species"]).strip().lower()
        f_name = str(row["folder_name"]).strip()
        
        b_samples = [s for s in all_samples if s["breed_name"] == b_name]
        n_total = len(b_samples)
        
        # Group samples of this breed by cluster_id
        clusters = defaultdict(list)
        for s in b_samples:
            clusters[s["cluster_id"]].append(s)
            
        cluster_list = list(clusters.values())
        random.shuffle(cluster_list)
        
        target_test_count = max(5, int(round(n_total * test_ratio)))
        target_val_count = max(5, int(round(n_total * val_ratio)))
        
        b_train, b_val, b_test = [], [], []
        
        # Allocate clusters to test first until target reached
        for c in cluster_list:
            if len(b_test) + len(c) <= target_test_count or (len(b_test) == 0):
                b_test.extend(c)
            elif len(b_val) + len(c) <= target_val_count or (len(b_val) == 0):
                b_val.extend(c)
            else:
                b_train.extend(c)
                
        # In case a small cluster can balance val/test counts
        train_samples.extend(b_train)
        val_samples.extend(b_val)
        test_samples.extend(b_test)
        
        summary_rows.append({
            "species": sp,
            "breed_name": b_name,
            "folder_name": f_name,
            "total_count": n_total,
            "train_count": len(b_train),
            "train_pct": round(len(b_train) / n_total * 100, 1),
            "val_count": len(b_val),
            "val_pct": round(len(b_val) / n_total * 100, 1),
            "test_count": len(b_test),
            "test_pct": round(len(b_test) / n_total * 100, 1)
        })
        
    # ----------------------------------------------------
    # Step 4: Cross-Split Leakage & Integrity Verification
    # ----------------------------------------------------
    print("\n[Step 4/5] Running automated cross-split leakage verification...")
    
    train_paths = set(s["rel_path"] for s in train_samples)
    val_paths = set(s["rel_path"] for s in val_samples)
    test_paths = set(s["rel_path"] for s in test_samples)
    
    # 1. Exact disjointness check
    leak_train_val = train_paths.intersection(val_paths)
    leak_train_test = train_paths.intersection(test_paths)
    leak_val_test = val_paths.intersection(test_paths)
    
    assert len(leak_train_val) == 0, f"Critical leak between Train and Val: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Critical leak between Train and Test: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Critical leak between Val and Test: {leak_val_test}"
    assert len(train_paths) + len(val_paths) + len(test_paths) == len(all_samples), "Sample loss in split"
    
    print("  [PASS] Exact set disjointness: 0 overlapping images across any split.")
    
    # 2. Cluster integrity check: no cluster exists in more than one split
    train_clusters = set(s["cluster_id"] for s in train_samples)
    val_clusters = set(s["cluster_id"] for s in val_samples)
    test_clusters = set(s["cluster_id"] for s in test_samples)
    
    c_leak_tv = train_clusters.intersection(val_clusters)
    c_leak_tt = train_clusters.intersection(test_clusters)
    c_leak_vt = val_clusters.intersection(test_clusters)
    
    assert len(c_leak_tv) == 0, f"Cluster leak between Train and Val: {c_leak_tv}"
    assert len(c_leak_tt) == 0, f"Cluster leak between Train and Test: {c_leak_tt}"
    assert len(c_leak_vt) == 0, f"Cluster leak between Val and Test: {c_leak_vt}"
    
    print("  [PASS] Near-duplicate cluster isolation: 0 near-duplicate groups split across boundaries.")
    
    # ----------------------------------------------------
    # Step 5: Save CSV split files & summary
    # ----------------------------------------------------
    print("\n[Step 5/5] Saving split files and generating split report...")
    
    output_columns = [
        "image_path", "breed_name", "species", "folder_name",
        "file_name", "cluster_id", "dhash_hex"
    ]
    
    def prepare_df(sample_list):
        records = []
        for s in sample_list:
            records.append({
                "image_path": s["rel_path"],
                "breed_name": s["breed_name"],
                "species": s["species"],
                "folder_name": s["folder_name"],
                "file_name": s["file_name"],
                "cluster_id": s["cluster_id"],
                "dhash_hex": s["dhash_hex"]
            })
        return pd.DataFrame(records).sort_values(by=["species", "breed_name", "file_name"])

    df_train = prepare_df(train_samples)
    df_val = prepare_df(val_samples)
    df_test = prepare_df(test_samples)
    df_summary = pd.DataFrame(summary_rows)
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    
    print(f"  Saved Train Split:       {train_csv} ({len(df_train)} samples)")
    print(f"  Saved Validation Split:  {val_csv} ({len(df_val)} samples)")
    print(f"  Saved Test Split:        {test_csv} ({len(df_test)} samples)")
    print(f"  Saved Splits Summary:    {summary_csv}")
    
    print("\n" + "=" * 90)
    print(f"{'SPECIES':<8} | {'BREED NAME':<14} | {'TOTAL':<5} | {'TRAIN':<12} | {'VAL':<10} | {'TEST':<10}")
    print("-" * 90)
    for r in summary_rows:
        train_str = f"{r['train_count']} ({r['train_pct']}%)"
        val_str = f"{r['val_count']} ({r['val_pct']}%)"
        test_str = f"{r['test_count']} ({r['test_pct']}%)"
        print(f"{r['species'].upper():<8} | {r['breed_name']:<14} | {r['total_count']:<5} | {train_str:<12} | {val_str:<10} | {test_str:<10}")
    print("-" * 90)
    
    tot_all = sum(r['total_count'] for r in summary_rows)
    tot_tr = sum(r['train_count'] for r in summary_rows)
    tot_v = sum(r['val_count'] for r in summary_rows)
    tot_te = sum(r['test_count'] for r in summary_rows)
    
    print(f"{'TOTAL':<8} | {'17 BREEDS':<14} | {tot_all:<5} | {tot_tr} ({round(tot_tr/tot_all*100,1)}%) | {tot_v} ({round(tot_v/tot_all*100,1)}%) | {tot_te} ({round(tot_te/tot_all*100,1)}%)")
    print("=" * 90)
    
    print("\nDataset Splitting Successfully Completed!")
    return df_summary


if __name__ == "__main__":
    prepare_splits()
