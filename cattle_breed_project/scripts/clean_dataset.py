#!/usr/bin/env python3
"""
Clean and Standardize Dataset Pipeline for Cattle & Buffalo Breed Classification.

This script:
1. Inspects data/raw for all 17 indigenous breeds (9 Buffalo, 8 Cattle).
2. Detects corrupt images, unreadable formats, and invalid dimensions.
3. Detects exact duplicates (within-breed redundant copies and cross-breed contradictory labels).
4. Converts all valid images into standardized RGB JPEGs (preserving transparency via white-background alpha compositing).
5. Retains all valid majority-class images (avoids naive downsampling/deletion).
6. Generates detailed audit reports:
   - annotations/rejected_images.csv
   - annotations/dataset_summary.csv
7. Ensures folder names and labels match annotations/breed_labels.csv.
"""

import os
import sys
import shutil
import hashlib
from collections import defaultdict
from PIL import Image
import pandas as pd

# Fix console encoding on Windows to avoid UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def compute_file_hash(filepath, block_size=65536):
    """Compute MD5 hash of a file for exact duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def convert_to_rgb(img):
    """
    Safely convert any image (RGBA, Palette, Grayscale, etc.) to RGB.
    Handles transparency by compositing onto a clean white background.
    """
    if img.mode == "RGB":
        return img.copy()
    
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(background, rgba)
        return composite.convert("RGB")
    
    return img.convert("RGB")


def clean_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))
    
    raw_dir = os.path.join(project_dir, "data", "raw")
    cleaned_dir = os.path.join(project_dir, "data", "cleaned")
    annotations_dir = os.path.join(project_dir, "annotations")
    labels_csv = os.path.join(annotations_dir, "breed_labels.csv")
    
    rejected_csv = os.path.join(annotations_dir, "rejected_images.csv")
    summary_csv = os.path.join(annotations_dir, "dataset_summary.csv")
    
    print("=" * 80)
    print("CATTLE & BUFFALO BREED DATASET CLEANING PIPELINE")
    print("=" * 80)
    print(f"Raw Data Directory:      {raw_dir}")
    print(f"Cleaned Data Directory:  {cleaned_dir}")
    print(f"Breed Labels Metadata:   {labels_csv}")
    print("-" * 80)
    
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"breed_labels.csv not found at {labels_csv}")
    
    df_labels = pd.read_csv(labels_csv)
    print(f"Loaded {len(df_labels)} target breeds from breed_labels.csv")
    
    # ----------------------------------------------------
    # Step 1: Scan all raw image files and map by hash
    # ----------------------------------------------------
    print("\n[Step 1/4] Scanning raw dataset files and computing content hashes...")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.jfif')
    all_raw_images = []
    hash_to_instances = defaultdict(list)
    
    for _, row in df_labels.iterrows():
        folder_name = str(row["folder_name"]).strip()
        breed_name = str(row["breed_name"]).strip().lower()
        species = str(row["species"]).strip().lower()
        
        # Check both lower and upper case directory names for cross-platform safety
        folder_path = None
        for sp_variant in [species, species.upper(), species.title()]:
            candidate = os.path.join(raw_dir, sp_variant, folder_name)
            if os.path.isdir(candidate):
                folder_path = candidate
                break
                
        if folder_path is None:
            print(f" [WARNING] Missing raw folder for breed: {breed_name} at {folder_name}")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        for fname in sorted(files):
            fpath = os.path.join(folder_path, fname)
            fsize = os.path.getsize(fpath)
            fhash = compute_file_hash(fpath)
            
            item = {
                "species": species,
                "folder_name": folder_name,
                "breed_name": breed_name,
                "file_name": fname,
                "file_path": fpath,
                "file_size": fsize,
                "hash": fhash
            }
            all_raw_images.append(item)
            hash_to_instances[fhash].append(item)
            
    print(f"  Total raw images indexed: {len(all_raw_images)}")
    print(f"  Total unique image hashes: {len(hash_to_instances)}")
    
    # ----------------------------------------------------
    # Step 2: Identify duplicates and conflicts
    # ----------------------------------------------------
    print("\n[Step 2/4] Identifying duplicate sets and cross-breed contradictory labels...")
    
    cross_breed_conflicts = set()   # Hash is associated with multiple distinct breeds
    within_breed_dups = set()       # Exact duplicate within the same breed
    kept_images = []
    rejected_records = []
    
    for fhash, instances in hash_to_instances.items():
        distinct_breeds = set(inst["breed_name"] for inst in instances)
        if len(distinct_breeds) > 1:
            # Contradictory labeling across breeds: reject all instances to avoid noisy ground truth
            breeds_str = ", ".join(sorted(distinct_breeds))
            for inst in instances:
                cross_breed_conflicts.add(inst["file_path"])
                rejected_records.append({
                    "image_path": os.path.relpath(inst["file_path"], project_dir),
                    "file_name": inst["file_name"],
                    "breed_name": inst["breed_name"],
                    "species": inst["species"],
                    "file_size_bytes": inst["file_size"],
                    "md5_hash": fhash,
                    "rejection_reason": f"cross_breed_contradictory_duplicate (conflicts with: {breeds_str})"
                })
        else:
            # All instances share the same breed. Keep the primary (first), reject redundant copies
            primary = instances[0]
            for redundant in instances[1:]:
                within_breed_dups.add(redundant["file_path"])
                rejected_records.append({
                    "image_path": os.path.relpath(redundant["file_path"], project_dir),
                    "file_name": redundant["file_name"],
                    "breed_name": redundant["breed_name"],
                    "species": redundant["species"],
                    "file_size_bytes": redundant["file_size"],
                    "md5_hash": fhash,
                    "rejection_reason": f"within_breed_redundant_duplicate (primary: {primary['file_name']})"
                })
                
    print(f"  Cross-breed contradictory files rejected: {len(cross_breed_conflicts)}")
    print(f"  Within-breed redundant duplicates rejected: {len(within_breed_dups)}")
    
    # ----------------------------------------------------
    # Step 3: Verify image integrity and copy to cleaned
    # ----------------------------------------------------
    print("\n[Step 3/4] Validating image integrity and standardizing to RGB in data/cleaned...")
    
    # Reset / recreate data/cleaned structure cleanly
    if os.path.exists(cleaned_dir):
        shutil.rmtree(cleaned_dir)
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Create species and breed directories
    for _, row in df_labels.iterrows():
        sp_dir = os.path.join(cleaned_dir, str(row["species"]).strip().upper())
        br_dir = os.path.join(sp_dir, str(row["folder_name"]).strip())
        os.makedirs(br_dir, exist_ok=True)
        
    breed_counts_raw = defaultdict(int)
    breed_counts_clean = defaultdict(int)
    breed_rejected_cross = defaultdict(int)
    breed_rejected_within = defaultdict(int)
    breed_rejected_corrupt = defaultdict(int)
    
    for item in all_raw_images:
        breed_name = item["breed_name"]
        species = item["species"]
        fpath = item["file_path"]
        fname = item["file_name"]
        folder_name = item["folder_name"]
        fhash = item["hash"]
        fsize = item["file_size"]
        
        breed_counts_raw[breed_name] += 1
        
        if fpath in cross_breed_conflicts:
            breed_rejected_cross[breed_name] += 1
            continue
            
        if fpath in within_breed_dups:
            breed_rejected_within[breed_name] += 1
            continue
            
        # File integrity and decodability validation
        if fsize < 500:
            breed_rejected_corrupt[breed_name] += 1
            rejected_records.append({
                "image_path": os.path.relpath(fpath, project_dir),
                "file_name": fname,
                "breed_name": breed_name,
                "species": species,
                "file_size_bytes": fsize,
                "md5_hash": fhash,
                "rejection_reason": "file_too_small_under_500_bytes"
            })
            continue
            
        try:
            with Image.open(fpath) as img:
                img.verify()
            
            with Image.open(fpath) as img:
                w, h = img.size
                if w < 50 or h < 50:
                    breed_rejected_corrupt[breed_name] += 1
                    rejected_records.append({
                        "image_path": os.path.relpath(fpath, project_dir),
                        "file_name": fname,
                        "breed_name": breed_name,
                        "species": species,
                        "file_size_bytes": fsize,
                        "md5_hash": fhash,
                        "rejection_reason": f"dimensions_too_small ({w}x{h})"
                    })
                    continue
                    
                rgb_img = convert_to_rgb(img)
                
            # Target output path: data/cleaned/<SPECIES_UPPER>/<folder_name>/<clean_name>.jpg
            clean_name = os.path.splitext(fname)[0] + ".jpg"
            dest_folder = os.path.join(cleaned_dir, species.upper(), folder_name)
            dest_path = os.path.join(dest_folder, clean_name)
            
            # Save standardized clean JPEG with 95% quality and optimized Huffman tables
            rgb_img.save(dest_path, "JPEG", quality=95, optimize=True)
            breed_counts_clean[breed_name] += 1
            
        except Exception as e:
            breed_rejected_corrupt[breed_name] += 1
            rejected_records.append({
                "image_path": os.path.relpath(fpath, project_dir),
                "file_name": fname,
                "breed_name": breed_name,
                "species": species,
                "file_size_bytes": fsize,
                "md5_hash": fhash,
                "rejection_reason": f"corrupt_or_unreadable: {str(e)}"
            })
            
    # ----------------------------------------------------
    # Step 4: Generate reports and dataset summary
    # ----------------------------------------------------
    print("\n[Step 4/4] Writing audit logs and dataset summaries...")
    
    df_rejected = pd.DataFrame(rejected_records)
    df_rejected.to_csv(rejected_csv, index=False)
    print(f"  Saved rejection audit log to: {rejected_csv} ({len(df_rejected)} records)")
    
    summary_rows = []
    for _, row in df_labels.iterrows():
        b_name = str(row["breed_name"]).strip().lower()
        f_name = str(row["folder_name"]).strip()
        sp = str(row["species"]).strip().lower()
        
        raw_count = breed_counts_raw[b_name]
        clean_count = breed_counts_clean[b_name]
        cross_rej = breed_rejected_cross[b_name]
        within_rej = breed_rejected_within[b_name]
        corrupt_rej = breed_rejected_corrupt[b_name]
        retention = (clean_count / raw_count * 100.0) if raw_count > 0 else 0.0
        
        summary_rows.append({
            "species": sp,
            "breed_name": b_name,
            "folder_name": f_name,
            "raw_count": raw_count,
            "cross_conflicts_rejected": cross_rej,
            "within_dups_rejected": within_rej,
            "corrupt_rejected": corrupt_rej,
            "cleaned_valid_count": clean_count,
            "retention_rate_pct": round(retention, 2)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv, index=False)
    print(f"  Saved dataset summary to:     {summary_csv}")
    
    print("\n" + "=" * 90)
    print(f"{'SPECIES':<10} | {'BREED NAME':<15} | {'RAW':<6} | {'CROSS REJ':<10} | {'DUP REJ':<8} | {'CORRUPT':<8} | {'CLEANED':<8} | {'RETENTION'}")
    print("-" * 90)
    for r in summary_rows:
        print(f"{r['species'].upper():<10} | {r['breed_name']:<15} | {r['raw_count']:<6} | {r['cross_conflicts_rejected']:<10} | {r['within_dups_rejected']:<8} | {r['corrupt_rejected']:<8} | {r['cleaned_valid_count']:<8} | {r['retention_rate_pct']}%")
    print("-" * 90)
    total_raw = sum(r['raw_count'] for r in summary_rows)
    total_clean = sum(r['cleaned_valid_count'] for r in summary_rows)
    print(f"{'TOTAL':<10} | {'17 BREEDS':<15} | {total_raw:<6} | {len(cross_breed_conflicts):<10} | {len(within_breed_dups):<8} | {sum(r['corrupt_rejected'] for r in summary_rows):<8} | {total_clean:<8} | {round(total_clean/total_raw*100, 2)}%")
    print("=" * 90)
    
    # Check minimum threshold
    min_count = min(r['cleaned_valid_count'] for r in summary_rows)
    min_breed = min(summary_rows, key=lambda x: x['cleaned_valid_count'])
    print(f"\nMinimum images per breed: {min_count} ({min_breed['breed_name']})")
    print(f"Maximum images per breed: {max(r['cleaned_valid_count'] for r in summary_rows)}")
    
    if min_count < 20:
        print(f"\n[ALERT] Low image count warning: Breed '{min_breed['breed_name']}' has only {min_count} images.")
    else:
        print(f"\n[SUCCESS] All 17 breeds have sufficient valid images (>= {min_count} samples) for training.")
        
    return df_summary


if __name__ == "__main__":
    clean_dataset()
