# Indigenous Cattle & Buffalo Breed Classification System

An end-to-end Machine Learning and Full-Stack system for identifying **17 indigenous Indian cattle and buffalo breeds** (8 Cattle + 9 Buffalo) using modern deep learning architectures.

Developed based on ICAR-NBAGR (*National Bureau of Animal Genetic Resources*) taxonomies.

---

## 1. Project Overview & Architecture

### System Architecture
- **Deep Learning Model**: Pretrained **ConvNeXt-Tiny** fine-tuned on 17 indigenous bovine breeds.
- **Model Baseline**: Pretrained **EfficientNet-B2** with differential learning rates.
- **Backend API**: **Flask 3.0** with singleton model initialization, CORS, and strict payload validation.
- **Frontend Application**: **React 18 + Vite + TypeScript + Tailwind CSS (Shadcn/UI)** with drag-and-drop upload and HTML5 live webcam scanning.

---

## 2. Dataset Summary & Class Mappings

The dataset consists of **1,820 verified, deduplicated RGB images** split into **70% Train (1,278)**, **15% Validation (271)**, and **15% Test (271)** using perceptual hash clustering (dHash, Hamming distance $\le 3$) to prevent data leakage across splits.

### 17 Supported Indigenous Breeds

| Index | Breed Name | Species | State of Origin | ICAR-NBAGR Accession Number | Primary Utility |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **0** | **Amritmahal** | Cattle | Karnataka | `INDIA_CATTLE_0800_AMRITMAHAL_03001` | Cavalry Draught |
| **1** | **Banni** | Buffalo | Gujarat | `INDIA_BUFFALO_0400_BANNI_01011` | High-Yield Dairy |
| **2** | **Bargur** | Buffalo | Tamil Nadu | `INDIA_BUFFALO_1800_BARGUR_01015` | Hill Draught & Milk |
| **3** | **Bhadawari** | Buffalo | Uttar Pradesh | `INDIA_BUFFALO_2010_BHADAWARI_01003` | High-Fat Dairy (up to 13%) |
| **4** | **Gir** | Cattle | Gujarat | `INDIA_CATTLE_0400_GIR_03007` | Premium A2 Dairy |
| **5** | **Hallikar** | Cattle | Karnataka | `INDIA_CATTLE_0800_HALLIKAR_03008` | Fast Trotting Draught |
| **6** | **Jaffarabadi** | Buffalo | Gujarat | `INDIA_BUFFALO_0400_JAFFARABADI_01006` | Heavy Draught & Dairy |
| **7** | **Kankrej** | Cattle | Gujarat | `INDIA_CATTLE_0417_KANKREJ_03011` | Dual Purpose (Sawai Chaal) |
| **8** | **Krishna Valley**| Cattle | Karnataka / Maharashtra | `INDIA_CATTLE_0800_KRISHNAVALLEY_03015` | Heavy Tillage Draught |
| **9** | **Mehsana** | Buffalo | Gujarat | `INDIA_BUFFALO_0400_MEHSANA_01004` | Persistent Dairy |
| **10** | **Murrah** | Buffalo | Haryana | `INDIA_BUFFALO_0500_MURRAH_01001` | Premier Dairy |
| **11** | **Nagpuri** | Buffalo | Maharashtra | `INDIA_BUFFALO_1100_NAGPURI_01007` | Arid Dual Purpose |
| **12** | **Nili Ravi** | Buffalo | Punjab | `INDIA_BUFFALO_1600_NILIRAVI_01002` | High-Yield Dairy |
| **13** | **Nimari** | Cattle | Madhya Pradesh | `INDIA_CATTLE_1000_NIMARI_03019` | Draught & Milk |
| **14** | **Ongole** | Cattle | Andhra Pradesh | `INDIA_CATTLE_0100_ONGOLE_03020` | Dual Purpose |
| **15** | **Toda** | Buffalo | Tamil Nadu | `INDIA_BUFFALO_0018_TODA_01010` | Sacred Dairy (Nilgiris) |
| **16** | **Vechur** | Cattle | Kerala | `INDIA_CATTLE_0900_VECHUR_03030` | Smallest Cattle / A2 Milk |

---

## 3. Verified Experimental Results

### Architecture Comparison on Validation Set
- **ConvNeXt-Tiny (Selected)**: **70.11% Val Accuracy** | **0.6788 Macro F1** | **0.6935 Weighted F1**
- **Swin-T (Transformer)**: 69.37% Val Accuracy | 0.6732 Macro F1 | 0.6811 Weighted F1
- **EfficientNet-B2 (Baseline)**: 64.21% Val Accuracy | 0.6233 Macro F1 | 0.6373 Weighted F1
- **Hierarchical Pipeline**: 57.93% End-to-End Accuracy | 0.5479 Macro F1

### Final Evaluation on Untouched Test Set (271 Images)
- **Overall Test Accuracy**: **73.43%**
- **Macro Precision**: **0.7394**
- **Macro Recall**: **0.7367**
- **Macro F1-Score**: **0.7240**
- **Weighted F1-Score**: **0.7276**
- **Top Performing Breeds**: Bargur (96.55% F1), Ongole (90.91% F1), Jaffarabadi (90.32% F1), Vechur (84.21% F1), Kankrej (80.00% F1).

---

## 4. Installation & Environment Setup (Windows)

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git

### Step 1: Clone Repository & Create Virtual Environment
```powershell
git clone https://github.com/Doreshwar/SIH.git
cd "Cattle Breed AIML"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install Python ML dependencies
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```powershell
cd cattle_breed_project\cattle-sense-main
npm install
npm run build
cd ..\..
```

---

## 5. Running the Application

### Option A: Start the Flask Backend API
```powershell
# From project root with virtual environment activated:
.\venv\Scripts\python cattle_breed_project\backend\app.py
```
- API will be accessible at: `http://localhost:5000`
- Health check: `http://localhost:5000/health`
- Breeds catalog: `http://localhost:5000/breeds`
- Inference endpoint: `http://localhost:5000/predict` (POST multipart/form-data)

### Option B: Start the React Frontend Development Server
In a second terminal:
```powershell
cd "cattle_breed_project\cattle-sense-main"
npm run dev
```
- Open browser at: `http://localhost:8080` (or `http://localhost:5173`)
- Features: Drag-and-drop image upload, real-time webcam scanning, species disambiguation, and complete 17-class probability distributions.

---

## 6. Training & Evaluation Scripts

| Script | Purpose | Command |
| :--- | :--- | :--- |
| **Data Cleaning** | Cleans raw dataset into 17 verified folders | `python cattle_breed_project/scripts/clean_dataset.py` |
| **Data Splitting** | Perceptual dHash cluster-aware stratification | `python cattle_breed_project/scripts/prepare_dataset_splits.py` |
| **Train EfficientNet** | Trains EfficientNet-B2 baseline | `python cattle_breed_project/scripts/train_cleaned_model.py` |
| **Train ConvNeXt** | Trains production ConvNeXt-Tiny | `python cattle_breed_project/scripts/train_convnext.py` |
| **Train Swin-T** | Trains Swin Transformer | `python cattle_breed_project/scripts/train_swin.py` |
| **Hierarchical Pipeline** | Trains Species + Specialists | `python cattle_breed_project/scripts/train_hierarchical.py` |
| **Final Test Evaluation** | Evaluates on untouched test set | `python cattle_breed_project/scripts/evaluate_final_model.py` |
| **System Audit** | Runs 37-check end-to-end verification | `python cattle_breed_project/scripts/run_end_to_end_audit.py` |

---

## 7. Model Checkpoints & Output Directory Structure

```
cattle_breed_project/
├── annotations/
│   ├── breed_labels.csv             # Verified ICAR-NBAGR class mappings
│   ├── train.csv                    # 1,278 train samples
│   ├── val.csv                      # 271 validation samples
│   └── test.csv                     # 271 holdout test samples (strictly isolated)
├── backend/
│   └── app.py                       # Production Flask API service
├── cattle-sense-main/               # React 18 / Vite / TypeScript Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Identify.tsx         # Upload, Webcam, & Classification results
│   │   │   └── Breeds.tsx           # 17-Breed Database with species filtering
│   │   └── lib/mockData.ts          # API interfaces and breed metadata
│   └── dist/                        # Compiled production frontend build
├── models/
│   ├── convnext_tiny_best.pth       # Selected production checkpoint (73.4% Acc)
│   ├── Best_model_final.pth         # EfficientNet-B2 baseline checkpoint
│   ├── swin_tiny_best.pth           # Swin Transformer checkpoint
│   ├── hierarchical_*.pth           # Hierarchical species & specialist models
│   └── training_curves.png          # Convergence loss & accuracy curves
└── results/
    ├── confusion_matrix_heatmap.png # High-res confusion matrix visualization
    ├── per_class_f1_chart.png       # Horizontal bar chart of breed F1-scores
    ├── prediction_gallery.png       # Sample correct vs misclassified images
    ├── confusion_pairs.csv          # Frequent misclassification pairs
    └── incorrect_predictions.csv    # All 72 test errors with confidence & reasons
```

---

## 8. Known Limitations & Recommendations

1. **Morphological Phenotype Convergence**:
   - Breeds belonging to identical genetic clusters (e.g. *Amritmahal* vs *Hallikar* draught cattle; *Murrah* vs *Banni* black river buffaloes) share near-identical coat colors and horn silhouettes. In head-only or low-contrast photos, visual classification alone cannot replace pedigree registration or genomic testing.
2. **Minority Class Support**:
   - *Amritmahal* (35 train / 7 test) and *Hallikar* (49 train / 10 test) have smaller sample sizes in the original dataset than majority breeds like *Gir* (123 train) or *Nagpuri* (119 train). Collecting additional field photography for Karnataka draught breeds is recommended.
3. **Hardware Considerations**:
   - Inference on CPU averages **~38 ms** per image on a modern multi-core processor. For high-volume server deployments, enabling CUDA GPU execution accelerates batch processing.
