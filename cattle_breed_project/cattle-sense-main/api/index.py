#!/usr/bin/env python3
"""
Vercel Serverless Function Backend for Cattle & Buffalo Breed Classification.
Uses ONNX Runtime with quantized ConvNeXt-Tiny for high-speed, lightweight serverless inference.
Total bundle size is ~110MB, well within Vercel's 250MB limit.
"""

import os
import io
import time
import base64
import logging
from typing import Dict, List, Any

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VercelCattleSenseAPI")

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.onnx")
LABELS_PATH = os.path.join(BASE_DIR, "breed_labels.csv")

# Verified ICAR-NBAGR Breed Characteristics & Taxonomy Metadata
BREED_INFO: Dict[str, Dict[str, str]] = {
    "amritmahal": {
        "species": "Cattle",
        "origin": "Hassan, Chikmagalur & Tumkur, Karnataka, India",
        "primary_use": "Draught work (renowned historical cavalry draught breed)",
        "average_weight": "380 - 450 kg (females) | 450 - 520 kg (males)",
        "distinctive_features": "Grey to dark iron-grey coat; long, sharp, upward-curving horns tapering to sharp points; compact muscular body frame."
    },
    "gir": {
        "species": "Cattle",
        "origin": "Saurashtra / Gir Hills, Gujarat, India",
        "primary_use": "Dairy (High A2 milk producer)",
        "average_weight": "380 - 450 kg (females) | 500 - 650 kg (males)",
        "distinctive_features": "Convex domed forehead shield; long pendulous folded ears resembling curled leaves; mottled red and white or pure red coat."
    },
    "hallikar": {
        "species": "Cattle",
        "origin": "Mysore, Mandya & Hassan, Karnataka, India",
        "primary_use": "Draught work (fast trotting and road haulage)",
        "average_weight": "340 - 420 kg (females) | 420 - 500 kg (males)",
        "distinctive_features": "Dark grey to grey coat with lighter belly; long, erect, backward-swept horns curving inward; agile high-speed gait."
    },
    "hariana": {
        "species": "Cattle",
        "origin": "Rohtak, Jind, Hisar & Gurgaon, Haryana, India",
        "primary_use": "Dual-purpose (milking and intensive field cultivation)",
        "average_weight": "360 - 440 kg (females) | 480 - 580 kg (males)",
        "distinctive_features": "White or light grey coat with darker shading over neck and hump in males; short to medium outward-curved horns; long coffin-shaped head."
    },
    "kangayam": {
        "species": "Cattle",
        "origin": "Coimbatore, Tiruppur & Erode, Tamil Nadu, India",
        "primary_use": "Heavy draught work and endurance carting",
        "average_weight": "350 - 420 kg (females) | 450 - 550 kg (males)",
        "distinctive_features": "Grey/white coat with dark grey/black points on head, neck, and hump in bulls; stout, nearly straight horns curving slightly backward."
    },
    "khillari": {
        "species": "Cattle",
        "origin": "Satara, Sangli, Kolhapur (Maharashtra) & Belagavi (Karnataka)",
        "primary_use": "Draught work (drought-hardy agricultural traction)",
        "average_weight": "350 - 420 kg (females) | 450 - 520 kg (males)",
        "distinctive_features": "Grey-white coat; characteristic long tapering head with prominent nasal bridge; long horns emerging close and sweeping backward parallel."
    },
    "krishnavalley": {
        "species": "Cattle",
        "origin": "Krishna river basin, Karnataka & Southern Maharashtra",
        "primary_use": "Heavy agricultural draught in deep black cotton soils",
        "average_weight": "450 - 550 kg (females) | 550 - 700 kg (males)",
        "distinctive_features": "Massive massive deep-chested body; greyish-white coat with dark forequarters in bulls; short, thick curved horns."
    },
    "ongole": {
        "species": "Cattle",
        "origin": "Prakasam & Guntur districts, Andhra Pradesh, India",
        "primary_use": "Dual-purpose (heavy traction and tropical milk yield)",
        "average_weight": "430 - 520 kg (females) | 550 - 700 kg (males)",
        "distinctive_features": "Glossy white coat with dark grey points in bulls; massive muscular hump; prominent dewlap folds; short, stout stumpy horns."
    },
    "banni": {
        "species": "Buffalo",
        "origin": "Banni grassland reserve, Kutch, Gujarat, India",
        "primary_use": "Dairy (extreme arid heat tolerance and night grazing)",
        "average_weight": "450 - 550 kg (females) | 550 - 650 kg (males)",
        "distinctive_features": "Predominantly jet black, rarely copper-brown; tightly coiled vertical horns forming closed rings; medium compact body."
    },
    "bhadawari": {
        "species": "Buffalo",
        "origin": "Agra & Etawah (Uttar Pradesh) and Bhind & Morena (Madhya Pradesh)",
        "primary_use": "High-fat dairy (milk fat reaches 8.5% to 13%)",
        "average_weight": "380 - 450 kg (females) | 450 - 520 kg (males)",
        "distinctive_features": "Characteristic copper/light coppery-brown coat; distinctive 'Janevar' (two white chevron collars) under the neck; flat sword horns."
    },
    "chilika": {
        "species": "Buffalo",
        "origin": "Chilika lake basin, Puri, Khordha & Ganjam, Odisha, India",
        "primary_use": "Dairy and brackish marsh grazing",
        "average_weight": "350 - 420 kg (females) | 400 - 480 kg (males)",
        "distinctive_features": "Brownish black or dark grey coat; medium body size; curved horns directed upward and backward; adapted to saline marsh vegetation."
    },
    "jaffarabadi": {
        "species": "Buffalo",
        "origin": "Gir forest and Saurashtra coast, Gujarat, India",
        "primary_use": "Heavy commercial dairy (massive milk capacity)",
        "average_weight": "600 - 750 kg (females) | 800 - 1000 kg (males)",
        "distinctive_features": "Massive forehead bone shield drooping down laterally; heavy drooping horns that turn downward then curve inward in a hook."
    },
    "marathwada": {
        "species": "Buffalo",
        "origin": "Marathwada region (Parbhani, Nanded, Beed, Latur), Maharashtra",
        "primary_use": "Draught and milk in semi-arid agrarian tracts",
        "average_weight": "350 - 420 kg (females) | 400 - 500 kg (males)",
        "distinctive_features": "Dark brown to jet black; medium build; horns flat, sweeping backward reaching shoulder point."
    },
    "mehsana": {
        "species": "Buffalo",
        "origin": "Mehsana, Banaskantha & Sabarkantha, Gujarat, India",
        "primary_use": "Commercial dairy (famous consistent milk yielder)",
        "average_weight": "480 - 560 kg (females) | 550 - 700 kg (males)",
        "distinctive_features": "Murrah x Surti stabilized cross; black to brown; horns curled into sickle or irregular spiral curves."
    },
    "murrah": {
        "species": "Buffalo",
        "origin": "Rohtak, Hisar, Jind (Haryana) & Nabha, Patiala (Punjab), India",
        "primary_use": "Premier commercial dairy breed of India",
        "average_weight": "500 - 650 kg (females) | 650 - 850 kg (males)",
        "distinctive_features": "Jet black body with switch of tail white; tightly spiraled horns curled like a ram's ring; wedge-shaped dairy conformation."
    },
    "nagpuri": {
        "species": "Buffalo",
        "origin": "Vidarbha region (Nagpur, Wardha, Amravati), Maharashtra, India",
        "primary_use": "Dual-purpose (draught traction and milk)",
        "average_weight": "400 - 500 kg (females) | 500 - 650 kg (males)",
        "distinctive_features": "Long, flat, sword-shaped horns curving backward down to shoulders; black coat with white patches on face, legs, and tail tip."
    },
    "toda": {
        "species": "Buffalo",
        "origin": "Nilgiri Hills plateau, Tamil Nadu, India",
        "primary_use": "Milk and sacred dairy pastoralism of the Toda community",
        "average_weight": "420 - 500 kg (females) | 480 - 580 kg (males)",
        "distinctive_features": "Wide crescent-shaped horns spreading outward and curving inward; thick silver-grey/fawn hair coat with dark dorsal stripe."
    }
}

# ---------------------------------------------------------------------------
# Global State & Lazy Loader
# ---------------------------------------------------------------------------
SESSION = None
DEFAULT_BREEDS = []

def init_onnx_model():
    global SESSION, DEFAULT_BREEDS
    if SESSION is not None:
        return
    
    # 1. Load Breed Names
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                header = lines[0]
                breed_idx = header.index("breed_name") if "breed_name" in header else 1
                DEFAULT_BREEDS = sorted(list(set([l[breed_idx].lower().strip() for l in lines[1:]])))
        except Exception as e:
            logger.warning(f"Error reading labels CSV: {e}")
            
    if not DEFAULT_BREEDS:
        DEFAULT_BREEDS = sorted(list(BREED_INFO.keys()))

    # 2. Load ONNX Session
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading ONNX Model from {MODEL_PATH}...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        SESSION = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
        logger.info("✅ ONNX Model successfully loaded.")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")

# Initialize model
try:
    init_onnx_model()
except Exception as exc:
    logger.error(f"Startup model load error: {exc}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image matching training: Resize 256, CenterCrop 224, Normalize."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    
    # Resize shortest side to 256
    short_side = 256
    if w < h:
        new_w = short_side
        new_h = int(h * (short_side / w))
    else:
        new_h = short_side
        new_w = int(w * (short_side / h))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Center crop 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    
    # Normalize with ImageNet mean and std
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC to CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0)  # (1, 3, 224, 224)


def run_inference(image_bytes: bytes) -> Dict[str, Any]:
    global SESSION, DEFAULT_BREEDS
    if SESSION is None:
        init_onnx_model()
    if SESSION is None:
        raise RuntimeError("Model session is not available.")

    t_start = time.time()
    input_tensor = preprocess_image(image_bytes)
    input_name = SESSION.get_inputs()[0].name
    logits = SESSION.run(None, {input_name: input_tensor})[0][0]
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    latency_ms = round((time.time() - t_start) * 1000, 2)
    
    # Sorted indices
    sorted_indices = np.argsort(probs)[::-1]
    
    # All ranked probabilities
    all_probabilities = []
    for rank, idx in enumerate(sorted_indices, start=1):
        breed_id = DEFAULT_BREEDS[idx] if idx < len(DEFAULT_BREEDS) else f"class_{idx}"
        confidence = float(probs[idx])
        species = BREED_INFO.get(breed_id, {}).get("species", "Cattle")
        all_probabilities.append({
            "rank": rank,
            "breed_id": breed_id,
            "breed_name": breed_id.capitalize(),
            "species": species,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%"
        })
        
    # Top 3 predictions
    top_predictions = []
    for item in all_probabilities[:3]:
        b_id = item["breed_id"]
        meta = BREED_INFO.get(b_id, {
            "origin": "India",
            "primary_use": "Dairy / Draught",
            "average_weight": "Unknown",
            "distinctive_features": "Verified Indigenous Breed"
        })
        top_predictions.append({
            "rank": item["rank"],
            "breed_id": b_id,
            "breed_name": item["breed_name"],
            "species": item["species"],
            "confidence": item["confidence"],
            "confidence_percentage": item["confidence_percentage"],
            "characteristics": meta
        })
        
    return {
        "success": True,
        "model_loaded": True,
        "architecture": "ConvNeXt-Tiny (ONNX Serverless)",
        "device": "Vercel Serverless CPU",
        "inference_time_ms": latency_ms,
        "predictions": top_predictions,
        "all_probabilities": all_probabilities,
        "disclaimer": "Automated breed predictions should be verified with registered ICAR-NBAGR accession standards."
    }


# ---------------------------------------------------------------------------
# API Routes (Compatible with and without /api prefix)
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
@app.route("/api", methods=["GET"])
@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "CattleSense Vercel Serverless API",
        "architecture": "ConvNeXt-Tiny ONNX",
        "classes": len(DEFAULT_BREEDS),
        "model_loaded": SESSION is not None
    }), 200


@app.route("/breeds", methods=["GET"])
@app.route("/api/breeds", methods=["GET"])
def get_breeds():
    catalog = []
    for breed_id, meta in BREED_INFO.items():
        catalog.append({
            "id": breed_id,
            "name": breed_id.capitalize(),
            "species": meta["species"],
            "image": f"/placeholder.svg",
            "characteristics": meta
        })
    return jsonify({
        "success": True,
        "count": len(catalog),
        "breeds": catalog
    }), 200


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No 'image' file field found in request."}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty filename provided."}), 400
            
        file_bytes = file.read()
        if len(file_bytes) < 100:
            return jsonify({"success": False, "error": "File too small or empty."}), 400
            
        result = run_inference(file_bytes)
        return jsonify(result), 200

    except UnidentifiedImageError:
        return jsonify({"success": False, "error": "Uploaded file is not a valid or readable image."}), 400
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict_base64", methods=["POST"])
@app.route("/api/predict_base64", methods=["POST"])
def predict_base64():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' key with base64 data."}), 400
            
        b64_str = data["image"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
            
        file_bytes = base64.b64decode(b64_str)
        result = run_inference(file_bytes)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Base64 error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
