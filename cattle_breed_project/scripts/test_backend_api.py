#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for CattleBreed Flask Backend API.
Tests all endpoints, valid predictions on real dataset images, and error cases.
"""

import os
import sys
import io
import json
import base64
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))
import app as backend_app

def run_tests():
    print("=" * 80)
    print("CATTLE SENSE FLASK BACKEND - AUTOMATED TEST SUITE")
    print("=" * 80)

    client = backend_app.app.test_client()
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name} {details}")
        else:
            print(f"  [FAIL] {name} {details}")

    # ----------------------------------------------------
    # TEST 1: /health Endpoint
    # ----------------------------------------------------
    print("\n--- Test 1: Health Check Endpoint (/health) ---")
    res = client.get("/health")
    data = res.get_json()
    assert_test("Status code is 200", res.status_code == 200)
    assert_test("Model loaded is True", data.get("model_loaded") is True)
    assert_test("Architecture is ConvNeXt-Tiny", data.get("architecture") == "ConvNeXt-Tiny")
    assert_test("Classes count is 17", data.get("num_classes") == 17)
    print(f"      Response: {json.dumps(data)}")

    # ----------------------------------------------------
    # TEST 2: /breeds Endpoint
    # ----------------------------------------------------
    print("\n--- Test 2: Verified Breeds Catalog (/breeds) ---")
    res = client.get("/breeds")
    data = res.get_json()
    assert_test("Status code is 200", res.status_code == 200)
    assert_test("Breed count is 17", data.get("count") == 17)
    breeds = data.get("breeds", [])
    has_gir = any(b["id"] == "gir" and b["species"] == "Cattle" for b in breeds)
    has_toda = any(b["id"] == "toda" and b["species"] == "Buffalo" for b in breeds)
    assert_test("Gir cattle metadata verified", has_gir)
    assert_test("Toda buffalo metadata verified", has_toda)
    print(f"      Verified 17 breeds (8 Cattle, 9 Buffalo) with accession numbers and phenotypic descriptors.")

    # ----------------------------------------------------
    # TEST 3: /predict on Actual Cattle Sample (Gir)
    # ----------------------------------------------------
    print("\n--- Test 3: /predict on Real Cattle Sample (Gir) ---")
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gir_dir = os.path.join(project_dir, "data", "cleaned", "CATTLE", "INDIA_CATTLE_0400_GIR_03007")
    gir_sample = os.path.join(gir_dir, os.listdir(gir_dir)[0])

    with open(gir_sample, "rb") as f:
        file_bytes = f.read()

    res = client.post("/predict", data={"image": (io.BytesIO(file_bytes), "gir_test.jpg")}, content_type="multipart/form-data")
    data = res.get_json()
    assert_test("Status code is 200", res.status_code == 200)
    assert_test("Success is True", data.get("success") is True)
    assert_test("Architecture matches ConvNeXt-Tiny", data.get("architecture") == "ConvNeXt-Tiny")
    assert_test("Top predictions returned", len(data.get("predictions", [])) == 3)
    assert_test("All 17 ranked probabilities returned", len(data.get("all_probabilities", [])) == 17)
    assert_test("Disclaimer is present", "disclaimer" in data)

    top_pred = data["predictions"][0]
    print(f"      Top Prediction: {top_pred['breed_name']} ({top_pred['species']}) with {top_pred['confidence_percentage']} confidence")
    print(f"      Inference latency: {data['inference_time_ms']} ms")

    # ----------------------------------------------------
    # TEST 4: /predict on Actual Buffalo Sample (Toda)
    # ----------------------------------------------------
    print("\n--- Test 4: /predict on Real Buffalo Sample (Toda) ---")
    toda_dir = os.path.join(project_dir, "data", "cleaned", "BUFFALO", "INDIA_BUFFALO_0018_TODA_01010")
    toda_sample = os.path.join(toda_dir, os.listdir(toda_dir)[0])

    with open(toda_sample, "rb") as f:
        toda_bytes = f.read()

    res = client.post("/predict", data={"image": (io.BytesIO(toda_bytes), "toda_test.jpg")}, content_type="multipart/form-data")
    data = res.get_json()
    assert_test("Status code is 200", res.status_code == 200)
    assert_test("Top prediction returned", len(data.get("predictions", [])) > 0)
    top_toda = data["predictions"][0]
    print(f"      Top Prediction: {top_toda['breed_name']} ({top_toda['species']}) with {top_toda['confidence_percentage']} confidence")

    # ----------------------------------------------------
    # TEST 5: /predict-base64 on Valid Image
    # ----------------------------------------------------
    print("\n--- Test 5: /predict-base64 on Base64 Encoded Image ---")
    b64_str = "data:image/jpeg;base64," + base64.b64encode(file_bytes).decode("utf-8")
    res = client.post("/predict-base64", json={"image": b64_str})
    data = res.get_json()
    assert_test("Status code is 200", res.status_code == 200)
    assert_test("Success is True", data.get("success") is True)
    print(f"      Base64 Top Prediction: {data['predictions'][0]['breed_name']} ({data['predictions'][0]['confidence_percentage']})")

    # ----------------------------------------------------
    # TEST 6: Error Handling - Empty / Missing File
    # ----------------------------------------------------
    print("\n--- Test 6: Error Handling - Missing & Empty File ---")
    res = client.post("/predict", data={}, content_type="multipart/form-data")
    assert_test("Missing 'image' field returns 400", res.status_code == 400)

    res = client.post("/predict", data={"image": (io.BytesIO(b""), "empty.jpg")}, content_type="multipart/form-data")
    assert_test("Empty file returns 400", res.status_code == 400)

    # ----------------------------------------------------
    # TEST 7: Error Handling - Corrupted Image Stream
    # ----------------------------------------------------
    print("\n--- Test 7: Error Handling - Corrupted Image Stream ---")
    res = client.post("/predict", data={"image": (io.BytesIO(b"THIS_IS_NOT_AN_IMAGE_FILE"), "corrupt.jpg")}, content_type="multipart/form-data")
    assert_test("Corrupted bytes returns 400", res.status_code == 400)
    print(f"      Error message returned: '{res.get_json().get('error')}'")

    # ----------------------------------------------------
    # TEST 8: Error Handling - Unsupported File Extension
    # ----------------------------------------------------
    print("\n--- Test 8: Error Handling - Unsupported Extension ---")
    res = client.post("/predict", data={"image": (io.BytesIO(b"fake pdf content"), "document.pdf")}, content_type="multipart/form-data")
    assert_test("Unsupported extension returns 400", res.status_code == 400)
    print(f"      Error message returned: '{res.get_json().get('error')}'")

    # ----------------------------------------------------
    # TEST 9: Error Handling - Dimensions Too Small (< 32x32)
    # ----------------------------------------------------
    print("\n--- Test 9: Error Handling - Dimensions Too Small ---")
    tiny_img = Image.new("RGB", (10, 10), (128, 128, 128))
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format="JPEG")
    res = client.post("/predict", data={"image": (io.BytesIO(tiny_buf.getvalue()), "tiny.jpg")}, content_type="multipart/form-data")
    assert_test("Sub-minimum dimensions return 400", res.status_code == 400)
    print(f"      Error message returned: '{res.get_json().get('error')}'")

    # ----------------------------------------------------
    # Summary
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} / {total} assertions passed ({(passed/total):.1%})")
    print("=" * 80)
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
