"""
predict.py — FloodNet Prediction Helper (Production Safe + UNCERTAIN FIX)
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# ── Load Models ONCE ──────────────────────────────────
try:
    TASK1_MODEL = joblib.load(os.path.join(MODELS_DIR, "task1_model.pkl"))
    TASK2_MODEL = joblib.load(os.path.join(MODELS_DIR, "task2_model.pkl"))
    MODELS_LOADED = True
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    MODELS_LOADED = False


# ── Config ────────────────────────────────────────────
IMG_SIZE   = (128, 128)
PATCH_SIZE = (32, 32)
STEP       = 16

SEG_COLORS = {
    0:(55,71,79), 1:(255,61,90), 2:(255,152,0), 3:(224,64,251),
    4:(124,77,255), 5:(0,229,255), 6:(0,230,118), 7:(255,214,0),
    8:(64,196,255), 9:(105,240,174)
}

CLASS_NAMES = ["Background","Building-Flooded","Building-Non-Flood",
               "Road-Flooded","Road-Non-Flood","Water",
               "Tree","Vehicle","Pool","Grass"]


# ── Feature Extraction ────────────────────────────────
def extract_features(img_bgr):
    img  = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, orientations=9,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   feature_vector=True)

    color_feat = []
    for ch in range(3):
        hist = cv2.calcHist([img],[ch],None,[32],[0,256])
        color_feat.extend(hist.flatten())

    return np.concatenate([hog_feat, color_feat])


# ── Task 1 ────────────────────────────────────────────
def predict_classification(img_bgr):
    if not MODELS_LOADED:
        return {"error": "Model not loaded"}

    try:
        feat  = extract_features(img_bgr)
        pred  = TASK1_MODEL.predict([feat])[0]
        proba = TASK1_MODEL.predict_proba([feat])[0]

        # ✅ UNCERTAINTY LOGIC
        CONF_THRESHOLD = 0.65  # Best value: Requires top prediction to be > 65%
        DIFF_THRESHOLD = 0.20  # Best value: Requires clear gap between top 2 classes (e.g., 60/40 is the minimum)
        
        max_prob = float(np.max(proba))
        
        if len(proba) >= 2:
            sorted_probs = np.sort(proba)
            prob_diff = float(sorted_probs[-1] - sorted_probs[-2])
        else:
            prob_diff = 1.0

        is_uncertain = (max_prob < CONF_THRESHOLD) or (prob_diff < DIFF_THRESHOLD)
        
        if is_uncertain:
            label = "Uncertain"
            flooded = False
        else:
            label = "Flooded" if pred == 0 else "Non-Flooded"
            flooded = bool(pred == 0)

        return {
            "label":      label,
            "confidence": round(max_prob * 100, 2),
            "class_idx":  int(pred) if not is_uncertain else -1,
            "flooded":    flooded,
            "uncertain":  bool(is_uncertain)
        }

    except Exception as e:
        return {"error": str(e)}


# ── Sub-class ─────────────────────────────────────────
def _sub_classify_patch(patch_bgr, binary_pred):
    p   = cv2.resize(patch_bgr, (32, 32)).astype(np.float32)
    hsv = cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_BGR2HSV)

    avg_h = float(np.median(hsv[:, :, 0]))
    avg_s = float(np.mean(hsv[:, :, 1]))
    avg_v = float(np.mean(hsv[:, :, 2]))

    if binary_pred == 1:
        if (85 < avg_h <= 140 and avg_s > 30) or (avg_v < 120 and avg_s < 60):
            return 5
        if 120 <= avg_v < 220 and avg_s < 55:
            return 3
        return 1
    else:
        if avg_v < 30:
            return 0
        if 35 <= avg_h <= 85 and avg_s > 25:
            return 9 if avg_v > 120 else 6
        if 85 < avg_h <= 130 and avg_s > 60:
            return 8
        if avg_s < 60:
            if np.max(hsv[:, :, 1]) > 160:
                return 7
        if (avg_h < 35 or avg_h > 150) and avg_s > 30:
            return 2
        if 90 <= avg_v < 220 and avg_s < 55:
            return 4
        return 2


# ── Task 2: Segmentation (FIXED) ─────────────────────
def predict_segmentation(img_bgr):
    import base64

    if not MODELS_LOADED:
        return {"error": "Model not loaded"}

    try:
        img = cv2.resize(img_bgr, (256, 256))
        h, w = img.shape[:2]
        ph, pw = PATCH_SIZE

        patches, coords = [], []
        for r in range(0, h - ph, STEP):
            for c in range(0, w - pw, STEP):
                patches.append(img[r:r+ph, c:c+pw].flatten()/255.0)
                coords.append((r, c))

        patch_arr = np.array(patches)

        probas = TASK2_MODEL.predict_proba(patch_arr)
        binary_preds = np.argmax(probas, axis=1)
        max_probas = np.max(probas, axis=1)

        class_counts = {name: 0 for name in CLASS_NAMES}
        overlay = img.copy()

        for (r, c), bp in zip(coords, binary_preds):
            cls = _sub_classify_patch(img[r:r+ph, c:c+pw], bp)
            overlay[r:r+ph, c:c+pw] = SEG_COLORS.get(cls, (128,128,128))
            class_counts[CLASS_NAMES[cls]] += 1

        blended = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
        _, buf = cv2.imencode(".png", blended)

        total_patches = sum(class_counts.values())

        class_pcts = {
            name: round(cnt / total_patches * 100, 1) if total_patches > 0 else 0.0
            for name, cnt in class_counts.items()
        }

        flooded_classes = {"Building-Flooded", "Road-Flooded", "Water"}
        flooded_pct = sum(class_pcts[n] for n in flooded_classes)
        safe_pct = 100.0 - flooded_pct

        avg_conf = float(np.mean(max_probas))
        conf_std = float(np.std(max_probas))

        # ✅ NEW UNCERTAINTY LOGIC
        CONF_THRESHOLD = 0.65
        CONF_STD_THRESHOLD = 0.18
        FLOOD_LOW = 20
        FLOOD_HIGH = 60

        low_conf = avg_conf < CONF_THRESHOLD
        high_variation = conf_std > CONF_STD_THRESHOLD
        ambiguous = FLOOD_LOW <= flooded_pct <= FLOOD_HIGH
    
        if low_conf or high_variation or ambiguous:
            verdict = "UNCERTAIN"
        elif flooded_pct > FLOOD_HIGH:
            verdict = "FLOODED"
        else:
            verdict = "NOT_FLOODED"

        return {
            "mask_b64": base64.b64encode(buf).decode("utf-8"),
            "class_dist": class_counts,
            "class_pcts": class_pcts,
            "total_patches": total_patches,
            "flooded_pct": round(flooded_pct, 1),
            "safe_pct": round(safe_pct, 1),
            "avg_conf": round(avg_conf * 100, 1),
            "conf_std": round(conf_std * 100, 1),
            "verdict": verdict,
            "uncertain": verdict == "UNCERTAIN"
        }

    except Exception as e:
        return {"error": str(e)}