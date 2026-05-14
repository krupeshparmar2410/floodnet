"""
dl_predict.py — Advanced Keras Prediction with Out-of-Distribution (OOD) Logic
Includes support for both Sigmoid and Softmax models, Entropy checking,
and an optional ImageNet pre-filter to reject non-aerial images.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import cv2

# ==========================================
# THRESHOLDS & CONFIGURATION
# ==========================================
CONFIDENCE_THRESHOLD = 0.70
PROB_DIFF_THRESHOLD = 0.20
ENTROPY_THRESHOLD = 0.50

# Class Mapping: Assuming 0 = Flooded, 1 = Non Flooded
CLASS_NAMES = ["Flooded", "Non Flooded"]

# Load ImageNet Filter (Optional: MobileNetV2 is lightweight)
try:
    imagenet_model = MobileNetV2(weights="imagenet")
    IMAGENET_LOADED = True
except Exception as e:
    print(f"Failed to load ImageNet model: {e}")
    IMAGENET_LOADED = False

# ==========================================
# ADVANCED IMAGENET PRE-FILTER (OPTIONAL)
# ==========================================
def run_imagenet_filter(img_bgr):
    """
    Uses MobileNetV2 to detect obvious non-aerial objects 
    (e.g., food, animals, humans, indoor scenes).
    Returns True if an unrelated object is detected.
    """
    if not IMAGENET_LOADED:
        return False, "ImageNet model not loaded."

    # Resize and preprocess for MobileNetV2 (224x224 RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = mobilenet_preprocess(img_array.astype(np.float32))

    # Predict ImageNet classes
    preds = imagenet_model.predict(img_preprocessed, verbose=0)
    decoded = decode_predictions(preds, top=3)[0]
    
    # We simply flag if the model is highly confident (>50%) about 
    # ANY specific ImageNet object since aerial flood shots rarely 
    # trigger strong, specific object classes in standard ImageNet 
    # (they might trigger 'valley', 'lakeside', but not 'pizza', 'dog', etc.)
    top_class_id, top_class_name, top_prob = decoded[0]
    
    # A robust approach: If the top prediction is very high confidence for 
    # an ImageNet class, we check if it's a known non-aerial category.
    # For simplicity, if MobileNet is >60% sure it's a specific object (e.g. food/animal)
    # we flag it. (You can refine this with a blocklist of ImageNet IDs).
    if top_prob > 0.60:
        return True, f"Detected unrelated object: {top_class_name} ({top_prob*100:.1f}%)"
    
    return False, "Passes ImageNet filter"


# ==========================================
# MAIN PREDICTION PIPELINE
# ==========================================
def predict_keras_model(model, img_bgr, target_size=(128, 128), use_imagenet_filter=True):
    """
    Production-ready prediction function handling Sigmoid/Softmax 
    and OOD uncertainty logic.
    """
    
    # 1. (Optional) Run ImageNet Pre-filter
    if use_imagenet_filter and IMAGENET_LOADED:
        is_unrelated, reason = run_imagenet_filter(img_bgr)
        if is_unrelated:
            return {
                "label": "Uncertain",
                "confidence": 0.0,
                "prob_diff": 0.0,
                "entropy": 0.0,
                "uncertain": True,
                "reason": f"ImageNet Filter Rejected: {reason}",
                "raw_probs": []
            }

    # 2. Preprocess Image
    # Resize to match training configuration
    img_resized = cv2.resize(img_bgr, target_size)
    
    # Image Normalization (required if training used it)
    img_array = img_resized.astype(np.float32)
    img_array = img_array / 255.0  # Normalized to [0, 1]
    
    # Add batch dimension: shape becomes (1, H, W, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Model Inference
    preds = model.predict(img_batch, verbose=0)[0] # Extract the 1D output
    
    # 4. Handle Sigmoid vs Softmax Outputs
    if len(preds) == 1:
        # Sigmoid Model: Output is a single probability (usually for Class 1)
        prob_class_1 = float(preds[0])
        prob_class_0 = 1.0 - prob_class_1
        # Convert properly into a 2-element array: [P(Flooded), P(Non Flooded)]
        probs = np.array([prob_class_0, prob_class_1])
    else:
        # Softmax Model: Output already has multiple probabilities
        probs = np.array(preds)
    
    # Ensure float types for JSON serialization
    probs = probs.astype(float)
    
    # 5. Calculate Metrics
    # Top 1 and Top 2 probabilities
    sorted_probs = np.sort(probs)[::-1] # Sort descending
    top_1_prob = sorted_probs[0]
    top_2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    
    # Probability Gap
    prob_diff = top_1_prob - top_2_prob
    
    # Shannon Entropy
    # Adding 1e-10 prevents log(0) which results in NaN
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    
    # 6. Final Uncertainty Logic
    uncertainty_reasons = []
    
    if top_1_prob < CONFIDENCE_THRESHOLD:
        uncertainty_reasons.append(f"Low Confidence ({top_1_prob*100:.1f}% < {CONFIDENCE_THRESHOLD*100}%)")
    
    if prob_diff < PROB_DIFF_THRESHOLD:
        uncertainty_reasons.append(f"Low Prob Gap ({prob_diff*100:.1f}% < {PROB_DIFF_THRESHOLD*100}%)")
        
    if entropy > ENTROPY_THRESHOLD:
        uncertainty_reasons.append(f"High Entropy ({entropy:.2f} > {ENTROPY_THRESHOLD})")

    # Determine Verdict
    is_uncertain = len(uncertainty_reasons) > 0
    
    if is_uncertain:
        label = "Uncertain"
        final_reason = " | ".join(uncertainty_reasons)
    else:
        predicted_class_idx = np.argmax(probs)
        label = CLASS_NAMES[predicted_class_idx]
        final_reason = "Clear Prediction"

    # 7. Return Result
    return {
        "label": label,
        "confidence": round(top_1_prob * 100, 2),
        "prob_diff": round(prob_diff * 100, 2),
        "entropy": round(entropy, 3),
        "uncertain": is_uncertain,
        "reason": final_reason,
        "raw_probs": [round(p * 100, 2) for p in probs]
    }
