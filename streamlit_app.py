import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Import our advanced DL prediction logic
from app.dl_predict import predict_keras_model, IMAGENET_LOADED

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="FloodNet ML - Advanced Predictor",
    page_icon="🌊",
    layout="centered"
)

st.title("🌊 Advanced Flood Classification")
st.markdown("""
Upload an image to classify whether it is **Flooded** or **Non Flooded**. 
This production-grade system uses advanced **Out-of-Distribution (OOD)** logic 
to reject unrelated images (food, animals, indoor scenes) by analyzing entropy, 
confidence thresholds, and probability gaps.
""")

# ==========================================
# MODEL LOADING
# ==========================================
@st.cache_resource
def load_classification_model():
    """
    Load the Keras model.
    In a real scenario, point this to your actual .h5 or .keras file.
    """
    model_path = "models/keras_flood_model.h5"
    try:
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception as e:
        # Provide a fallback/warning if model doesn't exist yet
        st.error(f"❌ Failed to load Keras model at `{model_path}`. Error: {e}")
        st.info("Please ensure you have trained and saved a Keras model to that path.")
        return None, False

keras_model, is_loaded = load_classification_model()

# ==========================================
# SIDEBAR SETTINGS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    use_imagenet = st.checkbox("Enable ImageNet Pre-filter", value=True)
    if use_imagenet and not IMAGENET_LOADED:
        st.warning("MobileNetV2 could not be loaded. Filter is disabled.")
        use_imagenet = False
        
    st.markdown("---")
    st.markdown("""
    **Current Thresholds:**
    - Confidence: `0.70`
    - Probability Gap: `0.20`
    - Max Entropy: `0.50`
    """)

# ==========================================
# MAIN UI FLOW
# ==========================================
uploaded_file = st.file_uploader("Upload Aerial Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and is_loaded:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Layout: Image on left, results on right
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        if st.button("🚀 Analyze Image", use_container_width=True):
            with st.spinner("Analyzing with OOD Logic..."):
                
                # Execute Production Pipeline
                # Change target_size if your model uses a different input shape
                result = predict_keras_model(
                    model=keras_model, 
                    img_bgr=img_bgr, 
                    target_size=(128, 128), 
                    use_imagenet_filter=use_imagenet
                )
                
                # ==========================================
                # DISPLAY RESULTS
                # ==========================================
                label = result["label"]
                
                st.markdown("### Prediction Result")
                
                # UI Behavior Based on Requirements
                if label == "Flooded":
                    st.error(f"🚨 **{label}**")
                elif label == "Non Flooded":
                    st.success(f"✅ **{label}**")
                else:
                    st.warning(f"⚠️ **{label}**")
                    
                # Display Metrics
                st.markdown("### Metrics Analysis")
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Top Confidence", f"{result['confidence']}%")
                    st.metric("Entropy", f"{result['entropy']}")
                with metrics_col2:
                    st.metric("Probability Gap", f"{result['prob_diff']}%")
                    st.metric("Raw Probs", f"{result['raw_probs']}")
                
                # Display Uncertainty Reason if applicable
                if result["uncertain"]:
                    st.markdown("### 🔍 Uncertainty Reason")
                    st.info(result["reason"])
                    st.caption("The model rejected this image because it did not pass the required OOD thresholds or the ImageNet pre-filter.")
