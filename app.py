import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Page Configuration
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("📸 Student Image Classification App")
st.write("Upload an image to predict its class using the trained MobileNetV2 model.")

# 2. Load Model and Class Names
@st.cache_resource # App එක reload වෙන හැම සැරේම model එක load නොවී memory එකේ තියාගන්න
def load_my_model():
    model = tf.keras.models.load_model("student_mobilenetv2_transfer_learning.keras")
    with open("class_names.json", "r") as f:
        classes = json.load(f)
        classes = sorted(list(set(classes)))
    return model, classes

try:
    model, class_names = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# 3. Image Upload Layer
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("👁️ Predicting...")
    
   # 4. Preprocessing (Training code එකට 100% සමාන කිරීම)
    img_resized = image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # (1, 160, 160, 3)
    
    # IMPORTANT FIX: Training එකේ Rescaling(scale=1.0) එකට match වෙන්න shape/type හදාගැනීම
    img_array = tf.cast(img_array, tf.float32)
    
    # MobileNetV2 preprocessing එක කෙලින්ම run කිරීම
    img_array = preprocess_input(img_array)
# ============================================================
    # 5. Model Prediction (Advanced Debug Mode)
    # ============================================================
    predictions = model.predict(img_array)
    raw_probabilities = predictions[0] # 1D array eka gannava
    
    # Hamama class ekatama model eken labunu values percentage vidiyatama screen eke penuvima:
    st.write("--- 🔍 Debug: Model Predictions for Each Class ---")
    for idx, name in enumerate(class_names):
        st.write(f"Class Index {idx} ({name}): **{raw_probabilities[idx]*100:.2f}%**")
    st.write("------------------------------------------------")
    
    # Real Prediction Logic
    predicted_index = np.argmax(raw_probabilities) 
    predicted_class = class_names[predicted_index]
    confidence = 100 * raw_probabilities[predicted_index] 
    
    # ============================================================
    # 6. Show Results
    # ============================================================
    st.subheader(f"Final Prediction: **{predicted_class}**")
    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence:.2f}%**")
