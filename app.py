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
    
    # 4. Preprocessing
    img_resized = image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Batch dimension එක එකතු කිරීම
    
    # Cast and Preprocess
    img_array = tf.cast(img_array, tf.float32)
    img_array = preprocess_input(img_array)
    
  # ============================================================
    # 5. Model Prediction (Diagnostic Mode)
    # ============================================================
    predictions = model.predict(img_array)
    
    # Model eka dena raw numbers okkoma screen eke print karanna 👇
    st.write("Raw Model Output Array:", predictions)
    st.write("After predictions[0]:", predictions[0])
    
    predicted_index = np.argmax(predictions[0]) 
    st.write("Predicted Index:", predicted_index)
    
    predicted_class = class_names[predicted_index]
    confidence = 100 * np.max(predictions[0])
    
    # ============================================================
    # 6. Show Results
    # ============================================================
    st.subheader(f"Prediction: **{predicted_class}**")
    
    # Confidence එක 0-100 අතර integer එකක් විය යුතු නිසා int() කලා
    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence:.2f}%**")
