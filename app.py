import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="PCOS Detection",
    page_icon="🧬",
    layout="centered"
)

# --------------------
# Hide Streamlit Default Menu & Style Background
# --------------------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp {
            background-color: #f9f2ec;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------
# Title and Subtitle
# --------------------
st.markdown("<h2 style='text-align: center; color: #1f3d7a;'>PCOS Detection from Ultrasound Image</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Upload a grayscale ovary ultrasound image to predict presence of PCOS</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------
# Sidebar Upload
# --------------------
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Accepted formats: JPG, PNG", type=["jpg", "jpeg", "png"])

# --------------------
# Download model from Google Drive if not present
# --------------------
model_path = 'pcos_mobilenet_model.h5'
gdrive_file_id = '1-0FilG8SexDWNvgxdg5cUsLuJGIiMcf3'  # Replace with your actual ID

os.makedirs('model', exist_ok=True)

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        url = f'https://drive.google.com/uc?id=1-0FilG8SexDWNvgxdg5cUsLuJGIiMcf3'
        gdown.download(url, model_path, quiet=True)

# --------------------
# Load the model
# --------------------
model = load_model(model_path)

# --------------------
# Prediction Section
# --------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prob = prediction[0][0]

    st.markdown("---")

    # Determine result and color
    result_text = "🩺 PCOS Detected" if prob > 0.5 else "✅ No PCOS Detected"
    result_color = "#b30000" if prob > 0.5 else "#006400"  # Deep red or green for contrast
    text_color = "#222"

    # Custom styled result box
    st.markdown(f"""
        <div style='
            padding: 1.2rem;
            background-color: #f9f2ec;
            border-radius: 10px;
            border-left: 5px solid {result_color};
        '>
            <h4 style='color: {result_color}; margin-bottom: 0.5rem;'>Prediction Result</h4>
            <p style='color: {text_color}; font-size: 18px; font-weight: 500;'>{result_text}</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Custom styled upload prompt
    st.markdown("""
    <div style='
        background-color: #e0f7fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #00acc1;
        font-size: 16px;
        color: #006064;
        margin-top: 2rem;
    '>
    📤 <strong>Please upload an image to begin.</strong>
    </div>
    """, unsafe_allow_html=True)
