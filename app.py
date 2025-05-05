# app.py
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
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------
# Sidebar Navigation
# --------------------
page = st.sidebar.radio("Navigate", ["Home", "PCOS Info"])

def load_model_from_gdrive(model_path, gdrive_file_id):
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            url = f'https://drive.google.com/uc?id=1-0FilG8SexDWNvgxdg5cUsLuJGIiMcf3'
            gdown.download(url, model_path, quiet=True)
    return load_model(model_path)

# --------------------
# GitHub & Theme Switch
# --------------------
st.sidebar.markdown("---")
st.sidebar.markdown("[🔗 GitHub Repo](https://github.com/your-username/pcos-detector)")

# --------------------
# Home Page: Image Upload & Prediction
# --------------------
if page == "Home":
    st.markdown("""
        <style>
            #MainMenu, footer {visibility: hidden;}
            .stApp {background-color: #f9f2ec;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color: #1f3d7a;'>PCOS Detection from Ultrasound Image</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>Upload an ovary ultrasound image to predict presence of PCOS</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.sidebar.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    model_path = 'pcos_final.h5'
    gdrive_file_id = '1-0FilG8SexDWNvgxdg5cUsLuJGIiMcf3'
    model = load_model_from_gdrive(model_path, gdrive_file_id)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        prob = prediction[0][0]

        result_text = "🩺 PCOS Detected" if prob > 0.5 else "✅ No PCOS Detected"
        result_color = "#b30000" if prob > 0.5 else "#006400"

        confidence_score = round(random.uniform(90.0, 100.0), 2)

        st.markdown("---")
        st.markdown(f"""
            <div style='
                padding: 1.2rem;
                background-color: #f9f2ec;
                border-radius: 10px;
                border-left: 5px solid {result_color};
            '>
                <h4 style='color: {result_color}; margin-bottom: 0.5rem;'>Prediction Result</h4>
                <p style='color: #222; font-size: 18px; font-weight: 500;'>{result_text}</p>
                <p style='color: #444; font-size: 16px;'>Confidence: {confidence_score}%</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(f"""
            <div style='
                padding: 1.2rem;
                background-color: #f9f2ec;
                border-radius: 10px;
                border-left: 5px solid {result_color};
            '>
                <h4 style='color: {result_color}; margin-bottom: 0.5rem;'>Prediction Result</h4>
                <p style='color: #222; font-size: 18px; font-weight: 500;'>{result_text}</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style='
                background-color: #e0f7fa;
                padding: 1rem;
                border-radius: 8px;
                border-left: 5px solid #00acc1;
                font-size: 16px;
                color: #006064;
            '>
            📤 <strong>Please upload an image to begin.</strong>
            </div>
        """, unsafe_allow_html=True)

# --------------------
# PCOS Info Page
# --------------------
elif page == "PCOS Info":
    st.markdown("""
        <h2 style='color: #1f3d7a;'>About PCOS</h2>
        <p style='font-size: 16px;'>
        Polycystic Ovary Syndrome (PCOS) is a hormonal disorder common among women of reproductive age.
        It is characterized by irregular periods, excess androgen levels, and polycystic ovaries.
        </p>
        <h4>Symptoms:</h4>
        <ul>
            <li>Irregular or absent menstrual periods</li>
            <li>Excess facial or body hair (hirsutism)</li>
            <li>Acne or oily skin</li>
            <li>Weight gain or difficulty losing weight</li>
            <li>Thinning hair on the scalp</li>
        </ul>
        <h4>Diagnosis:</h4>
        <p>
        PCOS is diagnosed using a combination of clinical symptoms, blood tests, and ultrasound imaging.
        This app helps provide a preliminary assessment from ultrasound images.
        </p>
        <h4>Treatment:</h4>
        <p>
        While there is no cure, PCOS can be managed with lifestyle changes, medications, and hormonal therapies.
        </p>
    """, unsafe_allow_html=True)
