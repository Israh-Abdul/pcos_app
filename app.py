import streamlit as st
import numpy as np
import os
import gdown
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --------------------
# Download model if not present
# --------------------
model_path = 'pcos_cnn_model.h5'
gdrive_file_id = '1HucmF4vFg_qG_tgeoEpneZB1qriIAGb7'  # Replace with your actual file ID

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
# Hide footer and menu
# --------------------
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp {
            background-color: #fafafa;
        }
    </style>
    """, unsafe_allow_html=True)

# --------------------
# Title and Subtitle
# --------------------
st.markdown("<h2 style='text-align: center; color: #1c4a1e;'>PCOS Detection from Ultrasound Image</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Upload a grayscale ovary ultrasound image to predict presence of PCOS</p>", unsafe_allow_html=True)
st.markdown("---")

# --------------------
# Sidebar: Upload
# --------------------
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Accepted formats: JPG, PNG", type=["jpg", "jpeg", "png"])

# --------------------
# Download model if not present
# --------------------
model_path = 'pcos_cnn_model.h5'
gdrive_file_id = '1HucmF4vFg_qG_tgeoEpneZB1qriIAGb7'  # 🔁 Replace with your actual Google Drive file ID

os.makedirs('model', exist_ok=True)

if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        url = f'https://drive.google.com/uc?id=1HucmF4vFg_qG_tgeoEpneZB1qriIAGb7'
        gdown.download(url, model_path, quiet=True)

# --------------------
# Load model
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
    st.subheader("Prediction Result")

    if prob > 0.5:
        st.success("🩺 PCOS Detected")
    else:
        st.info("✅ No PCOS Detected")
    
    st.caption(f"Confidence Score: {prob:.2f}")
else:
    st.info("Please upload an image to begin.")

