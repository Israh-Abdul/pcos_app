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
model_path = 'model/pcos_cnn_model.h5'
gdrive_file_id = 'YOUR_FILE_ID_HERE'  # Replace with your actual file ID

os.makedirs('model', exist_ok=True)

if not os.path.exists(model_path):
    st.info("Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?id=1HucmF4vFg_qG_tgeoEpneZB1qriIAGb7'
    gdown.download(url, model_path, quiet=False)
    st.success("Model downloaded successfully!")

# --------------------
# Load model
# --------------------
model = load_model(model_path)

# --------------------
# Streamlit UI
# --------------------
st.title("PCOS Detection App")
st.write("Upload a **grayscale** ultrasound image (e.g., PNG or JPG) for PCOS prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    prediction_label = "PCOS Detected" if probability > 0.5 else "No PCOS Detected"

    st.write(f"### Prediction: `{prediction_label}`")
    st.write(f"Confidence Score: `{probability:.2f}`")
