import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model('model/pcos_cnn_model.h5')

# Set up the Streamlit UI
st.title("PCOS Detection App")
st.write("Upload a grayscale ultrasound image for PCOS prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    prediction_label = "PCOS Detected" if prediction[0][0] > 0.5 else "No PCOS Detected"

    st.write(f"### Prediction: {prediction_label}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
