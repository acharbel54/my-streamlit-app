import streamlit as st
from PIL import Image
from keras.models import load_model
from util import set_background, classify

# --- Configuration Streamlit ---
st.set_page_config(page_title="Waste Classifier", layout="centered")

# --- Fond d'écran (optionnel) ---
set_background('./nani.png')

# --- Titre ---
st.title("♻️ Recyclable Waste Classifier")
st.subheader("Upload a recyclable waste image to classify it")

# --- Chargement du modèle et des labels ---
model = load_model("./converted_keras/keras_model.h5", compile=False)
with open("./converted_keras/labels.txt", "r") as f:
    class_names = f.readlines()

# --- Upload d'image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prédiction
    class_name, confidence = classify(image, model, class_names)

    st.markdown(f"### ✅ Predicted Class: **{class_name}**")
    st.markdown(f"### 📊 Confidence Score: **{confidence * 100:.2f}%**")
