import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    Classify an image using a trained model and return the predicted class and confidence score.
    """
    # Convert image to 224x224
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score
