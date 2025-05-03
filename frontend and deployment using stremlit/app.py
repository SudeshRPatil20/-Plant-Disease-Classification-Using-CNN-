import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("D:/sudesh/projects/plant desies detection/archive (4)/train_model.keras")

# Define class names based on your dataset
class_names = ['Healthy', 'Powdery', 'Rust']

# Streamlit app title
st.title("Plant Disease Detection")

# File uploader for image
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    """Preprocess the image to fit model input requirements."""
    image = image.resize((256, 256))  # resize to match model's input shape
    image_array = tf.keras.preprocessing.image.img_to_array(image)  # convert to array
    image_array = np.expand_dims(image_array, axis=0)  # add batch dimension
    image_array /= 255.0  # normalize pixel values to [0, 1]
    return image_array

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    # Display prediction
    st.write(f"Prediction: **{predicted_class}**")

    # Display prediction probabilities
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")
