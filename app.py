import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('resnet50_flower_classifier.h5')
image_size = (128, 128)  # The input size expected by the model

# Streamlit app title
st.title("Flower Recognition Example")

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Open the image file using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image
    st.write("Image Format:", image.format)
    st.write("Image Size:", image.size)
    st.write("Image Mode:", image.mode)

    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    
    # Resize the image to the required input size
    img = image.resize(image_size)
    
    # Convert to a NumPy array and preprocess
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    res = model.predict(img_array)
    max_index = res.argmax()

    # Define class names (adjust based on your classes)
    classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    predicted_class = classes[max_index]

    # Display prediction results
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {res[0][max_index] * 100:.2f}%")
