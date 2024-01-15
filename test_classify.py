import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import settings
import helper
import tempfile

# Assuming you have a YOLO model for classification
from ultralytics import YOLO

try:
    model_classification = helper.load_model(settings.CLASSIFICATION_MODEL)
except Exception as ex:
    st.error(f"Unable to load classification model. Check the specified path: {settings.CLASSIFICATION_MODEL}")
    st.error(ex)

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Check if image is uploaded
if uploaded_image is not None:
    try:
        # Create a temporary file
        temp_file = tempfile.TemporaryFile(delete=False, suffix=os.path.splitext(uploaded_image.name)[-1])

        # Write the uploaded content to the temporary file
        temp_file.write(uploaded_image.read())
        
        
        # Predict on the uploaded image
        results = model_classification(temp_file.name)

        # Extract names and probabilities
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        # Get the predicted class
        predicted_class = names_dict[np.argmax(probs)]
        
        if predicted_class=='xray':
            st.write('xray')
        else:
            st.write('non_xray')

        # # Display the image and the predicted label
        # img = Image.open(uploaded_image)
        # st.image(img, caption=f"Predicted Label: {predicted_class}", use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while processing the image.")
        st.error(ex)


