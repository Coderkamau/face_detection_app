import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit instructions
st.title("Face Detection with Viola-Jones Algorithm")
st.markdown("""
This app allows you to:
1. Upload an image.
2. Detect faces using the Viola-Jones algorithm.
3. Choose rectangle color.
4. Adjust minNeighbors and scaleFactor.
5. Save the image with detected faces.
""")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Choose rectangle color
color = st.color_picker("Choose rectangle color", "#00FF00")
bgr_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4))

# Adjust parameters
scale_factor = st.slider("Adjust scaleFactor", min_value=1.05, max_value=2.0, step=0.05, value=1.1)
min_neighbors = st.slider("Adjust minNeighbors", min_value=1, max_value=10, step=1, value=5)

# Process image
if uploaded_image:
    image = Image.open(uploaded_image)
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x+w, y+h), bgr_color, 2)

    st.image(img_array, caption="Detected Faces", use_column_width=True)

    # Save the image
    if st.button("Save Image"):
        result_filename = "detected_faces.jpg"
        cv2.imwrite(result_filename, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved as {result_filename}")