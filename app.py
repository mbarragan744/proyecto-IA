import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Cargar el modelo y el scaler
scaler = joblib.load("scaler.bin")
model = joblib.load("naive_bayes_model.bin")

# Subir una imagen
st.title("Clasificación de Imágenes")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Preprocesar la imagen
    image = image.convert("L")  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar a 28x28
    image_array = np.array(image).flatten().reshape(1, -1)  # Aplanar

    # Escalar la imagen
    image_scaled = scaler.transform(image_array)

    # Hacer predicción
    prediction = model.predict(image_scaled)
    st.write(f"Predicción del modelo: {prediction[0]}")
