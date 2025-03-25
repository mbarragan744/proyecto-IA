import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo y el scaler
scaler = joblib.load("scaler.bin")
model = joblib.load("naive_bayes_model.bin")

# Configurar la página
st.title("Modelo predicción calidad del clima con IA")
st.subheader("Realizado por Mariana Barragán y Camilo Muñoz")

# Introducción
st.write("""
Esta aplicación permite predecir la calidad del aire en función de varios factores ambientales.
Ingrese los valores en los deslizadores y obtendrá una predicción basada en un modelo de inteligencia artificial.
""")

# Definir los parámetros de entrada con sliders
pm10 = st.slider("PM10 (µg/m³)", 0, 200, 50)
no2 = st.slider("NO2 (ppb)", 0, 150, 30)
so2 = st.slider("SO2 (ppb)", 0, 100, 20)
co = st.slider("CO (ppm)", 0, 10, 1)
o3 = st.slider("O3 (ppb)", 0, 200, 50)
temp = st.slider("Temperature (°C)", -10, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (m/s)", 0, 20, 5)

# Crear dataframe con los datos de entrada
data = pd.DataFrame({
    'PM10 (µg/m³)': [pm10],
    'NO2 (ppb)': [no2],
    'SO2 (ppb)': [so2],
    'CO (ppm)': [co],
    'O3 (ppb)': [o3],
    'Temperature (°C)': [temp],
    'Humidity (%)': [humidity],
    'Wind Speed (m/s)': [wind_speed]
})

st.write("### Datos ingresados:")
st.write(data)

# Botón para realizar predicción
if st.button("Predecir Calidad del Aire"):
    # Normalizar los datos
    scaled_data = scaler.transform(data)

    # Realizar predicción
    prediction = model.predict(scaled_data)[0]

    # Mapear predicciones a etiquetas
    labels = {
        0: "Buena",
        1: "Moderada",
        2: "No saludable para grupos sensibles",
        3: "No saludable",
        4: "Muy no saludable",
        5: "Peligrosa"
    }

    # Mostrar resultado
    st.write("### Predicción de calidad del aire:")
    st.success(f"{labels[prediction]}")

# Línea divisoria
st.markdown("---")

# Símbolo de copyright
st.write("© UNAB 2025")
