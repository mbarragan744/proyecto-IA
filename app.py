import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo y el scaler
scaler = joblib.load("scaler.bin")
model = joblib.load("knn_model.bin")

# Configurar la página
st.title("Modelo predicción calidad del aire con IA")

# Introducción
st.write("""
Esta aplicación permite predecir la calidad del aire en función de varios factores ambientales.
Ingrese los valores en los deslizadores y obtendrá una predicción basada en un modelo de inteligencia artificial.
""")

# Definir los parámetros de entrada con sliders
pm25 = st.slider("PM2.5 (µg/m³)", 0, 200, 50)
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
    'PM2.5 (µg/m³)': [pm25],
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

# Mapear predicciones a etiquetas y emojis
labels = {
    0: "Buena",
    1: "Moderada",
    2: "No saludable para grupos sensibles",
    3: "No saludable",
    4: "Muy no saludable",
    5: "Peligrosa"
}

emojis = {
    0: "🙂",
    1: "😐",
    2: "😷",
    3: "😷⚠️",
    4: "😷🚨",
    5: "💀🚨"
}

recommendations = {
    0: "La calidad del aire es buena. No se requieren medidas especiales.",
    1: "La calidad del aire es moderada. Es seguro para la mayoría de las personas, pero personas con problemas respiratorios pueden querer limitar la exposición.",
    2: "No saludable para grupos sensibles. Las personas con enfermedades respiratorias o cardíacas deben evitar esfuerzos prolongados al aire libre.",
    3: "No saludable. Se recomienda limitar la exposición al aire libre para todas las personas.",
    4: "Muy no saludable. Evite salir al exterior, especialmente personas con condiciones de salud preexistentes.",
    5: "Peligrosa. Evite salir al exterior. La calidad del aire es extremadamente dañina para todos."
}

# Botón para realizar predicción
if st.button("Predecir Calidad del Aire"):
    # Normalizar los datos
    scaled_data = scaler.transform(data)

    # Realizar predicción
    prediction = model.predict(scaled_data)[0]

    # Mostrar resultado con emoji y recomendación
    st.write("### Predicción de calidad del aire:")
    st.success(f"{labels[prediction]} {emojis[prediction]}")

    st.write("### Recomendaciones:")
    st.info(recommendations[prediction])

# Línea divisoria
st.markdown("---")

st.write("Realizado por Camilo Muñoz y Mariana Barragán")

# Símbolo de copyright
st.write("© UNAB 2025")
