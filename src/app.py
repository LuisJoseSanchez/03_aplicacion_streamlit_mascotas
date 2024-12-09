import streamlit as st
import joblib
import pandas as pd
import json

st.image("src/mascotas.jpg", use_container_width=True)

# Cargar el modelo entrenado y las asignaciones
model = joblib.load("pets_model.joblib")
with open("category_mapping.json", "r") as f:
    category_mapping = json.load(f)

# Extraer los valores categóricos
eye_color_values = category_mapping["eye_color"]
fur_length_values = category_mapping["fur_length"]

# Título de la aplicación
st.title("Clasificación de Mascotas")

# Descripción de la app
st.write("Introduce los datos de la mascota para predecir su tipo.")

# Entrada de datos
weight = st.number_input("Peso (kg):", min_value=1.0, max_value=50.0, step=0.1)
height = st.number_input("Altura (cm):", min_value=10.0, max_value=100.0, step=0.1)
eye_color = st.selectbox("Color de ojos:", ["Marrón", "Azul", "Verde", "Gris"])
fur_length = st.selectbox("Longitud del pelo:", ["Corto", "Medio", "Largo"])

# Mapear selección de usuario a inglés (formato del modelo)
eye_color_map = {"Marrón": "brown", "Azul": "blue", "Verde": "green", "Gris": "gray"}
fur_length_map = {"Corto": "short", "Medio": "medium", "Largo": "long"}

selected_eye_color = eye_color_map[eye_color]
selected_fur_length = fur_length_map[fur_length]

# Generar las columnas binarias para eye_color
eye_color_binary = [int(color == selected_eye_color) for color in eye_color_values]

# Generar las columnas binarias para fur_length
fur_length_binary = [int(length == selected_fur_length) for length in fur_length_values]

# Crear el DataFrame de entrada con las columnas esperadas
input_data = [weight, height] + eye_color_binary + fur_length_binary
columns = ["weight_kg", "height_cm"] + [f"eye_color_{color}" for color in eye_color_values] + [f"fur_length_{length}" for length in fur_length_values]
input_df = pd.DataFrame([input_data], columns=columns)

# Mostrar los datos generados para depuración
# st.write("Datos generados para predicción (internos):")
# st.write(input_df)

# Predicción
if st.button("Predecir tipo de mascota"):
    prediction = model.predict(input_df)
    prediction_map = {"dog": "Perro", "cat": "Gato", "rabbit": "Conejo"}
    st.success(f"La mascota es un: **{prediction_map[prediction[0]]}**")
