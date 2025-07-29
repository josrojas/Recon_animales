import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore

# Cargar el modelo
model = load_model("model/model.h5")

# Configuración de la app
st.set_page_config(page_title="Reconocimiento de Animales", layout="centered")
st.title("🧠 Reconocimiento de Animales")
st.write("Sube una imagen y te diremos qué animal contiene.")

# Nombres de clase según dataset
class_names = [
    "perro", "caballo", "elefante", "mariposa", "gallina",
    "gato", "vaca", "oveja", "araña", "ardilla"
]

emoji_labels = {
    "perro": "🐶", "caballo": "🐴", "elefante": "🐘", "mariposa": "🦋", "gallina": "🐔",
    "gato": "🐱", "vaca": "🐮", "oveja": "🐑", "araña": "🕷️", "ardilla": "🐿️"
}

# Cargar imagen a analizar
uploaded_file = st.file_uploader("Selecciona una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_original = Image.open(uploaded_file).convert("RGB")
    st.image(image_original, caption="🖼 Imagen original")

    # Redimensionar para el modelo
    image_resized = image_original.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_names[predicted_index]
    emoji = emoji_labels.get(predicted_label, "❓")
    confianza = prediction[predicted_index]

    st.subheader("Resultado principal:")
    st.write(f"{emoji} **{predicted_label.capitalize()}** — confianza: `{confianza:.2%}`")

    # Top 3 predicciones
    st.subheader("🔍 Top 3 predicciones")
    top_indices = np.argsort(prediction)[::-1][:3]
    for i in top_indices:
        clase = class_names[i]
        emoji = emoji_labels.get(clase, "❓")
        prob = prediction[i]
        st.write(f"{emoji} **{clase.capitalize()}**: `{prob:.2%}`")