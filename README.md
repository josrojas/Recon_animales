# 🐶 Reconocimiento de Animales

Esta aplicación web permite identificar imagenes de animales usando un modelo de clasificación entrenado con el dataset Animals-10. Está construida con Streamlit y un modelo de Keras previamente entrenado.

## 🧠 Modelo

El modelo se entrenó con más de 28.000 imágenes del dataset Animals-10 de Kaggle. Se guardó como `model.h5` dentro de la carpeta `/model`.

`https://www.kaggle.com/datasets/alessiocorrado99/animals10`

El archivo `model.h5` ya contiene los pesos y arquitectura, por lo que no se requiere el dataset para ejecutar la app.

## ✨ Características

- Clasificación de imágenes en 10 categorías de animales.
- Visualización de la imagen original sin pérdida de resolución.
- Predicción principal con porcentaje de confianza.
- Top 3 predicciones.
- Interfaz sencilla y amigable.

## 🐍 Tecnologías usadas

- Python 3
- TensorFlow / Keras
- Streamlit
- PIL (Python Imaging Library)
- NumPy
- MobileNetV2