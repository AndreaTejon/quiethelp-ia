import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Leer CSV
df = pd.read_csv("mensajes.csv")

# Datos
texts = np.array(df["text"].astype(str).values)
labels = np.array(df["urgent"].astype("float32").values)

# Convertir texto a números
vectorizer = layers.TextVectorization(
    max_tokens=20000,
    output_sequence_length=200
)

vectorizer.adapt(texts)

# Modelo
model = tf.keras.Sequential([
    vectorizer,
    layers.Embedding(20000, 64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compilar
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Entrenar
model.fit(
    texts,
    labels,
    epochs=10
)

# Guardar modelo
tf.saved_model.save(model, "modelo_urgencia")

print("Modelo entrenado correctamente")