import os
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

if not os.path.exists("modelo_urgencia/saved_model.pb"):
    import train

app = FastAPI()

model = tf.saved_model.load("modelo_urgencia")
infer = model.signatures["serving_default"]

class Message(BaseModel):
    text: str

PALABRAS_FUERZA_URGENTE = [
    "morir", "matar", "suicidio", "suicidarme", "suicidar", "pastillas", "cortarme", "cortar",
    "sangre", "quitarme la vida", "acabar conmigo", "tirarme",
    "ahogarme", "no despertar", "última noche", "despedida", "adiós",
    "hacerme daño", "terminar con todo", "desaparecer", "no quiero seguir"
]

def forzar_urgente(texto: str) -> bool:
    """Devuelve True si el texto contiene alguna palabra clave"""
    texto_lower = texto.lower()
    for palabra in PALABRAS_FUERZA_URGENTE:
        if palabra in texto_lower:
            return True
    return False

@app.post("/predict")
def predict(msg: Message):
    #comprobar palabras clave 
    if forzar_urgente(msg.text):
        return {
            "urgent": True,
            "score": 1.0,
            "motivo": "palabra_clave"
        }
    
    # usar el modelo si no hay palabras clave
    result = infer(tf.constant([msg.text]))
    output = list(result.values())[0]
    score = float(output.numpy()[0][0])

    return {
        "urgent": score >= 0.7,
        "score": score,
        "motivo": "modelo"
    }