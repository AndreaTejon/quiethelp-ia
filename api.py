from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

app = FastAPI()

model = tf.saved_model.load("modelo_urgencia")
infer = model.signatures["serving_default"]

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict(msg: Message):
    text = msg.text.lower()
    result = infer(tf.constant([msg.text]))
    output = list(result.values())[0]
    score = float(output.numpy()[0][0])

    return {"urgent": score >= 0.5}