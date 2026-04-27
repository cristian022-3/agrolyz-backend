from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

modelo = tf.keras.models.load_model("agrolyz_v5_4clases.keras")

CLASES = ['No_Maiz', 'Otra_Enfermedad', 'Roya', 'Sana']


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):

    img = Image.open(io.BytesIO(await imagen.read())).convert("RGB")
    img = img.resize((300, 300))

    arr = np.array(img).astype(np.float32)

    # ✔ correcto para EfficientNet
    arr = preprocess_input(arr)

    arr = np.expand_dims(arr, axis=0)

    preds = modelo.predict(arr, verbose=0)[0]

    clase = CLASES[np.argmax(preds)]
    confianza = float(np.max(preds)) * 100

    print("PREDICCIONES:", preds)
    print("CLASE:", clase)
    print("CONFIANZA:", confianza)

    if clase == "No_Maiz" or confianza < 70:
        return {
            "valido": False,
            "mensaje": "No se detecta hoja de maíz clara"
        }

    return {
        "valido": True,
        "diagnostico": clase,
        "confianza": round(confianza, 2)
    }
