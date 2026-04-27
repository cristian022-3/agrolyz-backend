from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
modelo = tf.keras.models.load_model("agrolyz_v5_4clases.keras")

CLASES = ['No_Maiz', 'Otra_Enfermedad', 'Roya', 'Sana']


@app.get("/")
def home():
    return {"status": "Agrolyz API activa"}


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):

    img_bytes = await imagen.read()

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((300, 300))

    arr = np.array(img, dtype=np.float32)

    # 🔥 NORMALIZACIÓN SIMPLE (más estable en deploy)
    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    preds = modelo.predict(arr, verbose=0)[0]

    clase = CLASES[int(np.argmax(preds))]
    confianza = float(np.max(preds)) * 100

    # filtro de seguridad
    if clase == "No_Maiz" or confianza < 70:
        return {
            "valido": False,
            "mensaje": "No se detecta hoja de maíz clara"
        }

    return {
        "valido": True,
        "diagnostico": clase,
        "confianza": round(confianza, 2),
        "probabilidades": {
            c: round(float(preds[i]) * 100, 2)
            for i, c in enumerate(CLASES)
        }
    }