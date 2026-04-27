from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input
from supabase import create_client
import datetime

# ================================
# APP
# ================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# MODELO IA
# ================================
modelo = tf.keras.models.load_model("agrolyz_v5_4clases.keras")

CLASES = ['No_Maiz', 'Otra_Enfermedad', 'Roya', 'Sana']

# ================================
# SUPABASE CONFIG
# ================================
SUPABASE_URL = "https://ovvtjqwkfdwymbczcanm.supabase.co"
SUPABASE_KEY = "sb_publishable_-ug71BjVdFs9OLHKBBvCXA_U2FJkOrL"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================================
# ENDPOINT PRINCIPAL
# ================================
@app.get("/")
def home():
    return {"status": "Agrolyz API activa"}

# ================================
# PREDICCIÓN
# ================================
@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):

    # leer imagen
    img = Image.open(io.BytesIO(await imagen.read())).convert("RGB")
    img = img.resize((300, 300))

    # convertir a array
    arr = np.array(img).astype(np.float32)

    # preprocesamiento IA
    arr = preprocess_input(arr)

    # batch
    arr = np.expand_dims(arr, axis=0)

    # predicción
    preds = modelo.predict(arr, verbose=0)[0]

    clase = CLASES[np.argmax(preds)]
    confianza = float(np.max(preds)) * 100

    print("PREDICCIONES:", preds)
    print("CLASE:", clase)
    print("CONFIANZA:", confianza)

    # ================================
    # FILTRO DE SEGURIDAD
    # ================================
    if clase == "No_Maiz" or confianza < 70:
        return {
            "valido": False,
            "mensaje": "No se detecta hoja de maíz clara"
        }

    # ================================
    # GUARDAR EN SUPABASE
    # ================================
    try:
        supabase.table("diagnosticos").insert({
            "resultado_enfermedad": clase,
            "nivel_confianza": round(confianza, 2),
            "fecha_analisis": datetime.datetime.utcnow().isoformat()
        }).execute()

    except Exception as e:
        print("ERROR SUPABASE:", e)

    # ================================
    # RESPUESTA FINAL
    # ================================
    return {
        "valido": True,
        "diagnostico": clase,
        "confianza": round(confianza, 2),
        "probabilidades": {
            c: round(float(preds[i]) * 100, 2)
            for i, c in enumerate(CLASES)
        }
    }
