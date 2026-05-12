from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import base64
import os
import urllib.request

app = Flask(__name__)

MODEL_PATH = "models/modelo_senas.pkl"
LANDMARKER_PATH = "hand_landmarker.task"

# Descargar modelo si no existe
if not os.path.exists(LANDMARKER_PATH):
    print("Descargando modelo...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, LANDMARKER_PATH)

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

# Configurar MediaPipe
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

def extraer_landmarks(hand_landmarks):
    datos = []
    for lm in hand_landmarks:
        datos.extend([lm.x, lm.y, lm.z])
    return datos

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    img_data = base64.b64decode(data['frame'])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect(mp_image)

    resultados = {"izquierda": None, "derecha": None}

    if resultado.hand_landmarks and resultado.handedness:
        for hand_lm, handedness in zip(resultado.hand_landmarks, resultado.handedness):
            lado_raw = handedness[0].display_name
            lado_key = "derecha" if lado_raw == "Left" else "izquierda"

            lm_data = extraer_landmarks(hand_lm)
            proba = modelo.predict_proba([lm_data])[0]
            clase_idx = np.argmax(proba)
            clase = modelo.classes_[clase_idx]
            confianza = round(float(proba[clase_idx]) * 100, 1)

            # Bounding box
            h, w = frame.shape[:2]
            xs = [lm.x * w for lm in hand_lm]
            ys = [lm.y * h for lm in hand_lm]
            bbox = {
                "x1": max(0, int(min(xs)) - 20),
                "y1": max(0, int(min(ys)) - 20),
                "x2": min(w, int(max(xs)) + 20),
                "y2": min(h, int(max(ys)) + 20)
            }

            resultados[lado_key] = {
                "seña": clase,
                "confianza": confianza,
                "bbox": bbox,
                "landmarks": [[lm.x, lm.y] for lm in hand_lm]
            }

    return jsonify(resultados)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)