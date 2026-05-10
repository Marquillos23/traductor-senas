from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import threading

app = Flask(__name__)

# ─── CONFIGURACIÓN ───────────────────────────────────────────
MODEL_PATH = "models/modelo_senas.pkl"
LANDMARKER_PATH = "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────

CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Estado global compartido
estado = {"izquierda": None, "derecha": None}
lock = threading.Lock()

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

def dibujar_mano(frame, hand_landmarks):
    h, w, _ = frame.shape
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], (0, 255, 0), 2)
    for p in puntos:
        cv2.circle(frame, p, 4, (255, 255, 255), -1)

def procesar_camara():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect(mp_image)

        nuevo_estado = {"izquierda": None, "derecha": None}

        if resultado.hand_landmarks and resultado.handedness:
            for hand_lm, handedness in zip(resultado.hand_landmarks, resultado.handedness):
                lado_raw = handedness[0].display_name
                lado_key = "derecha" if lado_raw == "Left" else "izquierda"
                lado_label = "Right / Derecha" if lado_raw == "Left" else "Left / Izquierda"

                lm_data = extraer_landmarks(hand_lm)
                proba = modelo.predict_proba([lm_data])[0]
                clase_idx = np.argmax(proba)
                clase = modelo.classes_[clase_idx]
                confianza = round(proba[clase_idx] * 100, 1)

                dibujar_mano(frame, hand_lm)

                xs = [lm.x * w for lm in hand_lm]
                ys = [lm.y * h for lm in hand_lm]
                x1 = max(0, int(min(xs)) - 20)
                y1 = max(0, int(min(ys)) - 20)
                x2 = min(w, int(max(xs)) + 20)
                y2 = min(h, int(max(ys)) + 20)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                etiqueta = f"{clase}: {confianza}%"
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 255, 0), -1)
                cv2.putText(frame, etiqueta, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                cv2.putText(frame, lado_label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                nuevo_estado[lado_key] = {"seña": clase, "confianza": confianza}

        with lock:
            estado.update(nuevo_estado)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(procesar_camara(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/deteccion')
def deteccion():
    with lock:
        return jsonify(estado)

if __name__ == '__main__':
    # 0.0.0.0 permite acceso desde otros dispositivos en la misma red WiFi
    app.run(host='0.0.0.0', port=5000, debug=False)