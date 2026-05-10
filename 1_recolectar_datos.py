import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request

# ─── CONFIGURACIÓN ───────────────────────────────────────────
SEÑAS = ["hola", "bien", "mal", "te amo", "Que hora es"]
MUESTRAS_POR_SEÑA = 200
DATA_DIR = "data"
MODEL_PATH = "hand_landmarker.task"
# ─────────────────────────────────────────────────────────────

# Conexiones de la mano para dibujar manualmente
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo de MediaPipe...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Modelo descargado")

def crear_carpetas():
    os.makedirs(DATA_DIR, exist_ok=True)
    for seña in SEÑAS:
        os.makedirs(os.path.join(DATA_DIR, seña), exist_ok=True)
    print("Carpetas creadas")

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

def recolectar():
    descargar_modelo()
    crear_carpetas()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    for seña in SEÑAS:
        print(f"\nSena: '{seña.upper()}' — Presiona ESPACIO cuando estes listo")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Sena: {seña.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Presiona ESPACIO para iniciar", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Recolector de Senas", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        contador = 0
        dataset = []

        while contador < MUESTRAS_POR_SEÑA:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            resultado = detector.detect(mp_image)

            if resultado.hand_landmarks:
                for hand_lm in resultado.hand_landmarks:
                    dibujar_mano(frame, hand_lm)

                    # Bounding box
                    h, w, _ = frame.shape
                    xs = [lm.x * w for lm in hand_lm]
                    ys = [lm.y * h for lm in hand_lm]
                    x1 = int(min(xs)) - 15
                    y1 = int(min(ys)) - 15
                    x2 = int(max(xs)) + 15
                    y2 = int(max(ys)) + 15
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    lm_data = extraer_landmarks(hand_lm)
                    dataset.append(lm_data)
                    contador += 1

            progreso = int((contador / MUESTRAS_POR_SEÑA) * 100)
            cv2.putText(frame, f"Sena: {seña.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Capturas: {contador}/{MUESTRAS_POR_SEÑA} ({progreso}%)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Recolector de Senas", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(os.path.join(DATA_DIR, seña, "landmarks.npy"), np.array(dataset))
        print(f"   '{seña}' guardada — {len(dataset)} muestras")

    cap.release()
    cv2.destroyAllWindows()
    print("\nRecoleccion completada. Ahora ejecuta: 2_entrenar_modelo.py")

if __name__ == "__main__":
    recolectar()