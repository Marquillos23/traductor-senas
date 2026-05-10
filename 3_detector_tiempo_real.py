import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pickle
import os

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

def cargar_modelo():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

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

def obtener_bbox(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks]
    ys = [lm.y * h for lm in hand_landmarks]
    x1 = int(min(xs)) - 20
    y1 = int(min(ys)) - 20
    x2 = int(max(xs)) + 20
    y2 = int(max(ys)) + 20
    return x1, y1, x2, y2

def detectar():
    modelo = cargar_modelo()

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

    cap = cv2.VideoCapture(0)
    print("Detector iniciado. Presiona Q para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect(mp_image)

        info_manos = []  # para mostrar abajo

        if resultado.hand_landmarks and resultado.handedness:
            for i, (hand_lm, handedness) in enumerate(
                zip(resultado.hand_landmarks, resultado.handedness)
            ):
                # ── Lado de la mano ──
                lado_raw = handedness[0].display_name  # "Left" o "Right"
                # MediaPipe lo devuelve espejado, lo corregimos
                if lado_raw == "Left":
                    lado_es = "Derecha"
                    lado_en = "Right"
                else:
                    lado_es = "Izquierda"
                    lado_en = "Left"

                # ── Predicción ──
                lm_data = extraer_landmarks(hand_lm)
                proba = modelo.predict_proba([lm_data])[0]
                clase_idx = np.argmax(proba)
                clase = modelo.classes_[clase_idx]
                confianza = proba[clase_idx] * 100

                # ── Dibujar mano ──
                dibujar_mano(frame, hand_lm)

                # ── Bounding box ──
                x1, y1, x2, y2 = obtener_bbox(frame, hand_lm)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ── Etiqueta encima del bbox (como en la imagen) ──
                etiqueta = f"{clase}: {confianza:.0f}%"
                (tw, th), _ = cv2.getTextSize(
                    etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1),
                              (0, 255, 0), -1)
                cv2.putText(frame, etiqueta, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                # ── Guardar info para barra inferior ──
                info_manos.append((lado_es, lado_en, clase, confianza))

        # ── Barra inferior con Izquierda / Derecha ──
        bar_h = 60
        cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)

        if info_manos:
            # Ordenar: izquierda primero
            info_manos.sort(key=lambda x: 0 if x[1] == "Left" else 1)
            col_w = w // len(info_manos)

            for idx, (lado_es, lado_en, clase, confianza) in enumerate(info_manos):
                x_base = idx * col_w + 10
                texto1 = f"{lado_en} / {lado_es}"
                texto2 = f"{clase} ({confianza:.0f}%)"
                cv2.putText(frame, texto1, (x_base, h - bar_h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(frame, texto2, (x_base, h - bar_h + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Sin manos detectadas", (10, h - bar_h + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)

        cv2.imshow("Traductor de Senas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar()