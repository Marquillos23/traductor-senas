import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── CONFIGURACIÓN ───────────────────────────────────────────
SEÑAS = ["hola", "bien", "mal", "te amo", "Que hora es"]
DATA_DIR = "data"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_senas.pkl")
# ─────────────────────────────────────────────────────────────

def cargar_datos():
    X = []
    y = []
    print("Cargando datos...")
    for seña in SEÑAS:
        archivo = os.path.join(DATA_DIR, seña, "landmarks.npy")
        if not os.path.exists(archivo):
            print(f"   ADVERTENCIA: No se encontro '{seña}'")
            continue
        datos = np.load(archivo)
        X.extend(datos)
        y.extend([seña] * len(datos))
        print(f"   '{seña}' — {len(datos)} muestras cargadas")
    return np.array(X), np.array(y)

def entrenar():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = cargar_datos()

    if len(X) == 0:
        print("No hay datos para entrenar. Ejecuta primero 1_recolectar_datos.py")
        return

    print(f"\nTotal de muestras: {len(X)}")

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")

    # Entrenar modelo
    print("\nEntrenando modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrecision del modelo: {accuracy * 100:.2f}%")
    print("\nReporte detallado:")
    print(classification_report(y_test, y_pred))

    # Guardar modelo
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    entrenar()