from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

with open("models/modelo_senas.pkl", "rb") as f:
    modelo = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        data = request.get_json()
        resultados = {"izquierda": None, "derecha": None}

        for lado in ["izquierda", "derecha"]:
            landmarks = data.get(lado)
            if landmarks and len(landmarks) == 63:
                lm_array = np.array(landmarks).reshape(1, -1)
                proba = modelo.predict_proba(lm_array)[0]
                clase_idx = np.argmax(proba)
                clase = modelo.classes_[clase_idx]
                confianza = round(float(proba[clase_idx]) * 100, 1)
                resultados[lado] = {"seña": clase, "confianza": confianza}

        return jsonify(resultados)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)