from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open("models/modelo_senas.pkl", "rb") as f:
    modelo = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.get_json()
    resultados = {}

    for lado, landmarks in data.items():
        if landmarks:
            lm_array = np.array(landmarks).reshape(1, -1)
            proba = modelo.predict_proba(lm_array)[0]
            clase_idx = np.argmax(proba)
            clase = modelo.classes_[clase_idx]
            confianza = round(float(proba[clase_idx]) * 100, 1)
            resultados[lado] = {"seña": clase, "confianza": confianza}
        else:
            resultados[lado] = None

    return jsonify(resultados)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)